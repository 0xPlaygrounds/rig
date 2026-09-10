use crate::completion::{CompletionError, CompletionRequest};
use crate::http_client::HttpClientExt;
use crate::http_client::sse::GenericEventSource;
use crate::providers::cohere::CompletionModel;
use crate::providers::cohere::completion::{
    CohereCompletionRequest, FinishReason, PROVIDER_NAME, Usage, map_finish_reason,
};
use crate::providers::internal::adapter::{AdapterOutput, WireAdapter, WireFrame};
use crate::providers::internal::sse_transport::{
    OpenLog, SseTransportOptions, open_wire_stream, skip_blank_and_done,
};
use crate::providers::internal::wire;
use crate::streaming::{
    MintKind, RawStreamingChoice, RawStreamingResult, StreamFinal, StreamPartId,
    ToolCallDeltaContent, ToolInputEnd, UnparseableToolInput,
};
use crate::telemetry::{CompletionOperation, CompletionSpanBuilder, SpanCombinator};

/// Cohere thinking deltas carry no id; a per-stream constant minted identity
/// keys their accumulation and can never reach a request.
const REASONING_ID: StreamPartId = StreamPartId::minted(MintKind::Reasoning, 0);
use crate::{json_utils, streaming};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "type")]
enum StreamingEvent {
    MessageStart {
        #[serde(default)]
        id: Option<String>,
    },
    ContentStart,
    ContentDelta {
        delta: Option<Delta>,
    },
    ContentEnd,
    ToolPlan,
    ToolCallStart {
        delta: Option<Delta>,
    },
    ToolCallDelta {
        delta: Option<Delta>,
    },
    ToolCallEnd,
    MessageEnd {
        delta: Option<MessageEndDelta>,
    },
}

/// The kebab-case `type` values [`StreamingEvent`] can deserialize. A frame
/// whose `type` is in this set but fails the full parse has a data-level
/// defect and is surfaced as an `Err` item; a `type` outside this set is an
/// event this client doesn't know yet and is skipped.
const KNOWN_EVENT_TYPES: [&str; 9] = [
    "message-start",
    "content-start",
    "content-delta",
    "content-end",
    "tool-plan",
    "tool-call-start",
    "tool-call-delta",
    "tool-call-end",
    "message-end",
];

#[derive(Debug, Deserialize)]
struct MessageContentDelta {
    text: Option<String>,
    /// Cohere v2 reasoning models stream thought text as `content-delta`
    /// frames whose content carries `thinking` instead of `text`.
    thinking: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MessageToolFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MessageToolCallDelta {
    id: Option<String>,
    function: Option<MessageToolFunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct MessageDelta {
    content: Option<MessageContentDelta>,
    tool_calls: Option<MessageToolCallDelta>,
}

#[derive(Debug, Deserialize)]
struct Delta {
    message: Option<MessageDelta>,
}

#[derive(Debug, Deserialize)]
struct MessageEndDelta {
    usage: Option<Usage>,
    #[serde(default)]
    finish_reason: Option<FinishReason>,
}

/// Cohere's terminal stream record, kept provider-native for
/// [`CompletionModel::raw_stream`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamingCompletionResponse {
    pub usage: Option<Usage>,
    /// Cohere's own `finish_reason` from the `message-end` event, when reported.
    #[serde(default)]
    pub finish_reason: Option<FinishReason>,
    /// The `message-start` event's message identifier, when reported.
    #[serde(default)]
    pub message_id: Option<String>,
}

impl From<&StreamingCompletionResponse> for crate::completion::Usage {
    fn from(response: &StreamingCompletionResponse) -> crate::completion::Usage {
        response
            .usage
            .as_ref()
            .map(crate::completion::Usage::from)
            .unwrap_or_default()
    }
}

impl From<StreamingCompletionResponse> for StreamFinal {
    fn from(response: StreamingCompletionResponse) -> StreamFinal {
        // Cohere's streaming events carry no model identifier, so the
        // normalized `model` stays unset.
        StreamFinal::new(PROVIDER_NAME, crate::completion::Usage::from(&response))
            .with_optional_finish_reason(response.finish_reason.as_ref().map(map_finish_reason))
            .with_optional_response_id(response.message_id)
    }
}

/// The Cohere v2 chat SSE wire as a [`WireAdapter`].
///
/// Holds the per-stream state (open tool call, message id); frame-triage
/// policy (warn-skip `Unknown` for forward compatibility, in-band `Err` on
/// `Corrupt` so a later genuine `message-end` can still complete the stream)
/// lives in [`run_wire_stream`], not here.
struct CohereAdapter {
    /// Wire id of the open tool call, when one is streaming. Only the wire
    /// identity is tracked here; fragment assembly, internal-id minting, and
    /// finalize policy live in the shared accumulator.
    current_tool_call: Option<String>,
    message_id: Option<String>,
    /// Owns the constant-key reasoning lifecycle — the boundary end this
    /// wire never announces is derived, not hand-rolled here.
    reasoning: crate::providers::internal::chunk_lifecycle::MintedReasoningLifecycle,
}

impl Default for CohereAdapter {
    fn default() -> Self {
        Self {
            current_tool_call: None,
            message_id: None,
            reasoning: crate::providers::internal::chunk_lifecycle::MintedReasoningLifecycle::new(
                REASONING_ID,
            ),
        }
    }
}

impl WireAdapter for CohereAdapter {
    type Frame = WireFrame;
    type Event = StreamingEvent;
    type Response = StreamingCompletionResponse;

    fn classify(&self, frame: WireFrame) -> wire::WireEvent<StreamingEvent> {
        wire::classify_tagged_frame(&frame.as_str(), "type", |event_type| {
            KNOWN_EVENT_TYPES.contains(&event_type)
        })
    }

    fn interpret(&mut self, event: StreamingEvent, out: &mut AdapterOutput<Self::Response>) {
        match event {
            StreamingEvent::MessageStart { id: Some(id) } => {
                self.message_id = Some(id);
            }

            StreamingEvent::ContentDelta { delta: Some(delta) } => {
                let Some(message) = &delta.message else {
                    return;
                };
                let Some(content) = &message.content else {
                    return;
                };

                // Declare what the delta carried (thinking merges under the
                // per-stream constant minted key); the shared lifecycle
                // derives the canonical sequence, boundary end included.
                self.reasoning.emit_chunk(
                    crate::providers::internal::chunk_lifecycle::ChunkParts {
                        reasoning: content.thinking.clone(),
                        reasoning_signature: None,
                        text: content.text.clone(),
                        tool_events: Vec::new(),
                    },
                    out,
                );
            }

            StreamingEvent::MessageEnd { delta } => {
                // `message-end` is the genuine terminal even when its optional
                // payload is absent; usage and finish reason then default. The
                // driver stops consuming after the terminal record.
                let span = tracing::Span::current();
                let (usage, finish_reason) = match delta {
                    Some(delta) => (delta.usage, delta.finish_reason),
                    None => (None, None),
                };
                let recorded_usage = usage
                    .as_ref()
                    .map(crate::completion::Usage::from)
                    .unwrap_or_default();
                span.record_token_usage(&recorded_usage);
                out.push(Ok(RawStreamingChoice::FinalResponse(
                    StreamingCompletionResponse {
                        usage,
                        finish_reason,
                        message_id: self.message_id.take(),
                    },
                )));
            }

            StreamingEvent::ToolCallStart { delta: Some(delta) } => {
                let Some(message) = &delta.message else {
                    return;
                };
                let Some(tool_calls) = &message.tool_calls else {
                    return;
                };
                let Some(id) = tool_calls.id.clone() else {
                    return;
                };
                let Some(function) = &tool_calls.function else {
                    return;
                };
                let Some(name) = function.name.clone() else {
                    return;
                };
                let Some(arguments) = function.arguments.clone() else {
                    return;
                };

                self.current_tool_call = Some(id.clone());

                let mut tool_events = vec![RawStreamingChoice::ToolCallDelta {
                    id: StreamPartId::wire(id.clone()),
                    content: ToolCallDeltaContent::Name(name),
                }];
                // `tool-call-start` may carry initial argument text; on the
                // wire it is empty, but any payload must still enter assembly.
                if !arguments.is_empty() {
                    tool_events.push(RawStreamingChoice::ToolCallDelta {
                        id: StreamPartId::wire(id),
                        content: ToolCallDeltaContent::Delta(arguments),
                    });
                }
                // Tool content interleaving an open thinking block: the
                // shared lifecycle synthesizes the boundary end.
                self.reasoning.emit_chunk(
                    crate::providers::internal::chunk_lifecycle::ChunkParts {
                        reasoning: None,
                        reasoning_signature: None,
                        text: None,
                        tool_events,
                    },
                    out,
                );
            }

            StreamingEvent::ToolCallDelta { delta: Some(delta) } => {
                let Some(message) = &delta.message else {
                    return;
                };
                let Some(tool_calls) = &message.tool_calls else {
                    return;
                };
                let Some(function) = &tool_calls.function else {
                    return;
                };
                let Some(arguments) = function.arguments.clone() else {
                    return;
                };

                // A delta with no open call has nothing to extend; skip it, as
                // the wire never starts a call mid-delta.
                let Some(id) = self.current_tool_call.clone() else {
                    return;
                };

                // Emit the delta so UI can show progress
                out.push(Ok(RawStreamingChoice::ToolCallDelta {
                    id: StreamPartId::wire(id),
                    content: ToolCallDeltaContent::Delta(arguments),
                }));
            }

            StreamingEvent::ToolCallEnd => {
                let Some(id) = self.current_tool_call.take() else {
                    return;
                };
                // Unparseable assembled input drops in the accumulator,
                // matching the old skip.
                out.push(Ok(RawStreamingChoice::ToolInputEnd(ToolInputEnd::new(
                    id,
                    UnparseableToolInput::Drop,
                ))));
            }

            _ => {}
        }
    }

    fn finish(&mut self, _out: &mut AdapterOutput<Self::Response>) {
        // Only Cohere's `message-end` event counts as the provider completing
        // the turn. A stream that reached EOF without it (truncation) has no
        // terminal record to report; synthesizing one would present a partial
        // turn as a successful, zero-usage completion.
    }
}

impl<T> CompletionModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    /// Open a stream whose terminal record stays Cohere-native.
    ///
    /// This is the escape hatch for Cohere's own terminal payload; it shares the
    /// request builder, transport, telemetry, and error handling with
    /// [`CompletionModel::stream`](crate::completion::CompletionModel::stream),
    /// which calls it and normalizes the terminal record once through
    /// [`streaming::normalize_stream`] — one network request either way.
    pub async fn raw_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<RawStreamingResult<StreamingCompletionResponse>, CompletionError> {
        let system_instructions = request.system_instructions().map(str::to_owned);
        let record_telemetry_content = request.record_telemetry_content;
        let mut request = CohereCompletionRequest::try_from((self.model.as_ref(), request))?;
        let span = CompletionSpanBuilder::new(
            PROVIDER_NAME,
            &request.model,
            CompletionOperation::ChatStreaming,
        )
        .system_instructions(system_instructions.as_deref(), record_telemetry_content)
        .build();

        let params = json_utils::merge(
            request.additional_params.unwrap_or(serde_json::json!({})),
            serde_json::json!({"stream": true}),
        );

        request.additional_params = Some(params);

        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Streaming,
            "Cohere streaming completion input",
            &request,
        );

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("/v2/chat")?
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        Ok(open_wire_stream(
            GenericEventSource::new(self.client.clone(), req),
            SseTransportOptions {
                open_log: OpenLog::Trace,
                stream_ended_is_error: false,
                log_transport_errors: true,
            },
            |data: String| skip_blank_and_done(&data),
            CohereAdapter::default(),
            span,
        ))
    }

    pub(crate) async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse, CompletionError> {
        let stream = self.raw_stream(request).await?;
        let normalized =
            streaming::normalize_stream(stream, |response: StreamingCompletionResponse| {
                Ok(response.into())
            });

        Ok(streaming::StreamingCompletionResponse::stream(
            PROVIDER_NAME,
            normalized,
        ))
    }
}

#[cfg(test)]
mod tests;
