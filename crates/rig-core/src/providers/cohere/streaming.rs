use crate::completion::{CompletionError, CompletionRequest};
use crate::http_client::HttpClientExt;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::providers::cohere::CompletionModel;
use crate::providers::cohere::completion::{
    CohereCompletionRequest, FinishReason, PROVIDER_NAME, Usage, map_finish_reason,
};
use crate::providers::internal::adapter::{AdapterOutput, WireAdapter, WireFrame, run_wire_stream};
use crate::providers::internal::wire;
use crate::streaming::{
    MintKind, PartId, RawStreamingChoice, RawStreamingResult, StreamFinal, ToolCallDeltaContent,
    ToolInputEnd, UnparseableToolInput,
};
use crate::telemetry::{CompletionOperation, CompletionSpanBuilder, SpanCombinator};

/// Cohere thinking deltas carry no id; a per-stream constant minted identity
/// keys their accumulation and can never reach a request.
const REASONING_ID: PartId = PartId::Minted {
    kind: MintKind::Reasoning,
    index: 0,
};
use crate::{json_utils, streaming};
use async_stream::stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tracing::{Level, enabled};
use tracing_futures::Instrument;

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
#[derive(Default)]
struct CohereAdapter {
    /// Wire id of the open tool call, when one is streaming. Only the wire
    /// identity is tracked here; fragment assembly, internal-id minting, and
    /// finalize policy live in the shared accumulator.
    current_tool_call: Option<String>,
    message_id: Option<String>,
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

                // Thinking deltas carry no wire id and never interleave with
                // other output mid-item; the per-stream constant lives in the
                // minted namespace so the accumulator's boundary bump and the
                // upstream-serialization guard both apply.
                if let Some(thinking) = &content.thinking
                    && !thinking.is_empty()
                {
                    out.push(Ok(RawStreamingChoice::ReasoningDelta {
                        id: REASONING_ID,
                        reasoning: thinking.clone(),
                    }));
                }

                if let Some(text) = &content.text
                    && !text.is_empty()
                {
                    out.push(Ok(RawStreamingChoice::Message(text.clone())));
                }
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

                out.push(Ok(RawStreamingChoice::ToolCallDelta {
                    id: PartId::wire(id.clone()),
                    content: ToolCallDeltaContent::Name(name),
                }));
                // `tool-call-start` may carry initial argument text; on the
                // wire it is empty, but any payload must still enter assembly.
                if !arguments.is_empty() {
                    out.push(Ok(RawStreamingChoice::ToolCallDelta {
                        id: PartId::wire(id),
                        content: ToolCallDeltaContent::Delta(arguments),
                    }));
                }
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
                    id: PartId::wire(id),
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
        let system_instructions = request.preamble.clone();
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

        if enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::streaming",
                "Cohere streaming completion input: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("/v2/chat")?
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        let event_source = GenericEventSource::new(self.client.clone(), req);

        // Transport layer: SSE events → `WireFrame`s. Byte splitting and
        // framing only — classification and policy live downstream.
        let transport = stream! {
            let mut event_source = Box::pin(event_source);
            while let Some(event_result) = event_source.next().await {
                match event_result {
                    Ok(Event::Open) => {
                        tracing::trace!("SSE connection opened");
                    }
                    Ok(Event::Message(message)) => {
                        let data_str = message.data.trim();
                        if data_str.is_empty() || data_str == "[DONE]" {
                            continue;
                        }
                        yield Ok(WireFrame::Text(data_str.to_owned()));
                    }
                    Err(crate::http_client::Error::StreamEnded) => {
                        break;
                    }
                    Err(err) => {
                        tracing::error!(?err, "SSE error");
                        yield Err(CompletionError::from_stream_transport(err));
                        break;
                    }
                }
            }
            // Ensure event source is closed when stream ends
            event_source.close();
        };

        let stream: RawStreamingResult<StreamingCompletionResponse> =
            Box::pin(run_wire_stream(transport, CohereAdapter::default()).instrument(span));

        Ok(stream)
    }

    pub(crate) async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse, CompletionError> {
        let stream = self.raw_stream(request).await?;
        let normalized = streaming::normalize_stream(stream, |response| Ok(response.into()));

        Ok(streaming::StreamingCompletionResponse::stream(
            PROVIDER_NAME,
            normalized,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cohere_client<H>(http_client: H) -> crate::providers::cohere::Client<H>
    where
        H: HttpClientExt,
    {
        crate::providers::cohere::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("client should build")
    }

    fn classify(data: &str) -> wire::WireEvent<StreamingEvent> {
        wire::classify_tagged_frame(data, "type", |event_type| {
            KNOWN_EVENT_TYPES.contains(&event_type)
        })
    }

    #[test]
    fn classify_known_event_decodes() {
        let frame = json!({
            "type": "content-delta",
            "delta": {"message": {"content": {"text": "hi"}}},
        })
        .to_string();
        assert!(matches!(
            classify(&frame),
            wire::WireEvent::Known(StreamingEvent::ContentDelta { .. })
        ));
    }

    #[test]
    fn classify_unknown_event_type_is_unknown() {
        let frame = json!({"type": "citation-start"}).to_string();
        assert!(matches!(
            classify(&frame),
            wire::WireEvent::Unknown { event_type, .. } if event_type == "citation-start"
        ));
    }

    #[test]
    fn classify_invalid_json_is_corrupt() {
        assert!(matches!(classify("{not json"), wire::WireEvent::Corrupt(_)));
    }

    #[test]
    fn classify_known_event_with_defective_payload_is_corrupt() {
        let frame = json!({"type": "content-delta", "delta": 42}).to_string();
        assert!(matches!(classify(&frame), wire::WireEvent::Corrupt(_)));
    }

    #[tokio::test]
    async fn stream_terminal_record_is_normalized() {
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel as _;
        use crate::streaming::StreamedAssistantContent;
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        let sse_bytes = bytes::Bytes::from(
            [
                r#"{"type":"message-start","id":"msg_1"}"#,
                r#"{"type":"content-delta","delta":{"message":{"content":{"text":"hi"}}}}"#,
                r#"{"type":"message-end","delta":{"finish_reason":"MAX_TOKENS","usage":{"tokens":{"input_tokens":10,"output_tokens":4}}}}"#,
            ]
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>(),
        );

        let client = cohere_client(MockStreamingClient { sse_bytes });
        let model = client.completion_model(crate::providers::cohere::COMMAND_R);
        let request = model.completion_request("hello").build();

        let mut stream = crate::completion::CompletionModel::stream(&model, request)
            .await
            .expect("stream should open");

        let mut terminal = None;
        while let Some(item) = stream.next().await {
            if let StreamedAssistantContent::Final(final_response) =
                item.expect("stream item should be Ok")
            {
                terminal = Some(final_response);
            }
        }

        let terminal = terminal.expect("stream should yield a terminal record");
        assert_eq!(terminal.provider, PROVIDER_NAME);
        assert_eq!(terminal.response_id.as_deref(), Some("msg_1"));
        assert_eq!(terminal.message_id, None);
        assert_eq!(
            terminal.finish_reason,
            Some(crate::completion::FinishReason::Length)
        );
        assert_eq!(terminal.usage.input_tokens, 10);
        assert_eq!(terminal.usage.output_tokens, 4);
        assert_eq!(terminal.usage.total_tokens, 14);
        // Cohere's stream never names the model.
        assert_eq!(terminal.model, None);
    }

    #[tokio::test]
    async fn truncated_stream_does_not_synthesize_a_terminal_record() {
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel as _;
        use crate::streaming::StreamedAssistantContent;
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        // No `message-end`: the stream was cut off mid-response.
        let sse_bytes = bytes::Bytes::from(
            [
                r#"{"type":"message-start","id":"msg_1"}"#,
                r#"{"type":"content-delta","delta":{"message":{"content":{"text":"hi"}}}}"#,
            ]
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>(),
        );

        let client = cohere_client(MockStreamingClient { sse_bytes });
        let model = client.completion_model(crate::providers::cohere::COMMAND_R);
        let request = model.completion_request("hello").build();

        let mut stream = crate::completion::CompletionModel::stream(&model, request)
            .await
            .expect("stream should open");

        let mut texts = Vec::new();
        let mut saw_terminal = false;
        while let Some(item) = stream.next().await {
            match item.expect("stream item should be Ok") {
                StreamedAssistantContent::Text(text) => texts.push(text.text),
                StreamedAssistantContent::Final(_) => saw_terminal = true,
                _ => {}
            }
        }

        assert_eq!(texts, ["hi"]);
        assert!(
            !saw_terminal,
            "EOF without message-end must not synthesize a terminal record"
        );
        assert!(stream.response.is_none());
    }

    #[tokio::test]
    async fn malformed_frame_is_surfaced_and_the_terminal_still_arrives() {
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel as _;
        use crate::streaming::StreamedAssistantContent;
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        // A malformed frame between valid content and the genuine terminal
        // must surface as an `Err` item without derailing the rest of the
        // stream.
        let sse_bytes = bytes::Bytes::from(
            [
                r#"{"type":"message-start","id":"msg_1"}"#,
                r#"{"type":"content-delta","delta":{"message":{"content":{"text":"hi"}}}}"#,
                "{not json",
                r#"{"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":10,"output_tokens":4}}}}"#,
            ]
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>(),
        );

        let client = cohere_client(MockStreamingClient { sse_bytes });
        let model = client.completion_model(crate::providers::cohere::COMMAND_R);
        let request = model.completion_request("hello").build();

        let mut stream = crate::completion::CompletionModel::stream(&model, request)
            .await
            .expect("stream should open");

        let mut texts = Vec::new();
        let mut saw_error = false;
        let mut terminal = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(StreamedAssistantContent::Text(text)) => texts.push(text.text),
                Ok(StreamedAssistantContent::Final(final_response)) => {
                    terminal = Some(final_response)
                }
                Ok(_) => {}
                Err(_) => saw_error = true,
            }
        }

        assert_eq!(texts, ["hi"]);
        assert!(saw_error, "the malformed frame must reach the consumer");
        let terminal = terminal.expect("the genuine terminal record must still arrive");
        assert_eq!(terminal.usage.input_tokens, 10);
        assert_eq!(terminal.usage.output_tokens, 4);
    }

    #[tokio::test]
    async fn known_event_with_malformed_field_is_surfaced_as_an_error() {
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel as _;
        use crate::streaming::StreamedAssistantContent;
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        // A known `type` whose payload fails the full parse (text should be a
        // string) is a data-level defect, not a forward-compatibility event.
        let sse_bytes = bytes::Bytes::from(
            [
                r#"{"type":"message-start","id":"msg_1"}"#,
                r#"{"type":"content-delta","delta":{"message":{"content":{"text":42}}}}"#,
                r#"{"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":10,"output_tokens":4}}}}"#,
            ]
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>(),
        );

        let client = cohere_client(MockStreamingClient { sse_bytes });
        let model = client.completion_model(crate::providers::cohere::COMMAND_R);
        let request = model.completion_request("hello").build();

        let mut stream = crate::completion::CompletionModel::stream(&model, request)
            .await
            .expect("stream should open");

        let mut saw_error = false;
        let mut terminal = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(StreamedAssistantContent::Final(final_response)) => {
                    terminal = Some(final_response)
                }
                Ok(_) => {}
                Err(err) => {
                    assert!(
                        matches!(err, CompletionError::JsonError(_)),
                        "expected a JSON parse error item, got {err:?}"
                    );
                    saw_error = true;
                }
            }
        }

        assert!(
            saw_error,
            "a known event with a malformed field must surface an error item"
        );
        let terminal = terminal.expect("the genuine terminal record must still arrive");
        assert_eq!(terminal.usage.input_tokens, 10);
    }

    #[tokio::test]
    async fn unknown_event_type_is_skipped_and_the_terminal_still_arrives() {
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel as _;
        use crate::streaming::StreamedAssistantContent;
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        // An invented `type` is an event this client doesn't know yet: it is
        // skipped for forward compatibility, not surfaced as an error.
        let sse_bytes = bytes::Bytes::from(
            [
                r#"{"type":"message-start","id":"msg_1"}"#,
                r#"{"type":"citation-start","delta":{"whatever":true}}"#,
                r#"{"type":"content-delta","delta":{"message":{"content":{"text":"hi"}}}}"#,
                r#"{"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":10,"output_tokens":4}}}}"#,
            ]
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>(),
        );

        let client = cohere_client(MockStreamingClient { sse_bytes });
        let model = client.completion_model(crate::providers::cohere::COMMAND_R);
        let request = model.completion_request("hello").build();

        let mut stream = crate::completion::CompletionModel::stream(&model, request)
            .await
            .expect("stream should open");

        let mut texts = Vec::new();
        let mut terminal = None;
        while let Some(item) = stream.next().await {
            match item.expect("unknown event types must not surface errors") {
                StreamedAssistantContent::Text(text) => texts.push(text.text),
                StreamedAssistantContent::Final(final_response) => terminal = Some(final_response),
                _ => {}
            }
        }

        assert_eq!(texts, ["hi"]);
        let terminal = terminal.expect("the genuine terminal record must still arrive");
        assert_eq!(terminal.usage.output_tokens, 4);
    }

    #[tokio::test]
    async fn message_end_without_delta_still_emits_the_terminal_record() {
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel as _;
        use crate::streaming::StreamedAssistantContent;
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        // `message-end` with no payload is still the provider completing the
        // turn; the terminal record arrives with default usage.
        let sse_bytes = bytes::Bytes::from(
            [
                r#"{"type":"message-start","id":"msg_1"}"#,
                r#"{"type":"content-delta","delta":{"message":{"content":{"text":"hi"}}}}"#,
                r#"{"type":"message-end"}"#,
            ]
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>(),
        );

        let client = cohere_client(MockStreamingClient { sse_bytes });
        let model = client.completion_model(crate::providers::cohere::COMMAND_R);
        let request = model.completion_request("hello").build();

        let mut stream = crate::completion::CompletionModel::stream(&model, request)
            .await
            .expect("stream should open");

        let mut texts = Vec::new();
        let mut terminal = None;
        while let Some(item) = stream.next().await {
            match item.expect("stream item should be Ok") {
                StreamedAssistantContent::Text(text) => texts.push(text.text),
                StreamedAssistantContent::Final(final_response) => terminal = Some(final_response),
                _ => {}
            }
        }

        assert_eq!(texts, ["hi"]);
        let terminal = terminal.expect("message-end without a delta is still the terminal");
        assert_eq!(terminal.usage, crate::completion::Usage::default());
        assert_eq!(terminal.finish_reason, None);
        assert_eq!(terminal.response_id.as_deref(), Some("msg_1"));
    }

    #[tokio::test]
    async fn thinking_deltas_aggregate_into_one_reasoning_part_before_the_text() {
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel as _;
        use crate::message::AssistantContent;
        use crate::streaming::StreamedAssistantContent;
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        // Cohere v2 reasoning models stream `content-delta` frames carrying
        // `thinking` before the answer's `text` frames (documented `thinking`
        // deltas; #2258 F8 — previously these fell through the `text` guard
        // and the thought text was lost).
        let sse_bytes = bytes::Bytes::from(
            [
                r#"{"type":"message-start","id":"msg_1"}"#,
                r#"{"type":"content-delta","delta":{"message":{"content":{"thinking":"step one, "}}}}"#,
                r#"{"type":"content-delta","delta":{"message":{"content":{"thinking":"step two"}}}}"#,
                r#"{"type":"content-delta","delta":{"message":{"content":{"text":"answer"}}}}"#,
                r#"{"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":10,"output_tokens":4}}}}"#,
            ]
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>(),
        );

        let client = cohere_client(MockStreamingClient { sse_bytes });
        let model = client.completion_model(crate::providers::cohere::COMMAND_R);
        let request = model.completion_request("hello").build();

        let mut stream = crate::completion::CompletionModel::stream(&model, request)
            .await
            .expect("stream should open");

        let mut reasoning_deltas = Vec::new();
        while let Some(item) = stream.next().await {
            if let StreamedAssistantContent::ReasoningDelta { reasoning, .. } =
                item.expect("stream item should be Ok")
            {
                reasoning_deltas.push(reasoning);
            }
        }
        assert_eq!(reasoning_deltas, ["step one, ", "step two"]);

        let parts: Vec<_> = stream.choice.iter().cloned().collect();
        assert_eq!(parts.len(), 2, "one reasoning part, one text part");
        assert!(matches!(
            parts.first(),
            Some(AssistantContent::Reasoning(reasoning))
                if reasoning.content.iter().any(|content| matches!(
                    content,
                    crate::message::ReasoningContent::Text { text, .. }
                        if text == "step one, step two"
                ))
        ));
        assert!(matches!(
            parts.get(1),
            Some(AssistantContent::Text(text)) if text.text == "answer"
        ));
    }

    #[tokio::test]
    async fn errored_stream_does_not_synthesize_a_terminal_record() {
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel as _;
        use crate::streaming::StreamedAssistantContent;
        use crate::test_utils::HttpErrorStreamingClient;
        use futures::StreamExt;

        let client = cohere_client(HttpErrorStreamingClient::new(
            http::StatusCode::TOO_MANY_REQUESTS,
            r#"{"message":"slow down"}"#,
        ));
        let model = client.completion_model(crate::providers::cohere::COMMAND_R);
        let request = model.completion_request("hello").build();

        let mut stream = crate::completion::CompletionModel::stream(&model, request)
            .await
            .expect("stream should open");

        let mut saw_error = false;
        let mut saw_terminal = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(StreamedAssistantContent::Final(_)) => saw_terminal = true,
                Ok(_) => {}
                Err(_) => saw_error = true,
            }
        }

        assert!(saw_error, "the transport failure must reach the consumer");
        assert!(
            !saw_terminal,
            "a failed stream must not be reported as a successful, zero-usage completion"
        );
        assert!(stream.response.is_none());
    }

    #[test]
    fn test_message_content_delta_deserialization() {
        let json = json!({
            "type": "content-delta",
            "delta": {
                "message": {
                    "content": {
                        "text": "Hello world"
                    }
                }
            }
        });

        let event: StreamingEvent = serde_json::from_value(json).unwrap();
        match event {
            StreamingEvent::ContentDelta { delta } => {
                assert!(delta.is_some());
                let message = delta.unwrap().message.unwrap();
                let content = message.content.unwrap();
                assert_eq!(content.text, Some("Hello world".to_string()));
            }
            _ => panic!("Expected ContentDelta"),
        }
    }

    #[test]
    fn test_tool_call_start_deserialization() {
        let json = json!({
            "type": "tool-call-start",
            "delta": {
                "message": {
                    "tool_calls": {
                        "id": "call_123",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{"
                        }
                    }
                }
            }
        });

        let event: StreamingEvent = serde_json::from_value(json).unwrap();
        match event {
            StreamingEvent::ToolCallStart { delta } => {
                assert!(delta.is_some());
                let tool_call = delta.unwrap().message.unwrap().tool_calls.unwrap();
                assert_eq!(tool_call.id, Some("call_123".to_string()));
                assert_eq!(
                    tool_call.function.unwrap().name,
                    Some("get_weather".to_string())
                );
            }
            _ => panic!("Expected ToolCallStart"),
        }
    }

    #[test]
    fn test_tool_call_delta_deserialization() {
        let json = json!({
            "type": "tool-call-delta",
            "delta": {
                "message": {
                    "tool_calls": {
                        "function": {
                            "arguments": "\"location\""
                        }
                    }
                }
            }
        });

        let event: StreamingEvent = serde_json::from_value(json).unwrap();
        match event {
            StreamingEvent::ToolCallDelta { delta } => {
                assert!(delta.is_some());
                let tool_call = delta.unwrap().message.unwrap().tool_calls.unwrap();
                let function = tool_call.function.unwrap();
                assert_eq!(function.arguments, Some("\"location\"".to_string()));
            }
            _ => panic!("Expected ToolCallDelta"),
        }
    }

    #[test]
    fn test_tool_call_end_deserialization() {
        let json = json!({
            "type": "tool-call-end"
        });

        let event: StreamingEvent = serde_json::from_value(json).unwrap();
        match event {
            StreamingEvent::ToolCallEnd => {
                // Success
            }
            _ => panic!("Expected ToolCallEnd"),
        }
    }

    #[test]
    fn test_message_end_with_usage_deserialization() {
        let json = json!({
            "type": "message-end",
            "delta": {
                "usage": {
                    "tokens": {
                        "input_tokens": 100,
                        "output_tokens": 50
                    }
                }
            }
        });

        let event: StreamingEvent = serde_json::from_value(json).unwrap();
        match event {
            StreamingEvent::MessageEnd { delta } => {
                assert!(delta.is_some());
                let usage = delta.unwrap().usage.unwrap();
                let tokens = usage.tokens.unwrap();
                assert_eq!(tokens.input_tokens, Some(100.0));
                assert_eq!(tokens.output_tokens, Some(50.0));
            }
            _ => panic!("Expected MessageEnd"),
        }
    }

    #[test]
    fn test_streaming_event_order() {
        // Test that a typical sequence of events deserializes correctly
        let events = vec![
            json!({"type": "message-start"}),
            json!({"type": "content-start"}),
            json!({
                "type": "content-delta",
                "delta": {
                    "message": {
                        "content": {
                            "text": "Sure, "
                        }
                    }
                }
            }),
            json!({
                "type": "content-delta",
                "delta": {
                    "message": {
                        "content": {
                            "text": "I can help with that."
                        }
                    }
                }
            }),
            json!({"type": "content-end"}),
            json!({"type": "tool-plan"}),
            json!({
                "type": "tool-call-start",
                "delta": {
                    "message": {
                        "tool_calls": {
                            "id": "call_abc",
                            "function": {
                                "name": "search",
                                "arguments": ""
                            }
                        }
                    }
                }
            }),
            json!({
                "type": "tool-call-delta",
                "delta": {
                    "message": {
                        "tool_calls": {
                            "function": {
                                "arguments": "{\"query\":"
                            }
                        }
                    }
                }
            }),
            json!({
                "type": "tool-call-delta",
                "delta": {
                    "message": {
                        "tool_calls": {
                            "function": {
                                "arguments": "\"Rust\"}"
                            }
                        }
                    }
                }
            }),
            json!({"type": "tool-call-end"}),
            json!({
                "type": "message-end",
                "delta": {
                    "usage": {
                        "tokens": {
                            "input_tokens": 50,
                            "output_tokens": 25
                        }
                    }
                }
            }),
        ];

        for (i, event_json) in events.iter().enumerate() {
            let result = serde_json::from_value::<StreamingEvent>(event_json.clone());
            assert!(
                result.is_ok(),
                "Failed to deserialize event at index {}: {:?}",
                i,
                result.err()
            );
        }
    }
}
