use serde::{Deserialize, Serialize};

use super::completion::gemini_api_types::{
    ContentCandidate, FinishReason, Part, PartKind, UsageMetadata, map_finish_reason,
};
use super::completion::{
    CompletionModel, PROVIDER_NAME, create_request_body, function_call_finish_reason_error,
    resolve_request_model, streaming_endpoint,
};
use crate::completion::{CompletionError, CompletionRequest};
use crate::http_client::HttpClientExt;
use crate::http_client::sse::GenericEventSource;
use crate::providers::internal::adapter::{AdapterOutput, WireAdapter, WireFrame};
use crate::providers::internal::sse_transport::{
    OpenLog, SseTransportOptions, open_wire_stream, skip_blank_frames,
};
use crate::providers::internal::wire::{self, WireEvent};
use crate::streaming;
use crate::telemetry::{CompletionOperation, CompletionSpanBuilder, SpanCombinator};

/// Part-kind interpretation shared by the Gemini wires whose payloads
/// coincide: REST `streamGenerateContent` and the Interactions API both
/// deliver whole function calls and identity-less thought fragments.
pub(crate) mod shared_parts {
    use serde_json::Value;

    use crate::streaming::{MintKind, RawStreamingChoice, RawStreamingToolCall, StreamPartId};

    /// Gemini thought parts carry no id or block boundaries; a per-stream
    /// constant minted identity keeps all thought deltas merging into one
    /// item, and the core accumulator's minted-id boundary splits items
    /// around other output. Minted, so it can never reach a request.
    pub(crate) const REASONING_ID: StreamPartId = StreamPartId::minted(MintKind::Reasoning, 0);

    /// A whole function-call part as a canonical tool call (Gemini never
    /// streams arguments incrementally).
    pub(crate) fn function_call<R>(
        name: String,
        args: Value,
        wire_id: Option<String>,
        signature: Option<String>,
        tool_ids: &mut crate::streaming::SyntheticIds,
    ) -> RawStreamingChoice<R> {
        // Never fabricate the identifier that travels upstream: the wire's
        // own id (when Gemini supplies one) is both the part identity and
        // the correlation id; an id-less call keys the stream by a minted
        // identity — counted up per stream, so two id-less calls never
        // collide on one key — and replays with the id absent. The tool
        // *name* is never an identity — two calls to the same tool in one
        // turn must stay distinct, correlated by order and by the
        // rig-internal call id.
        let tool_id = wire_id.and_then(crate::streaming::WireId::new);
        let id = tool_id
            .as_ref()
            .map_or_else(|| tool_ids.mint(), |id| StreamPartId::wire(id.as_str()));
        let tool_call = RawStreamingToolCall {
            id,
            tool_id,
            internal_call_id: crate::id::InternalCallId::new(),
            // Gemini is a single-identifier wire: its one id travels as
            // `tool_id` and `call_id` stays unset. Filling both from the same
            // id would take the dual-wire arm downstream and fabricate an
            // item id Gemini never issued.
            call_id: None,
            name,
            arguments: args,
            signature,
            additional_params: None,
        };
        RawStreamingChoice::ToolCall(tool_call)
    }
}

/// The usage record on a `streamGenerateContent` chunk.
///
/// Identical to the unary wire's [`UsageMetadata`] — Gemini sends the same
/// `usageMetadata` object on streaming frames — so the streaming name is an
/// alias, not a second declaration that can drift from it.
pub type PartialUsage = UsageMetadata;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamGenerateContentResponse {
    pub response_id: Option<String>,
    /// Candidate responses from the model.
    #[serde(default)]
    pub candidates: Vec<ContentCandidate>,
    pub model_version: Option<String>,
    pub usage_metadata: Option<PartialUsage>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamingCompletionResponse {
    pub usage_metadata: PartialUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
}

impl From<&StreamingCompletionResponse> for crate::completion::Usage {
    fn from(value: &StreamingCompletionResponse) -> crate::completion::Usage {
        (&value.usage_metadata).into()
    }
}

impl From<StreamingCompletionResponse> for crate::completion::Usage {
    fn from(value: StreamingCompletionResponse) -> crate::completion::Usage {
        (&value).into()
    }
}

/// Normalize Gemini's terminal streaming record.
///
/// Infallible in practice, but stated as a `Result` because
/// [`crate::streaming::normalize_stream`] maps terminal records through a
/// fallible closure.
fn map_stream_final(
    response: StreamingCompletionResponse,
) -> Result<streaming::StreamFinal, CompletionError> {
    let finish_reason = response.finish_reason.as_ref().and_then(map_finish_reason);

    Ok(
        streaming::StreamFinal::new(PROVIDER_NAME, (&response.usage_metadata).into())
            .with_optional_finish_reason(finish_reason)
            .with_optional_response_id(response.response_id)
            .with_optional_model(response.model_version),
    )
}

fn tool_protocol_finish_reason_error(choice: &ContentCandidate) -> Option<CompletionError> {
    let reason = choice.finish_reason.as_ref()?;
    function_call_finish_reason_error(reason, choice.finish_message.as_deref())
}

/// The recognizability markers of a `streamGenerateContent` chunk: every
/// genuine frame carries `candidates` and/or `usageMetadata`. A frame with
/// either must fully decode (else `Corrupt`); other JSON is `Unknown`.
const RECOGNIZABLE_CHUNK_KEYS: &[&str] = &["candidates", "usageMetadata"];

/// The Gemini REST (`streamGenerateContent`) SSE wire as a [`WireAdapter`].
///
/// Holds the per-stream state (thought-restatement buffer, terminal
/// metadata); frame-triage policy lives in
/// [`run_wire_stream`](crate::providers::internal::adapter::run_wire_stream),
/// not here.
struct GeminiRestAdapter {
    /// Owns the constant-key thought lifecycle — the ends this wire never
    /// announces are derived by the shared lifecycle, not hand-rolled here.
    /// All accumulation lives in the shared accumulator.
    reasoning: crate::providers::internal::chunk_lifecycle::MintedReasoningLifecycle,
    /// Per-stream minter for id-less tool-call keys — a fresh key per call,
    /// so two id-less calls in one turn never collide on one identity.
    tool_ids: crate::streaming::SyntheticIds,
    final_usage: Option<PartialUsage>,
    final_finish_reason: Option<FinishReason>,
    final_finish_message: Option<String>,
    final_model_version: Option<String>,
    final_response_id: Option<String>,
    /// The provider sent a `finishReason` on some chunk.
    ///
    /// Gemini's `streamGenerateContent` sends an *intermediate* `finishReason`
    /// when a built-in tool runs a round — a recorded code-execution stream
    /// reads `[executableCode] [codeExecutionResult] [executableCode +
    /// finishReason:STOP] [codeExecutionResult] [text] [text +
    /// finishReason:STOP]` — so a `finishReason` chunk is not, on this wire, the
    /// provider completing the turn. The terminal record is therefore deferred
    /// to EOF (see [`WireAdapter::finish`], which names exactly this case);
    /// pushing it on the first such chunk made the driver stop reading there
    /// and silently drop the model's whole answer while still reporting a
    /// successful `STOP`.
    saw_finish_reason: bool,
    /// A tool-protocol finish reason ended the turn; later frames are dead —
    /// the provider aborted, and interpreting more output (or a terminal)
    /// would dress the failure up as a completed turn.
    failed: bool,
}

impl Default for GeminiRestAdapter {
    fn default() -> Self {
        Self {
            reasoning: crate::providers::internal::chunk_lifecycle::MintedReasoningLifecycle::new(
                shared_parts::REASONING_ID,
            ),
            tool_ids: crate::streaming::SyntheticIds::tool(),
            final_usage: None,
            final_finish_reason: None,
            final_finish_message: None,
            final_model_version: None,
            final_response_id: None,
            saw_finish_reason: false,
            failed: false,
        }
    }
}

impl WireAdapter for GeminiRestAdapter {
    type Frame = WireFrame;
    type Event = StreamGenerateContentResponse;
    type Response = StreamingCompletionResponse;

    fn classify(&self, frame: WireFrame) -> WireEvent<StreamGenerateContentResponse> {
        wire::classify_marker_keyed_frame(&frame.as_str(), RECOGNIZABLE_CHUNK_KEYS)
    }

    fn interpret(
        &mut self,
        data: StreamGenerateContentResponse,
        out: &mut AdapterOutput<Self::Response>,
    ) {
        if self.failed {
            return;
        }

        let span = tracing::Span::current();
        if let Some(response_id) = data.response_id.as_deref() {
            span.record("gen_ai.response.id", response_id);
            self.final_response_id = Some(response_id.to_owned());
        }
        if let Some(model_version) = &data.model_version {
            span.record("gen_ai.response.model", model_version.as_str());
            self.final_model_version = Some(model_version.clone());
        }
        if let Some(usage) = data.usage_metadata.as_ref() {
            span.record_token_usage(&crate::completion::Usage::from(usage));
            self.final_usage = Some(usage.clone());
        }

        let Some(choice) = data.candidates.into_iter().next() else {
            tracing::debug!("There is no content candidate");
            return;
        };

        if let Some(finish_reason) = &choice.finish_reason {
            // Last one wins: an intermediate `finishReason` is superseded by
            // the reason the turn actually ended on.
            self.saw_finish_reason = true;
            self.final_finish_reason = Some(finish_reason.clone());
        }
        if let Some(message) = &choice.finish_message {
            self.final_finish_message = Some(message.clone());
        }

        if let Some(err) = tool_protocol_finish_reason_error(&choice) {
            self.failed = true;
            out.push(Err(err));
            return;
        }

        match choice.content {
            Some(content) => {
                if content.parts.is_empty() {
                    tracing::trace!(reason = ?self.final_finish_reason, "There is no part in the streaming content");
                }
                for part in content.parts {
                    self.interpret_part(part, out);
                }
            }
            None => {
                // Gemini's final chunk may carry finishReason with no content.
                tracing::debug!(finish_reason = ?self.final_finish_reason, "Streaming candidate missing content");
            }
        }
    }

    fn finish(&mut self, out: &mut AdapterOutput<Self::Response>) {
        // EOF without a `finishReason` chunk is truncation: no terminal
        // record may be synthesized — it would report a successful completion
        // for a turn the provider aborted.
        if !self.saw_finish_reason {
            return;
        }

        // Deferral, not synthesis: the provider *did* signal the finish, on a
        // chunk that is not reliably its last (see `saw_finish_reason`).
        // Holding the record until EOF is what lets the driver read the rest
        // of the turn, and it means the terminal carries the last reason,
        // usage, and metadata the stream actually reported.
        out.push(Ok(streaming::RawStreamingChoice::FinalResponse(
            StreamingCompletionResponse {
                usage_metadata: self.final_usage.take().unwrap_or_default(),
                finish_reason: self.final_finish_reason.take(),
                finish_message: self.final_finish_message.take(),
                model_version: self.final_model_version.take(),
                response_id: self.final_response_id.take(),
            },
        )));
    }

    fn is_finished(&self) -> bool {
        // A tool-protocol terminal failure is the wire's own in-band
        // terminal: `interpret` already pushed the `Err` and gates itself on
        // `failed`, so the driver must stop reading rather than drain the
        // rest of the transport (and pass through post-error unknown frames).
        self.failed
    }
}

impl GeminiRestAdapter {
    fn interpret_part(&mut self, part: Part, out: &mut AdapterOutput<StreamingCompletionResponse>) {
        match part {
            Part {
                part: PartKind::Text(text),
                thought: Some(true),
                thought_signature,
                ..
            } => {
                // Declare what the part carried; the shared lifecycle
                // derives the sequence (a signature closes the block; the
                // shared accumulator signs the accumulated deltas or records
                // a signature-only part when nothing streamed).
                self.reasoning.emit_chunk(
                    crate::providers::internal::chunk_lifecycle::ChunkParts {
                        reasoning: Some(text),
                        reasoning_signature: thought_signature,
                        text: None,
                        tool_events: Vec::new(),
                    },
                    out,
                );
            }
            Part {
                part: PartKind::Text(text),
                thought_signature,
                ..
            } => {
                // The wire attaches `thoughtSignature` to a trailing part
                // that carries no `thought` flag at all — recorded traffic
                // shows `{"text":"","thoughtSignature":"..."}` — so the
                // signature must be recognized here as well as in the
                // `thought: true` arm above, which real streams never reach
                // for the signature. Dropping it costs the replay-required
                // provider state Gemini validates (`MISSING_THOUGHT_SIGNATURE`).
                // A trailing `thoughtSignature` rides a part with no
                // `thought` flag (recorded traffic:
                // `{"text":"","thoughtSignature":"..."}`); the shared
                // lifecycle emits its close before the text, and one end
                // covers every case — open block (sign the deltas),
                // already-closed block (sign the block that holds the
                // chain-of-thought, #2258 B4), nothing streamed
                // (signature-only part). No per-case branch to forget.
                self.reasoning.emit_chunk(
                    crate::providers::internal::chunk_lifecycle::ChunkParts {
                        reasoning: None,
                        reasoning_signature: thought_signature,
                        text: Some(text),
                        tool_events: Vec::new(),
                    },
                    out,
                );
            }
            Part {
                part: PartKind::FunctionCall(function_call),
                thought_signature,
                ..
            } => {
                // Tool content interleaving an open thought block: the
                // shared lifecycle synthesizes the boundary end.
                self.reasoning.emit_chunk(
                    crate::providers::internal::chunk_lifecycle::ChunkParts {
                        reasoning: None,
                        reasoning_signature: None,
                        text: None,
                        tool_events: vec![shared_parts::function_call(
                            function_call.name,
                            function_call.args,
                            function_call.id,
                            thought_signature,
                            &mut self.tool_ids,
                        )],
                    },
                    out,
                );
            }
            part => {
                // Structural metadata only: an unmodeled part can carry
                // model output, which must not leak into WARN logs.
                crate::providers::internal::adapter::warn_unmodeled("gemini_part", &part);
            }
        }
    }
}

impl<T> CompletionModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    /// Open a `streamGenerateContent` stream whose terminal record stays
    /// provider-native.
    ///
    /// The normalized [`CompletionModel::stream`](crate::completion::CompletionModel::stream)
    /// delegates here and maps only the terminal record, so both paths open
    /// exactly one stream over the same request, telemetry, and error handling.
    pub async fn raw_stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::RawStreamingResult<StreamingCompletionResponse>, CompletionError> {
        let request_model = resolve_request_model(&self.model, &completion_request);
        let span = CompletionSpanBuilder::new(
            PROVIDER_NAME,
            &request_model,
            CompletionOperation::ChatStreaming,
        )
        .system_instructions(
            completion_request.system_instructions(),
            completion_request.record_telemetry_content,
        )
        .build();
        let mut request = create_request_body(completion_request)?;
        if let Some(name) = self.cached_content.as_deref() {
            request.with_cached_content(name)?;
        }

        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Streaming,
            "Gemini streaming completion request",
            &request,
        );

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post(format!("{}?alt=sse", streaming_endpoint(&request_model)))?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        Ok(open_wire_stream(
            GenericEventSource::new(self.client.clone(), req),
            SseTransportOptions {
                open_log: OpenLog::Debug,
                stream_ended_is_error: false,
                log_transport_errors: true,
            },
            skip_blank_frames,
            GeminiRestAdapter::default(),
            span,
        ))
    }

    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse, CompletionError> {
        let inner = self.raw_stream(completion_request).await?;

        Ok(streaming::StreamingCompletionResponse::stream(
            PROVIDER_NAME,
            streaming::normalize_stream(inner, map_stream_final),
        ))
    }
}

#[cfg(test)]
mod tests;
