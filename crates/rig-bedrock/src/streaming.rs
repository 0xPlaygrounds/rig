use crate::types::assistant_content::{PROVIDER_NAME, map_stop_reason, normalize_usage};
use crate::types::completion_request::AwsCompletionRequest;
use crate::types::converse_output::{StopReason, TokenUsage};
use crate::{
    completion::{CompletionModel, resolve_request_model},
    types::errors::{AwsSdkConverseStreamError, converse_stream_output_completion_error},
};
use async_stream::stream;
use aws_sdk_bedrockruntime::types as aws_bedrock;
use base64::{Engine, prelude::BASE64_STANDARD};
use rig_core::providers::internal::adapter::{AdapterOutput, WireAdapter, run_wire_stream};
use rig_core::providers::internal::tool_call_bridge::ToolCallBridge;
use rig_core::providers::internal::wire::{self, TypedEvent, WireEvent};
use rig_core::streaming::{StreamFinal, StreamingCompletionResponse};
use rig_core::telemetry::{CompletionOperation, CompletionSpanBuilder, SpanCombinator};
use rig_core::{
    completion::CompletionError, message::ReasoningContent, streaming::UnparseableToolInput,
    wasm_compat::WasmCompatSend,
};
use serde::{Deserialize, Serialize};
use tracing_futures::Instrument;

#[derive(Clone, Deserialize, Serialize)]
pub struct BedrockStreamingResponse {
    pub usage: Option<TokenUsage>,
    /// Bedrock's own `stopReason` from the terminal `MessageStop` event, when
    /// the stream reported one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,
    /// The AWS request id from the converse-stream response's metadata
    /// (`x-amzn-RequestId`) — not part of any stream event; captured by
    /// `CompletionModel::stream` from the SDK operation output and carried on
    /// the adapter state, matching the unary surface's semantics. `None`
    /// when the SDK reported none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
}

impl From<&BedrockStreamingResponse> for rig_core::completion::Usage {
    fn from(response: &BedrockStreamingResponse) -> Self {
        response
            .usage
            .as_ref()
            .map(normalize_usage)
            .unwrap_or_default()
    }
}

/// Map Bedrock's terminal record onto rig's, serializing the native record
/// onto [`StreamFinal::raw`].
fn terminal_record(response: BedrockStreamingResponse) -> Result<StreamFinal, serde_json::Error> {
    let usage = (&response).into();
    let finish_reason = response.stop_reason.as_ref().map(map_stop_reason);
    let raw = serde_json::to_value(&response)?;
    Ok(StreamFinal::new(PROVIDER_NAME, usage)
        .with_optional_provider_request_id(response.provider_request_id)
        .with_optional_finish_reason(finish_reason)
        .with_raw(raw))
}

#[derive(Default)]
struct ReasoningState {
    /// Signature carried by this block's `signature` delta — delivered out
    /// of band from the thinking text. Thinking TEXT accumulates in the
    /// shared accumulator via reasoning deltas; no restatement buffer exists.
    signature: Option<String>,
    /// Whether a non-empty text delta was emitted for this block, so a
    /// wholly empty block (no text, no signature) can close without opening.
    streamed: bool,
}

/// Minted block identity for a Converse `contentBlockIndex`.
///
/// The wire index is `i32`; the minted index space is unsigned. The offset
/// map is injective over the whole `i32` domain, so even a (spec-violating)
/// negative index yields a distinct, well-formed minted identity instead of
/// a rendering the identity machinery could disagree about — the mint stays
/// total instead of trusting the wire.
fn block_id(content_block_index: i32) -> rig_core::streaming::BlockId {
    let index = (i64::from(content_block_index) - i64::from(i32::MIN)) as u64;
    rig_core::streaming::MintKind::Block.for_wire_index(index)
}

/// Close the open thinking block for `content_block_index`.
///
/// The end carries no restatement — the shared accumulator already holds every
/// reasoning delta this block streamed, so restating the text would supersede
/// the accumulation with a second copy of itself — only the signature, which
/// the wire never restates. Adaptive-thinking blocks can even be
/// signature-only (a `Signature` delta with no non-empty `Text` delta), and
/// dropping that signature makes the next turn fail with
/// `messages.N.content.0.thinking.signature: Field required` on replay. A
/// wholly empty block still lands nowhere: nothing was emitted for it, and
/// closing it would open it first (an end for an unseen id is preceded by
/// its start), which would publish an empty block instead of none.
fn reasoning_end(state: ReasoningState, content_block_index: i32, out: &mut AdapterOutput) {
    if !state.streamed && state.signature.is_none() {
        return;
    }
    out.reasoning_end(
        // Bedrock has no reasoning item id; the block's `contentBlockIndex`
        // is stable across its deltas and its close.
        block_id(content_block_index),
        None,
        state.signature,
        // Both call sites close on a frame the wire actually sent — its own
        // `contentBlockStop`, or the redacted sibling delta that ends the
        // plaintext block — so the completed block reaches the consumer.
        true,
    );
}

/// Accumulated per-stream state for [`process_event`].
///
/// In-flight tool calls are keyed by Bedrock's own `contentBlockIndex` via
/// the shared [`ToolCallBridge`]: the Converse stream indexes every content
/// block, and a message may open several tool-use blocks before it stops, so
/// a single "current" slot would let a later block silently overwrite an
/// earlier one. Fragment assembly, internal-id minting, and finalize policy
/// live in the shared accumulator; the bridge keeps only the index → identity
/// mapping (and the name, for the dropped-block warning).
#[derive(Default)]
struct StreamState {
    tool_calls: ToolCallBridge<i32>,
    current_reasoning: Option<ReasoningState>,
    final_stop_reason: Option<StopReason>,
    /// The AWS request id read off the SDK operation output before the event
    /// stream is opened; stamped onto the terminal record. `None` on the
    /// events-first seam, where no SDK operation exists.
    provider_request_id: Option<String>,
}

/// A static, log-safe label for a stop reason: known variants map to their
/// wire spelling, `Unknown` collapses to `"other"` so its carried wire
/// string (potentially model output) never reaches a log line.
fn stop_reason_label(stop_reason: &StopReason) -> &'static str {
    match stop_reason {
        StopReason::ContentFiltered => "content_filtered",
        StopReason::EndTurn => "end_turn",
        StopReason::GuardrailIntervened => "guardrail_intervened",
        StopReason::MaxTokens => "max_tokens",
        StopReason::StopSequence => "stop_sequence",
        StopReason::ToolUse => "tool_use",
        StopReason::Unknown(_) => "other",
    }
}

/// Handle one Converse stream event, pushing the items to yield in order.
///
/// Kept as a plain function over [`StreamState`] so the event bookkeeping can
/// be unit-tested without an AWS event receiver.
fn process_event(
    state: &mut StreamState,
    output: aws_bedrock::ConverseStreamOutput,
    out: &mut AdapterOutput,
) {
    match output {
        aws_bedrock::ConverseStreamOutput::ContentBlockDelta(event) => {
            let Some(delta) = event.delta else {
                tracing::warn!("skipping ContentBlockDelta with a missing delta");
                return;
            };
            match delta {
                aws_bedrock::ContentBlockDelta::Text(text) => {
                    out.text(text);
                }
                aws_bedrock::ContentBlockDelta::ToolUse(tool) => {
                    if let Some(tool_call) = state.tool_calls.get(event.content_block_index) {
                        // Emit the delta so UI can show progress; the shared
                        // accumulator assembles the fragments.
                        out.tool_arguments(tool_call.key(), tool.input());
                    }
                }
                aws_bedrock::ContentBlockDelta::ReasoningContent(reasoning) => match reasoning {
                    aws_bedrock::ReasoningContentBlockDelta::Text(text) => {
                        // Marks the block open so its stop emits an end; the
                        // text itself belongs to the shared accumulator.
                        let open = state
                            .current_reasoning
                            .get_or_insert_with(ReasoningState::default);

                        if !text.is_empty() {
                            open.streamed = true;
                            // Derive identity from `contentBlockIndex` (no
                            // wire id on Converse reasoning blocks).
                            out.reasoning_delta(&block_id(event.content_block_index), None, text);
                        }
                    }
                    aws_bedrock::ReasoningContentBlockDelta::Signature(signature) => {
                        state
                            .current_reasoning
                            .get_or_insert_with(ReasoningState::default)
                            .signature = Some(signature);
                    }
                    aws_bedrock::ReasoningContentBlockDelta::RedactedContent(blob) => {
                        // Opaque provider state the safety classifier
                        // encrypted. It is not part of any plaintext thinking
                        // block, so close an open one first: sharing the
                        // block index would otherwise make the redacted block
                        // *replace* the delta-built thinking part instead of
                        // landing beside it as a sibling.
                        if let Some(open) = state.current_reasoning.take() {
                            reasoning_end(open, event.content_block_index, out);
                        }

                        out.reasoning_block(
                            // Same minted block identity as the sibling
                            // reasoning paths, so provenance and boundary
                            // semantics stay uniform.
                            block_id(event.content_block_index),
                            None,
                            ReasoningContent::Redacted {
                                // The wire carries raw bytes; rig's canonical
                                // reasoning content is a string, so the blob
                                // travels base64-encoded and decodes back on
                                // the way out.
                                data: BASE64_STANDARD.encode(blob.as_ref()),
                            },
                        );
                    }
                    unknown => {
                        tracing::warn!(
                            delta = ?std::mem::discriminant(&unknown),
                            "skipping unrecognized Bedrock reasoning content delta variant"
                        );
                    }
                },
                unknown => {
                    tracing::warn!(
                        delta = ?std::mem::discriminant(&unknown),
                        "skipping unrecognized Bedrock content block delta variant"
                    );
                }
            }
        }
        aws_bedrock::ConverseStreamOutput::ContentBlockStart(event) => {
            let Some(start) = event.start else {
                tracing::warn!("skipping ContentBlockStart with no data");
                return;
            };
            match start {
                aws_bedrock::ContentBlockStart::ToolUse(tool_use) => {
                    // The wire always supplies a tool-use id here; the shared
                    // bridge fixes it as the assembly key (and would mint one
                    // in the reserved namespace if the wire ever omitted it).
                    let slot = state.tool_calls.open(
                        event.content_block_index,
                        Some(&tool_use.tool_use_id),
                        Some(&tool_use.name),
                    );
                    out.tool_name(slot.key(), tool_use.name);
                }
                // `ContentBlockStart` is a union: `toolUse` is the only
                // variant modeled today, and a future one is not a stream
                // failure. Failing the turn here contradicted every sibling
                // arm (which warn and skip) and the classify layer's
                // Unknown-frame policy, and the message ("Stream is empty")
                // described neither the frame nor the cause.
                unknown => tracing::warn!(
                    start = ?std::mem::discriminant(&unknown),
                    "skipping unrecognized Bedrock ContentBlockStart variant"
                ),
            }
        }
        aws_bedrock::ConverseStreamOutput::ContentBlockStop(event) => {
            if let Some(reasoning_state) = state.current_reasoning.take() {
                reasoning_end(reasoning_state, event.content_block_index, out);
            }
            // A closed tool-use block is complete: finalize and emit it here,
            // mirroring the reasoning close above, so every call in a
            // multi-tool-call message reaches the consumer. The shared
            // accumulator finalizes the assembled input: an empty accumulated
            // input means a tool with no parameters, and malformed JSON
            // surfaces as an error item (`UnparseableToolInput::Error`) rather
            // than a silent drop, so the terminal can never report tool use
            // whose calls the consumer never saw.
            if let Some(tool_call) = state.tool_calls.remove(event.content_block_index) {
                out.push(Ok(tool_call.end_event(UnparseableToolInput::Error)));
            }
        }
        aws_bedrock::ConverseStreamOutput::MessageStop(message_stop_event) => {
            // Remember Bedrock's own terminal reason so the final
            // record can report it; an unmapped SDK variant is kept
            // verbatim rather than dropped.
            state.final_stop_reason = Some(
                StopReason::try_from(message_stop_event.stop_reason.clone()).unwrap_or_else(|_| {
                    StopReason::Unknown(crate::types::converse_output::UnknownVariantValue(
                        message_stop_event.stop_reason.as_str().to_owned(),
                    ))
                }),
            );
            // Tool calls normally flush at their ContentBlockStop; when the
            // message genuinely stopped for tool use, drain any stragglers
            // here defensively (in block order) so a stream that omits the
            // stop event still delivers every call. Under any other stop
            // reason (notably MaxTokens) an in-flight block is one the model
            // never finished: drop it rather than fabricate a `{}`-args call
            // or a spurious error item — truncation is signaled to the
            // consumer by the mapped finish reason on the terminal record,
            // which the Metadata path emits via `final_stop_reason`.
            if matches!(state.final_stop_reason, Some(StopReason::ToolUse)) {
                for tool_call in state.tool_calls.drain_ordered() {
                    out.push(Ok(tool_call.end_event(UnparseableToolInput::Error)));
                }
            } else if !state.tool_calls.is_empty() {
                // Structural metadata only: tool names can be model-chosen
                // (a hallucinated call's name is model output) and the
                // `Unknown` variant carries a wire string, so neither may
                // reach the WARN log. Known variants log a static label —
                // unknown ones collapse to "other", never the wire value.
                let dropped = state.tool_calls.drain_ordered().len();
                tracing::warn!(
                    dropped_tool_calls = dropped,
                    stop_reason = state
                        .final_stop_reason
                        .as_ref()
                        .map_or("none", stop_reason_label),
                    "dropping unfinished tool-use blocks left in flight at MessageStop"
                );
            }
        }
        aws_bedrock::ConverseStreamOutput::Metadata(metadata_event) => {
            // Extract usage information from metadata; a missing usage still
            // yields a terminal record so the stream ends with a `Final`.
            let final_response = BedrockStreamingResponse {
                // The mirror conversion is infallible for `TokenUsage`.
                usage: metadata_event
                    .usage
                    .and_then(|usage| TokenUsage::try_from(usage).ok()),
                stop_reason: state.final_stop_reason.clone(),
                provider_request_id: state.provider_request_id.clone(),
            };
            match terminal_record(final_response) {
                Ok(record) => {
                    tracing::Span::current().record_token_usage(&record.usage);
                    out.final_record(record);
                }
                Err(err) => out.error(err.into()),
            }
        }
        _ => {}
    }
}

impl WireAdapter for StreamState {
    type Frame = aws_bedrock::ConverseStreamOutput;
    type Event = aws_bedrock::ConverseStreamOutput;

    fn classify(&self, frame: Self::Frame) -> WireEvent<Self::Event> {
        // The AWS SDK already deserialized the event-stream frame, so the
        // byte-level decode step collapses: an event-stream decode failure
        // surfaces as a receive error on the transport, and the only triage
        // left here is the SDK's own unknown-variant signal on its
        // non-exhaustive union.
        wire::classify_typed_event(if frame.is_unknown() {
            TypedEvent::Unrecognized {
                event_type: "unknown".to_string(),
                detail: format!("{frame:?}"),
            }
        } else {
            TypedEvent::Modeled(frame)
        })
    }

    fn interpret(&mut self, event: Self::Event, out: &mut AdapterOutput) {
        process_event(self, event, out);
    }

    fn finish(&mut self, _out: &mut AdapterOutput) {
        // EOF without Bedrock's `Metadata` terminal is truncation: in-flight
        // blocks drop and no terminal record may be synthesized.
    }
}

/// Drive already-typed Converse stream events through the full shared
/// pipeline — driver policy, canonical grammar, terminal normalization.
///
/// The events-first conformance seam: the adapter is a pure
/// `(state, event) → events` function, so grammar scenarios feed SDK events
/// directly with no AWS transport.
pub fn stream_from_events(
    events: impl futures::Stream<Item = Result<aws_bedrock::ConverseStreamOutput, CompletionError>>
    + WasmCompatSend
    + 'static,
) -> StreamingCompletionResponse {
    StreamingCompletionResponse::stream(
        PROVIDER_NAME,
        run_wire_stream(events, StreamState::default()),
    )
}

impl CompletionModel {
    /// Open a stream normalized to rig's terminal record; the adapter maps
    /// Bedrock's own terminal onto [`StreamFinal::raw`].
    pub(crate) async fn stream(
        &self,
        completion_request: rig_core::completion::CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        let request_model = resolve_request_model(&self.model, &completion_request);
        let system_instructions = completion_request.system_instructions().map(str::to_owned);
        let record_telemetry_content = completion_request.record_telemetry_content;
        let request = AwsCompletionRequest {
            inner: completion_request,
            prompt_caching: self.prompt_caching,
        };
        let span = CompletionSpanBuilder::new(
            "aws_bedrock",
            &request_model,
            CompletionOperation::ChatStreaming,
        )
        .system_instructions(system_instructions.as_deref(), record_telemetry_content)
        .build();

        let mut converse_builder = self
            .client
            .inner()
            .await
            .converse_stream()
            .model_id(request_model);

        let tool_config = request.tools_config()?;
        let output_config = request.output_config()?;
        let additional_params = request.additional_params();
        let inference_config = request.inference_config();
        let system_prompt = request.system_prompt()?;
        let prompt_with_history = request.messages()?;
        converse_builder = converse_builder
            .set_additional_model_request_fields(additional_params)
            .set_inference_config(Some(inference_config))
            .set_tool_config(tool_config)
            .set_system(system_prompt)
            .set_messages(Some(prompt_with_history))
            .set_output_config(output_config);

        let response = converse_builder
            .send()
            .instrument(span.clone())
            .await
            .map_err(|sdk_error| {
                Into::<CompletionError>::into(AwsSdkConverseStreamError(sdk_error))
            })?;

        // Read the AWS request id off the operation output *before* the event
        // stream is moved — `ConverseStreamOutput` implements the SDK
        // `RequestId` trait on the whole output, not on stream events. The
        // adapter stamps it onto the terminal record, mirroring the unary
        // surface (`InternalConverseOutput::request_id`).
        let provider_request_id =
            aws_sdk_bedrockruntime::operation::RequestId::request_id(&response).map(str::to_string);

        // Transport layer: SDK event-stream frames only — an event-stream
        // decode/receive failure is a transport error; classification and
        // policy live in the shared driver.
        let transport = stream! {
            let mut stream = response.stream;
            loop {
                match stream.recv().await {
                    Ok(Some(output)) => yield Ok(output),
                    Ok(None) => break,
                    Err(err) => {
                        yield Err(converse_stream_output_completion_error(err.into_service_error()));
                        break;
                    }
                }
            }
        };

        let state = StreamState {
            provider_request_id,
            ..StreamState::default()
        };
        let stream = run_wire_stream(transport, state).instrument(span);
        Ok(StreamingCompletionResponse::stream(
            PROVIDER_NAME,
            Box::pin(stream),
        ))
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;

#[cfg(test)]
mod response_identity_tests;
