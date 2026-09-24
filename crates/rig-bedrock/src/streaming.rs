use crate::completion::ConverseFrame;
use crate::types::assistant_content::{
    PROVIDER_NAME, map_stop_reason, normalize_usage, reasoning_issuer,
};
use crate::types::converse_output::{InternalConverseOutput, StopReason, TokenUsage};
use crate::types::message::RigMessage;
use aws_sdk_bedrockruntime::types as aws_bedrock;
use base64::{Engine, prelude::BASE64_STANDARD};
use rig_core::error::ProviderError;
use rig_core::operation::{AdapterOutput, Completion};
use rig_core::providers::internal::tool_call_bridge::ToolCallBridge;
use rig_core::providers::internal::wire::{self, TypedEvent, WireEvent};
use rig_core::streaming::StreamFinal;
use rig_core::{message::ReasoningContent, streaming::UnparseableToolInput};
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize, Serialize)]
pub struct BedrockStreamingResponse {
    pub usage: Option<TokenUsage>,
    /// Bedrock's own `stopReason` from the terminal `MessageStop` event, when
    /// the stream reported one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,
    /// AWS request ID from SDK operation metadata, or `None` when absent.
    /// Individual stream events do not contain this identifier.
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
    Ok(StreamFinal::new(PROVIDER_NAME, usage, raw)
        .with_optional_provider_request_id(response.provider_request_id)
        .with_optional_finish_reason(finish_reason))
}

#[derive(Default)]
struct ReasoningState {
    /// Signature delivered separately from the text held by the shared accumulator.
    signature: Option<String>,
    /// Whether a non-empty text delta was emitted for this block, so a
    /// wholly empty block (no text, no signature) can close without opening.
    streamed: bool,
}

/// Maps every signed Converse block index to a distinct minted identity,
/// including invalid negative wire indices.
fn block_id(content_block_index: i32) -> rig_core::streaming::BlockId {
    let index = (i64::from(content_block_index) - i64::from(i32::MIN)) as u64;
    rig_core::streaming::MintKind::Block.for_wire_index(index)
}

/// Closes a reasoning block without restating accumulated text.
/// Retains signature-only blocks required for replay and omits blocks with
/// neither text nor signature.
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
        // Both callers close on an observed wire boundary, making the block
        // eligible for delivery.
        true,
    );
}

/// Per-reply state with independently indexed tool calls.
/// The shared accumulator owns argument fragments and finalization.
#[derive(Default)]
pub struct StreamState {
    tool_calls: ToolCallBridge<i32>,
    current_reasoning: Option<ReasoningState>,
    final_stop_reason: Option<StopReason>,
    /// The AWS request id read off the SDK operation output before the event
    /// stream is opened; stamped onto the terminal record.
    provider_request_id: Option<String>,
    /// The issuer the reply's reasoning records when it is not
    /// [`PROVIDER_NAME`] (see [`reasoning_issuer`]).
    reasoning_issuer: Option<&'static str>,
    /// The unary reply's document, for the response's `raw`.
    document: Option<serde_json::Value>,
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

/// Processes a Converse event and appends normalized output in delivery order.
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
                        // Close plaintext reasoning first so redacted content
                        // becomes a sibling instead of replacing accumulated text.
                        if let Some(open) = state.current_reasoning.take() {
                            reasoning_end(open, event.content_block_index, out);
                        }

                        out.reasoning_block(
                            block_id(event.content_block_index),
                            None,
                            ReasoningContent::Redacted {
                                // Base64 preserves opaque bytes for replay.
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
                // Unknown union variants do not invalidate recognized content.
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
            // Finalize each closed call independently; malformed arguments must
            // surface as errors rather than silently dropping completed calls.
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
            // A tool-use stop completes remaining calls even without block stops.
            // Other stop reasons leave them incomplete; drop them and preserve
            // the terminal reason instead of fabricating arguments.
            if matches!(state.final_stop_reason, Some(StopReason::ToolUse)) {
                for tool_call in state.tool_calls.drain_ordered() {
                    out.push(Ok(tool_call.end_event(UnparseableToolInput::Error)));
                }
            } else if !state.tool_calls.is_empty() {
                // Log only counts and static reason labels: tool names and
                // unknown reason strings can contain model output.
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
                    let record = match state.reasoning_issuer {
                        Some(issuer) => record.with_reasoning_issuer(issuer),
                        None => record,
                    };
                    out.final_record(record);
                }
                Err(err) => out.error(err.into()),
            }
        }
        _ => {}
    }
}

/// Emits a whole Converse reply as the events a stream sends for it.
fn whole(state: &mut StreamState, output: InternalConverseOutput, out: &mut AdapterOutput) {
    // The provider's own document, captured before the output is consumed
    // into normalized content.
    match serde_json::to_value(&output) {
        Ok(document) => state.document = Some(document),
        Err(error) => return out.error(error.into()),
    }
    let choice = match assistant_content(&output) {
        Ok(choice) => choice,
        Err(error) => return out.error(error),
    };
    out.content(&choice);
    let usage = output.usage().map(normalize_usage).unwrap_or_default();
    let record = StreamFinal::new(PROVIDER_NAME, usage, serde_json::Value::Null)
        .with_optional_provider_request_id(output.request_id())
        .with_finish_reason(map_stop_reason(&output.stop_reason));
    out.final_record(match state.reasoning_issuer {
        Some(issuer) => record.with_reasoning_issuer(issuer),
        None => record,
    });
}

/// The assistant content of a Converse reply.
fn assistant_content(
    output: &InternalConverseOutput,
) -> Result<Vec<rig_core::message::AssistantContent>, ProviderError> {
    let message: RigMessage = output
        .output
        .clone()
        .ok_or(ProviderError::Provider(
            "Model didn't return any output".into(),
        ))?
        .as_message()
        .map_err(|_| {
            ProviderError::Provider("Failed to extract message from converse output".into())
        })?
        .to_owned()
        .try_into()?;
    match message.0 {
        rig_core::completion::Message::Assistant { content, .. } => Ok(content),
        _ => Err(ProviderError::Response(
            "Converse output message was not an assistant message".to_owned(),
        )),
    }
}

impl rig_core::wire::Decoder<Completion, ConverseFrame> for StreamState {
    type Event = ConverseFrame;

    fn classify(&self, frame: ConverseFrame) -> WireEvent<Self::Event> {
        // The SDK handles byte decoding; only unknown union variants need
        // classification here.
        let unknown = matches!(&frame, ConverseFrame::Event(event) if event.is_unknown());
        wire::classify_typed_event(if unknown {
            TypedEvent::Unrecognized {
                event_type: "unknown".to_string(),
                detail: match &frame {
                    ConverseFrame::Event(event) => format!("{event:?}"),
                    _ => String::new(),
                },
            }
        } else {
            TypedEvent::Modeled(frame)
        })
    }

    fn interpret(&mut self, event: Self::Event, out: &mut AdapterOutput) {
        match event {
            ConverseFrame::Opened { model, request_id } => {
                let issuer = reasoning_issuer(&model);
                self.reasoning_issuer = (issuer != PROVIDER_NAME).then_some(issuer);
                self.provider_request_id = request_id;
            }
            ConverseFrame::Whole(output) => whole(self, *output, out),
            ConverseFrame::Event(event) => process_event(self, event, out),
        }
    }

    fn finish(&mut self, _out: &mut AdapterOutput) {
        // EOF without Bedrock's `Metadata` terminal is truncation: in-flight
        // blocks drop and no terminal record may be synthesized.
    }

    fn document(&self) -> Option<serde_json::Value> {
        self.document.clone()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;

#[cfg(test)]
mod response_identity_tests;
