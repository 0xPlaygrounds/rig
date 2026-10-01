use crate::completion::ConverseFrame;
use crate::types::assistant_content::{
    PROVIDER_NAME, map_stop_reason, normalize_usage, reasoning_issuer,
};
use crate::types::converse_output::{InternalConverseOutput, StopReason, TokenUsage};
use crate::types::message;
use aws_sdk_bedrockruntime::types as aws_bedrock;
use base64::{Engine, prelude::BASE64_STANDARD};
use rig_core::error::ProviderError;
use rig_core::message::ReasoningContent;
use rig_core::operation::{
    CallFragment, Completion, Finish, IfMalformed, ReasoningPart, Seal, TextPart,
};
use rig_core::wire::{Flow, Out, WireEvent};
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

/// Bedrock's terminal record as the provider's end of the reply.
fn finish_of(response: &BedrockStreamingResponse) -> Finish {
    Finish {
        usage: response.into(),
        reason: response.stop_reason.as_ref().map(map_stop_reason),
        ..Finish::default()
    }
}

/// The buffer index of a Converse content block: every signed index,
/// negative ones included, maps to a distinct one.
fn block_index(content_block_index: i32) -> usize {
    (i64::from(content_block_index) - i64::from(i32::MIN)) as usize
}

/// One Converse reply's state: a whole reply or a stream of events.
#[derive(Default)]
pub struct StreamState<'id> {
    /// The text part text deltas extend; another block closes it.
    text: Option<TextPart<'id>>,
    /// The open reasoning block and the signature delivered for it.
    reasoning: Option<(ReasoningPart<'id>, Option<String>)>,
    final_stop_reason: Option<StopReason>,
    /// The AWS request id read off the SDK operation output before the event
    /// stream is opened; the reply's end carries it.
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

impl<'id> StreamState<'id> {
    /// Close the open reasoning block. A signature-only block is kept for
    /// replay; one with neither text nor signature is dropped.
    fn close_reasoning(&mut self, out: &mut Out<'id, Completion>) {
        if let Some((part, signature)) = self.reasoning.take() {
            out.close_reasoning(
                part,
                Seal {
                    signature,
                    ..Seal::default()
                },
            );
        }
    }

    /// Write one Converse event in delivery order.
    fn process_event(
        &mut self,
        output: aws_bedrock::ConverseStreamOutput,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match output {
            aws_bedrock::ConverseStreamOutput::ContentBlockDelta(event) => {
                let Some(delta) = event.delta else {
                    tracing::warn!("skipping ContentBlockDelta with a missing delta");
                    return Ok(Flow::More);
                };
                match delta {
                    aws_bedrock::ContentBlockDelta::Text(text) => {
                        out.extend_text(&mut self.text, &text);
                    }
                    aws_bedrock::ContentBlockDelta::ToolUse(tool) => {
                        let index = block_index(event.content_block_index);
                        if out.pending_calls().contains(&index) {
                            out.call_fragment(
                                index,
                                CallFragment {
                                    arguments: Some(tool.input()),
                                    ..CallFragment::default()
                                },
                            )?;
                        }
                    }
                    aws_bedrock::ContentBlockDelta::ReasoningContent(reasoning) => {
                        match reasoning {
                            aws_bedrock::ReasoningContentBlockDelta::Text(text) => {
                                out.close_open_text(&mut self.text);
                                let (part, _) = self
                                    .reasoning
                                    .get_or_insert_with(|| (out.reasoning(), None));
                                out.push_reasoning(part, &text);
                            }
                            aws_bedrock::ReasoningContentBlockDelta::Signature(signature) => {
                                out.close_open_text(&mut self.text);
                                self.reasoning
                                    .get_or_insert_with(|| (out.reasoning(), None))
                                    .1 = Some(signature);
                            }
                            aws_bedrock::ReasoningContentBlockDelta::RedactedContent(blob) => {
                                // Close plaintext reasoning first, so redacted
                                // content is a sibling rather than a replacement.
                                out.close_open_text(&mut self.text);
                                self.close_reasoning(&mut out);
                                out.reasoning_block(rig_core::message::Reasoning {
                                    id: None,
                                    content: vec![ReasoningContent::Redacted {
                                        // Base64 preserves opaque bytes for replay.
                                        data: BASE64_STANDARD.encode(blob.as_ref()),
                                    }],
                                    native: None,
                                });
                            }
                            unknown => {
                                tracing::warn!(
                                    delta = ?std::mem::discriminant(&unknown),
                                    "skipping unrecognized Bedrock reasoning content delta variant"
                                );
                            }
                        }
                    }
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
                    return Ok(Flow::More);
                };
                match start {
                    aws_bedrock::ContentBlockStart::ToolUse(tool_use) => {
                        out.close_open_text(&mut self.text);
                        out.call_fragment(
                            block_index(event.content_block_index),
                            CallFragment {
                                id: Some(tool_use.tool_use_id.as_str()),
                                name: Some(tool_use.name.as_str()),
                                ..CallFragment::default()
                            },
                        )?;
                    }
                    // Unknown union variants do not invalidate recognized content.
                    unknown => tracing::warn!(
                        start = ?std::mem::discriminant(&unknown),
                        "skipping unrecognized Bedrock ContentBlockStart variant"
                    ),
                }
            }
            aws_bedrock::ConverseStreamOutput::ContentBlockStop(event) => {
                self.close_reasoning(&mut out);
                // Each closed call finalizes on its own; malformed arguments
                // fail the reply rather than silently dropping the call.
                let index = block_index(event.content_block_index);
                if out.pending_calls().contains(&index) {
                    out.close_pending(index, IfMalformed::Fail)?;
                }
            }
            aws_bedrock::ConverseStreamOutput::MessageStop(message_stop_event) => {
                // Bedrock's own terminal reason; an unmapped SDK variant is
                // kept verbatim rather than dropped.
                self.final_stop_reason = Some(
                    StopReason::try_from(message_stop_event.stop_reason.clone()).unwrap_or_else(
                        |_| {
                            StopReason::Unknown(crate::types::converse_output::UnknownVariantValue(
                                message_stop_event.stop_reason.as_str().to_owned(),
                            ))
                        },
                    ),
                );
                // A tool-use stop completes remaining calls even without block
                // stops. Other stop reasons leave them incomplete: they drop,
                // and the terminal reason stands instead of fabricated
                // arguments.
                let pending = out.pending_calls();
                if matches!(self.final_stop_reason, Some(StopReason::ToolUse)) {
                    for index in pending {
                        out.close_pending(index, IfMalformed::Fail)?;
                    }
                } else if !pending.is_empty() {
                    // Log only counts and static reason labels: tool names and
                    // unknown reason strings can contain model output.
                    tracing::warn!(
                        dropped_tool_calls = pending.len(),
                        stop_reason = self
                            .final_stop_reason
                            .as_ref()
                            .map_or("none", stop_reason_label),
                        "dropping unfinished tool-use blocks left in flight at MessageStop"
                    );
                    for index in pending {
                        out.drop_pending(index);
                    }
                }
            }
            aws_bedrock::ConverseStreamOutput::Metadata(metadata_event) => {
                // The provider's end; a missing usage still ends the reply.
                let native = BedrockStreamingResponse {
                    // The mirror conversion is infallible for `TokenUsage`.
                    usage: metadata_event
                        .usage
                        .and_then(|usage| TokenUsage::try_from(usage).ok()),
                    stop_reason: self.final_stop_reason.clone(),
                    provider_request_id: self.provider_request_id.clone(),
                };
                out.close_open_text(&mut self.text);
                self.close_reasoning(&mut out);
                out.raw(serde_json::to_value(&native)?);
                return Ok(out.end(finish_of(&native)));
            }
            _ => {}
        }
        Ok(Flow::More)
    }
}

/// A whole Converse reply, written as the parts a stream sends for it.
fn whole(
    output: InternalConverseOutput,
    mut out: Out<'_, Completion>,
) -> Result<Flow, ProviderError> {
    // The provider's own document, captured before the output is consumed
    // into normalized content.
    out.raw(serde_json::to_value(&output)?);
    for content in assistant_content(&output)? {
        match content {
            // The conversion seals to Bedrock; the reply's issuer is the
            // model's, as it is for streamed reasoning.
            rig_core::message::AssistantContent::Reasoning(reasoning) => {
                if let Some(reasoning) = reasoning.open(reasoning.issuer()) {
                    out.reasoning_block(reasoning.clone());
                }
            }
            content => out.content(content)?,
        }
    }
    let usage = output.usage().map(normalize_usage).unwrap_or_default();
    Ok(out.end(Finish {
        usage,
        reason: Some(map_stop_reason(&output.stop_reason)),
        ..Finish::default()
    }))
}

/// The assistant content of a Converse reply.
fn assistant_content(
    output: &InternalConverseOutput,
) -> Result<Vec<rig_core::message::AssistantContent>, ProviderError> {
    let reply = output
        .output
        .as_ref()
        .ok_or(ProviderError::Provider(
            "Model didn't return any output".into(),
        ))?
        .as_message()
        .map_err(|_| {
            ProviderError::Provider("Failed to extract message from converse output".into())
        })?
        .to_owned();
    message::assistant_reply(reply)
}

impl<'id> rig_core::wire::Decoder<'id, Completion, ConverseFrame> for StreamState<'id> {
    type Event = ConverseFrame;

    fn classify(&self, frame: ConverseFrame) -> WireEvent<Self::Event> {
        // The SDK handles byte decoding; only unknown union variants need
        // classification here.
        match &frame {
            ConverseFrame::Event(event) if event.is_unknown() => {
                WireEvent::unrecognized("unknown", format!("{event:?}"))
            }
            _ => WireEvent::Known(frame),
        }
    }

    /// EOF without Bedrock's `Metadata` event is truncation.
    fn decode(
        &mut self,
        event: ConverseFrame,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match event {
            ConverseFrame::Opened { model, request_id } => {
                let issuer = reasoning_issuer(&model);
                if issuer != PROVIDER_NAME {
                    out.issued_by(issuer);
                }
                self.provider_request_id = request_id;
                Ok(Flow::More)
            }
            ConverseFrame::Whole(output) => whole(*output, out),
            ConverseFrame::Event(event) => self.process_event(event, out),
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
