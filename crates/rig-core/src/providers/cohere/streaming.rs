//! Cohere chat event decoding and terminal metadata for unary and streamed replies.
//!
//! ```
//! use rig_core::providers::cohere::streaming::StreamingEvent;
//! let event: StreamingEvent = serde_json::from_str(r#"{"type":"message-end"}"#)?;
//! assert!(matches!(event, StreamingEvent::MessageEnd { delta: None }));
//! # Ok::<(), serde_json::Error>(())
//! ```

use crate::operation::AdapterOutput;
use crate::operation::Completion;
use crate::providers::cohere::completion::{
    AssistantContent, CompletionResponse, FinishReason, PROVIDER_NAME, Usage, map_finish_reason,
};
use crate::providers::internal::wire;
use crate::streaming::{BlockId, MintKind, StreamFinal, ToolCallEnd, UnparseableToolInput};
use crate::wire::WireFrame;
use serde::{Deserialize, Serialize};

/// One streamed frame of Cohere's `/v2/chat`, named by its `type`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "type")]
pub enum StreamingEvent {
    MessageStart {
        /// The message identifier, when the wire named one.
        #[serde(default)]
        id: Option<String>,
    },
    /// A content block opens.
    ContentStart,
    /// One content fragment: text, or a reasoning model's thought text.
    ContentDelta {
        /// The fragment, absent on a frame that carries none.
        delta: Option<Delta>,
    },
    /// A content block closes.
    ContentEnd,
    /// The model's plan for the tool calls that follow.
    ToolPlan,
    /// A tool call opens, naming the function it calls.
    ToolCallStart {
        /// The call's identity and name.
        delta: Option<Delta>,
    },
    /// One argument fragment of the open tool call.
    ToolCallDelta {
        /// The fragment.
        delta: Option<Delta>,
    },
    /// The open tool call closes.
    ToolCallEnd,
    /// The turn ends: the wire's genuine terminal.
    MessageEnd {
        /// Usage and finish reason, absent on a bare terminal.
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

/// One content fragment of a `content-delta` frame.
#[derive(Debug, Deserialize)]
pub struct MessageContentDelta {
    /// Assistant text.
    pub text: Option<String>,
    /// Cohere v2 reasoning models stream thought text as `content-delta`
    /// frames whose content carries `thinking` instead of `text`.
    pub thinking: Option<String>,
}

/// The function half of a tool-call frame.
#[derive(Debug, Deserialize)]
pub struct MessageToolFunctionDelta {
    /// The tool's name, on the frame that opens the call.
    pub name: Option<String>,
    /// One fragment of the call's JSON arguments.
    pub arguments: Option<String>,
}

/// The tool-call half of a message delta.
#[derive(Debug, Deserialize)]
pub struct MessageToolCallDelta {
    /// The call's wire id, on the frame that opens it.
    pub id: Option<String>,
    /// The function the call names.
    pub function: Option<MessageToolFunctionDelta>,
}

/// What one frame's message delta carried.
#[derive(Debug, Deserialize)]
pub struct MessageDelta {
    /// A content fragment.
    pub content: Option<MessageContentDelta>,
    /// A tool-call fragment.
    pub tool_calls: Option<MessageToolCallDelta>,
}

/// One frame's delta envelope.
#[derive(Debug, Deserialize)]
pub struct Delta {
    /// The message the delta applies to.
    pub message: Option<MessageDelta>,
}

/// The `message-end` payload: what the turn cost and why it stopped.
#[derive(Debug, Deserialize)]
pub struct MessageEndDelta {
    /// Token counters, when Cohere reported them.
    pub usage: Option<Usage>,
    /// Cohere's own finish reason.
    #[serde(default)]
    pub finish_reason: Option<FinishReason>,
}

/// Cohere's terminal stream record: the `message-end` payload as rig parsed
/// it, serialized onto [`StreamFinal::raw`] by the adapter's terminal
/// mapping.
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

/// Stateful decoder for unary and streaming v2 chat replies. Tracks open calls,
/// message identity, and reasoning boundaries; the driver handles corrupt frames.
pub struct ChatDecoder {
    /// Wire id of the open tool call, when one is streaming. Only the wire
    /// identity is tracked here; fragment assembly, internal-id minting, and
    /// finalize policy live in the shared accumulator.
    current_tool_call: Option<BlockId>,
    /// Keys for calls whose wire id is empty: an absent id is not an id.
    tool_ids: crate::streaming::SyntheticIds,
    message_id: Option<String>,
    /// Derives reasoning closure when subsequent content changes block type.
    reasoning: crate::providers::internal::chunk_lifecycle::MintedReasoningLifecycle,
}

impl Default for ChatDecoder {
    fn default() -> Self {
        Self {
            current_tool_call: None,
            tool_ids: crate::streaming::SyntheticIds::tool(),
            message_id: None,
            reasoning: crate::providers::internal::chunk_lifecycle::MintedReasoningLifecycle::new(
                MintKind::Reasoning,
            ),
        }
    }
}

/// Tagged streaming event or untagged unary reply from `/v2/chat`.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ChatEvent {
    /// One SSE frame of `POST /v2/chat` with `stream: true`.
    Stream(StreamingEvent),
    /// The whole reply of `POST /v2/chat` without `stream`.
    Reply(CompletionResponse),
}

impl ChatDecoder {
    /// Interpret one streamed `/v2/chat` frame.
    fn interpret_stream(&mut self, event: StreamingEvent, out: &mut AdapterOutput) {
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

                self.reasoning.emit_chunk(
                    crate::providers::internal::chunk_lifecycle::ChunkParts {
                        reasoning: content.thinking.clone(),
                        reasoning_signature: None,
                        text: content.text.clone(),
                        text_meta: None,
                        tool_events: Vec::new(),
                    },
                    out,
                );
            }

            StreamingEvent::MessageEnd { delta } => {
                // A bare message-end still completes the turn with unknown usage and reason.
                let (usage, finish_reason) = match delta {
                    Some(delta) => (delta.usage, delta.finish_reason),
                    None => (None, None),
                };
                let message_id = self.message_id.take();
                self.terminal(usage, finish_reason, message_id, out);
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

                let key = crate::streaming::non_empty_id(id)
                    .map_or_else(|| self.tool_ids.mint(), BlockId::wire);
                self.current_tool_call = Some(key.clone());
                let mut tool_events = AdapterOutput::new();
                tool_events.tool_name(&key, name);
                // `tool-call-start` may carry initial argument text; on the
                // wire it is empty, but any payload must still enter assembly.
                if !arguments.is_empty() {
                    tool_events.tool_arguments(&key, arguments);
                }
                // Tool content interleaving an open thinking block: the
                // shared lifecycle synthesizes the boundary end.
                self.reasoning.emit_chunk(
                    crate::providers::internal::chunk_lifecycle::ChunkParts {
                        reasoning: None,
                        reasoning_signature: None,
                        text: None,
                        text_meta: None,
                        tool_events: tool_events
                            .into_items()
                            .into_iter()
                            .filter_map(Result::ok)
                            .collect(),
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
                let Some(key) = self.current_tool_call.clone() else {
                    return;
                };

                out.tool_arguments(&key, arguments);
            }

            StreamingEvent::ToolCallEnd => {
                let Some(key) = self.current_tool_call.take() else {
                    return;
                };
                // This endpoint drops calls whose assembled arguments are unparseable.
                out.tool_end(key, ToolCallEnd::new(UnparseableToolInput::Drop));
            }

            _ => {}
        }
    }

    /// Interpret the unary reply by *synthesizing the stream* it would have
    /// been: one block per content part, each tool call whole, then the
    /// terminal the `message-end` event carries.
    fn interpret_reply(&mut self, reply: CompletionResponse, out: &mut AdapterOutput) {
        let response_id = crate::streaming::non_empty_id(reply.id.clone());
        let finish_reason = Some(reply.finish_reason.clone());
        let usage = reply.usage;
        let (content, _citations, tool_calls) = match reply.message() {
            Ok(message) => message,
            Err(error) => {
                out.error(error);
                return;
            }
        };

        for part in content {
            match part {
                AssistantContent::Text { text } => out.text(text),
                AssistantContent::Thinking { thinking } => out.reasoning(thinking),
            }
        }
        for call in tool_calls {
            let Some(function) = call.function else {
                continue;
            };
            // Mint absent IDs rather than using tool names, which cannot distinguish
            // repeated calls and are not provider-issued identity.
            let key = call
                .id
                .and_then(crate::streaming::non_empty_id)
                .map_or_else(|| self.tool_ids.mint(), BlockId::wire);
            let mut end = ToolCallEnd::whole(function.name, function.arguments);
            if let Some(wire_id) = key.wire_str() {
                end = end.with_tool_id(wire_id);
            }
            out.tool_call(key, end);
        }

        out.close_active_blocks();
        self.terminal(usage, finish_reason, response_id, out);
    }

    /// The terminal record both replies end with: Cohere's usage, its finish
    /// reason, and the message id it named.
    fn terminal(
        &self,
        usage: Option<Usage>,
        finish_reason: Option<FinishReason>,
        message_id: Option<String>,
        out: &mut AdapterOutput,
    ) {
        let recorded_usage = usage
            .as_ref()
            .map(crate::completion::Usage::from)
            .unwrap_or_default();
        let native = StreamingCompletionResponse {
            usage,
            finish_reason,
            message_id,
        };
        let raw = match serde_json::to_value(&native) {
            Ok(raw) => raw,
            Err(error) => {
                out.error(error.into());
                return;
            }
        };
        // Cohere's `/v2/chat` reports no model identifier in either mode, so
        // the normalized `model` stays unset.
        out.final_record(
            StreamFinal::new(PROVIDER_NAME, recorded_usage, raw)
                .with_optional_finish_reason(native.finish_reason.as_ref().map(map_finish_reason))
                .with_optional_response_id(native.message_id),
        );
    }
}

impl crate::wire::Decoder<Completion> for ChatDecoder {
    type Event = ChatEvent;

    fn classify(&self, frame: WireFrame) -> wire::WireEvent<ChatEvent> {
        // One classifier for both shapes: a modeled `type` decodes as a
        // streamed frame, an unmodeled one stays skippable, and a body with
        // no `type` at all can only be the unary reply.
        wire::classify_tagged_frame(&frame.as_str(), "type", |event_type| {
            KNOWN_EVENT_TYPES.contains(&event_type)
        })
    }

    fn interpret(&mut self, event: ChatEvent, out: &mut AdapterOutput) {
        match event {
            ChatEvent::Stream(event) => self.interpret_stream(event, out),
            ChatEvent::Reply(reply) => self.interpret_reply(reply, out),
        }
    }

    fn finish(&mut self, _out: &mut AdapterOutput) {
        // EOF without message-end is truncation, not successful completion.
    }
}

#[cfg(test)]
mod tests;
