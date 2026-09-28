//! Cohere chat event decoding and terminal metadata for unary and streamed replies.
//!
//! ```
//! use rig_core::providers::cohere::streaming::StreamingEvent;
//! let event: StreamingEvent = serde_json::from_str(r#"{"type":"message-end"}"#)?;
//! assert!(matches!(event, StreamingEvent::MessageEnd { delta: None }));
//! # Ok::<(), serde_json::Error>(())
//! ```

use crate::error::ProviderError;
use crate::operation::{CallFragment, Completion, Finish, IfMalformed, TextPart};
use crate::providers::cohere::completion::{
    AssistantContent, CompletionResponse, FinishReason, Usage, map_finish_reason,
};
use crate::providers::internal::thoughts::Thoughts;
use crate::providers::internal::wire;
use crate::wire::{Flow, Out, WireFrame};
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
/// it: a streamed response's `raw`.
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

/// The `/v2/chat` decoder: one state machine for the whole reply and its
/// stream of events.
pub struct ChatDecoder<'id> {
    /// The wire index the open tool call's fragments are buffered under.
    current_tool_call: Option<usize>,
    /// Tool calls opened so far.
    calls: usize,
    message_id: Option<String>,
    /// Reasoning closes when subsequent content changes block type.
    thoughts: Thoughts<'id>,
    text: Option<TextPart<'id>>,
}

impl Default for ChatDecoder<'_> {
    fn default() -> Self {
        Self {
            current_tool_call: None,
            calls: 0,
            message_id: None,
            thoughts: Thoughts::new(),
            text: None,
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

impl<'id> ChatDecoder<'id> {
    fn close_text(&mut self, out: &mut Out<'id, Completion>) {
        if let Some(part) = self.text.take() {
            out.close_text(part);
        }
    }

    /// Reasoning then text, as one content fragment carries them.
    fn content(
        &mut self,
        out: &mut Out<'id, Completion>,
        thinking: Option<&str>,
        text: Option<&str>,
    ) {
        if let Some(thinking) = thinking.filter(|thinking| !thinking.is_empty()) {
            self.close_text(out);
            self.thoughts.fragment(out, thinking);
        }
        if let Some(text) = text.filter(|text| !text.is_empty()) {
            self.thoughts.boundary();
            let part = self.text.get_or_insert_with(|| out.text());
            out.push_text(part, text);
        }
    }

    /// Interpret one streamed `/v2/chat` frame.
    fn interpret_stream(
        &mut self,
        event: StreamingEvent,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match event {
            StreamingEvent::MessageStart { id: Some(id) } => {
                self.message_id = Some(id);
            }

            StreamingEvent::ContentDelta { delta: Some(delta) } => {
                if let Some(content) = delta
                    .message
                    .as_ref()
                    .and_then(|message| message.content.as_ref())
                {
                    self.content(
                        &mut out,
                        content.thinking.as_deref(),
                        content.text.as_deref(),
                    );
                }
            }

            StreamingEvent::MessageEnd { delta } => {
                // A bare message-end still completes the turn with unknown usage and reason.
                let (usage, finish_reason) = match delta {
                    Some(delta) => (delta.usage, delta.finish_reason),
                    None => (None, None),
                };
                let message_id = self.message_id.take();
                return self.end(usage, finish_reason, message_id, out, true);
            }

            StreamingEvent::ToolCallStart { delta: Some(delta) } => {
                let Some(tool_calls) = delta
                    .message
                    .as_ref()
                    .and_then(|message| message.tool_calls.as_ref())
                else {
                    return Ok(Flow::More);
                };
                let (Some(id), Some(function)) = (&tool_calls.id, &tool_calls.function) else {
                    return Ok(Flow::More);
                };
                let (Some(name), Some(arguments)) = (&function.name, &function.arguments) else {
                    return Ok(Flow::More);
                };
                // Tool content interleaving an open thinking part stops it.
                self.thoughts.boundary();
                self.close_text(&mut out);
                let index = self.calls;
                self.calls += 1;
                self.current_tool_call = Some(index);
                // `tool-call-start` may carry initial argument text; on the
                // wire it is empty, but any payload is part of the call.
                out.call_fragment(
                    index,
                    CallFragment {
                        id: Some(id.as_str()),
                        name: Some(name.as_str()),
                        arguments: Some(arguments.as_str()),
                        ..CallFragment::default()
                    },
                )?;
            }

            StreamingEvent::ToolCallDelta { delta: Some(delta) } => {
                let Some(arguments) = delta
                    .message
                    .as_ref()
                    .and_then(|message| message.tool_calls.as_ref())
                    .and_then(|tool_calls| tool_calls.function.as_ref())
                    .and_then(|function| function.arguments.as_deref())
                else {
                    return Ok(Flow::More);
                };
                // A delta with no open call has nothing to extend; the wire
                // never starts a call mid-delta.
                if let Some(index) = self.current_tool_call {
                    out.call_fragment(
                        index,
                        CallFragment {
                            arguments: Some(arguments),
                            ..CallFragment::default()
                        },
                    )?;
                }
            }

            StreamingEvent::ToolCallEnd => {
                // This endpoint drops calls whose assembled arguments are unparseable.
                if let Some(index) = self.current_tool_call.take() {
                    out.close_pending(index, IfMalformed::Drop)?;
                }
            }

            _ => {}
        }
        Ok(Flow::More)
    }

    /// The unary reply, written as the stream it would have been: its
    /// content parts, each tool call whole, then the end the `message-end`
    /// event carries.
    fn interpret_reply(
        &mut self,
        reply: CompletionResponse,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        let response_id = Some(reply.id.clone()).filter(|id| !id.is_empty());
        let finish_reason = Some(reply.finish_reason.clone());
        let usage = reply.usage;
        let (content, _citations, tool_calls) = reply.message()?;

        for part in content {
            match part {
                AssistantContent::Text { text } => self.content(&mut out, None, Some(&text)),
                AssistantContent::Thinking { thinking } => {
                    self.content(&mut out, Some(&thinking), None);
                }
            }
        }
        self.thoughts.boundary();
        self.close_text(&mut out);
        for call in tool_calls {
            let Some(function) = call.function else {
                continue;
            };
            // An absent id is issued by rig, never taken from the tool name,
            // which cannot tell repeated calls apart.
            let index = self.calls;
            self.calls += 1;
            out.call_fragment(
                index,
                CallFragment {
                    id: call.id.as_deref(),
                    name: Some(function.name.as_str()),
                    ..CallFragment::default()
                },
            )?;
            out.announce_pending(index, function.arguments);
            out.close_pending(index, IfMalformed::Fail)?;
        }
        self.end(usage, finish_reason, response_id, out, false)
    }

    /// The end both replies finish with: Cohere's usage, its finish reason,
    /// and the message id it named. A stream's `raw` is this native record;
    /// a whole reply's is the reply itself.
    fn end(
        &mut self,
        usage: Option<Usage>,
        finish_reason: Option<FinishReason>,
        message_id: Option<String>,
        mut out: Out<'id, Completion>,
        streamed: bool,
    ) -> Result<Flow, ProviderError> {
        self.close_text(&mut out);
        self.thoughts.close(&mut out, None);
        let recorded_usage = usage
            .as_ref()
            .map(crate::completion::Usage::from)
            .unwrap_or_default();
        let native = StreamingCompletionResponse {
            usage,
            finish_reason,
            message_id,
        };
        if streamed {
            out.raw(serde_json::to_value(&native)?);
        }
        // Cohere's `/v2/chat` reports no model identifier in either mode, so
        // the normalized `model` stays unset.
        Ok(out.end(
            Finish::new(recorded_usage)
                .with_optional_reason(native.finish_reason.as_ref().map(map_finish_reason))
                .with_optional_response_id(native.message_id),
        ))
    }
}

impl<'id> crate::wire::Decoder<'id, Completion> for ChatDecoder<'id> {
    type Event = ChatEvent;

    fn classify(&self, frame: WireFrame) -> crate::wire::WireEvent<ChatEvent> {
        // One classifier for both shapes: a modeled `type` decodes as a
        // streamed frame, an unmodeled one stays skippable, and a body with
        // no `type` at all can only be the unary reply.
        wire::classify_tagged_frame(&frame.as_str(), "type", |event_type| {
            KNOWN_EVENT_TYPES.contains(&event_type)
        })
    }

    /// EOF without message-end is truncation, not successful completion.
    fn decode(
        &mut self,
        event: ChatEvent,
        out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match event {
            ChatEvent::Stream(event) => self.interpret_stream(event, out),
            ChatEvent::Reply(reply) => self.interpret_reply(reply, out),
        }
    }
}

#[cfg(test)]
mod tests;
