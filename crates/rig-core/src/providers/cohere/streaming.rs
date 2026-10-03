//! Cohere chat event decoding and terminal metadata for unary and streamed replies.
//!
//! ```
//! use rig_core::providers::cohere::streaming::StreamingEvent;
//! let event: StreamingEvent = serde_json::from_str(r#"{"type":"message-end"}"#)?;
//! assert!(matches!(event, StreamingEvent::MessageEnd { delta: None }));
//! # Ok::<(), serde_json::Error>(())
//! ```

use crate::error::ProviderError;
use crate::operation::{Block, CallFragment, Completion, Finish};
use crate::providers::cohere::completion::{
    CompletionResponse, FinishReason, map_finish_reason, usage_of,
};
use crate::providers::internal::wire;
use crate::providers::openai::wire::dto::{merge_fields, open_once};
use crate::wire::{Flow, Out, WireFrame};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One streamed frame of Cohere's `/v2/chat`, named by its `type`. Content
/// and tool calls are numbered by `index` within their own kind.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "type")]
pub enum StreamingEvent {
    /// The turn opens, with the message's empty fields.
    MessageStart {
        /// The message identifier, when the wire named one.
        #[serde(default)]
        id: Option<String>,
        /// The message's opening fields.
        #[serde(default)]
        delta: Option<Delta>,
    },
    /// A content item opens.
    ContentStart {
        /// The item's position in the message's content.
        #[serde(default)]
        index: usize,
        /// The item as it opens.
        delta: Option<Delta>,
    },
    /// One fragment of a content item: text, or a reasoning model's thought
    /// text.
    ContentDelta {
        /// The item's position in the message's content.
        #[serde(default)]
        index: usize,
        /// The fragment.
        delta: Option<Delta>,
    },
    /// A content item closes.
    ContentEnd {
        /// The item's position in the message's content.
        #[serde(default)]
        index: usize,
    },
    /// One fragment of the model's plan for the tool calls that follow.
    ToolPlanDelta {
        /// The fragment.
        delta: Option<Delta>,
    },
    /// A tool call opens, naming the function it calls.
    ToolCallStart {
        /// The call's position in the message's tool calls.
        #[serde(default)]
        index: usize,
        /// The call's identity and name.
        delta: Option<Delta>,
    },
    /// One argument fragment of an open tool call.
    ToolCallDelta {
        /// The call's position in the message's tool calls.
        #[serde(default)]
        index: usize,
        /// The fragment.
        delta: Option<Delta>,
    },
    /// A tool call closes.
    ToolCallEnd {
        /// The call's position in the message's tool calls.
        #[serde(default)]
        index: usize,
    },
    /// A citation of the text opens, whole.
    CitationStart {
        /// The citation's position in the message's citations.
        #[serde(default)]
        index: usize,
        /// The citation.
        delta: Option<Delta>,
    },
    /// A citation closes.
    CitationEnd,
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
const KNOWN_EVENT_TYPES: [&str; 11] = [
    "message-start",
    "content-start",
    "content-delta",
    "content-end",
    "tool-plan-delta",
    "tool-call-start",
    "tool-call-delta",
    "tool-call-end",
    "citation-start",
    "citation-end",
    "message-end",
];

/// One frame's delta envelope: the fields of the message it adds to, as
/// Cohere sent them.
#[derive(Debug, Default, Deserialize)]
pub struct Delta {
    /// The message fields the delta carries.
    #[serde(default)]
    pub message: serde_json::Map<String, Value>,
}

impl Delta {
    /// The object the delta carries under `key`.
    fn take(self, key: &str) -> serde_json::Map<String, Value> {
        match self.message.into_iter().find(|(name, _)| name == key) {
            Some((_, Value::Object(fields))) => fields,
            _ => serde_json::Map::new(),
        }
    }
}

/// The `message-end` payload: what the turn cost and why it stopped.
#[derive(Debug, Deserialize)]
pub struct MessageEndDelta {
    /// Token counters as Cohere sent them, when it reported them.
    #[serde(default)]
    pub usage: Option<Value>,
    /// Cohere's own finish reason.
    #[serde(default)]
    pub finish_reason: Option<FinishReason>,
}

/// Cohere's terminal stream record: the `message-end` payload as rig parsed
/// it: a streamed response's `raw`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamingCompletionResponse {
    /// The usage as Cohere sent it.
    pub usage: Option<Value>,
    /// Cohere's own `finish_reason` from the `message-end` event, when reported.
    #[serde(default)]
    pub finish_reason: Option<FinishReason>,
    /// The `message-start` event's message identifier, when reported.
    #[serde(default)]
    pub message_id: Option<String>,
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

/// The writer index of tool call `index`: content items keep their own.
const CALL_INDEX: usize = 1 << 24;

/// The `/v2/chat` decoder: one state machine for the stream of events and
/// the whole reply, which is restated as the events of its message. Each
/// content item and tool call is a block holding its item; the tool plan is
/// reasoning holding its field; the assembled message is the turn's native.
#[derive(Default)]
pub struct ChatDecoder {
    /// The message as assembled so far, without its content, tool calls and
    /// citations.
    message: serde_json::Map<String, Value>,
    content: Items,
    tool_calls: Items,
    citations: Items,
    /// The writer index of the block each content index currently extends:
    /// its own index, or a fresh one once a tool call interrupted it.
    writer: std::collections::BTreeMap<usize, usize>,
    /// The writer index of the tool plan's reasoning.
    plan: Option<usize>,
    message_id: Option<String>,
}

#[deny(clippy::wildcard_enum_match_arm)]
impl ChatDecoder {
    /// Interpret one streamed `/v2/chat` frame.
    fn interpret_stream(
        &mut self,
        event: StreamingEvent,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        match event {
            StreamingEvent::MessageStart { id, delta } => {
                self.message_id = id.or(self.message_id.take());
                merge_fields(&mut self.message, &delta.unwrap_or_default().message);
            }
            StreamingEvent::ContentStart { index, delta } => {
                self.open_content(index, delta.unwrap_or_default().take("content"), out)?;
            }
            StreamingEvent::ContentDelta { index, delta } => {
                let mut item = delta.unwrap_or_default().take("content");
                if out.is_open(self.writer_of(index)) {
                    self.content_fragment(index, item, out)?;
                } else {
                    // A delta of an item whose start never came opens it, and
                    // one that continues an item a tool call interrupted opens
                    // the next block in arrival order.
                    let kind = self
                        .content
                        .get(index)
                        .and_then(|item| item.get("type"))
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                        .unwrap_or_else(|| {
                            if item.contains_key("thinking") {
                                "thinking".to_owned()
                            } else {
                                "text".to_owned()
                            }
                        });
                    item.insert("type".to_owned(), kind.into());
                    self.open_content(index, item, out)?;
                }
            }
            StreamingEvent::ContentEnd { index } => self.close_content(index, out)?,
            StreamingEvent::ToolPlanDelta { delta } => {
                let fragment = delta.unwrap_or_default().message;
                if let Some(plan) = fragment
                    .get("tool_plan")
                    .and_then(Value::as_str)
                    .filter(|plan| !plan.is_empty())
                {
                    let index =
                        open_once(&mut self.plan, Block::Reasoning { redacted: false }, out)?;
                    out.push(index, plan)?;
                }
                merge_fields(&mut self.message, &fragment);
            }
            StreamingEvent::ToolCallStart { index, delta } => {
                // A call ends the plan and the content before it, so content
                // that follows the call is a later block.
                self.close_plan(out)?;
                for content in self.content.open() {
                    self.close_content(content, out)?;
                }
                self.tool_calls
                    .start(index, Value::Object(serde_json::Map::new()));
                self.call_fragment(index, delta.unwrap_or_default().take("tool_calls"), out)?;
            }
            StreamingEvent::ToolCallDelta { index, delta } => {
                if out.is_open(CALL_INDEX + index) {
                    self.call_fragment(index, delta.unwrap_or_default().take("tool_calls"), out)?;
                }
            }
            StreamingEvent::ToolCallEnd { index } => self.close_call(index, out)?,
            StreamingEvent::CitationStart { index, delta } => {
                self.citations.start(
                    index,
                    Value::Object(delta.unwrap_or_default().take("citations")),
                );
            }
            StreamingEvent::CitationEnd => {}
            StreamingEvent::MessageEnd { .. } => {}
        }
        Ok(())
    }

    /// Open content item `index` as `item` states it: text, thinking, or an
    /// item kind rig does not model, which replays as it came. It ends the
    /// plan before it.
    fn open_content(
        &mut self,
        index: usize,
        item: serde_json::Map<String, Value>,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        self.close_plan(out)?;
        let block = match item.get("type").and_then(Value::as_str) {
            Some("text") => Block::Text,
            Some("thinking") => Block::Reasoning { redacted: false },
            _ => Block::Opaque { replay: true },
        };
        let writer = if self.writer.contains_key(&index) {
            out.fresh_index()
        } else {
            index
        };
        self.writer.insert(index, writer);
        out.open(writer, block, Value::Null)?;
        self.content
            .start(index, Value::Object(serde_json::Map::new()));
        self.content_fragment(index, item, out)
    }

    /// Close the tool plan's reasoning, holding its field.
    fn close_plan(&mut self, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        if let Some(index) = self.plan.take() {
            let mut plan = serde_json::Map::new();
            if let Some(text) = self.message.get("tool_plan") {
                plan.insert("tool_plan".to_owned(), text.clone());
            }
            out.edit(index, |item| *item = Value::Object(plan))?;
            out.finish(index)?;
        }
        Ok(())
    }

    /// The writer index content item `index` currently extends.
    fn writer_of(&self, index: usize) -> usize {
        self.writer.get(&index).copied().unwrap_or(index)
    }

    /// Close content item `index`'s open block, holding the item.
    fn close_content(
        &mut self,
        index: usize,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let writer = self.writer_of(index);
        if out.is_open(writer) {
            let item = self.content.get(index).cloned().unwrap_or_default();
            out.edit(writer, |slot| *slot = item)?;
            out.finish(writer)?;
        }
        Ok(())
    }

    /// A fragment of content item `index`: its text or thinking grows the
    /// block, and every field the item. A text that is not a string is a
    /// malformed item.
    fn content_fragment(
        &mut self,
        index: usize,
        fragment: serde_json::Map<String, Value>,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let opaque = self
            .content
            .get(index)
            .and_then(|item| item.get("type"))
            .or_else(|| fragment.get("type"))
            .and_then(Value::as_str)
            .is_none_or(|kind| !matches!(kind, "text" | "thinking"));
        let writer = self.writer_of(index);
        for key in ["text", "thinking"] {
            match fragment.get(key) {
                Some(Value::String(text)) if !opaque => out.push(writer, text)?,
                Some(Value::String(_)) | None => {}
                Some(other) => {
                    return Err(ProviderError::from(
                        <serde_json::Error as serde::de::Error>::custom(format!(
                            "Cohere content `{key}` is not a string: {other}"
                        )),
                    ));
                }
            }
        }
        if let Some(Value::Object(item)) = self.content.get_mut(index) {
            merge_fields(item, &fragment);
        }
        Ok(())
    }

    /// A fragment of tool call `index`, buffered until the call ends. A call
    /// Cohere sent without an id gets one rig issues.
    fn call_fragment(
        &mut self,
        index: usize,
        fragment: serde_json::Map<String, Value>,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let call = Value::Object(fragment);
        let text = |pointer: &str| call.pointer(pointer).and_then(Value::as_str);
        out.fragment(
            Some(CALL_INDEX + index),
            CallFragment {
                id: text("/id"),
                name: text("/function/name"),
                arguments: text("/function/arguments"),
            },
        )?;
        if let (Some(Value::Object(existing)), Value::Object(fragment)) =
            (self.tool_calls.get_mut(index), &call)
        {
            merge_fields(existing, fragment);
        }
        Ok(())
    }

    /// Close tool call `index`, holding its call. Arguments that never parse
    /// keep what they state, and a call that names no tool is dropped by the
    /// writer.
    fn close_call(
        &mut self,
        index: usize,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        if !out.is_open(CALL_INDEX + index) {
            return Ok(());
        }
        let call = self.tool_calls.get(index).cloned().unwrap_or_default();
        out.finish_with(CALL_INDEX + index, call)
    }

    /// The unary reply, restated as the events of its message, then the end
    /// the `message-end` event carries.
    fn interpret_reply(
        &mut self,
        reply: CompletionResponse,
        mut out: Out<'_, Completion>,
    ) -> Result<Flow, ProviderError> {
        let CompletionResponse {
            id,
            finish_reason,
            message,
            usage,
        } = reply;
        let Value::Object(mut message) = message else {
            return Err(ProviderError::Response(
                "completion response did not contain an assistant message".into(),
            ));
        };
        let mut items = |key: &str| match message.get_mut(key) {
            Some(Value::Array(items)) => std::mem::take(items),
            _ => Vec::new(),
        };
        let (content, calls, citations) =
            (items("content"), items("tool_calls"), items("citations"));
        let plan = message.shift_remove("tool_plan");
        let delta = |key: &str, value: Value| {
            let mut message = serde_json::Map::new();
            message.insert(key.to_owned(), value);
            Some(Delta { message })
        };
        let mut events = vec![StreamingEvent::MessageStart {
            id: Some(id).filter(|id| !id.is_empty()),
            delta: Some(Delta { message }),
        }];
        if let Some(plan) = plan {
            events.push(StreamingEvent::ToolPlanDelta {
                delta: delta("tool_plan", plan),
            });
        }
        for (index, item) in content.into_iter().enumerate() {
            events.push(StreamingEvent::ContentStart {
                index,
                delta: delta("content", item),
            });
            events.push(StreamingEvent::ContentEnd { index });
        }
        for (index, call) in calls.into_iter().enumerate() {
            events.push(StreamingEvent::ToolCallStart {
                index,
                delta: delta("tool_calls", call),
            });
            events.push(StreamingEvent::ToolCallEnd { index });
        }
        for (index, citation) in citations.into_iter().enumerate() {
            events.push(StreamingEvent::CitationStart {
                index,
                delta: delta("citations", citation),
            });
        }
        for event in events {
            self.interpret_stream(event, &mut out)?;
        }
        self.end(usage, Some(finish_reason), out, false)
    }

    /// The end both replies finish with: the tool plan closes holding its
    /// field, the assembled message becomes the turn's native, and Cohere's
    /// usage, finish reason and message id end the reply. A stream's `raw`
    /// is this native record; a whole reply's is the reply itself.
    fn end(
        &mut self,
        usage: Option<Value>,
        finish_reason: Option<FinishReason>,
        mut out: Out<'_, Completion>,
        streamed: bool,
    ) -> Result<Flow, ProviderError> {
        self.close_plan(&mut out)?;
        for index in self.tool_calls.open() {
            self.close_call(index, &mut out)?;
        }
        for index in self.content.open() {
            self.close_content(index, &mut out)?;
        }
        let mut message = std::mem::take(&mut self.message);
        for (key, items) in [
            ("content", std::mem::take(&mut self.content).into_values()),
            (
                "tool_calls",
                std::mem::take(&mut self.tool_calls).into_values(),
            ),
            (
                "citations",
                std::mem::take(&mut self.citations).into_values(),
            ),
        ] {
            if !items.is_empty() || message.contains_key(key) {
                message.insert(key.to_owned(), Value::Array(items));
            }
        }
        let recorded_usage = usage.as_ref().map(usage_of).unwrap_or_default();
        let native = StreamingCompletionResponse {
            usage,
            finish_reason,
            message_id: self.message_id.take(),
        };
        if streamed {
            out.raw(serde_json::to_value(&native)?);
        }
        // Cohere's `/v2/chat` reports no model identifier in either mode, so
        // the normalized `model` stays unset.
        Ok(out.end(Finish {
            usage: recorded_usage,
            reason: native.finish_reason.as_ref().map(map_finish_reason),
            response_id: native.message_id,
            ..Finish::default()
        }))
    }
}

/// One kind of the message's items in arrival order, keyed by the wire
/// index of the item each currently extends.
#[derive(Default)]
struct Items {
    values: Vec<Value>,
    at: std::collections::BTreeMap<usize, usize>,
}

impl Items {
    /// A new item at wire `index`, after every item so far.
    fn start(&mut self, index: usize, item: Value) {
        self.at.insert(index, self.values.len());
        self.values.push(item);
    }

    fn get(&self, index: usize) -> Option<&Value> {
        self.values.get(*self.at.get(&index)?)
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut Value> {
        self.values.get_mut(*self.at.get(&index)?)
    }

    /// The wire indices items were last started at.
    fn open(&self) -> Vec<usize> {
        self.at.keys().copied().collect()
    }

    /// The items, without the ones dropped.
    fn into_values(self) -> Vec<Value> {
        self.values
            .into_iter()
            .filter(|item| !item.is_null())
            .collect()
    }
}

#[deny(clippy::wildcard_enum_match_arm)]
impl<'id> crate::wire::Decoder<'id, Completion> for ChatDecoder {
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
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match event {
            // A bare message-end still completes the turn with unknown usage
            // and reason.
            ChatEvent::Stream(StreamingEvent::MessageEnd { delta }) => {
                let (usage, finish_reason) = match delta {
                    Some(delta) => (delta.usage, delta.finish_reason),
                    None => (None, None),
                };
                self.end(usage, finish_reason, out, true)
            }
            ChatEvent::Stream(event) => {
                self.interpret_stream(event, &mut out)?;
                Ok(Flow::More)
            }
            ChatEvent::Reply(reply) => self.interpret_reply(reply, out),
        }
    }
}

#[cfg(test)]
mod tests;
