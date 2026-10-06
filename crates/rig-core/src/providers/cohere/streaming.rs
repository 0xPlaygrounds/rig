//! The native chat reply decoder, for a whole reply and for the event
//! stream alike. Each content part, the tool plan and each tool call
//! becomes one block in the order it arrives. A block's provider item is
//! the part as Cohere stated it, with the citations that point at it, so
//! citations and the tool plan round-trip through history.
//!
//! ```
//! use rig_core::providers::cohere::streaming::ChatDecoder;
//!
//! let decoder = ChatDecoder::default();
//! # let _ = decoder;
//! ```

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value, json};

use super::chat::PLAN;
use crate::completion::{FinishReason, Usage};
use crate::error::ProviderError;
use crate::json_utils::Lenient;
use crate::message::{CallId, ToolName};
use crate::operation::{Block, CallFragment, Completion, Finish};
use crate::providers::internal::wire;
use crate::wire::{
    AdapterEvent, AdapterUsage, AdapterVerdict, Decoder, Flow, ObservationSink, Out, WireEvent,
    WireFrame,
};

/// The stream's event tags; any other tag classifies as unknown.
const KNOWN_EVENT_TYPES: &[&str] = &[
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
    "debug",
];

/// The wire index of the tool plan. Content parts keep their own indices,
/// and calls theirs from [`CALLS`] on.
const PLAN_INDEX: usize = 1 << 21;

/// The wire index of the first tool call. A content or call index Cohere
/// states must stay below it.
const CALLS: usize = 1 << 20;

/// `index` as Cohere stated it for a content part or call, failing when it
/// would reach the indices the decoder keeps for calls and the plan.
fn checked(index: usize) -> Result<usize, ProviderError> {
    if index < CALLS {
        Ok(index)
    } else {
        Err(ProviderError::Response(format!(
            "Cohere stated index {index}, past the {CALLS} parts or calls a reply may hold"
        )))
    }
}

/// One stream event, or the whole reply a unary call answers (`kind` is
/// then empty), as Cohere sent it.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatEvent {
    /// The event as sent.
    pub fields: Value,
}

impl ChatEvent {
    fn kind(&self) -> &str {
        self.fields.str("type").unwrap_or_default()
    }

    /// The part or call the event addresses.
    fn index(&self) -> Result<usize, ProviderError> {
        self.fields
            .u64("index")
            .and_then(|index| usize::try_from(index).ok())
            .ok_or_else(|| {
                ProviderError::Response(format!("Cohere `{}` names no index", self.kind()))
            })
            .and_then(checked)
    }

    /// The object at `pointer` under the event's `delta.message`.
    fn message(&self, key: &str) -> Option<&Map<String, Value>> {
        self.fields
            .at(&format!("/delta/message/{key}"))
            .and_then(Value::as_object)
    }
}

/// What an open block is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Text,
    Thinking,
    Plan,
    Call,
    Opaque,
}

/// Decodes native chat replies, a whole reply or a stream of events. An
/// `*-end` event states its block complete, and `message-end` states every
/// block still open complete, but a call whose arguments are not JSON.
#[derive(Debug, Default)]
pub struct ChatDecoder {
    /// Each open block's kind and, for a call, the argument text streamed.
    open: BTreeMap<usize, (Kind, String)>,
    /// Every index a block opened at, so a citation finds its block.
    started: BTreeSet<usize>,
    message_id: Option<String>,
}

impl ChatDecoder {
    /// Open the content part at `index` as Cohere states it.
    fn content(
        &mut self,
        index: usize,
        part: &Map<String, Value>,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let index = checked(index)?;
        let part = Value::Object(part.clone());
        let (kind, block, key) = match part.str("type") {
            Some("text") | None => (Kind::Text, Block::Text, "text"),
            Some("thinking") => (
                Kind::Thinking,
                Block::Reasoning { redacted: false },
                "thinking",
            ),
            // A part rig does not know goes back as it came.
            Some(_) => (Kind::Opaque, Block::Opaque { replay: true }, ""),
        };
        let text = part.str(key).unwrap_or_default().to_owned();
        self.open.insert(index, (kind, String::new()));
        self.started.insert(index);
        out.open(index, block, part)?;
        out.push(index, &text)
    }

    /// Grow the open part at `index`, and its item, by a delta's text.
    fn grow(
        &mut self,
        index: usize,
        delta: &Map<String, Value>,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let Some((kind, _)) = self.open.get(&index) else {
            return Err(ProviderError::Response(format!(
                "Cohere streamed content to part {index}, which is not open"
            )));
        };
        let key = match kind {
            Kind::Thinking => "thinking",
            Kind::Plan => PLAN,
            Kind::Text | Kind::Call => "text",
            // An unknown part's deltas merge into its item.
            Kind::Opaque => {
                return out.edit(index, |item| {
                    crate::operation::completion::merge(item, delta)
                });
            }
        };
        let Some(text) = delta.get(key).and_then(Value::as_str) else {
            return Ok(());
        };
        out.push(index, text)?;
        let delta = Map::from_iter([(key.to_owned(), Value::from(text))]);
        out.edit(index, |item| {
            crate::operation::completion::merge(item, &delta)
        })
    }

    /// Append a fragment of the tool plan, opening it at its first.
    fn plan(&mut self, text: &str, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        if !self.started.contains(&PLAN_INDEX) {
            self.open.insert(PLAN_INDEX, (Kind::Plan, String::new()));
            self.started.insert(PLAN_INDEX);
            let item = json!({"type": PLAN, PLAN: ""});
            out.open(PLAN_INDEX, Block::Reasoning { redacted: false }, item)?;
        }
        let delta = Map::from_iter([(PLAN.to_owned(), Value::from(text))]);
        self.grow(PLAN_INDEX, &delta, out)
    }

    /// Open the tool call at `index` from its stated id, name and the
    /// arguments it opens with.
    fn call(
        &mut self,
        index: usize,
        call: &Map<String, Value>,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        // The plan is whole once the calls it plans begin.
        if self.open.contains_key(&PLAN_INDEX) {
            self.stop(PLAN_INDEX, out)?;
        }
        let index = CALLS + checked(index)?;
        let item = Value::Object(call.clone());
        let id = item.str("id").unwrap_or_default().to_owned();
        let arguments = item
            .at("/function/arguments")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        self.open.insert(index, (Kind::Call, String::new()));
        self.started.insert(index);
        match ToolName::new(
            item.at("/function/name")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        ) {
            Ok(name) => {
                let id = CallId::from_wire(&id);
                out.open(index, Block::Call { id, name }, item)?;
            }
            // A nameless call: the writer drops it with a warning.
            Err(_) => out.fragment(
                Some(index),
                CallFragment {
                    id: Some(&id),
                    ..CallFragment::default()
                },
            )?,
        }
        self.arguments(index, &arguments, out)
    }

    /// Append argument text to the open call at `index`.
    fn arguments(
        &mut self,
        index: usize,
        fragment: &str,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let Some((Kind::Call, json)) = self.open.get_mut(&index) else {
            return Err(ProviderError::Response(format!(
                "Cohere streamed arguments to call {}, which is not open",
                index.saturating_sub(CALLS)
            )));
        };
        json.push_str(fragment);
        out.push(index, fragment)
    }

    /// Attach `citation` to the item of the block it cites: the tool plan,
    /// or the content part at its `content_index` (the first by default).
    /// A citation of a block that never opened has nowhere to go, and one
    /// whose `content_index` reaches [`CALLS`] fails the reply.
    fn cite(&self, citation: Value, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        let index = if citation.str("type") == Some("PLAN") {
            PLAN_INDEX
        } else {
            checked(
                citation
                    .u64("content_index")
                    .map_or(Ok(0), usize::try_from)
                    .unwrap_or(usize::MAX),
            )?
        };
        if !self.started.contains(&index) {
            tracing::warn!(
                index,
                "Cohere cited a block the reply never opened; dropping it"
            );
            return Ok(());
        }
        out.edit(index, |item| {
            if let Some(item) = item.as_object_mut() {
                match item.get_mut("citations") {
                    Some(Value::Array(citations)) => citations.push(citation),
                    _ => {
                        item.insert("citations".to_owned(), Value::Array(vec![citation]));
                    }
                }
            }
        })
    }

    /// End the block at `index` as stated complete. Its item becomes the
    /// block's native unless it would replay nothing: blank text, or a call
    /// whose arguments are not a JSON object.
    fn stop(&mut self, index: usize, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        let Some((kind, json)) = self.open.remove(&index) else {
            return Err(ProviderError::Response(format!(
                "Cohere ended block {index}, which is not open"
            )));
        };
        out.edit(index, |item| {
            let kept = match kind {
                Kind::Text => !item.str("text").unwrap_or_default().trim().is_empty(),
                Kind::Thinking | Kind::Plan | Kind::Opaque => true,
                Kind::Call => {
                    let arguments = if json.trim().is_empty() {
                        "{}"
                    } else {
                        json.as_str()
                    };
                    let object = crate::json_utils::parse_tool_arguments(arguments)
                        .is_ok_and(|parsed| parsed.is_object());
                    if let Some(function) = item.get_mut("function").and_then(Value::as_object_mut)
                    {
                        function.insert("arguments".to_owned(), Value::from(arguments));
                    }
                    object
                }
            };
            if !kept {
                *item = Value::Null;
            }
        })?;
        out.finish(index)
    }

    /// End the reply: every block still open is complete, but a call whose
    /// arguments are not JSON yet, which the end closes unfinished.
    fn end(
        &mut self,
        usage: Option<&Value>,
        reason: Option<&str>,
        error: Option<&str>,
        mut out: Out<'_, Completion>,
    ) -> Result<Flow, ProviderError> {
        let open: Vec<usize> = self
            .open
            .iter()
            .filter(|(_, (kind, json))| {
                *kind != Kind::Call
                    || json.trim().is_empty()
                    || crate::json_utils::parse_tool_arguments(json).is_ok_and(|v| v.is_object())
            })
            .map(|(index, _)| *index)
            .collect();
        for index in open {
            self.stop(index, &mut out)?;
        }
        let error = error
            .filter(|error| !error.is_empty())
            .map(str::to_owned)
            .or_else(|| {
                (reason == Some("ERROR")).then(|| "Cohere ended the reply with an error".to_owned())
            });
        Ok(out.end(Finish {
            usage: usage_of(usage),
            reason: reason.map(finish_of),
            response_id: self.message_id.clone(),
            model: None,
            error,
        }))
    }

    /// A whole reply, written block by block through the calls a stream
    /// makes, then ended.
    fn whole(
        &mut self,
        reply: &Value,
        mut out: Out<'_, Completion>,
    ) -> Result<Flow, ProviderError> {
        self.message_id = reply.str("id").map(str::to_owned);
        // A `message` string is Cohere's error body, sent with a success
        // status.
        let message = match reply.get("message") {
            Some(Value::String(_)) => {
                return Err(ProviderError::from_provider_body(reply.to_string()));
            }
            message => message.unwrap_or(&Value::Null),
        };
        // The plan leads, as it streams first.
        if let Some(plan) = message.str(PLAN).filter(|plan| !plan.is_empty()) {
            self.plan(plan, &mut out)?;
            self.stop(PLAN_INDEX, &mut out)?;
        }
        // A part or call that is not an object states nothing to keep.
        for (index, part) in message.arr("content").iter().enumerate() {
            if let Some(part) = part.as_object() {
                self.content(index, part, &mut out)?;
                self.stop(index, &mut out)?;
            }
        }
        for (index, call) in message.arr("tool_calls").iter().enumerate() {
            if let Some(call) = call.as_object() {
                self.call(index, call, &mut out)?;
                self.stop(CALLS + index, &mut out)?;
            }
        }
        for citation in message.arr("citations") {
            self.cite(citation.clone(), &mut out)?;
        }
        self.end(reply.get("usage"), reply.str("finish_reason"), None, out)
    }
}

/// How a native `finish_reason` ends the turn. `ERROR`, `TIMEOUT` and any
/// reason Cohere does not document are [`FinishReason::Other`], which fails
/// the turn.
fn finish_of(reason: &str) -> FinishReason {
    match reason {
        "COMPLETE" | "STOP_SEQUENCE" => FinishReason::Stop,
        "MAX_TOKENS" => FinishReason::Length,
        "TOOL_CALL" => FinishReason::ToolCalls,
        other => FinishReason::Other(other.to_owned()),
    }
}

/// Rig's usage from Cohere's: the `tokens` the model read and wrote,
/// else the `billed_units`, with `cached_tokens` among the input.
fn usage_of(usage: Option<&Value>) -> Usage {
    let count = |pointer: &str| usage?.at(pointer)?.as_u64_lenient();
    let input = count("/tokens/input_tokens").or_else(|| count("/billed_units/input_tokens"));
    let output = count("/tokens/output_tokens").or_else(|| count("/billed_units/output_tokens"));
    Usage {
        input_tokens: input,
        output_tokens: output,
        cached_input_tokens: count("/cached_tokens"),
        cache_creation_input_tokens: None,
        reasoning_tokens: count("/tokens/reasoning_tokens"),
        total_tokens: input.zip(output).map(|(input, output)| input + output),
        tool_use_prompt_tokens: None,
        cost: None,
    }
}

impl<'id> Decoder<'id, Completion> for ChatDecoder {
    type Event = ChatEvent;

    /// A stream event by its `type`; a frame without one is the whole
    /// reply, recognized by its `message`.
    fn classify(&self, frame: WireFrame) -> WireEvent<ChatEvent> {
        let data = frame.as_str();
        wire::classify_or_untagged(
            &data,
            "type",
            |data| {
                wire::classify_tagged_frame::<Value>(data, "type", |tag| {
                    KNOWN_EVENT_TYPES.contains(&tag)
                })
            },
            |data| wire::classify_marker_keyed_frame::<Value>(data, &["message"]),
        )
        .map(|fields| ChatEvent { fields })
    }

    fn decode(
        &mut self,
        event: ChatEvent,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match event.kind() {
            "" => return self.whole(&event.fields, out),
            "message-start" => self.message_id = event.fields.str("id").map(str::to_owned),
            "content-start" => {
                let part = event.message("content").cloned().unwrap_or_default();
                self.content(event.index()?, &part, &mut out)?;
            }
            "content-delta" => {
                let delta = event.message("content").cloned().unwrap_or_default();
                self.grow(event.index()?, &delta, &mut out)?;
            }
            "content-end" => self.stop(event.index()?, &mut out)?,
            "tool-plan-delta" => {
                if let Some(text) = event
                    .fields
                    .at("/delta/message/tool_plan")
                    .and_then(Value::as_str)
                {
                    self.plan(text, &mut out)?;
                }
            }
            "tool-call-start" => {
                let call = event.message("tool_calls").cloned().unwrap_or_default();
                self.call(event.index()?, &call, &mut out)?;
            }
            "tool-call-delta" => {
                let fragment = event
                    .fields
                    .at("/delta/message/tool_calls/function/arguments")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                self.arguments(CALLS + event.index()?, fragment, &mut out)?;
            }
            "tool-call-end" => self.stop(CALLS + event.index()?, &mut out)?,
            "citation-start" => {
                if let Some(citation) = event.message("citations") {
                    self.cite(Value::Object(citation.clone()), &mut out)?;
                }
            }
            "message-end" => {
                let delta = event.fields.get("delta");
                return self.end(
                    delta.and_then(|delta| delta.get("usage")),
                    delta.and_then(|delta| delta.str("finish_reason")),
                    delta.and_then(|delta| delta.str("error")),
                    out,
                );
            }
            // `citation-end` and `debug`: the classifier passes only the
            // listed tags.
            _ => {}
        }
        Ok(Flow::More)
    }
}

impl ChatDecoder {
    /// Native chat metadata projected before normalization can discard it:
    /// the finish reason, the reply id and the usage, on the unary reply
    /// and on the stream's `message-start` and `message-end` events.
    pub(crate) fn project(payload: &[u8], sink: &mut ObservationSink<'_>) {
        let Ok(payload) = serde_json::from_slice::<Value>(payload) else {
            return;
        };
        let end = payload.get("delta").unwrap_or(&payload);
        if let Some(usage) = end.get("usage") {
            let usage = usage_of(Some(usage));
            sink.emit(AdapterEvent::Usage {
                usage: AdapterUsage {
                    input_tokens: usage.input_tokens,
                    output_tokens: usage.output_tokens,
                    total_tokens: usage.total_tokens,
                    cached_input_tokens: usage.cached_input_tokens,
                    reasoning_tokens: usage.reasoning_tokens,
                    tool_input_tokens: None,
                },
            });
        }
        let verdict = AdapterVerdict {
            finish_reason: end.str("finish_reason").map(|reason| sink.scrub(reason)),
            block_reason: None,
            detail: None,
            model: None,
        };
        let response_id = payload.str("id").map(|id| sink.scrub(id));
        sink.provider(verdict, response_id);
    }
}

pub(crate) mod document;

#[cfg(test)]
mod tests;
