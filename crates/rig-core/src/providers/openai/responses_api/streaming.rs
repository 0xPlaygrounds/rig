//! Responses frame classification and event decoding: one block per output
//! item, at its output index, each holding the item as the provider stated it
//! complete.
//!
//! Frames are read as JSON and classified by their `type` alone. The decoder
//! takes only the fields a block or the finish is built from, so a gateway
//! that omits or retypes any other field never fails a reply.
//!
//! ```
//! use rig_core::providers::openai::responses_api::streaming::ResponsesDecoder;
//! let decoder = ResponsesDecoder::new();
//! # let _ = decoder;
//! ```

use std::collections::{HashMap, HashSet};

use serde_json::Value;

use crate::completion::FinishReason;
use crate::error::ProviderError;
use crate::operation::{Block, CallFragment, Completion, Finish};
use crate::providers::internal::wire;
use crate::wire::{Decoder, Flow, Out, WireEvent, WireFrame};

/// Whether `kind` is a Responses event type this decoder reads. A frame of
/// any other type passes through as unknown.
fn is_known_responses_event_type(kind: &str) -> bool {
    matches!(
        kind,
        "error"
            | "response.created"
            | "response.queued"
            | "response.in_progress"
            | "response.completed"
            | "response.failed"
            | "response.incomplete"
            | "response.output_item.added"
            | "response.output_item.done"
            | "response.content_part.added"
            | "response.content_part.done"
            | "response.output_text.delta"
            | "response.output_text.done"
            | "response.refusal.delta"
            | "response.refusal.done"
            | "response.function_call_arguments.delta"
            | "response.function_call_arguments.done"
            | "response.custom_tool_call_input.delta"
            | "response.custom_tool_call_input.done"
            | "response.reasoning_summary_part.added"
            | "response.reasoning_summary_part.done"
            | "response.reasoning_summary_text.delta"
            | "response.reasoning_summary_text.done"
            | "response.reasoning_text.delta"
            | "response.reasoning_text.done"
    )
}

/// Whether `kind` is a response lifecycle event, which carries the response
/// object under `response`.
pub(crate) fn is_lifecycle_event(kind: &str) -> bool {
    matches!(
        kind,
        "response.created"
            | "response.queued"
            | "response.in_progress"
            | "response.completed"
            | "response.failed"
            | "response.incomplete"
    )
}

/// A classified Responses payload.
#[derive(Debug)]
pub enum ResponsesEvent {
    /// A stream event of a known `type`, as the provider sent it.
    Frame {
        /// The event's `type`.
        kind: String,
        /// The event.
        frame: Value,
        /// The frame's payload, verbatim, for the error a failed reply
        /// reports.
        raw: String,
    },
    /// The unary reply: the response object itself, which carries no `type`
    /// because it is not an event.
    Whole(Value),
    /// The provider's error, as the stream's `error` event or a success
    /// body holding an error envelope instead of a response.
    Failure(String),
    /// The `[DONE]` sentinel. The provider's end is `response.completed`.
    Sentinel,
}

/// The keys a response object, or the error envelope a success body can
/// hold instead, always has one of.
const WHOLE_BODY_MARKERS: &[&str] = &["object", "output", "status", "id", "error"];

/// A payload that names its `type`, read as the JSON it is.
struct Tagged(Value);

impl<'de> serde::Deserialize<'de> for Tagged {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        if value.get("type").is_some_and(Value::is_string) {
            Ok(Self(value))
        } else {
            Err(serde::de::Error::custom("the payload names no `type`"))
        }
    }
}

/// Classify one payload by its `type`; a payload without one is the unary
/// response object or an error envelope.
pub fn classify_responses_payload(data: &str) -> WireEvent<ResponsesEvent> {
    if data.trim() == "[DONE]" {
        return WireEvent::Known(ResponsesEvent::Sentinel);
    }
    wire::classify_or_untagged(
        data,
        "type",
        |data| {
            wire::classify_tagged_frame::<Tagged>(data, "type", is_known_responses_event_type)
                .map(|Tagged(frame)| event_of(frame, data))
        },
        |data| {
            wire::classify_marker_keyed_frame::<Value>(data, WHOLE_BODY_MARKERS)
                .map(|body| body_of(body, data))
        },
    )
}

/// A known event: the stream's `error` event is the provider's failure.
fn event_of(frame: Value, data: &str) -> ResponsesEvent {
    match string(&frame, "type") {
        Some("error") => ResponsesEvent::Failure(data.to_owned()),
        kind => ResponsesEvent::Frame {
            kind: kind.unwrap_or_default().to_owned(),
            frame,
            raw: data.to_owned(),
        },
    }
}

/// A body: the error envelope when it holds an error and no response.
fn body_of(body: Value, data: &str) -> ResponsesEvent {
    let error = body.get("error").is_some_and(|error| !error.is_null());
    if error && body.get("output").is_none() && body.get("status").is_none() {
        ResponsesEvent::Failure(data.to_owned())
    } else {
        ResponsesEvent::Whole(body)
    }
}

/// The OpenAI Responses wire's decoder: one state machine for the SSE
/// stream, the unary body and the websocket session.
///
/// Each output item becomes one block at its `output_index`. It opens at
/// `output_item.added` with no provider item, and `output_item.done` closes
/// it with the done item as its native: an item the provider never stated
/// complete replays from its canonical fields. A call opens when it is
/// announced and its argument deltas accumulate, so a call never done is
/// still delivered. A terminal response restates the whole output: the
/// items no stream event carried are written from it at their index, which
/// is how a unary body decodes.
#[derive(Default)]
pub struct ResponsesDecoder {
    /// Whether the reply's blocks were asked to follow output indices.
    ordered: bool,
    /// The output indices an item opened or was written at.
    seen: HashSet<usize>,
    /// What each open index holds, as the decoder opened it.
    kinds: HashMap<usize, Kind>,
    /// The id of the item `output_item.added` opened at each index.
    added_ids: HashMap<usize, String>,
    /// The output indices and item ids whose done item was written.
    written: HashSet<usize>,
    written_ids: HashSet<String>,
    /// The text deltas wrote to each open item.
    texts: HashMap<usize, Streamed>,
    /// The input deltas of each open custom tool call.
    custom_inputs: HashMap<usize, String>,
    /// Reasoning items done without their ciphertext, by index and id. Azure
    /// states `encrypted_content` only in the terminal response, so these
    /// close there.
    awaiting_ciphertext: Vec<(usize, String)>,
}

/// The block an open index holds.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Message,
    Reasoning,
    Call,
    Opaque,
}

/// What deltas wrote to one item: the field and part that last extended
/// it, since a new part of reasoning starts a new paragraph and a second
/// field restating the same reasoning is not appended, and the text so far.
struct Streamed {
    field: TextField,
    part: u64,
    text: String,
}

/// The field a text delta arrived in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TextField {
    Message,
    Summary,
    Reasoning,
}

/// The item types the client executes, which nothing in rig answers: they
/// are kept in history and never sent back.
const CLIENT_EXECUTED: &[&str] = &[
    "computer_call",
    "local_shell_call",
    "shell_call",
    "apply_patch_call",
    "mcp_approval_request",
];

fn kind_of(item: &Value) -> Option<&str> {
    item.get("type").and_then(Value::as_str)
}

fn string<'a>(item: &'a Value, key: &str) -> Option<&'a str> {
    item.get(key).and_then(Value::as_str)
}

fn item_id(item: &Value) -> Option<&str> {
    string(item, "id").filter(|id| !id.is_empty())
}

fn has_ciphertext(item: &Value) -> bool {
    string(item, "encrypted_content").is_some_and(|ciphertext| !ciphertext.is_empty())
}

/// The index or part number `key` states, zero when absent: a gateway that
/// omits its indices streams one item at a time.
fn number(frame: &Value, key: &str) -> u64 {
    frame.get(key).and_then(Value::as_u64).unwrap_or(0)
}

fn output_index(frame: &Value) -> Result<usize, ProviderError> {
    let index = number(frame, "output_index");
    usize::try_from(index)
        .map_err(|_| ProviderError::Response(format!("output_index {index} is out of range")))
}

/// Whether `item` is a call, and of which form.
fn is_call(item: &Value) -> bool {
    matches!(kind_of(item), Some("function_call" | "custom_tool_call"))
}

/// The argument JSON a call item states: a function call's `arguments`, or
/// a custom call's free-form `input` as `{"input": ...}`.
fn arguments_of(item: &Value) -> String {
    match kind_of(item) {
        Some("custom_tool_call") => {
            serde_json::json!({ "input": string(item, "input").unwrap_or_default() }).to_string()
        }
        _ => match item.get("arguments") {
            Some(Value::String(arguments)) => arguments.clone(),
            Some(Value::Null) | None => String::new(),
            Some(arguments) => arguments.to_string(),
        },
    }
}

/// The text a done item states: a message's content parts joined, refusals
/// included; reasoning's summary, or its raw content when it has none.
fn text_of(item: &Value) -> String {
    let texts = |key: &str| -> Vec<String> {
        match item.get(key) {
            Some(Value::Array(parts)) => parts
                .iter()
                .filter_map(|part| match part {
                    Value::String(text) => Some(text.clone()),
                    part => string(part, "text")
                        .or_else(|| string(part, "refusal"))
                        .map(str::to_owned),
                })
                .collect(),
            Some(Value::String(text)) => vec![text.clone()],
            _ => Vec::new(),
        }
    };
    match kind_of(item) {
        Some("message") => texts("content").concat(),
        Some("reasoning") => {
            let summary = texts("summary");
            if summary.is_empty() {
                texts("content").join("\n\n")
            } else {
                summary.join("\n\n")
            }
        }
        _ => String::new(),
    }
}

/// The provider's own words for a failed or cancelled response.
fn provider_message(response: &Value, fallback: &str) -> String {
    let error = response.get("error");
    let code = error.and_then(|error| string(error, "code"));
    let message = error.and_then(|error| string(error, "message"));
    match (code, message) {
        (Some(code), Some(message)) => format!("{code}: {message}"),
        (None, Some(message)) => message.to_owned(),
        (Some(code), None) => code.to_owned(),
        (None, None) => fallback.to_owned(),
    }
}

/// How the turn ended, for every status the API documents. A response
/// without a status ended as the provider sent it. A status that is not a
/// documented end, including one still `queued` or `in_progress`, is a
/// failed turn.
pub(crate) fn finish_reason_of(response: &Value) -> (Option<FinishReason>, Option<String>) {
    let reason = response
        .pointer("/incomplete_details/reason")
        .and_then(Value::as_str)
        .filter(|reason| !reason.is_empty());
    match string(response, "status") {
        None => (None, None),
        Some("completed") => (Some(FinishReason::Stop), None),
        Some("incomplete") => match reason {
            Some("max_output_tokens") => (Some(FinishReason::Length), None),
            Some("content_filter") => (Some(FinishReason::ContentFilter), None),
            Some(reason) => (
                Some(FinishReason::Other(format!("incomplete: {reason}"))),
                Some(format!("Response incomplete: {reason}")),
            ),
            None => (
                Some(FinishReason::Other("incomplete".to_owned())),
                Some("Response incomplete without a provider reason".to_owned()),
            ),
        },
        Some(status @ ("failed" | "cancelled")) => (
            Some(FinishReason::Other(status.to_owned())),
            Some(provider_message(response, &format!("Response {status}"))),
        ),
        Some(status @ ("queued" | "in_progress")) => (
            Some(FinishReason::Other(status.to_owned())),
            Some(format!("Response ended while {status}")),
        ),
        Some(status) => (
            Some(FinishReason::Other(status.to_owned())),
            Some(format!("Response ended with the unknown status `{status}`")),
        ),
    }
}

fn finish_of(response: &Value) -> Finish {
    let (reason, error) = finish_reason_of(response);
    Finish {
        usage: response
            .get("usage")
            .map(super::wire::usage_of)
            .unwrap_or_default(),
        reason,
        response_id: item_id(response).map(str::to_owned),
        model: string(response, "model")
            .filter(|model| !model.is_empty())
            .map(str::to_owned),
        error,
    }
}

impl ResponsesDecoder {
    /// A decoder for one reply of a Responses endpoint.
    pub fn new() -> Self {
        Self::default()
    }

    /// A streamed reply's blocks follow output indices, so an item only the
    /// terminal states lands at its index. A unary body writes its items in
    /// output order already.
    fn order(&mut self, out: &mut Out<'_, Completion>) {
        if !self.ordered {
            self.ordered = true;
            out.order_by_index();
        }
    }

    /// The item at `index` started: it opens with no provider item, since
    /// only its done item is complete. An opaque item keeps what it states
    /// so far, which is its only content when it never finishes.
    fn added(
        &mut self,
        index: usize,
        item: Value,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        self.vacate(index, out)?;
        self.seen.insert(index);
        match item_id(&item) {
            Some(id) => self.added_ids.insert(index, id.to_owned()),
            None => self.added_ids.remove(&index),
        };
        self.texts.remove(&index);
        let kind = match kind_of(&item) {
            Some("message") => {
                out.open(index, Block::Text, Value::Null)?;
                Kind::Message
            }
            Some("reasoning") => {
                out.open(index, Block::Reasoning { redacted: false }, Value::Null)?;
                Kind::Reasoning
            }
            Some("function_call" | "custom_tool_call") => {
                let arguments =
                    (kind_of(&item) == Some("function_call")).then(|| arguments_of(&item));
                out.fragment(
                    index,
                    CallFragment {
                        id: string(&item, "call_id"),
                        name: string(&item, "name"),
                        arguments: arguments.as_deref(),
                    },
                )?;
                if kind_of(&item) == Some("custom_tool_call") {
                    let input = string(&item, "input").unwrap_or_default().to_owned();
                    self.custom_inputs.insert(index, input);
                }
                Kind::Call
            }
            other => {
                let replay = !other.is_some_and(|kind| CLIENT_EXECUTED.contains(&kind));
                out.open(index, Block::Opaque { replay }, item)?;
                Kind::Opaque
            }
        };
        self.kinds.insert(index, kind);
        Ok(())
    }

    /// Close the item still open at `index` before another opens there: a
    /// stream that omits its indices puts every item at zero. The item was
    /// never stated complete, so it has no native.
    fn vacate(&mut self, index: usize, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        if !out.is_open(index) {
            return Ok(());
        }
        self.awaiting_ciphertext.retain(|(at, _)| *at != index);
        self.custom_inputs.remove(&index);
        out.close(index)
    }

    /// Append a text delta to the item at `index`.
    fn extend(
        &mut self,
        index: usize,
        field: TextField,
        part: u64,
        delta: &str,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        if delta.is_empty() {
            return Ok(());
        }
        // Text where reasoning or another item is open, or reasoning where
        // text is, is a stream without indices moving to its next item.
        let message = field == TextField::Message;
        let wanted = if message {
            Kind::Message
        } else {
            Kind::Reasoning
        };
        if out.is_open(index) && self.kinds.get(&index).is_some_and(|kind| *kind != wanted) {
            self.vacate(index, out)?;
            self.texts.remove(&index);
            self.seen.remove(&index);
        }
        // A gateway that streams an item without announcing it gets one
        // opened at its first delta; a delta for an item that already
        // closed is left to the item, which states its text.
        if !out.is_open(index) {
            if !self.seen.insert(index) {
                return Ok(());
            }
            let block = if message {
                Block::Text
            } else {
                Block::Reasoning { redacted: false }
            };
            out.open(index, block, Value::Null)?;
            self.kinds.insert(index, wanted);
        }
        let streamed = self.texts.entry(index).or_insert(Streamed {
            field,
            part,
            text: String::new(),
        });
        if streamed.field != field {
            return Ok(());
        }
        if streamed.part != part && !message && !streamed.text.is_empty() {
            streamed.text.push_str("\n\n");
            out.push(index, "\n\n")?;
        }
        streamed.part = part;
        streamed.text.push_str(delta);
        out.push(index, delta)
    }

    /// Append an argument delta to the call announced at `index`.
    fn arguments(
        &mut self,
        index: usize,
        delta: &str,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        if out.is_open(index) && self.kinds.get(&index) == Some(&Kind::Call) {
            if let Some(input) = self.custom_inputs.get_mut(&index) {
                input.push_str(delta);
                return Ok(());
            }
            out.push(index, delta)?;
        }
        Ok(())
    }

    /// The item at `index` is done: `item` is its block's native, and the
    /// text it states completes the block's. A call is written whole from
    /// it, replacing what its deltas announced.
    fn done(
        &mut self,
        index: usize,
        item: Value,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        self.seen.insert(index);
        self.written.insert(index);
        if let Some(id) = item_id(&item) {
            self.written_ids.insert(id.to_owned());
        }
        if is_call(&item) {
            if out.is_open(index) && self.kinds.get(&index) == Some(&Kind::Call) {
                // Nothing of an open call is visible yet.
                out.discard(index);
                self.custom_inputs.remove(&index);
            } else {
                self.vacate(index, out)?;
            }
            let arguments = arguments_of(&item);
            out.fragment(
                index,
                CallFragment {
                    id: string(&item, "call_id"),
                    name: string(&item, "name"),
                    arguments: Some(&arguments),
                },
            )?;
            self.kinds.insert(index, Kind::Call);
            return out.finish_with(index, item);
        }
        // An item done without being added opens here; so does one at an
        // index another item holds, or whose previous item still waits for
        // its ciphertext.
        let kind = match kind_of(&item) {
            Some("message") => Kind::Message,
            Some("reasoning") => Kind::Reasoning,
            _ => Kind::Opaque,
        };
        if !out.is_open(index)
            || self.kinds.get(&index) != Some(&kind)
            || self.awaiting_ciphertext.iter().any(|(at, _)| *at == index)
        {
            self.added(index, item.clone(), out)?;
        }
        // The item states the whole text: what the deltas left out of it
        // is pushed, and text that diverged from it stays as it streamed.
        let text = text_of(&item);
        let streamed = self.texts.remove(&index).map(|streamed| streamed.text);
        if let Some(rest) = streamed
            .as_deref()
            .map_or(Some(text.as_str()), |streamed| text.strip_prefix(streamed))
        {
            out.push(index, rest)?;
        }
        let awaits = kind == Kind::Reasoning && !has_ciphertext(&item);
        match item_id(&item).map(str::to_owned) {
            Some(id) if awaits => {
                out.edit(index, |native| *native = item)?;
                self.awaiting_ciphertext.push((index, id));
                Ok(())
            }
            _ => out.finish_with(index, item),
        }
    }

    /// Write one output-item event into the reply.
    fn item_event(
        &mut self,
        kind: &str,
        frame: Value,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        self.order(out);
        let index = output_index(&frame)?;
        let delta = string(&frame, "delta").unwrap_or_default();
        match kind {
            "response.output_item.added" | "response.output_item.done" => {
                let Some(item) = frame.get("item").filter(|item| item.is_object()).cloned() else {
                    return Ok(());
                };
                if kind == "response.output_item.added" {
                    self.added(index, item, out)
                } else {
                    self.done(index, item, out)
                }
            }
            // A refusal is the assistant's message for that turn.
            "response.output_text.delta" | "response.refusal.delta" => self.extend(
                index,
                TextField::Message,
                number(&frame, "content_index"),
                delta,
                out,
            ),
            "response.reasoning_summary_text.delta" => self.extend(
                index,
                TextField::Summary,
                number(&frame, "summary_index"),
                delta,
                out,
            ),
            "response.reasoning_text.delta" => self.extend(
                index,
                TextField::Reasoning,
                number(&frame, "content_index"),
                delta,
                out,
            ),
            "response.function_call_arguments.delta" | "response.custom_tool_call_input.delta" => {
                self.arguments(index, delta, out)
            }
            // Restatements repeat what the deltas carried; the done item
            // states the whole item.
            _ => Ok(()),
        }
    }

    /// The terminal response: finish the output items no done event
    /// carried, give reasoning that waited its ciphertext, deliver a custom
    /// call that was never done, then end with how the turn ended. A body
    /// that reports its reasoning as one top-level string and no reasoning
    /// item is written that reasoning first.
    fn finish(
        &mut self,
        response: Value,
        mut out: Out<'_, Completion>,
    ) -> Result<Flow, ProviderError> {
        let output = response
            .get("output")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let structured_reasoning = output.iter().any(|item| kind_of(item) == Some("reasoning"));
        if !structured_reasoning
            && self.seen.is_empty()
            && let Some(reasoning) = string(&response, "reasoning").filter(|text| !text.is_empty())
        {
            out.content(crate::message::AssistantContent::Reasoning(
                crate::message::Reasoning::new(reasoning),
            ))?;
        }
        // Only the stream's items are matched by id: a provider may give two
        // items of one reply the same id.
        let streamed_ids = std::mem::take(&mut self.written_ids);
        for (index, item) in output.into_iter().enumerate() {
            if !item.is_object() {
                continue;
            }
            let id = item_id(&item).map(str::to_owned);
            if let Some(at) = self
                .awaiting_ciphertext
                .iter()
                .find(|(_, awaiting)| Some(awaiting) == id.as_ref())
                .map(|(at, _)| *at)
                && let Some(ciphertext) = item
                    .get("encrypted_content")
                    .filter(|_| has_ciphertext(&item))
            {
                let ciphertext = ciphertext.clone();
                out.edit(at, |native| {
                    if let Some(native) = native.as_object_mut() {
                        native.insert("encrypted_content".to_owned(), ciphertext);
                    }
                })?;
            }
            let written = self.written.contains(&index)
                || id.as_ref().is_some_and(|id| streamed_ids.contains(id));
            if written {
                continue;
            }
            // An item the stream opened and never finished is done as the
            // terminal states it; one the stream never stated is written
            // from it whole, at its index.
            let opened = out.is_open(index);
            if opened
                && self
                    .added_ids
                    .get(&index)
                    .is_some_and(|added| Some(added) != id.as_ref())
            {
                continue;
            }
            self.done(index, item, &mut out)?;
        }
        for (index, _) in std::mem::take(&mut self.awaiting_ciphertext) {
            out.finish(index)?;
        }
        for (index, input) in std::mem::take(&mut self.custom_inputs) {
            if out.is_open(index) {
                out.push(index, &serde_json::json!({ "input": input }).to_string())?;
            }
        }
        let end = finish_of(&response);
        out.raw(response);
        Ok(out.end(end))
    }
}

impl<'id> Decoder<'id, Completion> for ResponsesDecoder {
    type Event = ResponsesEvent;

    fn classify(&self, frame: WireFrame) -> WireEvent<ResponsesEvent> {
        classify_responses_payload(&frame.as_str())
    }

    fn decode(
        &mut self,
        event: ResponsesEvent,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match event {
            ResponsesEvent::Frame { kind, frame, raw } => match kind.as_str() {
                // `response.incomplete` is a genuine terminal (e.g. hitting
                // `max_output_tokens`): the partial output and usage are
                // kept.
                "response.completed" | "response.incomplete" => {
                    let response = frame
                        .get("response")
                        .cloned()
                        .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
                    self.finish(response, out)
                }
                "response.failed" => Err(ProviderError::from_provider_body(raw)),
                kind if is_lifecycle_event(kind) => Ok(Flow::More),
                kind => {
                    self.item_event(kind, frame, &mut out)?;
                    Ok(Flow::More)
                }
            },
            // The unary reply is the terminal response with no stream before it.
            ResponsesEvent::Whole(body) => self.finish(body, out),
            ResponsesEvent::Failure(raw) => Err(ProviderError::from_provider_body(raw)),
            // Nothing to write: the provider's end is `response.completed`.
            ResponsesEvent::Sentinel => Ok(Flow::More),
        }
    }
}

#[cfg(test)]
mod tests;
