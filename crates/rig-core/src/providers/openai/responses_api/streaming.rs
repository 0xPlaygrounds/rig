//! Responses frame classification and event decoding: one block per output
//! item, in the order the items arrive, each holding the item as the
//! provider stated it complete.
//!
//! Frames are read as JSON and classified by their `type` alone, and every
//! field is read leniently, so a gateway that omits or retypes a field no
//! block or finish needs never fails a reply.
//!
//! ```
//! use rig_core::providers::openai::responses_api::streaming::ResponsesDecoder;
//! let decoder = ResponsesDecoder::new();
//! # let _ = decoder;
//! ```

use std::collections::{HashMap, HashSet};

use serde_json::{Value, json};

use crate::completion::{FinishReason, Usage};
use crate::error::ProviderError;
use crate::json_utils::Lenient;
use crate::operation::{Block, CallFragment, Completion, Finish};
use crate::providers::internal::wire;
use crate::wire::{Decoder, Flow, Out, WireEvent, WireFrame};

/// The item events this decoder reads, after their `response.` prefix.
const ITEM_EVENTS: &str = "output_item.added output_item.done content_part.added content_part.done \
    output_text.delta output_text.done refusal.delta refusal.done function_call_arguments.delta \
    function_call_arguments.done custom_tool_call_input.delta custom_tool_call_input.done \
    reasoning_summary_part.added reasoning_summary_part.done reasoning_summary_text.delta \
    reasoning_summary_text.done reasoning_text.delta reasoning_text.done";

/// Whether `kind` is a Responses event type this decoder reads. A frame of
/// any other type passes through as unknown.
fn is_known_responses_event_type(kind: &str) -> bool {
    kind == "error"
        || is_lifecycle_event(kind)
        || kind
            .strip_prefix("response.")
            .is_some_and(|event| ITEM_EVENTS.split_whitespace().any(|known| known == event))
}

/// Whether `kind` is a response lifecycle event, which carries the response
/// object under `response`.
pub(crate) fn is_lifecycle_event(kind: &str) -> bool {
    kind.strip_prefix("response.").is_some_and(|event| {
        "created queued in_progress completed failed incomplete"
            .split(' ')
            .any(|known| known == event)
    })
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
        if value.str("type").is_some() {
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
            wire::classify_tagged_frame::<Tagged>(data, "type", is_known_responses_event_type).map(
                |Tagged(frame)| match frame.str("type") {
                    Some("error") => ResponsesEvent::Failure(data.to_owned()),
                    kind => ResponsesEvent::Frame {
                        kind: kind.unwrap_or_default().to_owned(),
                        raw: data.to_owned(),
                        frame,
                    },
                },
            )
        },
        |data| {
            wire::classify_marker_keyed_frame::<Value>(data, WHOLE_BODY_MARKERS).map(|body| {
                let error = body.get("error").is_some_and(|error| !error.is_null());
                if error && body.get("output").is_none() && body.get("status").is_none() {
                    ResponsesEvent::Failure(data.to_owned())
                } else {
                    ResponsesEvent::Whole(body)
                }
            })
        },
    )
}

/// What an output item becomes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Message,
    Reasoning,
    Call,
    Opaque,
}

impl Kind {
    fn of(item: &Value) -> Self {
        match item.str("type") {
            Some("message") => Self::Message,
            Some("reasoning") => Self::Reasoning,
            Some("function_call" | "custom_tool_call") => Self::Call,
            _ => Self::Opaque,
        }
    }
}

/// One output item of the reply, as the decoder saw it.
struct Slot {
    /// The writer index its block opened at.
    at: usize,
    kind: Kind,
    /// The item id it states.
    id: Option<String>,
    open: bool,
    /// The text its deltas wrote, and the field and part that last
    /// extended it: a new part of reasoning starts a paragraph, and a second
    /// field restating the same reasoning is not appended.
    text: String,
    field: Option<&'static str>,
    part: u64,
    /// A call's streamed argument text, or a custom call's input.
    arguments: String,
    /// The argument JSON already written to the call.
    sent: String,
    custom: bool,
    /// Whether the call's id was stated.
    named: bool,
    /// Whether the call's tool name was stated.
    titled: bool,
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

fn item_id(item: &Value) -> Option<&str> {
    item.str("id").filter(|id| !id.is_empty())
}

/// The argument JSON a done call states: a function call's `arguments`, a
/// custom call's `input` as `{"input": ...}`; `None` when it states none.
fn arguments_of(item: &Value) -> Option<String> {
    if item.str("type") == Some("custom_tool_call") {
        return item
            .str("input")
            .map(|input| json!({ "input": input }).to_string());
    }
    match item.get("arguments")? {
        Value::String(arguments) if arguments.is_empty() => None,
        Value::String(arguments) => Some(arguments.clone()),
        Value::Null => None,
        arguments => Some(arguments.to_string()),
    }
}

/// The text a done item states: a message's parts joined, refusals
/// included; reasoning's summary, or its raw content when it has none.
fn text_of(item: &Value) -> String {
    let texts = |key: &str| -> Vec<&str> {
        match item.get(key) {
            Some(Value::String(text)) => vec![text.as_str()],
            _ => item
                .arr(key)
                .iter()
                .filter_map(|part| {
                    part.as_str()
                        .or_else(|| part.str("text"))
                        .or_else(|| part.str("refusal"))
                })
                .collect(),
        }
    };
    match Kind::of(item) {
        Kind::Message => texts("content").concat(),
        Kind::Reasoning => Some(texts("summary"))
            .filter(|summary| !summary.is_empty())
            .unwrap_or_else(|| texts("content"))
            .join("\n\n"),
        Kind::Call | Kind::Opaque => String::new(),
    }
}

/// `item`, a done item that states no text, stating `text`, so the item
/// replays as the block it is.
fn stating(mut item: Value, kind: Kind, text: &str) -> Value {
    let (key, part) = match kind {
        Kind::Message => (
            "content",
            json!({"type": "output_text", "text": text, "annotations": []}),
        ),
        _ => ("summary", json!({"type": "summary_text", "text": text})),
    };
    if let Some(fields) = item.as_object_mut() {
        fields.insert(key.to_owned(), json!([part]));
    }
    item
}

/// The provider's own words for a failed or cancelled response.
fn provider_message(response: &Value, fallback: &str) -> String {
    let parts: Vec<&str> = ["/error/code", "/error/message"]
        .iter()
        .filter_map(|pointer| response.at(pointer).and_then(Value::as_str))
        .collect();
    if parts.is_empty() {
        fallback.to_owned()
    } else {
        parts.join(": ")
    }
}

/// How the turn ended, for every status the API documents. A response
/// without a status ended as pi reads it, with a stop. A status that is
/// not a documented end, including one still `queued` or `in_progress`, is
/// a failed turn.
pub(crate) fn finish_reason_of(response: &Value) -> (FinishReason, Option<String>) {
    let reason = response
        .at("/incomplete_details/reason")
        .and_then(Value::as_str)
        .filter(|reason| !reason.is_empty());
    let other = |reason: &str, error: String| (FinishReason::Other(reason.to_owned()), Some(error));
    match response.str("status") {
        None | Some("completed") => (FinishReason::Stop, None),
        Some("incomplete") => match reason {
            Some("max_output_tokens") => (FinishReason::Length, None),
            Some("content_filter") => (FinishReason::ContentFilter, None),
            Some(reason) => other(
                &format!("incomplete: {reason}"),
                format!("Response incomplete: {reason}"),
            ),
            None => other(
                "incomplete",
                "Response incomplete without a provider reason".to_owned(),
            ),
        },
        Some(status @ ("failed" | "cancelled")) => other(
            status,
            provider_message(response, &format!("Response {status}")),
        ),
        Some(status @ ("queued" | "in_progress")) => {
            other(status, format!("Response ended while {status}"))
        }
        Some(status) => other(
            status,
            format!("Response ended with the unknown status `{status}`"),
        ),
    }
}

/// The usage a Responses `usage` value reports; a counter that is absent or
/// not a count is unreported.
pub(crate) fn usage_of(usage: &Value) -> Usage {
    let count = |pointer: &str| usage.at(pointer).and_then(Lenient::as_u64_lenient);
    Usage {
        input_tokens: count("/input_tokens"),
        output_tokens: count("/output_tokens"),
        total_tokens: count("/total_tokens"),
        cached_input_tokens: count("/input_tokens_details/cached_tokens"),
        cache_creation_input_tokens: count("/input_tokens_details/cache_write_tokens"),
        reasoning_tokens: count("/output_tokens_details/reasoning_tokens"),
        ..Usage::default()
    }
}

/// The OpenAI Responses wire's decoder: one state machine for the SSE
/// stream, the unary body and the websocket session.
///
/// Each output item becomes one block. It opens when it is announced, or at
/// its first delta, with no provider item, and its done item closes it,
/// becoming the block's native: an item the provider never stated complete
/// replays from its canonical fields. A done item merges into what streamed:
/// a call keeps the id and name it was announced with and the arguments it
/// streamed when its done item leaves them out, and text the done item
/// leaves out stays, written into the item. The terminal response is the
/// whole output: it finishes the items no done event carried, writes the
/// ones the stream never stated, which is how a unary body decodes, and
/// backfills the ciphertext of reasoning done without it.
#[derive(Default)]
pub struct ResponsesDecoder {
    /// Every item, in the order it arrived.
    slots: Vec<Slot>,
    /// The latest slot at each output index the provider named.
    indexed: HashMap<usize, usize>,
    /// The slot the last item event addressed.
    current: Option<usize>,
    /// The writer indices of reasoning done and waiting for the item after
    /// it to complete.
    held: Vec<usize>,
}

impl ResponsesDecoder {
    /// A decoder for one reply of a Responses endpoint.
    pub fn new() -> Self {
        Self::default()
    }

    /// The slot `frame` addresses, an event for an item of `kind`: the one
    /// at its output index, else the one its item id names, else the open
    /// one the stream is on, when the frame names no index or that item
    /// was opened by a frame that named none (an envelope-less delta, then
    /// its envelope-full done item). `None` for an item no slot holds yet.
    fn addressed(&self, frame: &Value, kind: Kind) -> Result<Option<usize>, ProviderError> {
        let index = output_index(frame)?;
        if let Some(slot) = index.and_then(|index| self.indexed.get(&index)) {
            return Ok(Some(*slot));
        }
        let id = frame
            .at("/item/id")
            .and_then(Value::as_str)
            .or_else(|| frame.str("item_id"));
        if let Some(slot) = id.filter(|id| !id.is_empty()).and_then(|id| {
            self.slots
                .iter()
                .rposition(|slot| slot.id.as_deref() == Some(id))
        }) {
            return Ok(Some(slot));
        }
        Ok(self.current.filter(|current| {
            (index.is_none() || !self.indexed.values().any(|slot| slot == current))
                && self
                    .slots
                    .get(*current)
                    .is_some_and(|slot| slot.open && slot.kind == kind)
        }))
    }

    /// Open the block of `item` as a new slot at output `index`, closing what
    /// is still open there.
    fn added(
        &mut self,
        index: Option<usize>,
        item: &Value,
        out: &mut Out<'_, Completion>,
    ) -> Result<usize, ProviderError> {
        if let Some(previous) = index.and_then(|index| self.indexed.get(&index).copied()) {
            self.vacate(previous, out)?;
        }
        let at = match index {
            Some(index) if !self.slots.iter().any(|slot| slot.at == index) => index,
            _ => out.fresh_index(),
        };
        let kind = Kind::of(item);
        let custom = item.str("type") == Some("custom_tool_call");
        match kind {
            Kind::Message => out.open(at, Block::Text, Value::Null)?,
            Kind::Reasoning => out.open(at, Block::Reasoning { redacted: false }, Value::Null)?,
            Kind::Call => out.fragment(
                Some(at),
                CallFragment {
                    id: item.str("call_id"),
                    name: item.str("name"),
                    arguments: None,
                },
            )?,
            // An item that names no type is kept and never sent back.
            Kind::Opaque => {
                let replay = item
                    .str("type")
                    .is_some_and(|kind| !CLIENT_EXECUTED.contains(&kind));
                out.open(at, Block::Opaque { replay }, item.clone())?;
            }
        }
        let arguments = item.str(if custom { "input" } else { "arguments" });
        self.slots.push(Slot {
            at,
            kind,
            id: item_id(item).map(str::to_owned),
            open: true,
            text: String::new(),
            field: None,
            part: 0,
            arguments: arguments.unwrap_or_default().to_owned(),
            sent: String::new(),
            custom,
            named: item.str("call_id").is_some_and(|id| !id.is_empty()),
            titled: item.str("name").is_some_and(|name| !name.is_empty()),
        });
        let slot = self.slots.len() - 1;
        if let Some(index) = index {
            self.indexed.insert(index, slot);
        }
        self.current = Some(slot);
        if let Some(call) = self
            .slots
            .get_mut(slot)
            .filter(|slot| slot.kind == Kind::Call)
        {
            send(call, out)?;
        }
        Ok(slot)
    }

    /// Close the slot's block before another item takes its place: the
    /// provider never stated it complete. A call keeps what streamed.
    fn vacate(&mut self, slot: usize, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        let Some(open) = self.slots.get_mut(slot).filter(|slot| slot.open) else {
            return Ok(());
        };
        open.open = false;
        let at = open.at;
        if open.kind == Kind::Call {
            flush(open, out)?;
        }
        out.close(at)?;
        self.release(false, out)
    }

    /// End the reasoning held for the item after it: complete when that
    /// item completed, else as never stated complete.
    fn release(
        &mut self,
        complete: bool,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        std::mem::take(&mut self.held)
            .into_iter()
            .try_for_each(|at| {
                if complete {
                    out.finish(at)
                } else {
                    out.close(at)
                }
            })
    }

    /// Append a text delta of `field` to the item `frame` addresses,
    /// opening a block for an item the stream never announced.
    fn text(
        &mut self,
        frame: &Value,
        field: &'static str,
        part: &str,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let delta = frame.str("delta").unwrap_or_default();
        if delta.is_empty() {
            return Ok(());
        }
        let part = frame.u64(part).unwrap_or(0);
        let kind = match field {
            "message" => Kind::Message,
            _ => Kind::Reasoning,
        };
        let known = self.addressed(frame, kind)?.and_then(|at| {
            let slot = self.slots.get(at)?;
            Some((at, slot.open, slot.kind == kind))
        });
        let slot = match known {
            Some((slot, true, true)) => slot,
            // A delta for an item that already closed is left to the item,
            // which states its text.
            Some((_, false, true)) => return Ok(()),
            _ => self.added(
                output_index(frame)?,
                &json!({"type": if kind == Kind::Message { "message" } else { "reasoning" }}),
                out,
            )?,
        };
        let Some(streamed) = self.slots.get_mut(slot).filter(|slot| slot.open) else {
            return Ok(());
        };
        if streamed.field.is_some_and(|known| known != field) {
            return Ok(());
        }
        if streamed.part != part && kind != Kind::Message && !streamed.text.is_empty() {
            streamed.text.push_str("\n\n");
            out.push(streamed.at, "\n\n")?;
        }
        streamed.field = Some(field);
        streamed.part = part;
        streamed.text.push_str(delta);
        out.push(streamed.at, delta)
    }

    /// Append an argument delta to the call `frame` addresses and write
    /// it to the call, which streams it.
    fn arguments(
        &mut self,
        frame: &Value,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let slot = self.addressed(frame, Kind::Call)?;
        if let Some(call) = slot
            .and_then(|slot| self.slots.get_mut(slot))
            .filter(|slot| slot.open && slot.kind == Kind::Call)
        {
            call.arguments
                .push_str(frame.str("delta").unwrap_or_default());
            send(call, out)?;
        }
        Ok(())
    }

    /// The item at `slot` is done: it merges into what streamed and becomes
    /// the block's native.
    fn done(
        &mut self,
        slot: usize,
        item: Value,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let Some(done) = self.slots.get_mut(slot) else {
            return Ok(());
        };
        done.open = false;
        if done.id.is_none() {
            done.id = item_id(&item).map(str::to_owned);
        }
        let at = done.at;
        let mut item = item;
        let mut nameless = false;
        match done.kind {
            Kind::Call => {
                nameless = !done.titled && item.str("name").is_none_or(str::is_empty);
                let arguments = arguments_of(&item).unwrap_or_else(|| streamed_arguments(done));
                let rest = remainder(done, &arguments, out)?;
                out.fragment(
                    Some(at),
                    CallFragment {
                        id: item.str("call_id").filter(|_| !done.named),
                        name: item.str("name"),
                        arguments: Some(&rest),
                    },
                )?;
            }
            Kind::Message | Kind::Reasoning => {
                let text = text_of(&item);
                if text.is_empty() {
                    // Raw reasoning text the done item leaves out stays out:
                    // written in as a summary, it would replay one the
                    // provider never produced.
                    if !done.text.is_empty() && done.field != Some("reasoning") {
                        item = stating(item, done.kind, &done.text);
                    }
                } else if let Some(rest) = text.strip_prefix(done.text.as_str()) {
                    out.push(at, rest)?;
                } else {
                    // The done item states the whole text, as pi takes it.
                    out.restate(at, &text)?;
                }
            }
            Kind::Opaque => {}
        }
        let reasoning = done.kind == Kind::Reasoning;
        out.edit(at, |native| *native = item)?;
        if reasoning {
            // Reasoning is sent only with the item it precedes, so it is
            // complete only once that item is.
            self.held.push(at);
            return Ok(());
        }
        out.finish(at)?;
        // The writer drops a call with no name, so the reasoning it held
        // has no item to go with.
        self.release(!nameless, out)
    }

    /// Write one output-item event into the reply.
    fn item_event(
        &mut self,
        kind: &str,
        frame: Value,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        match kind {
            "response.output_item.added" | "response.output_item.done" => {
                let Some(item) = frame.get("item").filter(|item| item.is_object()) else {
                    return Ok(());
                };
                let index = output_index(&frame)?;
                if kind == "response.output_item.added" {
                    return self.added(index, item, out).map(|_| ());
                }
                let known = self.addressed(&frame, Kind::of(item))?;
                // At a named output index the item is the one there, as pi
                // reads it, whatever id it states.
                let restates = |slot: &Slot| {
                    slot.kind == Kind::of(item)
                        && (index.is_some()
                            || item_id(item)
                                .zip(slot.id.as_deref())
                                .is_none_or(|(id, known)| id == known))
                };
                let known = known.and_then(|at| {
                    let slot = self.slots.get(at)?;
                    let same = item_id(item).is_some() && slot.id.as_deref() == item_id(item);
                    Some((at, slot.open && restates(slot), same))
                });
                let slot = match known {
                    Some((slot, true, _)) => slot,
                    // The item this one restates is already done.
                    Some((_, false, true)) => return Ok(()),
                    _ => self.added(index, item, out)?,
                };
                self.done(slot, item.clone(), out)
            }
            // A refusal is the assistant's message for that turn.
            "response.output_text.delta" | "response.refusal.delta" => {
                self.text(&frame, "message", "content_index", out)
            }
            "response.reasoning_summary_text.delta" => {
                self.text(&frame, "summary", "summary_index", out)
            }
            "response.reasoning_text.delta" => self.text(&frame, "reasoning", "content_index", out),
            "response.function_call_arguments.delta" | "response.custom_tool_call_input.delta" => {
                self.arguments(&frame, out)
            }
            _ => Ok(()),
        }
    }

    /// The slot the terminal's item at output `index` restates, among those
    /// no earlier terminal item took: the one with its id; else the one the
    /// stream gave that index when it is of the item's kind, as pi reads an
    /// index whatever id it states; else the first of its kind the stream
    /// gave no index and no other id; else the first of its kind with no
    /// other id that the stream put at an index the terminal holds no item
    /// of that kind at, a stream whose indices are shifted against it.
    fn restated(
        &self,
        index: usize,
        item: &Value,
        output: &[Value],
        taken: &HashSet<usize>,
    ) -> Option<usize> {
        let id = item_id(item);
        let free = |at: &usize| {
            !taken.contains(at)
                && self
                    .slots
                    .get(*at)
                    .is_some_and(|slot| slot.kind == Kind::of(item))
        };
        let find = |fits: &dyn Fn(usize, &Slot) -> bool| {
            self.slots
                .iter()
                .enumerate()
                .find(|(at, slot)| free(at) && fits(*at, slot))
                .map(|(at, _)| at)
        };
        id.and_then(|id| find(&|_, slot| slot.id.as_deref() == Some(id)))
            .or_else(|| self.indexed.get(&index).copied().filter(free))
            .or_else(|| {
                find(&|at, slot| {
                    !self.indexed.values().any(|indexed| *indexed == at)
                        && (id.is_none() || slot.id.is_none())
                })
            })
            .or_else(|| {
                find(&|at, slot| {
                    (id.is_none() || slot.id.is_none())
                        && self.indexed.iter().any(|(streamed, indexed)| {
                            *indexed == at
                                && output
                                    .get(*streamed)
                                    .is_none_or(|other| Kind::of(other) != slot.kind)
                        })
                })
            })
    }

    /// The terminal response: finish or write each item of its output,
    /// backfill the ciphertext of reasoning done without it, then end with
    /// how the turn ended. A body that reports its reasoning as one
    /// top-level string and no reasoning item is written that reasoning
    /// first.
    fn finish(
        &mut self,
        response: Value,
        mut out: Out<'_, Completion>,
    ) -> Result<Flow, ProviderError> {
        let output = response.arr("output");
        if self.slots.is_empty()
            && !output.iter().any(|item| Kind::of(item) == Kind::Reasoning)
            && let Some(reasoning) = response.str("reasoning").filter(|text| !text.is_empty())
        {
            // Index 0 orders it before the output; it closes at once, so the
            // output's first item may open there after it.
            out.whole(
                0,
                Block::Reasoning { redacted: false },
                Value::Null,
                reasoning,
            )?;
        }
        let mut taken = HashSet::new();
        for (index, item) in output
            .iter()
            .enumerate()
            .filter(|(_, item)| item.is_object())
        {
            let known = self.restated(index, item, output, &taken).and_then(|at| {
                let slot = self.slots.get(at)?;
                Some((at, slot.open, slot.kind, slot.at))
            });
            let slot = match known {
                Some((slot, false, kind, at)) => {
                    taken.insert(slot);
                    let ciphertext = item
                        .get("encrypted_content")
                        .filter(|cipher| cipher.as_str().is_some_and(|cipher| !cipher.is_empty()));
                    if let (Kind::Reasoning, Some(ciphertext)) = (kind, ciphertext) {
                        out.edit(at, |native| {
                            let stated = native
                                .str("encrypted_content")
                                .is_some_and(|cipher| !cipher.is_empty());
                            if let (false, Some(fields)) = (stated, native.as_object_mut()) {
                                fields.insert("encrypted_content".to_owned(), ciphertext.clone());
                            }
                        })?;
                    }
                    continue;
                }
                Some((slot, ..)) => slot,
                None => self.added(
                    Some(index).filter(|index| !self.indexed.contains_key(index)),
                    item,
                    &mut out,
                )?,
            };
            taken.insert(slot);
            self.done(slot, item.clone(), &mut out)?;
        }
        // Reasoning still waiting is complete when nothing after it is left
        // open.
        let open = self.slots.iter().any(|slot| slot.open);
        self.release(!open, &mut out)?;
        // A call never done shows what streamed, in a turn that fails.
        for slot in self.slots.iter_mut().filter(|slot| slot.open) {
            if slot.kind == Kind::Call {
                flush(slot, &mut out)?;
            }
        }
        let (reason, error) = finish_reason_of(&response);
        let end = Finish {
            usage: response.get("usage").map(usage_of).unwrap_or_default(),
            reason: Some(reason),
            response_id: item_id(&response).map(str::to_owned),
            model: response
                .str("model")
                .filter(|model| !model.is_empty())
                .map(str::to_owned),
            error,
        };
        Ok(out.end(end))
    }
}

/// Give `slot`'s call what it streamed, before it closes undone.
fn flush(slot: &mut Slot, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
    let arguments = streamed_arguments(slot);
    let rest = remainder(slot, &arguments, out)?;
    let fragment = CallFragment {
        arguments: Some(&rest),
        ..CallFragment::default()
    };
    out.fragment(Some(slot.at), fragment)
}

/// Write the argument JSON `slot`'s call streamed so far that the call has
/// not been given yet. A custom call's input streams inside its
/// `{"input": ...}` object, escaped as the object's string, so its
/// fragments join to the object its end states.
fn send(slot: &mut Slot, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
    if slot.arguments.is_empty() {
        return Ok(());
    }
    let streamed = if slot.custom {
        let quoted = Value::String(slot.arguments.clone()).to_string();
        // The string without its closing quote: later input extends it.
        format!("{{\"input\":{}", &quoted[..quoted.len() - 1])
    } else {
        slot.arguments.clone()
    };
    let Some(rest) = streamed
        .strip_prefix(slot.sent.as_str())
        .filter(|rest| !rest.is_empty())
    else {
        return Ok(());
    };
    let fragment = CallFragment {
        arguments: Some(rest),
        ..CallFragment::default()
    };
    out.fragment(Some(slot.at), fragment)?;
    slot.sent = streamed;
    Ok(())
}

/// What of `arguments`, the whole JSON `slot`'s call states, the call has
/// not been given yet. When `arguments` does not extend what streamed, it
/// replaces the call's text and nothing more streams: the call's end states
/// it.
fn remainder(
    slot: &mut Slot,
    arguments: &str,
    out: &mut Out<'_, Completion>,
) -> Result<String, ProviderError> {
    let rest = match arguments.strip_prefix(slot.sent.as_str()) {
        Some(rest) => rest.to_owned(),
        None => {
            out.restate(slot.at, arguments)?;
            String::new()
        }
    };
    arguments.clone_into(&mut slot.sent);
    Ok(rest)
}

/// The argument JSON of what `slot`'s call streamed.
fn streamed_arguments(slot: &Slot) -> String {
    if slot.custom {
        json!({ "input": slot.arguments }).to_string()
    } else {
        slot.arguments.clone()
    }
}

/// The output index `frame` names.
fn output_index(frame: &Value) -> Result<Option<usize>, ProviderError> {
    frame
        .u64("output_index")
        .map(|index| {
            usize::try_from(index).map_err(|_| {
                ProviderError::Response(format!("output_index {index} is out of range"))
            })
        })
        .transpose()
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
        // A terminal may state items the stream never announced; they take
        // their place by output index.
        out.order_by_index();
        match event {
            ResponsesEvent::Frame { kind, frame, raw } => match kind.as_str() {
                // `response.incomplete` is a genuine terminal that keeps the
                // partial output and usage. It is incomplete whatever its
                // `status` says, where pi reads a missing status as a stop.
                "response.completed" | "response.incomplete" => {
                    let mut response = frame
                        .get("response")
                        .filter(|response| response.is_object())
                        .cloned()
                        .unwrap_or_else(|| json!({}));
                    if let (true, Some(fields)) =
                        (kind == "response.incomplete", response.as_object_mut())
                    {
                        fields.insert("status".to_owned(), json!("incomplete"));
                    }
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

pub(crate) mod document;

#[cfg(test)]
mod tests;
