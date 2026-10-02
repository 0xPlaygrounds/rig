//! Responses frame classification and event decoding: one block per output
//! item, in output order, each holding the item as the provider stated it.
//!
//! ```
//! use rig_core::providers::openai::responses_api::streaming::ResponsesDecoder;
//! let decoder = ResponsesDecoder::new().with_envelope_repair();
//! # let _ = decoder;
//! ```

use std::collections::{HashMap, HashSet};

use crate::error::ProviderError;
use crate::message::{CallId, ToolName};
use crate::operation::{Block, Completion, Finish};
use crate::providers::internal::wire;
use crate::providers::openai::responses_api::{ReasoningSummary, ResponseStatus};
use crate::wire::{Decoder, Flow, Out, WireEvent, WireFrame};
use serde::{Deserialize, Serialize};

use super::{CompletionResponse, Output};

/// Response lifecycle event or output-item event.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum StreamingCompletionChunk {
    Response(ResponseChunk),
    Delta(ItemChunk),
}

/// A response chunk from OpenAI's response API.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResponseChunk {
    /// The response chunk type
    #[serde(rename = "type")]
    pub kind: ResponseChunkKind,
    /// The response itself
    pub response: CompletionResponse,
    /// The item sequence
    pub sequence_number: u64,
}

/// Response chunk type.
/// Renames are used to ensure that this type gets (de)serialized properly.
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum ResponseChunkKind {
    #[serde(rename = "response.created")]
    ResponseCreated,
    #[serde(rename = "response.in_progress")]
    ResponseInProgress,
    #[serde(rename = "response.completed")]
    ResponseCompleted,
    #[serde(rename = "response.failed")]
    ResponseFailed,
    #[serde(rename = "response.incomplete")]
    ResponseIncomplete,
}

/// Whether `kind` is a Responses SSE event type this client models.
///
/// The union of [`ResponseChunkKind`]'s and [`ItemChunkKind`]'s wire names: a
/// frame carrying one of these that still fails to deserialize is a data-level
/// defect in a known event, not an unknown event type, and must surface as an
/// error rather than be skipped.
fn is_known_responses_event_type(kind: &str) -> bool {
    matches!(
        kind,
        "response.created"
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
            | "response.reasoning_summary_part.added"
            | "response.reasoning_summary_part.done"
            | "response.reasoning_summary_text.delta"
            | "response.reasoning_summary_text.done"
            | "response.reasoning_text.delta"
            | "response.reasoning_text.done"
    )
}

/// Classify a tagged Responses event with strict decoding for known event types.
/// Callers must handle `error` and WebSocket `response.done` separately.
#[doc(hidden)]
pub fn classify_responses_frame(data: &str) -> WireEvent<StreamingCompletionChunk> {
    wire::classify_tagged_frame(data, "type", is_known_responses_event_type)
}

/// Fill absent sequence, output, content, and summary indices with zero.
/// Preserve existing fields and content. Return `None` for invalid or non-object
/// JSON or serialization failure. Missing output indices can merge distinct items
/// into slot zero; callers must enable repair only for compatible dialects.
fn repair_envelope_less_frame(data: &str) -> Option<String> {
    let mut value = serde_json::from_str::<serde_json::Value>(data).ok()?;
    let object = value.as_object_mut()?;
    for field in [
        "sequence_number",
        "output_index",
        "content_index",
        "summary_index",
    ] {
        object
            .entry(field)
            .or_insert_with(|| serde_json::Value::from(0));
    }
    serde_json::to_string(&value).ok()
}

/// Classified Responses stream frame, whole reply, error, or sentinel.
pub enum ResponsesEvent {
    /// Decoded stream frame with raw bytes retained for provider errors.
    Frame {
        /// The frame's payload, verbatim.
        raw: String,
        /// The frame, decoded.
        chunk: StreamingCompletionChunk,
    },
    /// The unary reply: the response object itself, which carries no `type`
    /// discriminator because it is not an event.
    Whole(Box<ResponseBody>),
    /// A success whose body is the provider's error envelope instead of a
    /// response, with the raw body the error preserves. Both the stream's
    /// own `error` event and a 200 whose whole body is an envelope reach
    /// this.
    Failure(String),
    /// `[DONE]` sentinel. Does not independently establish successful completion.
    Sentinel,
}

/// Top-level presence markers for response bodies, including error envelopes.
const WHOLE_BODY_MARKERS: &[&str] = &["object", "output", "status", "error"];

/// Whether valid JSON carries the `error` event tag.
fn is_error_event(data: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(data)
        .is_ok_and(|value| value.get("type").and_then(serde_json::Value::as_str) == Some("error"))
}

/// The error envelope a success body can carry instead of a response. The
/// error itself is built from the raw body; this only proves the shape.
#[derive(Deserialize)]
struct ErrorEnvelope {
    // Validate envelope presence while retaining raw bytes for the reported error.
    #[allow(dead_code)]
    error: serde_json::Value,
}

/// A whole response object: typed for its metadata, with its output items
/// exactly as the provider stated them.
pub struct ResponseBody {
    /// The response, typed.
    pub response: CompletionResponse,
    /// The response's `output` items, verbatim.
    pub output: Vec<serde_json::Value>,
}

impl<'de> Deserialize<'de> for ResponseBody {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let output = output_items(&value);
        let response = CompletionResponse::deserialize(value).map_err(serde::de::Error::custom)?;
        Ok(Self { response, output })
    }
}

/// The `output` items of a response object.
fn output_items(response: &serde_json::Value) -> Vec<serde_json::Value> {
    response
        .get("output")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default()
}

/// The OpenAI Responses wire's decoder: one state machine for the SSE
/// stream, the unary body and the websocket session.
///
/// Each output item becomes one block at its `output_index`, opened by
/// `output_item.added` and closed by `output_item.done`, whose item is the
/// block's native. A terminal response restates the whole output: the items
/// no stream event carried are written from it, which is how a unary body
/// decodes.
#[derive(Default)]
pub struct ResponsesDecoder {
    /// Whether to repair absent envelope indices before retrying classification.
    /// Selected by dialect, independently of unary or streaming mode.
    repair_envelopes: bool,
    /// The output indices an item opened or was written at.
    seen: HashSet<usize>,
    /// The id of the item `output_item.added` opened at each index.
    added_ids: HashMap<usize, String>,
    /// The output indices and item ids whose done item was written.
    written: HashSet<usize>,
    written_ids: HashSet<String>,
    /// The text deltas wrote to each open item.
    texts: HashMap<usize, Streamed>,
    /// Reasoning items done without their ciphertext, by index and id. Azure
    /// states `encrypted_content` only in the terminal response, so these
    /// close there.
    awaiting_ciphertext: Vec<(usize, String)>,
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

/// What an output item becomes. Every typed variant is named, so a new
/// one must choose its block.
#[deny(clippy::wildcard_enum_match_arm)]
fn block_of(item: &Output) -> Option<Block> {
    match item {
        Output::Message(_) => Some(Block::Text),
        Output::Reasoning { .. } => Some(Block::Reasoning { redacted: false }),
        // Calls are written whole when they are done.
        Output::FunctionCall(_) | Output::CustomToolCall(_) => None,
        Output::Unknown(value) => Some(Block::Opaque {
            replay: !value
                .get("type")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|kind| CLIENT_EXECUTED.contains(&kind)),
        }),
    }
}

/// The text a done item states: a message's content parts joined, refusals
/// included; reasoning's summary, or its raw content when it has none.
#[deny(clippy::wildcard_enum_match_arm)]
fn text_of(item: &Output) -> String {
    match item {
        Output::Message(message) => message
            .content
            .iter()
            .map(|part| match part {
                super::AssistantContent::OutputText(text) => text.text.as_str(),
                super::AssistantContent::Refusal { refusal } => refusal.as_str(),
                super::AssistantContent::Unknown(_) => "",
            })
            .collect(),
        Output::Reasoning {
            summary, content, ..
        } => {
            let summary: Vec<&str> = summary.iter().map(ReasoningSummary::text).collect();
            if summary.is_empty() {
                content.join("\n\n")
            } else {
                summary.join("\n\n")
            }
        }
        Output::FunctionCall(_) | Output::CustomToolCall(_) | Output::Unknown(_) => String::new(),
    }
}

fn typed(item: &serde_json::Value) -> Result<Output, ProviderError> {
    Output::deserialize(item).map_err(|error| {
        ProviderError::Response(format!("malformed Responses output item: {error}"))
    })
}

fn item_id(item: &serde_json::Value) -> Option<&str> {
    item.get("id")
        .and_then(serde_json::Value::as_str)
        .filter(|id| !id.is_empty())
}

fn has_ciphertext(item: &serde_json::Value) -> bool {
    item.get("encrypted_content")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|ciphertext| !ciphertext.is_empty())
}

/// How the turn ended: the status as a finish reason, and pi's explanation
/// for an incomplete turn that was neither cut by the token limit nor
/// filtered, which is never replayed.
fn finish_of(response: &CompletionResponse) -> Finish {
    use crate::completion::FinishReason;
    let reason = super::map_finish_reason(&response.status, response.incomplete_details.as_ref());
    let incomplete = response
        .incomplete_details
        .as_ref()
        .map(|details| details.reason.as_str())
        .filter(|reason| !reason.is_empty());
    let error = match (&response.status, &reason) {
        (ResponseStatus::Incomplete, Some(FinishReason::Length | FinishReason::ContentFilter)) => {
            None
        }
        (ResponseStatus::Incomplete, _) => Some(incomplete.map_or_else(
            || "Response incomplete without a provider reason".to_owned(),
            |reason| format!("Response incomplete: {reason}"),
        )),
        // A failed or cancelled reply never replays (pi's rule).
        (ResponseStatus::Failed, _) => Some("Response failed".to_owned()),
        (ResponseStatus::Cancelled, _) => Some("Response cancelled".to_owned()),
        _ => None,
    };
    Finish {
        usage: response
            .usage
            .as_ref()
            .map(crate::completion::Usage::from)
            .unwrap_or_default(),
        reason,
        response_id: Some(response.id.clone()).filter(|id| !id.is_empty()),
        model: Some(response.model.clone()).filter(|model| !model.is_empty()),
        error,
    }
}

impl ResponsesDecoder {
    /// A decoder for one reply of a Responses endpoint.
    pub fn new() -> Self {
        Self::default()
    }

    /// Salvage replayed frames that omit their envelope bookkeeping.
    pub fn with_envelope_repair(mut self) -> Self {
        self.repair_envelopes = true;
        self
    }

    /// Recognize sentinels and error events, then try stream, whole-body, and
    /// error-envelope classifiers in order. Fall through only on corrupt results.
    fn classify_payload(&self, data: &str) -> WireEvent<ResponsesEvent> {
        if data.trim() == "[DONE]" {
            return WireEvent::Known(ResponsesEvent::Sentinel);
        }
        if is_error_event(data) {
            return WireEvent::Known(ResponsesEvent::Failure(data.to_owned()));
        }
        let body = |data: &str| {
            wire::classify_marker_keyed_frame::<ResponseBody>(data, WHOLE_BODY_MARKERS)
                .map(|body| ResponsesEvent::Whole(Box::new(body)))
        };
        let envelope = |data: &str| {
            // Error envelopes must retain the provider's diagnostic on every dialect.
            wire::classify_marker_keyed_frame::<ErrorEnvelope>(data, &["error"])
                .map(|_| ResponsesEvent::Failure(data.to_owned()))
        };
        wire::classify_or(
            data,
            |data| {
                classify_responses_frame(data).map(|chunk| ResponsesEvent::Frame {
                    raw: data.to_owned(),
                    chunk,
                })
            },
            |data| wire::classify_or(data, body, envelope),
        )
    }

    /// The item at `index` started. A call opens when it is done; any other
    /// item opens here, as `item` states it so far.
    fn added(
        &mut self,
        index: usize,
        item: serde_json::Value,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        self.vacate(index, out)?;
        self.seen.insert(index);
        match item_id(&item) {
            Some(id) => self.added_ids.insert(index, id.to_owned()),
            None => self.added_ids.remove(&index),
        };
        self.texts.remove(&index);
        match block_of(&typed(&item)?) {
            Some(block) => out.open(index, block, item),
            None => Ok(()),
        }
    }

    /// Close the item still open at `index` before another opens there: a
    /// stream whose indices were repaired to zero puts every item at one.
    fn vacate(&mut self, index: usize, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        if !out.is_open(index) {
            return Ok(());
        }
        self.awaiting_ciphertext.retain(|(at, _)| *at != index);
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
        // Text after reasoning at one index, or reasoning after text, is a
        // stream whose indices were repaired to zero moving to its next item.
        let message = field == TextField::Message;
        if self
            .texts
            .get(&index)
            .is_some_and(|streamed| (streamed.field == TextField::Message) != message)
        {
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
            out.open(index, block, serde_json::Value::Null)?;
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

    /// The item at `index` is done: `item` is its block's native, and the
    /// text it states completes the block's. A call is written whole.
    #[deny(clippy::wildcard_enum_match_arm)]
    fn done(
        &mut self,
        index: usize,
        item: serde_json::Value,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let output = typed(&item)?;
        self.seen.insert(index);
        self.written.insert(index);
        if let Some(id) = item_id(&item) {
            self.written_ids.insert(id.to_owned());
        }
        // A call the provider cut short (`status: incomplete`) keeps what its
        // arguments state; the turn then ends with `Length`.
        let call = match &output {
            Output::FunctionCall(call) => Some((
                &call.call_id,
                &call.name,
                call.arguments.as_str().to_owned(),
            )),
            Output::CustomToolCall(call) => Some((
                &call.call_id,
                &call.name,
                serde_json::json!({ "input": call.input }).to_string(),
            )),
            Output::Message(_) | Output::Reasoning { .. } | Output::Unknown(_) => None,
        };
        if let Some((call_id, name, arguments)) = call {
            // A call with no name is not a call anything can answer.
            let Ok(name) = ToolName::new(name.as_str()) else {
                tracing::warn!(
                    index,
                    "Responses tool call without a name; nothing can answer it"
                );
                return Ok(());
            };
            let block = Block::Call {
                id: CallId::from_wire(call_id.as_str()),
                name,
            };
            self.vacate(index, out)?;
            out.open(index, block, item)?;
            out.push(index, &arguments)?;
            return out.finish(index);
        }
        // An item done without being added opens here; so does one at an
        // index whose previous item still waits for its ciphertext.
        if !out.is_open(index) || self.awaiting_ciphertext.iter().any(|(at, _)| *at == index) {
            self.added(index, item.clone(), out)?;
        }
        // The item states the whole text: what the deltas left out of it
        // is pushed, and text that diverged from it stays as it streamed.
        let text = text_of(&output);
        let streamed = self.texts.remove(&index).map(|streamed| streamed.text);
        if let Some(rest) = streamed
            .as_deref()
            .map_or(Some(text.as_str()), |streamed| text.strip_prefix(streamed))
        {
            out.push(index, rest)?;
        }
        let awaits = matches!(output, Output::Reasoning { .. }) && !has_ciphertext(&item);
        let id = item_id(&item).map(str::to_owned);
        out.edit(index, |native| *native = item)?;
        match id {
            Some(id) if awaits => {
                self.awaiting_ciphertext.push((index, id));
                Ok(())
            }
            _ => out.finish(index),
        }
    }

    /// Write one output-item event into the reply.
    fn item_event(
        &mut self,
        chunk: ItemChunk,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let index = usize::try_from(chunk.output_index).map_err(|_| {
            ProviderError::Response(format!(
                "output_index {} is out of range",
                chunk.output_index
            ))
        })?;
        match chunk.data {
            ItemChunkKind::OutputItemAdded(StreamingItemDoneOutput { item, .. }) => {
                self.added(index, item, out)
            }
            ItemChunkKind::OutputItemDone(StreamingItemDoneOutput { item, .. }) => {
                self.done(index, item, out)
            }
            // A refusal is the assistant's message for that turn.
            ItemChunkKind::OutputTextDelta(DeltaTextChunk {
                delta,
                content_index,
                ..
            })
            | ItemChunkKind::RefusalDelta(DeltaTextChunk {
                delta,
                content_index,
                ..
            }) => self.extend(index, TextField::Message, content_index, &delta, out),
            ItemChunkKind::ReasoningSummaryTextDelta(SummaryTextChunk {
                delta,
                summary_index,
                ..
            }) => self.extend(index, TextField::Summary, summary_index, &delta, out),
            ItemChunkKind::ReasoningTextDelta(DeltaTextChunkWithItemId {
                delta,
                content_index,
                ..
            }) => self.extend(
                index,
                TextField::Reasoning,
                content_index.unwrap_or_default(),
                &delta,
                out,
            ),
            // Call arguments are written from the done item, which states
            // them whole; restatements repeat what the deltas carried.
            ItemChunkKind::FunctionCallArgsDelta(_)
            | ItemChunkKind::FunctionCallArgsDone(_)
            | ItemChunkKind::ContentPartAdded(_)
            | ItemChunkKind::ContentPartDone(_)
            | ItemChunkKind::OutputTextDone(_)
            | ItemChunkKind::RefusalDone(_)
            | ItemChunkKind::ReasoningSummaryPartAdded(_)
            | ItemChunkKind::ReasoningSummaryPartDone(_)
            | ItemChunkKind::ReasoningSummaryTextDone(_)
            | ItemChunkKind::ReasoningTextDone(_) => Ok(()),
        }
    }

    /// The terminal response: finish the output items no done event
    /// carried, give reasoning that waited its ciphertext, then end with how
    /// the turn ended. A body that reports its reasoning as one top-level string and
    /// no reasoning item is written that reasoning first.
    fn finish(
        &mut self,
        response: CompletionResponse,
        output: Vec<serde_json::Value>,
        mut out: Out<'_, Completion>,
    ) -> Result<Flow, ProviderError> {
        let structured_reasoning = output
            .iter()
            .any(|item| item.get("type").and_then(serde_json::Value::as_str) == Some("reasoning"));
        if !structured_reasoning
            && self.seen.is_empty()
            && let Some(reasoning) = response
                .provider_reasoning
                .as_deref()
                .filter(|reasoning| !reasoning.is_empty())
        {
            out.content(crate::message::AssistantContent::Reasoning(
                crate::message::Reasoning::new(reasoning),
            ))?;
        }
        // Only the stream's items are matched by id: a provider may give two
        // items of one reply the same id.
        let streamed_ids = std::mem::take(&mut self.written_ids);
        for (index, item) in output.into_iter().enumerate() {
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
            // from it whole.
            let opened = out.is_open(index);
            if opened
                && self
                    .added_ids
                    .get(&index)
                    .is_some_and(|added| Some(added) != id.as_ref())
            {
                continue;
            }
            if !opened {
                self.added(index, item.clone(), &mut out)?;
            }
            self.done(index, item, &mut out)?;
        }
        for (index, _) in std::mem::take(&mut self.awaiting_ciphertext) {
            out.finish(index)?;
        }
        out.raw(serde_json::to_value(&response)?);
        Ok(out.end(finish_of(&response)))
    }
}

impl<'id> Decoder<'id, Completion> for ResponsesDecoder {
    type Event = ResponsesEvent;

    fn classify(&self, frame: WireFrame) -> WireEvent<ResponsesEvent> {
        let data = frame.as_str().into_owned();
        if !self.repair_envelopes {
            return self.classify_payload(&data);
        }
        // Replayed bodies omit envelope bookkeeping fields; salvage them
        // through the same classifier, with the error wording a buffered
        // body reports.
        wire::classify_with_repair(
            &data,
            |data| self.classify_payload(data),
            repair_envelope_less_frame,
            |corrupt| {
                <serde_json::Error as serde::de::Error>::custom(format!(
                    "invalid JSON frame in buffered Responses SSE body: {corrupt}"
                ))
            },
            || {
                let kind = serde_json::from_str::<serde_json::Value>(&data)
                    .ok()
                    .and_then(|value| {
                        value
                            .get("type")
                            .and_then(serde_json::Value::as_str)
                            .map(ToOwned::to_owned)
                    })
                    .unwrap_or_default();
                <serde_json::Error as serde::de::Error>::custom(format!(
                    "malformed `{kind}` event in buffered Responses SSE body"
                ))
            },
        )
    }

    fn decode(
        &mut self,
        event: ResponsesEvent,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match event {
            ResponsesEvent::Frame {
                chunk: StreamingCompletionChunk::Delta(chunk),
                ..
            } => {
                self.item_event(chunk, &mut out)?;
                Ok(Flow::More)
            }
            ResponsesEvent::Frame {
                raw,
                chunk: StreamingCompletionChunk::Response(chunk),
            } => {
                let ResponseChunk { kind, response, .. } = chunk;
                match kind {
                    // `response.incomplete` is a genuine terminal (e.g.
                    // hitting `max_output_tokens`): the partial output and
                    // usage are kept.
                    ResponseChunkKind::ResponseCompleted
                    | ResponseChunkKind::ResponseIncomplete => {
                        let output = serde_json::from_str::<serde_json::Value>(&raw)
                            .ok()
                            .and_then(|frame| frame.get("response").map(output_items))
                            .unwrap_or_default();
                        self.finish(response, output, out)
                    }
                    ResponseChunkKind::ResponseFailed => {
                        Err(crate::error::ProviderError::from_provider_body(&raw))
                    }
                    ResponseChunkKind::ResponseCreated | ResponseChunkKind::ResponseInProgress => {
                        Ok(Flow::More)
                    }
                }
            }
            // The unary reply is the terminal response with no stream before it.
            ResponsesEvent::Whole(body) => {
                let ResponseBody { response, output } = *body;
                self.finish(response, output, out)
            }
            ResponsesEvent::Failure(raw) => {
                Err(crate::error::ProviderError::from_provider_body(&raw))
            }
            // Nothing to write: the provider's end is `response.completed`.
            ResponsesEvent::Sentinel => Ok(Flow::More),
        }
    }
}

/// Output-item event with its slot index and optional provider item ID.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ItemChunk {
    /// Item ID. Optional.
    pub item_id: Option<String>,
    /// The output index of the item from a given streamed response.
    pub output_index: u64,
    /// The item type chunk, as well as the inner data.
    #[serde(flatten)]
    pub data: ItemChunkKind,
}

/// The item chunk type from OpenAI's Responses API.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ItemChunkKind {
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded(StreamingItemDoneOutput),
    #[serde(rename = "response.output_item.done")]
    OutputItemDone(StreamingItemDoneOutput),
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded(ContentPartChunk),
    #[serde(rename = "response.content_part.done")]
    ContentPartDone(ContentPartChunk),
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta(DeltaTextChunk),
    #[serde(rename = "response.output_text.done")]
    OutputTextDone(OutputTextChunk),
    #[serde(rename = "response.refusal.delta")]
    RefusalDelta(DeltaTextChunk),
    #[serde(rename = "response.refusal.done")]
    RefusalDone(RefusalTextChunk),
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgsDelta(DeltaTextChunkWithItemId),
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgsDone(ArgsTextChunk),
    #[serde(rename = "response.reasoning_summary_part.added")]
    ReasoningSummaryPartAdded(SummaryPartChunk),
    #[serde(rename = "response.reasoning_summary_part.done")]
    ReasoningSummaryPartDone(SummaryPartChunk),
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta(SummaryTextChunk),
    #[serde(rename = "response.reasoning_summary_text.done")]
    ReasoningSummaryTextDone(SummaryTextChunk),
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta(DeltaTextChunkWithItemId),
    /// Raw-reasoning text restatement. Decoded but not emitted to avoid
    /// duplicating accumulated reasoning deltas.
    #[serde(rename = "response.reasoning_text.done")]
    ReasoningTextDone(OutputTextChunk),
    // No `#[serde(other)]` catch-all: unknown event types are triaged by the
    // classify layer (`classify_responses_frame` checks the `type` tag against
    // `is_known_responses_event_type` BEFORE decoding), so a frame that
    // reaches this decoder with an unmodeled tag is a known-set/enum drift
    // and must fail loudly (`Corrupt`) rather than be silently absorbed.
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamingItemDoneOutput {
    pub sequence_number: u64,
    /// The item as the provider stated it.
    pub item: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ContentPartChunk {
    pub content_index: u64,
    pub sequence_number: u64,
    pub part: ContentPartChunkPart,
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPartChunkPart {
    OutputText {
        text: String,
    },
    SummaryText {
        text: String,
    },
    /// Unmodeled content part retained verbatim without emitting content.
    /// Visible content is delivered by the corresponding delta events.
    #[serde(untagged)]
    Unknown(serde_json::Value),
}

/// Decode known tags only with a string text field; preserve unknown or absent tags.
/// Nonstring tags return an error. Duplicate keys use the last value retained by
/// `serde_json::Value`.
impl<'de> Deserialize<'de> for ContentPartChunkPart {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let text_field = |part: &str| -> Result<String, D::Error> {
            value
                .get("text")
                .and_then(serde_json::Value::as_str)
                .map(ToOwned::to_owned)
                .ok_or_else(|| {
                    serde::de::Error::custom(format!(
                        "`{part}` content part is missing a string `text` field"
                    ))
                })
        };
        match value.get("type").cloned() {
            Some(serde_json::Value::String(tag)) => match tag.as_str() {
                "output_text" => Ok(Self::OutputText {
                    text: text_field("output_text")?,
                }),
                "summary_text" => Ok(Self::SummaryText {
                    text: text_field("summary_text")?,
                }),
                _ => Ok(Self::Unknown(value)),
            },
            Some(_) => Err(serde::de::Error::custom(
                "content part `type` must be a string",
            )),
            None => Ok(Self::Unknown(value)),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DeltaTextChunk {
    pub content_index: u64,
    pub sequence_number: u64,
    pub delta: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DeltaTextChunkWithItemId {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_index: Option<u64>,
    pub sequence_number: u64,
    pub delta: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OutputTextChunk {
    pub content_index: u64,
    pub sequence_number: u64,
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RefusalTextChunk {
    pub content_index: u64,
    pub sequence_number: u64,
    pub refusal: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ArgsTextChunk {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_index: Option<u64>,
    pub sequence_number: u64,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SummaryPartChunk {
    pub summary_index: u64,
    pub sequence_number: u64,
    pub part: SummaryPartChunkPart,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SummaryTextChunk {
    pub summary_index: u64,
    pub sequence_number: u64,
    // `response.reasoning_summary_text.delta` carries `delta`;
    // the `.done` sibling carries the full `text` under the same shape.
    #[serde(alias = "text")]
    pub delta: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SummaryPartChunkPart {
    SummaryText { text: String },
}

#[cfg(test)]
mod tests;
