//! The streaming module for the OpenAI Responses API.
//! Please see the `openai_streaming` or `openai_streaming_with_tools` example for more practical usage.
use crate::completion::CompletionError;
use crate::operation::AdapterOutput;
use crate::operation::Completion;
use crate::providers::internal::wire::{self, WireEvent};
use crate::providers::openai::responses_api::{
    IncompleteDetailsReason, ReasoningSummary, ResponseStatus, ResponsesUsage,
};
use crate::streaming::{BlockId, StreamFinal, ToolCallEnd, UnparseableToolInput};
use crate::wire::Decoder;
use crate::wire::WireFrame;
use serde::{Deserialize, Serialize};

use super::{CompletionResponse, Output};

// ================================================================
// OpenAI Responses Streaming API
// ================================================================

/// A streaming completion chunk.
/// Streaming chunks can come in one of two forms:
/// - A response chunk (where the completed response will have the total token usage)
/// - An item chunk commonly referred to as a delta. In the completions API this would be referred to as the message delta.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum StreamingCompletionChunk {
    Response(ResponseChunk),
    Delta(ItemChunk),
}

/// The final streaming response from the OpenAI Responses API.
///
/// This is the provider-native terminal record. The adapter maps it once,
/// through `terminal_record`, into the [`StreamFinal`] the stream yields,
/// and serializes it onto [`StreamFinal::raw`] — the escape hatch for
/// Responses-API terminal fields rig does not normalize.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamingCompletionResponse {
    /// Token usage from the terminal response event; `None` when the event
    /// carried no `usage` object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponsesUsage>,
    /// The complete object-shaped reasoning metadata from the terminal response event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_metadata: Option<serde_json::Map<String, serde_json::Value>>,
    /// The effective reasoning context from the terminal response event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_context: Option<String>,
    /// The `status` reported by the terminal `response.completed` event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<ResponseStatus>,
    /// Why the response stopped short, when the provider said so.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<IncompleteDetailsReason>,
    /// The assistant message ID (`msg_...`) carried by the terminal response's
    /// output items.
    ///
    /// Distinct from [`Self::response_id`] (`resp_...`), which names the whole
    /// response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    /// The response ID (`resp_...`) reported by the terminal
    /// `response.completed` event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// The model identifier reported by the terminal response event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The transport request id from the SSE connection's `x-request-id`
    /// response header — not part of any stream frame. The transport stamps
    /// it onto the normalized [`StreamFinal`] after the adapter has mapped
    /// this record, so here it is `None` unless a caller filled it in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
}

impl StreamingCompletionResponse {
    /// Create a terminal record carrying only usage; the remaining metadata is
    /// filled in from the terminal `response.completed` event as it arrives.
    pub fn new(usage: Option<ResponsesUsage>) -> Self {
        Self {
            usage,
            provider_request_id: None,
            reasoning_metadata: None,
            reasoning_context: None,
            status: None,
            incomplete_details: None,
            message_id: None,
            response_id: None,
            model: None,
        }
    }
}

/// Normalize the Responses API's terminal stream record.
///
/// The provider descriptor name is an input for the same reason it is on the
/// unary conversion: ChatGPT and Copilot stream this exact wire shape, so a
/// baked-in `"openai"` would mislabel them.
///
/// The finish reason is left exactly as the provider reported it;
/// [`crate::streaming::StreamingCompletionResponse`] applies the tool-call
/// reconciliation afterwards, using the calls the stream actually emitted.
///
/// The native record is serialized onto [`StreamFinal::raw`]; a
/// serialization failure is the caller's to surface as an in-band error.
fn terminal_record(
    provider: &str,
    response: StreamingCompletionResponse,
) -> Result<StreamFinal, CompletionError> {
    let raw = serde_json::to_value(&response)?;
    let finish_reason = response
        .status
        .as_ref()
        .and_then(|status| super::map_finish_reason(status, response.incomplete_details.as_ref()));

    Ok(
        StreamFinal::new(provider, crate::completion::Usage::from(&response))
            .with_optional_finish_reason(finish_reason)
            .with_optional_message_id(response.message_id)
            .with_optional_response_id(response.response_id)
            .with_optional_provider_request_id(response.provider_request_id)
            .with_optional_model(response.model)
            .with_raw(raw),
    )
}

/// The done item's blocks as ONE authoritative end-of-part restatement.
///
/// Every block — summaries, content texts, `encrypted_content` — belongs to
/// one `rs_*` reasoning item, so it must land in one part: emitting a
/// whole-block end per entry made every block after the first a sibling
/// part under the same key, and history then replayed duplicate reasoning
/// input items carrying the identical `rs_*` id. The restatement supersedes
/// the delta-built part in place (wire field order: summary, content,
/// encrypted); the caller closes the block with it as a wire-sent
/// `BlockEnd`. `None` when the item carries no blocks — an empty done item
/// says nothing at the boundary.
pub(crate) fn reasoning_from_done_item(
    provider_id: Option<&str>,
    summary: Vec<ReasoningSummary>,
    content: Vec<String>,
    encrypted_content: Option<String>,
) -> Option<crate::message::Reasoning> {
    // Same builder as the unary decode, so the restatement and the
    // non-streaming conversion of one item cannot drift.
    let blocks = super::reasoning_content_blocks(summary, content, encrypted_content);

    if blocks.is_empty() {
        return None;
    }

    Some(crate::message::Reasoning {
        id: provider_id.map(str::to_owned),
        content: blocks,
    })
}

impl From<&StreamingCompletionResponse> for crate::completion::Usage {
    fn from(response: &StreamingCompletionResponse) -> Self {
        response.usage.as_ref().map(Self::from).unwrap_or_default()
    }
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

/// Classify one Responses SSE frame; see
/// [`crate::providers::internal::wire`] for the dispatch contract.
///
/// Shared by the live SSE loop, the buffered [`stream_events_from_sse_body`]
/// path, and the websocket session so all apply the same known/unknown
/// boundary. Provider `error` events (and the websocket-only `response.done`)
/// are checked separately before this, because their `type` is outside the
/// modeled set yet must not be skipped as unknown.
#[doc(hidden)]
pub fn classify_responses_frame(data: &str) -> WireEvent<StreamingCompletionChunk> {
    wire::classify_tagged_frame(data, "type", is_known_responses_event_type)
}

#[derive(Clone, Copy)]
#[doc(hidden)]
pub enum ResponsesStreamOptions {
    Strict,
    StrictWithImmediateToolCalls,
}

impl ResponsesStreamOptions {
    #[doc(hidden)]
    pub const fn strict() -> Self {
        Self::Strict
    }

    pub(crate) const fn strict_with_immediate_tool_calls() -> Self {
        Self::StrictWithImmediateToolCalls
    }

    const fn emits_completed_tool_calls_immediately(self) -> bool {
        matches!(self, Self::StrictWithImmediateToolCalls)
    }
}

#[doc(hidden)]
pub struct RawChoiceAccumulator {
    /// Stable descriptor name stamped on the terminal record: ChatGPT and
    /// Copilot stream this exact wire shape, so it is an input rather than
    /// a baked-in `"openai"`.
    provider: String,
    final_usage: Option<ResponsesUsage>,
    reasoning_metadata: Option<serde_json::Map<String, serde_json::Value>>,
    reasoning_context: Option<String>,
    status: Option<ResponseStatus>,
    incomplete_details: Option<IncompleteDetailsReason>,
    message_id: Option<String>,
    response_id: Option<String>,
    model: Option<String>,
    /// Buffered tool-call ends for calls delivered whole by
    /// `output_item.done`, flushed at the terminal (or before a terminal
    /// error) as `BlockEnd`s keyed by the slot's assembly id. Assembly and
    /// internal-id correlation live in the shared accumulator, keyed by the
    /// function-call item id the added/delta/done events share.
    tool_calls: Vec<(BlockId, ToolCallEnd)>,
    /// Whether a genuine terminal event (`response.completed` or
    /// `response.incomplete`) arrived. Without one the stream was truncated,
    /// and `finish` withholds the terminal record.
    saw_terminal: bool,
    /// Slot-scoped reasoning identity, mirroring `tool_slots`: one assembly
    /// key per output slot, fixed at the slot's FIRST reasoning event (wire
    /// `rs_*` id when it carries one, else minted `output-{index}`) and
    /// reused by every later frame regardless of the id it carries.
    /// Gateways and ChatGPT's envelope-less replay bodies omit the id on a
    /// subset of a slot's events; per-event resolution split one slot into
    /// `Wire("rs_1")` and `Minted(Output, i)` halves, and the done item
    /// superseded only one of them — the other survived as an orphaned
    /// partial part carrying the same provider id (#2258 F3 and its mixed
    /// generalization).
    reasoning_slots: std::collections::HashMap<u64, crate::streaming::BlockId>,
    /// Tool-call identities minted for function-call items whose wire events
    /// carried no `fc_*` id (gateways and the ChatGPT envelope-less replay
    /// bodies), keyed by output slot. Mirrors `minted_reasoning_ids`: the
    /// added/delta/done events of one item must all share one assembly key —
    /// forwarding `""` verbatim would let two parallel id-less calls share the
    /// empty key, and an id-less delta whose done restates a real `fc_*` id
    /// would leave the fragments dangling under a different key.
    /// Slot-scoped tool identity: one assembly key per output slot, fixed at
    /// the slot's first event (wire `fc_*` id, else minted `output-{index}`),
    /// reused by every later event regardless of the id it carries — mixed
    /// id/id-less events on one slot can no longer split assembly keys.
    tool_slots: crate::providers::internal::tool_call_bridge::ToolCallBridge<u64>,
    /// The `call_…` correlator each open slot announced on
    /// `output_item.added`, kept beside the bridge so a slot closed by the
    /// terminal drain (its `output_item.done` frame was lost) still
    /// finalizes with the dual-wire identity Responses replay pairs on.
    pending_call_ids: std::collections::HashMap<u64, String>,
    /// The message item whose text block is currently open. A text or
    /// refusal delta carrying a different `item_id` opens a new text block
    /// (a text `BlockStart` keyed by that item id), so two `message` output
    /// items aggregate as two distinct text parts instead of concatenating.
    /// Deltas without an `item_id` (ChatGPT's envelope-less replays) extend
    /// the open block, or open a boundary-minted one in the output helper.
    current_text_item: Option<String>,
    /// The message items whose visible text a delta already delivered, and
    /// whether any fragment arrived that could not be attributed to one.
    /// The terminal restates the whole turn's output, so its message text
    /// is published only where no delta delivered it: this trio is the fact
    /// `merge_terminal_body_text` reads to decide that.
    delta_text_items: std::collections::HashSet<String>,
    /// The output slots whose visible text a delta already delivered.
    ///
    /// Item ids cannot decide this alone: Copilot's Responses route stamps
    /// a FRESH `item_id` on every delta and a different one again on the
    /// terminal's message item, so id equality reports "never delivered"
    /// for text the deltas streamed in full and the turn's answer lands
    /// twice (`crates/rig-cassette/fixtures/cassettes/copilot/reasoning_roundtrip/streaming.yaml`
    /// record 2 replays it once). `output_index` is the wire's positional
    /// correlator for output items — it is what the terminal's `output[]`
    /// array is indexed by, and what `tool_slots`/`reasoning_slots`
    /// already key their assemblies on for the same reason.
    delta_text_slots: std::collections::HashSet<u64>,
    unattributed_text_delta: bool,
}

/// The assistant message ID (`msg_...`) a terminal response object carries,
/// which is deliberately not the response's own `resp_...` id.
fn message_id_from_response(response: &CompletionResponse) -> Option<String> {
    response.output.iter().find_map(|item| match item {
        Output::Message(message) => Some(message.id.clone()),
        _ => None,
    })
}

impl RawChoiceAccumulator {
    /// `initial_usage` seeds the terminal's usage for replayed bodies whose
    /// SSE frames may not carry one (the unary Responses body's own `usage`).
    #[doc(hidden)]
    pub fn new(provider: impl Into<String>, initial_usage: Option<ResponsesUsage>) -> Self {
        Self {
            provider: provider.into(),
            final_usage: initial_usage,
            reasoning_metadata: None,
            reasoning_context: None,
            status: None,
            incomplete_details: None,
            message_id: None,
            response_id: None,
            model: None,
            tool_calls: Vec::new(),
            saw_terminal: false,
            reasoning_slots: std::collections::HashMap::new(),
            tool_slots:
                crate::providers::internal::tool_call_bridge::ToolCallBridge::with_minted_namespace(
                    crate::streaming::SyntheticIds::output(),
                ),
            pending_call_ids: std::collections::HashMap::new(),
            current_text_item: None,
            delta_text_items: std::collections::HashSet::new(),
            delta_text_slots: std::collections::HashSet::new(),
            unattributed_text_delta: false,
        }
    }

    /// Open the text block for the message item a text/refusal delta belongs
    /// to, when the wire identifies it and it differs from the open one.
    fn start_text_item(&mut self, item_id: Option<&str>, out: &mut AdapterOutput) {
        if let Some(item_id) = item_id.filter(|id| !id.is_empty())
            && self.current_text_item.as_deref() != Some(item_id)
        {
            self.current_text_item = Some(item_id.to_string());
            out.text_start(BlockId::wire(item_id.to_string()), None);
        }
    }

    /// Record that a delta delivered the visible text of a message item.
    ///
    /// The output slot is always recorded; the item id is recorded on top
    /// of it, because a delta the wire did not attribute extends whichever
    /// text block is open and is credited to that item, and with no block
    /// open there is nothing to attribute it to at all.
    fn note_text_delta(&mut self, output_index: u64, item_id: Option<&str>) {
        self.delta_text_slots.insert(output_index);
        match item_id
            .filter(|id| !id.is_empty())
            .map(str::to_owned)
            .or_else(|| self.current_text_item.clone())
        {
            Some(id) => {
                self.delta_text_items.insert(id);
            }
            None => self.unattributed_text_delta = true,
        }
    }

    /// Whether a delta already delivered the visible text of the message
    /// item at `output_index` carrying `item_id`.
    ///
    /// An unattributable fragment counts for every item: its text is
    /// already in the choice and nothing on the wire says which item the
    /// terminal restates, so the merge withholds rather than risk stating
    /// one turn's text twice.
    fn delta_delivered_text(&self, output_index: u64, item_id: &str) -> bool {
        self.unattributed_text_delta
            || self.delta_text_slots.contains(&output_index)
            || self.delta_text_items.contains(item_id)
    }

    /// Publish one message item's visible text as the deltas that built it,
    /// recording what it delivered so a terminal restating the same item
    /// merges nothing.
    fn publish_message_text(
        &mut self,
        output_index: u64,
        message: &super::OutputMessage,
        out: &mut AdapterOutput,
    ) {
        // The stream opens the item's text block on its first delta and
        // sends one delta per content part; a part's own-wire extras ride
        // the block's metadata, where the accumulator merges them into the
        // one block the item published.
        self.start_text_item(Some(&message.id), out);
        if !message.content.is_empty() {
            self.note_text_delta(output_index, Some(&message.id));
        }
        for content in message.content.iter().cloned() {
            let mut text = super::text_block(content);
            super::stamp_phase(&mut text, message.phase.as_deref());
            out.text(text.text);
            if let Some(additional_params) = text.additional_params {
                out.text_meta(additional_params);
            }
        }
    }

    /// Merge the terminal body's own message text into the choice.
    ///
    /// The terminal restates the whole turn, so **its content is published
    /// only where no delta delivered it** — the same principle
    /// `reasoning_from_done_item`'s `None` implements for a restated
    /// reasoning part. A gateway answering a unary call with a replayed
    /// event stream can state a message's text *only* here (no
    /// `output_text.delta`, no `output_item.done` for it), and that text is
    /// the turn's answer; a gateway that streamed the text first restates
    /// it, and the restatement must add nothing. An empty restatement says
    /// nothing at the boundary either.
    ///
    /// Dialect-independent on purpose: a body-only terminal is a shape any
    /// Responses dialect can send, and every dialect's conformance suite
    /// runs it.
    fn merge_terminal_body_text(&mut self, response: &CompletionResponse, out: &mut AdapterOutput) {
        // The item's position in `output[]` IS the `output_index` its
        // stream events carried, which is how a restatement is matched to
        // the deltas that already delivered it.
        for (output_index, item) in response.output.iter().enumerate() {
            let output_index = output_index as u64;
            let Output::Message(message) = item else {
                continue;
            };
            if message.content.is_empty() || self.delta_delivered_text(output_index, &message.id) {
                continue;
            }
            self.publish_message_text(output_index, message, out);
        }
    }

    /// The slot's reasoning assembly key, fixed at its first reasoning
    /// event: the wire's `rs_*` id when that first frame carries one, else
    /// a minted `output-{index}` identity. Every later frame on the slot
    /// reuses the stored key regardless of the id it carries — the same
    /// discipline as `tool_slots` — so mixed id/id-less frames cannot
    /// split one slot's assembly. The durable `provider_id` is fixed on
    /// the block's start; the done item's wire-sent restatement (which
    /// always carries the real `rs_*` id) supersedes it, never the
    /// accumulation key.
    fn reasoning_slot_key(
        &mut self,
        output_index: u64,
        item_id: Option<&str>,
    ) -> crate::streaming::BlockId {
        if let Some(key) = self.reasoning_slots.get(&output_index) {
            return key.clone();
        }
        // Minted from the bridge's ONE counter (tool_call_bridge's own
        // invariant): a second sequence stamping `Minted{Output, index}`
        // could collide with an assembly the bridge minted the same value
        // for. The per-slot map above, not the mint, is what keeps the key
        // stable across the slot's frames.
        let key = item_id.map_or_else(
            || self.tool_slots.minted_ids().mint(),
            crate::streaming::BlockId::wire,
        );
        self.reasoning_slots.insert(output_index, key.clone());
        key
    }

    /// Map one item/delta event onto grammar events, pushed to `out`.
    #[doc(hidden)]
    pub fn decode_item_chunk(
        &mut self,
        chunk: ItemChunk,
        options: ResponsesStreamOptions,
        out: &mut AdapterOutput,
    ) {
        let ItemChunk {
            item_id: outer_item_id,
            output_index,
            data: item,
        } = chunk;

        match item {
            ItemChunkKind::OutputItemAdded(StreamingItemDoneOutput {
                item: Output::FunctionCall(func),
                ..
            }) => {
                // A function-call item interleaving a message item closes the
                // open text block; forget it so a later delta for that message
                // re-emits its text `BlockStart` and reactivates its block
                // downstream.
                self.current_text_item = None;
                // Slot identity is established here once (wire `fc_*` id,
                // else a minted `output-{index}`) and reused for every later
                // event on this slot — gateways and ChatGPT's envelope-less
                // replay bodies can omit the id on any subset of a slot's
                // events, and event-scoped resolution would split the
                // assembly key.
                // The `fc_*` item id keys the slot only when the item names
                // a `call_id`: on this dual-id wire the correlator is the
                // `call_id`, and a wire-keyed assembly would publish the item
                // id AS the correlator (the shared accumulator reads the key
                // as the wire's id), which replay would then send where the
                // API wants the real one.
                let wire_id = (!func.call_id.is_empty()).then_some(func.id.as_str());
                let key = self
                    .tool_slots
                    .open(output_index, wire_id, Some(&func.name))
                    .key()
                    .to_owned();
                if !func.call_id.is_empty() {
                    self.pending_call_ids
                        .insert(output_index, func.call_id.clone());
                }
                out.tool_name(&key, func.name);
            }
            ItemChunkKind::OutputItemDone(message) => {
                // Any completed item ends the block it carried; a text delta
                // arriving afterwards belongs to a (re)opened block.
                self.current_text_item = None;
                self.push_output_item_done(
                    message.item,
                    output_index,
                    out,
                    options.emits_completed_tool_calls_immediately(),
                );
            }
            // Text and refusal deltas are the same visible-text stream: a
            // refusal is the assistant's message for that turn, and both
            // (re)open the item's text block before their fragment.
            ItemChunkKind::OutputTextDelta(DeltaTextChunk { delta, .. })
            | ItemChunkKind::RefusalDelta(DeltaTextChunk { delta, .. }) => {
                self.start_text_item(outer_item_id.as_deref(), out);
                self.note_text_delta(output_index, outer_item_id.as_deref());
                out.text(delta);
            }
            // Summary and raw-reasoning deltas differ only in which wire
            // event carries them; both are fragments of the output item's
            // reasoning block and accumulate under its slot identity.
            ItemChunkKind::ReasoningSummaryTextDelta(SummaryTextChunk { delta, .. })
            | ItemChunkKind::ReasoningTextDelta(DeltaTextChunkWithItemId { delta, .. }) => {
                // Reasoning interleaving text closes the open text block
                // downstream (a non-text block event is a boundary for
                // anonymous text); forget the open message item so a later
                // delta for the *same* item re-emits its text `BlockStart`
                // and reactivates its block instead of silently opening a
                // boundary-minted sibling (#2258 P2).
                self.current_text_item = None;
                let id = self.reasoning_slot_key(output_index, outer_item_id.as_deref());
                out.reasoning_delta(
                    &id,
                    outer_item_id
                        .clone()
                        .and_then(crate::streaming::non_empty_id),
                    delta,
                );
            }
            ItemChunkKind::FunctionCallArgsDelta(delta) => {
                // Tool output interleaving text is a block boundary too.
                self.current_text_item = None;
                // The slot's established identity keys the fragment; an
                // id-less delta on a never-opened slot mints it here so the
                // fragments survive truncation before the authoritative
                // `output_item.done` restatement (#2258 P3). A late wire id
                // updates the slot's reported id without moving the key.
                let slot = self
                    .tool_slots
                    .open(output_index, outer_item_id.as_deref(), None);
                slot.saw_arguments_delta = true;
                let key = slot.key().clone();
                out.tool_arguments(&key, delta.delta);
            }
            _ => {}
        }
    }

    #[doc(hidden)]
    pub fn record_response_chunk(
        &mut self,
        kind: ResponseChunkKind,
        response: CompletionResponse,
        raw_event_data: &str,
        out: &mut AdapterOutput,
    ) -> Result<(), CompletionError> {
        match kind {
            // `response.incomplete` is a genuine terminal (e.g. hitting
            // `max_output_tokens`): the partial output and usage are kept, and
            // the recorded status/incomplete_details map to the finish reason
            // downstream, matching the unary path's `map_finish_reason`.
            ResponseChunkKind::ResponseCompleted | ResponseChunkKind::ResponseIncomplete => {
                self.saw_terminal = true;
                // The terminal restates the whole turn, so the message text
                // no delta delivered is published here: a gateway that
                // states its answer only in the terminal body still lands
                // it in the choice, and one that streamed the text first
                // does not state it twice.
                self.merge_terminal_body_text(&response, out);
                // The provider proved the turn ended, so a slot still open
                // here lost only its `output_item.done` frame — the same
                // terminal-drain the sibling adapters ship (Interactions at
                // `interaction.completed`, chat-compat at `finish_reason`).
                // Closing it lets the shared accumulator finalize the call
                // from its streamed fragments (parse-or-drop), instead of
                // discarding a provider-completed call as truncation.
                for (index, slot) in self.tool_slots.drain_ordered_indexed() {
                    let mut end = slot.end(UnparseableToolInput::Drop);
                    end.call_id = self.pending_call_ids.remove(&index);
                    self.tool_calls.push((slot.key().clone(), end));
                }
                // The terminal event is the only place the stream learns how the
                // turn ended, which model answered, and which assistant message
                // (`msg_...`, not the response's `resp_...`) carried the output.
                if let Some(message_id) = message_id_from_response(&response) {
                    self.message_id = Some(message_id);
                }
                if !response.id.is_empty() {
                    self.response_id = Some(response.id.clone());
                }
                if !response.model.is_empty() {
                    self.model = Some(response.model.clone());
                }
                self.status = Some(response.status);
                if response.incomplete_details.is_some() {
                    self.incomplete_details = response.incomplete_details;
                }
                if response.usage.is_some() {
                    self.final_usage = response.usage;
                }
                if response.reasoning_metadata.is_some() {
                    self.reasoning_metadata = response.reasoning_metadata;
                }
                if response.reasoning_context.is_some() {
                    self.reasoning_context = response.reasoning_context;
                }
                Ok(())
            }
            ResponseChunkKind::ResponseFailed => Err(
                crate::provider_response::completion_error_from_body(raw_event_data),
            ),
            _ => Ok(()),
        }
    }

    fn push_output_item_done(
        &mut self,
        item: Output,
        output_index: u64,
        out: &mut AdapterOutput,
        emit_completed_tool_calls_immediately: bool,
    ) {
        match item {
            Output::FunctionCall(func) => {
                // The done item restates the call whole; its fields are
                // authoritative over any assembled fragments, and the shared
                // accumulator correlates by the shared item id (minting the
                // internal id if no fragments preceded).
                //
                // Identity mirrors the reasoning arm below: when this slot's
                // added/delta events carried no `fc_*` id they were keyed by
                // the minted `output-{index}` identity, and the done event
                // must find that same key (even when it restates a real id)
                // or the assembled fragments dangle. A slot with no minted
                // identity keeps the wire id, minting only when it is empty.
                let slot = self.tool_slots.remove(output_index);
                // The done item restates its own call_id; the announce-time
                // copy is only for slots the terminal drain must close.
                self.pending_call_ids.remove(&output_index);
                let item_id = match &slot {
                    // The slot's established key wins even when the done item
                    // restates a real `fc_*` id — assembled fragments must
                    // not dangle under a different key.
                    Some(slot) => slot.key().clone(),
                    // Minted from the bridge's ONE counter: a done-only call
                    // stamping `Minted{Output, index}` from a second sequence
                    // could collide with a mid-assembly key the bridge minted
                    // the same value for, consuming that assembly under the
                    // wrong call.
                    // Uncorrelated (or id-less) calls key on a minted
                    // identity for the same reason `output_item.added` does.
                    None if func.id.is_empty() || func.call_id.is_empty() => {
                        self.tool_slots.minted_ids().mint()
                    }
                    None => BlockId::wire(func.id.clone()),
                };
                let mut end = ToolCallEnd::new(UnparseableToolInput::Drop);
                end.name = Some(func.name);
                // An empty `call_id` is this wire's "absent" spelling: with
                // no correlator the call carries no provider identity at all,
                // and the `fc_*` item id must NOT stand in for one — replay
                // would send it where the API wants the real `call_id`, which
                // it rejects.
                end.call_id = crate::streaming::non_empty_id(func.call_id.clone());
                // The finalized call reports the authoritative wire id even
                // when assembly keyed on a minted slot identity (the
                // accumulator honors the override), but only as the item half
                // of a correlated pair.
                end.tool_id = end
                    .call_id
                    .as_ref()
                    .and_then(|_| crate::streaming::non_empty_id(func.id.clone()));
                // The restated arguments are authoritative when they parse. A
                // turn cut by `max_output_tokens` mid-tool-call restates them
                // truncated mid-JSON (item status `incomplete`); routing the
                // raw string through the assembly buffer instead lets the
                // shared accumulator apply the settled truncation policy
                // (`UnparseableToolInput::Drop` — partial arguments never
                // fabricate a call), including when no argument fragments
                // preceded the done item.
                match func.arguments.parse() {
                    Ok(arguments) => end.arguments = Some(arguments),
                    // Fragments already streamed these bytes into the
                    // assembly buffer — re-emitting the restatement doubled
                    // them (rendered twice by delta consumers and
                    // double-charged against the accumulation bound). Only a
                    // fragment-less done item (pure replay of a truncated
                    // restatement) routes its raw string through the buffer,
                    // so the truncation policy still has bytes to judge.
                    Err(_) => {
                        let saw_fragments =
                            slot.as_ref().is_some_and(|slot| slot.saw_arguments_delta);
                        if !saw_fragments {
                            out.tool_arguments(&item_id, func.arguments.as_str());
                        }
                    }
                }

                if emit_completed_tool_calls_immediately {
                    out.tool_end(item_id, end);
                } else {
                    self.tool_calls.push((item_id, end));
                }
            }
            Output::Reasoning {
                id,
                summary,
                content,
                encrypted_content,
                ..
            } => {
                // The done item resolves through the slot map: its full
                // blocks must share whatever identity the slot's deltas
                // established (wire or minted) to supersede the delta-built
                // part — keying them by the item's own `rs_*` id would
                // append the restated content beside a minted-keyed part.
                // A slot with no established identity keeps the wire id
                // (the pure-replay shape). The durable handle is the item's
                // real `rs_*` id regardless of the accumulation key.
                let provider_id = crate::streaming::non_empty_id(id.clone());
                let slot = self.reasoning_slots.remove(&output_index);
                // No deltas preceded this item, so there is no part to
                // supersede and the item is all there is.
                let pure_replay = slot.is_none();
                let key = match slot {
                    Some(key) => key,
                    // No slot and no id (an envelope-less done item with
                    // nothing before it): mint from the bridge's ONE
                    // counter, as a delta would have.
                    None if id.is_empty() => self.tool_slots.minted_ids().mint(),
                    None => BlockId::wire(id),
                };
                let reasoning = reasoning_from_done_item(
                    provider_id.as_deref(),
                    summary,
                    content,
                    encrypted_content,
                )
                // A contentless reasoning item is still an item: its `rs_*`
                // id is the durable handle the next turn has to replay.
                // Copilot's Responses route answers a tool-calling turn
                // with `{"id":…,"summary":[]}` and then requires it back —
                // `crates/rig-cassette/fixtures/cassettes/copilot/typed_prompt_tools/
                // prompt_typed_with_tool_call_roundtrip.yaml` — and without
                // it turn two's `input` is missing an element the provider
                // sent. An item with no id at all still says nothing at the
                // boundary, and neither does an empty restatement of a part
                // the deltas already built.
                .or_else(|| {
                    let id = provider_id.filter(|_| pure_replay)?;
                    Some(crate::message::Reasoning {
                        id: Some(id),
                        content: Vec::new(),
                    })
                });
                if let Some(reasoning) = reasoning {
                    out.reasoning_end(key, Some(reasoning), None, true);
                }
            }
            Output::Message(message) => {
                // A message item with no id starts no block: there is
                // nothing to key it on.
                if let Some(id) = crate::streaming::non_empty_id(message.id) {
                    out.message_id(id);
                }
            }
            // An unmodeled output item (e.g. a hosted-tool result such as
            // `web_search_call`) arriving on `response.output_item.done`. Surface
            // the raw item to stream consumers, mirroring how the non-streaming
            // decode preserves it on `CompletionResponse.output`.
            Output::Unknown(value) => {
                out.unknown(value.into());
            }
            // A compaction item mid-stream: surfaced raw like an unmodeled
            // item so a stateless consumer can capture it from the stream.
            Output::Compaction(fields) => {
                let mut map = fields;
                map.insert(
                    "type".to_string(),
                    serde_json::Value::String("compaction".to_string()),
                );
                out.unknown(serde_json::Value::Object(map).into());
            }
        }
    }

    /// Replay one whole Responses body as the events its stream delivers.
    ///
    /// The unary reply is the same turn stated at once, so this is not a
    /// second interpreter: each `output[]` item is the
    /// `response.output_item.done` the stream sends for it — a message item
    /// preceded by the `output_text.delta`s that built its text — and the
    /// body itself is the terminal `response.completed`/`response.incomplete`
    /// event. Everything downstream is the code the stream already runs.
    pub(crate) fn replay_whole_response(
        &mut self,
        response: CompletionResponse,
        out: &mut AdapterOutput,
    ) {
        // A compatible backend that reports its reasoning as one top-level
        // string has no stream event for it, so the body is the only place it
        // is stated. Structured `reasoning` items supersede it: publishing
        // both would carry one chain of thought twice.
        let structured_reasoning = response
            .output
            .iter()
            .any(|item| matches!(item, Output::Reasoning { .. }));
        if !structured_reasoning
            && let Some(reasoning) = response
                .provider_reasoning
                .as_deref()
                .filter(|reasoning| !reasoning.is_empty())
        {
            // Minted from the bridge's ONE counter, as a delta would be:
            // this block has no wire id to key it on.
            let key = self.tool_slots.minted_ids().mint();
            out.reasoning_block(
                key,
                None,
                crate::message::ReasoningContent::Text {
                    text: reasoning.to_owned(),
                    signature: None,
                },
            );
        }

        for (output_index, item) in response.output.iter().cloned().enumerate() {
            let output_index = output_index as u64;
            if let Output::Message(message) = &item {
                self.publish_message_text(output_index, message, out);
            }
            // Published where the item appears rather than buffered to the
            // terminal: the body states every item in order, and the
            // accumulator registers a part at its START — which is where a
            // streamed call registers too (its `output_item.added`) — so
            // both paths order the choice identically.
            self.push_output_item_done(item, output_index, out, true);
        }

        // `response.completed` and `response.incomplete` are the wire's two
        // genuine terminals; any other status (`failed`, `cancelled`) rides
        // through `map_finish_reason` verbatim on the completed path, exactly
        // as the unary conversion always did.
        let kind = if matches!(response.status, ResponseStatus::Incomplete) {
            ResponseChunkKind::ResponseIncomplete
        } else {
            ResponseChunkKind::ResponseCompleted
        };
        // The raw body is read only for a `response.failed` error payload,
        // which neither of those kinds is.
        if let Err(error) = self.record_response_chunk(kind, response, "", out) {
            out.error(error);
        }
    }

    /// Flush the buffered fully-delivered tool calls without finishing the
    /// stream. The errored-terminal path flushes these before the error and
    /// must not produce a terminal record.
    #[doc(hidden)]
    pub fn flush_tool_calls(&mut self, out: &mut AdapterOutput) {
        for (id, end) in std::mem::take(&mut self.tool_calls) {
            out.tool_end(id, end);
        }
    }

    /// Flush the buffered tool calls, then the terminal record when a
    /// genuine terminal event arrived.
    #[doc(hidden)]
    pub fn finish(mut self, out: &mut AdapterOutput) {
        self.flush_tool_calls(out);
        // Only a genuine terminal event (`response.completed` or
        // `response.incomplete`) counts as the provider ending the turn; a
        // stream that ended without one was truncated,
        // and a synthesized terminal record would present the partial turn as
        // a successful, default-usage completion.
        if !self.saw_terminal {
            return;
        }
        let native = StreamingCompletionResponse {
            usage: self.final_usage,
            // The transport stamps the normalized record.
            provider_request_id: None,
            reasoning_metadata: self.reasoning_metadata,
            reasoning_context: self.reasoning_context,
            status: self.status,
            incomplete_details: self.incomplete_details,
            message_id: self.message_id,
            response_id: self.response_id,
            model: self.model,
        };
        match terminal_record(&self.provider, native) {
            Ok(record) => out.final_record(record),
            Err(error) => out.error(error),
        }
    }
}

/// Repair an envelope-less Responses frame so the shared typed decode can
/// interpret it.
///
/// ChatGPT's replayed (unary) SSE bodies omit envelope bookkeeping fields
/// (`sequence_number`, `output_index`, `content_index`, `summary_index`)
/// that the typed frame decode requires. Those fields are bookkeeping only —
/// no semantic decision reads them beyond the reasoning-identity fallback,
/// which treats a missing `output_index` as `0` anyway — so injecting
/// neutral zeros where they are absent turns salvage into a preprocessing
/// step in front of the ONE event interpreter instead of a second one.
/// Data-level fields (`delta`, `item`, `response`, …) are never touched, so
/// a frame that is defective in its content still fails the re-decode.
///
/// **Policy decision — buffered-only, deliberately asymmetric with the live
/// loop (#2258 F8):** a *live* SSE or websocket frame with a known `type` but
/// a missing envelope field classifies `Corrupt` and surfaces as an in-band
/// `Err` item; it is never repaired. Only ChatGPT's replayed unary bodies
/// verifiably omit the envelope bookkeeping (every recorded live Copilot and
/// OpenAI cassette carries full envelopes), so on a live wire an
/// envelope-less known frame is evidence of a defective gateway, and
/// silently repairing it would mask the defect the `Corrupt` classification
/// exists to surface. The old live behavior (skip) hid the frame entirely;
/// the `Err` item is the stated uniform policy for defective known frames.
///
/// **Known limit — identity collapse, and why the obvious fix is worse
/// (#2258 G2):** the injected `output_index: 0` is the reasoning-identity
/// fallback's key when `item_id` is also absent (see `reasoning_item_id` in
/// [`RawChoiceAccumulator::decode_item_chunk`]). A body that omits BOTH
/// `item_id` *and* `output_index` across two or more items therefore collapses
/// them onto the single minted identity `output-0`, merging what the provider
/// sent as separate reasoning parts. A sweep of every recorded cassette found
/// **zero** bodies of that shape: ChatGPT's replayed bodies drop the
/// bookkeeping fields but keep `item_id`, and every body that drops `item_id`
/// carries `output_index`. So the collapse is reachable in principle and
/// unobserved in practice.
///
/// It is deliberately NOT fixed with a per-frame counter (mint `0, 1, 2, …` as
/// frames arrive). Envelope-less bodies are exactly the ones where consecutive
/// frames belong to the SAME item: a counter would hand every delta of one
/// reasoning block a different index, shattering one item into N single-delta
/// parts. That is a real regression against a real recorded shape — it breaks
/// `envelope_less_reasoning_deltas_are_superseded_by_their_done_item`, whose
/// whole point is that the deltas and their `output_item.done` share one
/// identity. Any future fix must key on something the body actually carries
/// (item boundaries), not on arrival order.
///
/// Returns `None` when the frame is not a JSON object (nothing to repair).
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

/// One classified Responses frame.
///
/// The stream's SSE frames and the unary body are two shapes of the same
/// reply, so they are variants of ONE event type: the unary variant's
/// `interpret` synthesizes the very frames the stream sends — that is what the
/// accumulator's crate-internal `replay_whole_response` does — and everything after
/// classification is shared.
pub enum ResponsesEvent {
    /// One stream frame, with its raw payload: `response.failed` preserves
    /// the raw event body as the provider error body, exactly as the
    /// pre-migration loop did.
    Frame {
        /// The frame's payload, verbatim.
        raw: String,
        /// The frame, decoded.
        chunk: StreamingCompletionChunk,
    },
    /// The unary reply: the response object itself, which carries no `type`
    /// discriminator because it is not an event.
    Whole(Box<CompletionResponse>),
    /// A success whose body is the provider's error envelope instead of a
    /// response, with the raw body the error preserves. Both the stream's
    /// own `error` event and a 200 whose whole body is an envelope reach
    /// this.
    Failure(String),
    /// The wire's `[DONE]` terminal sentinel: not JSON, and it says
    /// nothing `response.completed` has not already said, so it is a
    /// modeled no-op rather than a frame to parse. Known by construction —
    /// classifying it as unknown would warn on every stream, and routing
    /// it into the typed decode would fail every stream.
    Sentinel,
}

/// The top-level keys that recognize a Responses reply body: `object`
/// (`"response"`), the two fields every reply carries whatever the gateway
/// omits, and `error` — a success whose body is nothing but the provider's
/// error envelope is a reply too, and recognizing it here is what lets the
/// body decode fail and hand the frame to the envelope classifier. A stream
/// event carries none of them at top level (its response object is nested
/// under `response`), so the shapes cannot be confused.
const WHOLE_BODY_MARKERS: &[&str] = &["object", "output", "status", "error"];

/// Whether a frame is the Responses stream's own `error` event.
///
/// `error` is outside the modeled event set on purpose — it is not one of
/// the turn's `response.*`/item events — but it is the protocol's in-band
/// failure on every dialect, so it must never be skipped as unmodeled.
fn is_error_event(data: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(data)
        .is_ok_and(|value| value.get("type").and_then(serde_json::Value::as_str) == Some("error"))
}

/// The error envelope a success body can carry instead of a response. The
/// error itself is built from the raw body; this only proves the shape.
#[derive(Deserialize)]
struct ErrorEnvelope {
    // Decoding it is the whole point — it proves the body is an envelope and
    // nothing else — but the error the consumer sees is built from the raw
    // body, so the provider's payload rides out verbatim.
    #[allow(dead_code)]
    error: serde_json::Value,
}

/// The OpenAI Responses wire's decoder: one state machine for the SSE
/// stream, the unary body and the websocket session.
///
/// Holds the per-reply assembly state ([`RawChoiceAccumulator`]); frame
/// triage policy lives in the driver, not here.
pub struct ResponsesDecoder {
    /// The reply's own envelope, captured from the terminal event.
    ///
    /// A unary call on a dialect that always streams answers with an event
    /// stream, so there is no reply document for the driver to parse; the
    /// terminal `response.completed` carries it, and this is what makes
    /// `CompletionResponse::raw` the reply rather than a summary of it.
    document: Option<serde_json::Value>,
    accumulator: RawChoiceAccumulator,
    options: ResponsesStreamOptions,
    /// Envelope salvage for replayed bodies that omit the bookkeeping
    /// fields. A dialect flag, not a mode flag: only the ChatGPT gateway
    /// verifiably sends them, and it sends the same bytes for a unary and a
    /// streamed turn, so there is no mode to key it on. Off for OpenAI and
    /// xAI, whose live frames carry full envelopes — see
    /// [`repair_envelope_less_frame`] for why repairing those would hide a
    /// defect.
    repair_envelopes: bool,
    /// A `response.failed` event (or a success-status error envelope) ended
    /// the turn: the flush-then-`Err` sequence has been pushed and the
    /// driver stops consuming.
    finished: bool,
}

impl ResponsesDecoder {
    /// A decoder for one reply of `provider`'s Responses endpoint.
    pub fn new(provider: &str, options: ResponsesStreamOptions) -> Self {
        Self {
            document: None,
            accumulator: RawChoiceAccumulator::new(provider, None),
            options,
            repair_envelopes: false,
            finished: false,
        }
    }

    /// Salvage replayed frames that omit their envelope bookkeeping.
    pub fn with_envelope_repair(mut self) -> Self {
        self.repair_envelopes = true;
        self
    }

    /// Seed the terminal's usage for a replayed body whose frames may not
    /// carry one (the unary Responses body's own `usage`).
    pub fn with_initial_usage(mut self, usage: Option<ResponsesUsage>) -> Self {
        self.accumulator.final_usage = usage;
        self
    }

    /// The shapes one Responses reply can take, in the order that
    /// distinguishes them: the `[DONE]` sentinel and the stream's own
    /// `error` event, both of which pre-empt classification because their
    /// payloads are outside the modeled event set; then a tagged stream
    /// event; then the unary body, which is the response object itself and
    /// so carries no `type` (the tagged classifier reports that as
    /// `Corrupt`); and finally a bare error envelope, on a gateway that
    /// answers a success with one. Everything past the pre-emptions is
    /// composed through the classify layer's own combinator, so no triage
    /// policy is restated here.
    ///
    /// The two pre-emptions are not stylistic. [`wire::classify_or`] falls
    /// through to its second classifier only on `Corrupt`, and the tagged
    /// classifier reports an unmodeled `type` as `Unknown` — so a frame
    /// whose `type` is `"error"` would be *skipped* rather than handed to
    /// the envelope branch below, and EOF would become the diagnostic
    /// instead of the provider's own message. `[DONE]` is not an object at
    /// all. The chat wire's decoder pre-empts both for the same reason.
    fn classify_payload(&self, data: &str) -> WireEvent<ResponsesEvent> {
        if data.trim() == "[DONE]" {
            return WireEvent::Known(ResponsesEvent::Sentinel);
        }
        if is_error_event(data) {
            return WireEvent::Known(ResponsesEvent::Failure(data.to_owned()));
        }
        let body = |data: &str| {
            wire::classify_marker_keyed_frame::<CompletionResponse>(data, WHOLE_BODY_MARKERS)
                .map(|response| ResponsesEvent::Whole(Box::new(response)))
        };
        let envelope = |data: &str| {
            // An `error` payload is the Responses protocol's own in-band
            // failure on EVERY dialect — the stream's `error` event — so it
            // is recognized unconditionally. `error_envelope_in_success` is
            // a different fact about a different shape: a gateway answering
            // a 200 with an error envelope as the whole BODY. Gating the
            // event on that quirk made a standard protocol frame
            // unreadable, and EOF became the diagnostic instead of the
            // provider's own message.
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

    fn interpret_frame(
        &mut self,
        raw: String,
        chunk: StreamingCompletionChunk,
        out: &mut AdapterOutput,
    ) {
        match chunk {
            StreamingCompletionChunk::Delta(chunk) => {
                self.accumulator.decode_item_chunk(chunk, self.options, out);
            }
            StreamingCompletionChunk::Response(chunk) => {
                let ResponseChunk { kind, response, .. } = chunk;
                // The reply's whole envelope, and on a unary call over this
                // wire there is no other document: the bytes were an event
                // stream. Every `response.*` frame carries a snapshot of the
                // same envelope, so the LAST one wins — `response.created`
                // and `response.in_progress` precede the usage and the final
                // status, and latching the first would hand back a
                // pre-completion snapshot.
                self.document = serde_json::to_value(&response).ok();
                if matches!(kind, ResponseChunkKind::ResponseCompleted) {
                    // Inert under the driver, which records the same fields
                    // off the terminal record; the client layer's stream
                    // loop has no other recording site.
                    let span = tracing::Span::current();
                    span.record("gen_ai.response.id", response.id.as_str());
                    span.record("gen_ai.response.model", response.model.as_str());
                }
                if let Err(error) = self
                    .accumulator
                    .record_response_chunk(kind, response, &raw, out)
                {
                    // `response.failed`: fully-delivered tool calls flush
                    // before the terminal error, which ends the reply with
                    // no terminal record, preserving the failure signal.
                    self.accumulator.flush_tool_calls(out);
                    out.error(error);
                    self.finished = true;
                }
            }
        }
    }

    /// Flush what the accumulator still holds: the buffered tool calls, then
    /// the terminal record when a genuine terminal arrived.
    fn flush(&mut self, out: &mut AdapterOutput) {
        let provider = self.accumulator.provider.clone();
        let accumulator = std::mem::replace(
            &mut self.accumulator,
            RawChoiceAccumulator::new(provider, None),
        );
        accumulator.finish(out);
    }
}

impl Decoder<Completion> for ResponsesDecoder {
    type Event = ResponsesEvent;

    fn classify(&self, frame: WireFrame) -> WireEvent<ResponsesEvent> {
        let data = frame.as_str().into_owned();
        if !self.repair_envelopes {
            return self.classify_payload(&data);
        }
        // Replayed bodies omit envelope bookkeeping fields; salvage through
        // the SAME interpreter, with the operation-error wording the
        // buffered driver surfaces verbatim.
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

    fn interpret(&mut self, event: ResponsesEvent, out: &mut AdapterOutput) {
        if self.finished {
            return;
        }

        match event {
            ResponsesEvent::Frame { raw, chunk } => self.interpret_frame(raw, chunk, out),
            // The unary reply is the same turn stated at once: replay it as
            // the events the stream sends, then close it with the terminal
            // the body itself is.
            ResponsesEvent::Whole(response) => {
                self.document = serde_json::to_value(&*response).ok();
                self.accumulator.replay_whole_response(*response, out);
                self.flush(out);
            }
            ResponsesEvent::Failure(raw) => {
                self.accumulator.flush_tool_calls(out);
                out.error(crate::provider_response::completion_error_from_body(&raw));
                self.finished = true;
            }
            // Nothing to interpret: the terminal record comes from
            // `response.completed`, or from the driver's EOF flush.
            ResponsesEvent::Sentinel => {}
        }
    }

    fn finish(&mut self, out: &mut AdapterOutput) {
        self.flush(out);
    }

    fn flush_before_terminal_error(&mut self, out: &mut AdapterOutput) {
        // Tool calls the provider fully delivered are content: they flush
        // before the terminal error reaches the consumer.
        self.accumulator.flush_tool_calls(out);
    }

    fn document(&self) -> Option<serde_json::Value> {
        self.document.clone()
    }

    fn project(&self, payload: &[u8], sink: &mut dyn crate::wire::ObservationSink) {
        super::wire::project_payload(payload, sink);
    }

    fn is_finished(&self) -> bool {
        self.finished
    }
}

/// An item message chunk from OpenAI's Responses API.
/// See
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
    /// Terminator for a raw-reasoning block, restating the text the
    /// `response.reasoning_text.delta` events already streamed.
    ///
    /// Modeled but not acted on — it falls into `decode_item_chunk`'s no-op
    /// arm exactly like [`Self::ReasoningSummaryTextDone`], because the
    /// accumulated deltas are already the authoritative content and replaying
    /// the restatement would double the reasoning text.
    ///
    /// The variant is what makes naming the tag in
    /// `is_known_responses_event_type` safe: the classify layer sends every
    /// KNOWN tag straight to `decode_known`, so listing the tag without a
    /// variant to decode into would turn today's benign warn-and-skip into an
    /// in-band `Corrupt`/`Err` on every raw-reasoning block (#2258 G4). The
    /// two edits only make sense together.
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
    pub item: Output,
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
    /// Any part type this client doesn't model — `refusal` and
    /// `reasoning_text` parts appear on real refusal/reasoning-text turns,
    /// and new part types ship without notice. Content-part events are
    /// bookkeeping (the content itself arrives via the corresponding delta
    /// events, e.g. `response.refusal.delta`), so an unmodeled part must
    /// parse as a no-op rather than fail the whole chunk — the same shape as
    /// [`Output::Unknown`](super::Output).
    #[serde(untagged)]
    Unknown(serde_json::Value),
}

/// Hand-written tag dispatch instead of a trailing `#[serde(untagged)]`
/// variant: on an internally-tagged enum the untagged fallback also swallows
/// a *known* tag with an invalid payload, silently demoting a data-level
/// defect to a skippable unknown part
/// (`rig-2257-code-review-findings-34ee8ba5.md` P2). Here a known part tag
/// must decode fully or error; only an unmodeled (or absent) tag falls back
/// to [`ContentPartChunkPart::Unknown`], preserving the value verbatim.
///
/// Two documented edges of the hand dispatch (#2258 F8):
/// - A part with **duplicate `type` keys** dispatches on the **last**
///   occurrence, because `serde_json::Value` keeps the last duplicate, while
///   a derived internally-tagged enum takes the first. Duplicate keys are
///   not something any Responses gateway emits; the divergence is accepted
///   and pinned by test rather than papered over with a custom map visitor.
/// - A **non-string `type`** is a data-level defect of the tagged shape, not
///   an unmodeled part kind: it errors (classifying the frame `Corrupt`)
///   instead of degrading to an `Unknown` no-op.
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
