//! The streaming module for the OpenAI Responses API.
//! Please see the `openai_streaming` or `openai_streaming_with_tools` example for more practical usage.
use crate::completion::{self, CompletionError};
use crate::http_client::HttpClientExt;
use crate::http_client::sse::GenericEventSource;
use crate::providers::internal::adapter::{
    AdapterOutput, WireAdapter, WireFrame, run_wire_buffered,
};
use crate::providers::internal::sse_transport::{
    FrameDisposition, OpenLog, SseTransportOptions, open_wire_stream, stamp_terminal_request_id,
};
use crate::providers::internal::wire::{self, WireEvent};
use crate::providers::openai::responses_api::{
    IncompleteDetailsReason, ReasoningSummary, ResponseStatus, ResponsesUsage,
};
use crate::streaming::{
    self, BlockId, StreamEvent, StreamFinal, StreamingResult, ToolCallEnd, UnparseableToolInput,
};
use crate::telemetry::{CompletionOperation, CompletionSpanBuilder};
use crate::wasm_compat::WasmCompatSend;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use super::{CompletionResponse, GenericResponsesCompletionModel, Output, ResponsesProviderExt};

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
    Response(Box<ResponseChunk>),
    Delta(ItemChunk),
}

/// The final streaming response from the OpenAI Responses API.
///
/// This is the provider-native terminal record. The adapter maps it once,
/// through [`terminal_record`], into the [`StreamFinal`] the stream yields,
/// and serializes it onto [`StreamFinal::raw`] — the escape hatch for
/// Responses-API terminal fields rig does not normalize.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamingCompletionResponse {
    /// Token usage
    pub usage: ResponsesUsage,
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
    pub fn new(usage: ResponsesUsage) -> Self {
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
/// [`streaming::StreamingCompletionResponse`] applies the tool-call
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
        StreamFinal::new(provider, crate::completion::Usage::from(&response.usage))
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
        Self::from(&response.usage)
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

fn provider_response_from_responses_error_value(
    value: &serde_json::Value,
    data: &str,
) -> CompletionError {
    if let Some(message) = value
        .get("error")
        .and_then(|error| error.get("message"))
        .and_then(serde_json::Value::as_str)
    {
        tracing::warn!(message, "provider returned a streaming error event");
    }

    crate::provider_response::completion_error_from_body(data)
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

fn provider_response_from_responses_sse_data(data: &str) -> Option<CompletionError> {
    let value = serde_json::from_str::<serde_json::Value>(data).ok()?;
    (value.get("type").and_then(serde_json::Value::as_str) == Some("error"))
        .then(|| provider_response_from_responses_error_value(&value, data))
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

/// The payload of every content-bearing `data:` line in a buffered SSE body.
///
/// Blank lines, non-`data:` fields (SSE comments, `event:`), and the `[DONE]`
/// sentinel are skipped, so both buffered readers below see exactly the frame
/// payloads a live transport would deliver.
fn sse_data_frames(body: &str) -> impl Iterator<Item = &str> {
    body.lines()
        .map(|line| {
            line.strip_prefix("data:")
                .map(str::trim)
                .unwrap_or_default()
        })
        .filter(|data| !data.is_empty() && *data != "[DONE]")
}

pub(crate) fn parse_sse_completion_body(
    body: &str,
    provider_name: &str,
) -> Result<CompletionResponse, CompletionError> {
    let mut completed = None;

    for data in sse_data_frames(body) {
        if let Ok(chunk) = serde_json::from_str::<StreamingCompletionChunk>(data) {
            if let StreamingCompletionChunk::Response(chunk) = chunk {
                let ResponseChunk { kind, response, .. } = *chunk;
                match kind {
                    // `response.incomplete` is a genuine terminal; the unary
                    // conversion maps its status to a finish reason.
                    ResponseChunkKind::ResponseCompleted
                    | ResponseChunkKind::ResponseIncomplete => {
                        completed = Some(response);
                        break;
                    }
                    ResponseChunkKind::ResponseFailed => {
                        return Err(crate::provider_response::completion_error_from_body(data));
                    }
                    _ => {}
                }
            }
            continue;
        }

        let Ok(value) = serde_json::from_str::<serde_json::Value>(data) else {
            continue;
        };

        match value.get("type").and_then(serde_json::Value::as_str) {
            Some("response.completed") | Some("response.incomplete") => {
                if let Some(response) = value.get("response") {
                    completed = Some(serde_json::from_value(response.clone())?);
                    break;
                }
            }
            Some("response.failed") => {
                return Err(crate::provider_response::completion_error_from_body(data));
            }
            Some("error") => {
                return Err(provider_response_from_responses_error_value(&value, data));
            }
            _ => {}
        }
    }

    completed.ok_or_else(|| {
        CompletionError::ProviderError(format!(
            "{provider_name} stream did not yield a terminal response event (response.completed or response.incomplete)"
        ))
    })
}

#[doc(hidden)]
pub struct RawChoiceAccumulator {
    /// Stable descriptor name stamped on the terminal record: ChatGPT and
    /// Copilot stream this exact wire shape, so it is an input rather than
    /// a baked-in `"openai"`.
    provider: String,
    final_usage: ResponsesUsage,
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
}

impl RawChoiceAccumulator {
    #[doc(hidden)]
    pub fn new(provider: impl Into<String>, initial_usage: ResponsesUsage) -> Self {
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
        }
    }

    /// Open the text block for the message item a text/refusal delta belongs
    /// to, when the wire identifies it and it differs from the open one.
    fn start_text_item(&mut self, item_id: Option<&str>, out: &mut AdapterOutput) {
        if let Some(item_id) = item_id
            && self.current_text_item.as_deref() != Some(item_id)
        {
            self.current_text_item = Some(item_id.to_string());
            out.text_start(BlockId::wire(item_id.to_string()), None);
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
                let key = self
                    .tool_slots
                    .open(output_index, Some(&func.id), Some(&func.name))
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
    ) -> Result<(), CompletionError> {
        match kind {
            // `response.incomplete` is a genuine terminal (e.g. hitting
            // `max_output_tokens`): the partial output and usage are kept, and
            // the recorded status/incomplete_details map to the finish reason
            // downstream, matching the unary path's `map_finish_reason`.
            ResponseChunkKind::ResponseCompleted | ResponseChunkKind::ResponseIncomplete => {
                self.saw_terminal = true;
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
                if let Some(usage) = response.usage {
                    self.final_usage = usage;
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
                    None if func.id.is_empty() => self.tool_slots.minted_ids().mint(),
                    None => BlockId::wire(func.id.clone()),
                };
                let mut end = ToolCallEnd::new(UnparseableToolInput::Drop);
                end.name = Some(func.name);
                // The finalized call reports the authoritative wire id even
                // when assembly keyed on a minted slot identity (the
                // accumulator honors the override).
                end.tool_id = crate::streaming::non_empty_id(func.id.clone());
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
                end.call_id = Some(func.call_id);

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
                let key = self
                    .reasoning_slots
                    .remove(&output_index)
                    .unwrap_or(BlockId::wire(id));
                if let Some(reasoning) = reasoning_from_done_item(
                    provider_id.as_deref(),
                    summary,
                    content,
                    encrypted_content,
                ) {
                    out.reasoning_end(key, Some(reasoning), None, true);
                }
            }
            Output::Message(message) => {
                out.message_id(message.id);
            }
            // An unmodeled output item (e.g. a hosted-tool result such as
            // `web_search_call`) arriving on `response.output_item.done`. Surface
            // the raw item to stream consumers, mirroring how the non-streaming
            // decode preserves it on `CompletionResponse.output`.
            Output::Unknown(value) => {
                out.unknown(value.into());
            }
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

pub(crate) fn stream_events_from_sse_body(
    provider: &str,
    body: &str,
    initial_usage: ResponsesUsage,
) -> Result<Vec<StreamEvent>, CompletionError> {
    // Framing layer for the buffered (unary) Responses SSE body: line
    // splitting, sentinel skipping, and the provider `error` envelope
    // pre-check (which fails the operation, mirroring the live transport).
    // Classification and policy live in the buffered driver.
    let mut frames = Vec::new();
    for data in sse_data_frames(body) {
        if let Some(error) = provider_response_from_responses_sse_data(data) {
            return Err(error);
        }

        frames.push(WireFrame::Text(data.to_owned()));
    }

    // The SAME interpreter as the live loop (`classify_responses_frame`
    // feeding `RawChoiceAccumulator`), under [`run_wire_buffered`]'s
    // no-stream policy: there is no stream to carry `Err` items, so `Corrupt`
    // frames — and adapter-detected data errors like `response.failed` — fail
    // the whole operation instead of returning a silently partial completion.
    // Buffered classification adds the envelope-repair salvage; see
    // [`ResponsesAdapter::buffered`].
    run_wire_buffered(frames, ResponsesAdapter::buffered(provider, initial_usage))
}

pub(crate) async fn completion_response_from_sse_body(
    provider: &str,
    body: &str,
    raw_response: CompletionResponse,
) -> Result<completion::CompletionResponse, CompletionError> {
    let events = stream_events_from_sse_body(
        provider,
        body,
        raw_response.usage.unwrap_or_else(ResponsesUsage::new),
    )?;
    completion_response_from_stream_events(provider, events, &raw_response)
        .await?
        .ok_or_else(|| CompletionError::ResponseError("Response contained no parts".to_owned()))
}

/// Replay accumulated stream events through
/// [`streaming::StreamingCompletionResponse`] and merge the result with the
/// parsed terminal response body.
///
/// The replayed stream is authoritative where it reported something; the
/// terminal body fills any gap it left (usage, message ID, finish reason,
/// model). Returns `Ok(None)` when the replay produced no content, leaving the
/// caller to decide how to fall back.
#[doc(hidden)]
pub async fn completion_response_from_stream_events(
    provider: &str,
    events: Vec<StreamEvent>,
    raw_response: &CompletionResponse,
) -> Result<Option<completion::CompletionResponse>, CompletionError> {
    let stream: StreamingResult = Box::pin(futures::stream::iter(
        events.into_iter().map(Ok::<_, CompletionError>),
    ));
    let mut stream = streaming::StreamingCompletionResponse::stream(provider, stream);

    while let Some(item) = stream.next().await {
        item?;
    }

    let mut choice = stream.snapshot();
    if choice_is_empty(&choice) {
        return Ok(None);
    }

    // Merge per content kind: the replayed choice is authoritative for what it
    // carried (reasoning, tool calls, streamed text), but some backends emit
    // message text only in the terminal body while streaming other kinds as
    // deltas. A replay with no message text takes the body's message content;
    // everything replayed is kept.
    // Presence of ANY streamed text — even whitespace — means the deltas were
    // the content channel; merging the body then would duplicate it.
    let replay_has_message_text = choice.iter().any(|content| {
        matches!(
            content,
            completion::AssistantContent::Text(text) if !text.text.is_empty()
        )
    });
    if !replay_has_message_text {
        choice.extend(
            raw_response
                .output
                .iter()
                .filter(|item| matches!(item, Output::Message(_)))
                .cloned()
                .flat_map(<Vec<completion::AssistantContent>>::from),
        );
    }

    let terminal = stream.response.clone();
    let usage = terminal.as_ref().map_or_else(
        || usage_from_raw_response(raw_response),
        |terminal| terminal.usage,
    );
    let message_id = stream
        .message_id
        .clone()
        .or_else(|| message_id_from_response(raw_response));
    let finish_reason = terminal
        .as_ref()
        .and_then(|terminal| terminal.finish_reason.clone())
        .or_else(|| {
            super::map_finish_reason(
                &raw_response.status,
                raw_response.incomplete_details.as_ref(),
            )
        });
    let model = terminal
        .as_ref()
        .and_then(|terminal| terminal.model.clone())
        .or_else(|| Some(raw_response.model.clone()).filter(|model| !model.is_empty()));

    let response_id = stream
        .response
        .as_ref()
        .and_then(|terminal| terminal.response_id.clone())
        .or_else(|| Some(raw_response.id.clone()).filter(|id| !id.is_empty()));

    Ok(Some(
        completion::CompletionResponse::new(choice, usage, provider)
            .with_optional_message_id(message_id)
            .with_optional_response_id(response_id)
            .with_optional_model(model)
            .with_optional_finish_reason(finish_reason),
    ))
}

fn choice_is_empty(choice: &[completion::AssistantContent]) -> bool {
    choice.iter().all(|content| match content {
        completion::AssistantContent::Text(text) => text.text.trim().is_empty(),
        completion::AssistantContent::Reasoning(reasoning) => reasoning.content.is_empty(),
        completion::AssistantContent::Image(_) => false,
        completion::AssistantContent::ToolCall(_) => false,
    })
}

fn message_id_from_response(response: &CompletionResponse) -> Option<String> {
    response.output.iter().find_map(|item| match item {
        Output::Message(message) => Some(message.id.clone()),
        _ => None,
    })
}

fn usage_from_raw_response(response: &CompletionResponse) -> completion::Usage {
    response
        .usage
        .as_ref()
        .map(completion::Usage::from)
        .unwrap_or_default()
}

/// Open a Responses SSE stream for `provider`, as the grammar events
/// [`completion::CompletionModel::stream`] wraps in a
/// [`streaming::StreamingCompletionResponse`].
pub(crate) fn responses_stream_from_event_source<HttpClient, RequestBody>(
    provider: &str,
    event_source: GenericEventSource<HttpClient, RequestBody>,
    span: tracing::Span,
) -> StreamingResult
where
    HttpClient: HttpClientExt + Clone + 'static,
    RequestBody: Into<bytes::Bytes> + Clone + WasmCompatSend + 'static,
{
    responses_stream_from_event_source_with_options(
        provider,
        event_source,
        span,
        ResponsesStreamOptions::strict(),
    )
}

pub(crate) fn responses_stream_from_event_source_with_options<HttpClient, RequestBody>(
    provider: &str,
    event_source: GenericEventSource<HttpClient, RequestBody>,
    span: tracing::Span,
    options: ResponsesStreamOptions,
) -> StreamingResult
where
    HttpClient: HttpClientExt + Clone + 'static,
    RequestBody: Into<bytes::Bytes> + Clone + WasmCompatSend + 'static,
{
    // The wire's in-band provider `error` envelope is a terminal transport
    // condition, detected pre-classification exactly as an HTTP failure
    // would be.
    open_wire_stream(
        event_source,
        SseTransportOptions {
            open_log: OpenLog::Trace,
            stream_ended_is_error: false,
            log_transport_errors: true,
        },
        |data| {
            if data.trim().is_empty() || data == "[DONE]" {
                return FrameDisposition::Skip;
            }
            if let Some(error) = provider_response_from_responses_sse_data(&data) {
                // A terminal failure: the driver flushes fully-delivered
                // content, yields this error last, and emits no terminal
                // record.
                return FrameDisposition::Fail(error);
            }
            FrameDisposition::Frame(data)
        },
        ResponsesAdapter::live(provider, options),
        span,
    )
}

/// One classified Responses frame, carrying its raw payload alongside the
/// decoded chunk: `response.failed` preserves the raw event body as the
/// provider error body, exactly as the pre-migration loop did.
pub(crate) struct ResponsesFrameEvent {
    raw: String,
    chunk: StreamingCompletionChunk,
}

/// The OpenAI Responses SSE wire as a [`WireAdapter`], shared by the live
/// loop ([`run_wire_stream`]) and the buffered unary path
/// ([`run_wire_buffered`]).
///
/// Holds the per-stream assembly state ([`RawChoiceAccumulator`]); frame
/// triage policy lives in the drivers, not here. The two modes differ only in
/// classification: the buffered mode adds the envelope-repair salvage for
/// ChatGPT's replayed bodies (see [`repair_envelope_less_frame`] for why the
/// live wire deliberately does NOT repair).
pub(crate) struct ResponsesAdapter {
    accumulator: RawChoiceAccumulator,
    options: ResponsesStreamOptions,
    /// Buffered-only envelope salvage; `false` on the live wire.
    repair_envelopes: bool,
    /// A `response.failed` event ended the turn: the flush-then-`Err`
    /// sequence has been pushed and the driver stops consuming.
    finished: bool,
}

impl ResponsesAdapter {
    fn live(provider: &str, options: ResponsesStreamOptions) -> Self {
        Self {
            accumulator: RawChoiceAccumulator::new(provider, ResponsesUsage::new()),
            options,
            repair_envelopes: false,
            finished: false,
        }
    }

    fn buffered(provider: &str, initial_usage: ResponsesUsage) -> Self {
        Self {
            accumulator: RawChoiceAccumulator::new(provider, initial_usage),
            options: ResponsesStreamOptions::strict(),
            repair_envelopes: true,
            finished: false,
        }
    }
}

impl WireAdapter for ResponsesAdapter {
    type Frame = WireFrame;
    type Event = ResponsesFrameEvent;

    fn classify(&self, frame: WireFrame) -> WireEvent<ResponsesFrameEvent> {
        let data = frame.as_str().into_owned();
        let event = if self.repair_envelopes {
            // Buffered bodies (ChatGPT's replayed unary SSE) omit envelope
            // bookkeeping fields; salvage through the SAME interpreter, with
            // the operation-error wording the buffered driver surfaces
            // verbatim.
            wire::classify_with_repair(
                &data,
                classify_responses_frame,
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
        } else {
            classify_responses_frame(&data)
        };
        event.map(|chunk| ResponsesFrameEvent { raw: data, chunk })
    }

    fn interpret(&mut self, event: ResponsesFrameEvent, out: &mut AdapterOutput) {
        if self.finished {
            return;
        }

        match event.chunk {
            StreamingCompletionChunk::Delta(chunk) => {
                self.accumulator.decode_item_chunk(chunk, self.options, out);
            }
            StreamingCompletionChunk::Response(chunk) => {
                let ResponseChunk { kind, response, .. } = *chunk;
                if matches!(kind, ResponseChunkKind::ResponseCompleted) {
                    let span = tracing::Span::current();
                    span.record("gen_ai.response.id", response.id.as_str());
                    span.record("gen_ai.response.model", response.model.as_str());
                }
                if let Err(error) = self
                    .accumulator
                    .record_response_chunk(kind, response, &event.raw)
                {
                    // `response.failed`: fully-delivered tool calls flush
                    // before the terminal error, which ends the stream with
                    // no terminal record, preserving the failure signal.
                    self.accumulator.flush_tool_calls(out);
                    out.error(error);
                    self.finished = true;
                }
            }
        }
    }

    fn finish(&mut self, out: &mut AdapterOutput) {
        let provider = self.accumulator.provider.clone();
        let accumulator = std::mem::replace(
            &mut self.accumulator,
            RawChoiceAccumulator::new(provider, ResponsesUsage::new()),
        );
        let final_usage = accumulator.final_usage;

        // Flush buffered tool calls, then the terminal record when a genuine
        // terminal event arrived; EOF without one is truncation and the
        // accumulator withholds the record (deferral, never synthesis).
        accumulator.finish(out);

        let span = tracing::Span::current();
        span.record("gen_ai.usage.input_tokens", final_usage.input_tokens);
        span.record("gen_ai.usage.output_tokens", final_usage.output_tokens);
        let cached_tokens = final_usage
            .input_tokens_details
            .as_ref()
            .map_or(0, |d| d.cached_tokens);
        span.record("gen_ai.usage.cache_read.input_tokens", cached_tokens);
    }

    fn flush_before_terminal_error(&mut self, out: &mut AdapterOutput) {
        // Tool calls the provider fully delivered are content: they flush
        // before the terminal error reaches the consumer.
        self.accumulator.flush_tool_calls(out);
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

impl<Ext, H> GenericResponsesCompletionModel<Ext, H>
where
    crate::client::Client<Ext, H>: HttpClientExt + Clone + WasmCompatSend + 'static,
    Ext: crate::client::Provider + ResponsesProviderExt + Clone + 'static,
    H: Clone + WasmCompatSend + 'static,
{
    /// Open a Responses stream.
    ///
    /// The terminal record's provider-native form — the escape hatch for
    /// Responses-API terminal fields rig does not normalize — rides on
    /// [`StreamFinal::raw`] as the serialized [`StreamingCompletionResponse`].
    pub(crate) async fn stream(
        &self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse, CompletionError> {
        let system_instructions = completion_request.system_instructions().map(str::to_owned);
        let record_telemetry_content = completion_request.record_telemetry_content;
        let (request_model, request) = self.create_provider_request(completion_request, true)?;

        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            "Responses streaming completion request",
            &request,
        );

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post(Ext::RESPONSES_PATH)?
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        let span = CompletionSpanBuilder::new(
            Ext::PROVIDER_NAME,
            &request_model,
            CompletionOperation::ChatStreaming,
        )
        .system_instructions(system_instructions.as_deref(), record_telemetry_content)
        .build();
        let client = self.client.clone();
        let event_source = GenericEventSource::new(client, req);
        let (event_source, request_id_slot) = match Ext::REQUEST_ID_HEADER {
            Some(header) => {
                let (event_source, slot) = event_source.capture_request_id(header);
                (event_source, Some(slot))
            }
            None => (event_source, None),
        };

        let options = if Ext::EMITS_COMPLETE_TOOL_CALLS_IMMEDIATELY {
            ResponsesStreamOptions::strict_with_immediate_tool_calls()
        } else {
            ResponsesStreamOptions::strict()
        };
        let stream = responses_stream_from_event_source_with_options(
            Ext::PROVIDER_NAME,
            event_source,
            span,
            options,
        );
        Ok(streaming::StreamingCompletionResponse::stream(
            Ext::PROVIDER_NAME,
            stamp_terminal_request_id(stream, request_id_slot, Ext::REQUEST_ID_HEADER),
        ))
    }
}

#[cfg(test)]
mod tests;
