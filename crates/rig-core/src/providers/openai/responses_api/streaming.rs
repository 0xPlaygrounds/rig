//! The streaming module for the OpenAI Responses API.
//! Please see the `openai_streaming` or `openai_streaming_with_tools` example for more practical usage.
use crate::completion::{self, CompletionError};
use crate::http_client::HttpClientExt;
use crate::http_client::sse::GenericEventSource;
use crate::providers::internal::adapter::{
    AdapterOutput, WireAdapter, WireFrame, run_wire_buffered,
};
use crate::providers::internal::sse_transport::{
    FrameDisposition, OpenLog, SseTransportOptions, open_wire_stream,
};
use crate::providers::internal::wire::{self, WireEvent};
use crate::providers::openai::responses_api::{
    IncompleteDetailsReason, ReasoningSummary, ResponseStatus, ResponsesUsage,
};
use crate::streaming;
use crate::streaming::RawStreamingChoice;
use crate::telemetry::{CompletionOperation, CompletionSpanBuilder};
use crate::wasm_compat::WasmCompatSend;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use super::{CompletionResponse, GenericResponsesCompletionModel, Output, ResponsesProviderExt};

type StreamingRawChoice = RawStreamingChoice<StreamingCompletionResponse>;

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
/// This is the provider-native terminal record carried by
/// [`GenericResponsesCompletionModel::raw_stream`]. The normalized path maps it
/// once, through [`crate::streaming::normalize_stream`], into a
/// [`streaming::StreamFinal`].
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
    /// response header — not part of any stream frame; stamped by the
    /// transport. `None` when the provider did not report one.
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
/// [`crate::streaming::normalize_stream`] applies the tool-call reconciliation
/// afterwards, using the calls the stream actually emitted.
impl From<(&str, StreamingCompletionResponse)> for streaming::StreamFinal {
    fn from((provider, response): (&str, StreamingCompletionResponse)) -> Self {
        let finish_reason = response.status.as_ref().and_then(|status| {
            super::map_finish_reason(status, response.incomplete_details.as_ref())
        });

        streaming::StreamFinal::new(provider, crate::completion::Usage::from(&response.usage))
            .with_optional_finish_reason(finish_reason)
            .with_optional_message_id(response.message_id)
            .with_optional_response_id(response.response_id)
            .with_optional_provider_request_id(response.provider_request_id)
            .with_optional_model(response.model)
    }
}

/// Normalize a provider-native Responses stream for `provider`.
///
/// Maps only the terminal record; every incremental event passes through
/// untouched.
pub(crate) fn normalize_responses_stream(
    provider: &str,
    raw: streaming::RawStreamingResult<StreamingCompletionResponse>,
) -> streaming::StreamingCompletionResponse {
    let provider = provider.to_owned();
    let mapped_provider = provider.clone();
    let normalized = streaming::normalize_stream(raw, move |response| {
        Ok(streaming::StreamFinal::from((
            mapped_provider.as_str(),
            response,
        )))
    });

    streaming::StreamingCompletionResponse::stream(provider, normalized)
}

/// The done item's blocks as ONE authoritative end-of-part restatement.
///
/// Every block — summaries, content texts, `encrypted_content` — belongs to
/// one `rs_*` reasoning item, so it must land in one part: emitting a
/// whole-block choice per entry made every block after the first a sibling
/// part under the same key, and history then replayed duplicate reasoning
/// input items carrying the identical `rs_*` id. The restatement supersedes
/// the delta-built part in place (wire field order: summary, content,
/// encrypted). `None` when the item carries no blocks — an empty done item
/// says nothing at the boundary.
pub(crate) fn reasoning_end_from_done_item(
    id: &crate::streaming::StreamPartId,
    provider_id: Option<&crate::streaming::WireId>,
    summary: Vec<ReasoningSummary>,
    content: Vec<String>,
    encrypted_content: Option<String>,
) -> Option<RawStreamingChoice<StreamingCompletionResponse>> {
    // Same builder as the unary decode, so the restatement and the
    // non-streaming conversion of one item cannot drift.
    let blocks = super::reasoning_content_blocks(summary, content, encrypted_content);

    if blocks.is_empty() {
        return None;
    }

    Some(RawStreamingChoice::ReasoningEnd {
        id: id.clone(),
        reasoning: Some(crate::message::Reasoning {
            id: provider_id.map(|provider_id| provider_id.as_str().to_owned()),
            content: blocks,
        }),
        signature: None,
        wire_sent: true,
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
#[derive(Debug, Serialize, Deserialize, Clone)]
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
/// Shared by the live SSE loop, the buffered [`raw_choices_from_sse_body`]
/// path, and the websocket session so all apply the same known/unknown
/// boundary. Provider `error` events (and the websocket-only `response.done`)
/// are checked separately before this, because their `type` is outside the
/// modeled set yet must not be skipped as unknown.
pub(super) fn classify_responses_frame(data: &str) -> WireEvent<StreamingCompletionChunk> {
    wire::classify_tagged_frame(data, "type", is_known_responses_event_type)
}

fn provider_response_from_responses_sse_data(data: &str) -> Option<CompletionError> {
    let value = serde_json::from_str::<serde_json::Value>(data).ok()?;
    (value.get("type").and_then(serde_json::Value::as_str) == Some("error"))
        .then(|| provider_response_from_responses_error_value(&value, data))
}

#[derive(Clone, Copy)]
pub(crate) enum ResponsesStreamOptions {
    Strict,
    StrictWithImmediateToolCalls,
}

impl ResponsesStreamOptions {
    pub(crate) const fn strict() -> Self {
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

        let value = match serde_json::from_str::<serde_json::Value>(data) {
            Ok(value) => value,
            Err(_) => continue,
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

pub(crate) struct RawChoiceAccumulator {
    final_usage: ResponsesUsage,
    reasoning_metadata: Option<serde_json::Map<String, serde_json::Value>>,
    reasoning_context: Option<String>,
    status: Option<ResponseStatus>,
    incomplete_details: Option<IncompleteDetailsReason>,
    message_id: Option<String>,
    response_id: Option<String>,
    model: Option<String>,
    /// Buffered tool-input end events for calls delivered whole by
    /// `output_item.done`, flushed at the terminal (or before a terminal
    /// error). Assembly and internal-id correlation live in the shared
    /// accumulator, keyed by the function-call item id the added/delta/done
    /// events share.
    tool_calls: Vec<StreamingRawChoice>,
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
    reasoning_slots: std::collections::HashMap<u64, crate::streaming::StreamPartId>,
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
    /// (`TextStart` keyed by that item id), so two `message` output items
    /// aggregate as two distinct text parts instead of concatenating.
    /// Deltas without an `item_id` (ChatGPT's envelope-less replays) extend
    /// the open block, or open a boundary-minted one downstream.
    current_text_item: Option<String>,
}

impl RawChoiceAccumulator {
    pub(crate) fn new(initial_usage: ResponsesUsage) -> Self {
        Self {
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
    fn start_text_item(
        &mut self,
        item_id: &Option<String>,
        immediate: &mut Vec<StreamingRawChoice>,
    ) {
        if let Some(item_id) = item_id
            && self.current_text_item.as_deref() != Some(item_id)
        {
            self.current_text_item = Some(item_id.clone());
            immediate.push(streaming::RawStreamingChoice::TextStart {
                id: crate::streaming::StreamPartId::wire(item_id.clone()),
                additional_params: None,
            });
        }
    }

    /// The slot's reasoning assembly key, fixed at its first reasoning
    /// event: the wire's `rs_*` id when that first frame carries one, else
    /// a minted `output-{index}` identity. Every later frame on the slot
    /// reuses the stored key regardless of the id it carries — the same
    /// discipline as `tool_slots` — so mixed id/id-less frames cannot
    /// split one slot's assembly. A late-arriving wire id upgrades the
    /// part's durable `provider_id` (carried as data on each event), never
    /// the accumulation key.
    fn reasoning_slot_key(
        &mut self,
        output_index: u64,
        item_id: Option<&str>,
    ) -> crate::streaming::StreamPartId {
        if let Some(key) = self.reasoning_slots.get(&output_index) {
            return key.clone();
        }
        // Minted from the bridge's ONE counter (tool_call_bridge's own
        // invariant): a second sequence stamping `Minted{Output, index}`
        // could collide with an assembly the bridge minted the same value
        // for. The per-slot map above, not the mint, is what keeps the key
        // stable across the slot's frames.
        let key = item_id
            .map(crate::streaming::StreamPartId::wire)
            .unwrap_or_else(|| self.tool_slots.minted_ids().mint());
        self.reasoning_slots.insert(output_index, key.clone());
        key
    }

    pub(crate) fn decode_item_chunk(
        &mut self,
        chunk: ItemChunk,
        options: ResponsesStreamOptions,
    ) -> Vec<StreamingRawChoice> {
        let mut immediate = Vec::new();

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
                // re-emits `TextStart` and reactivates its block downstream.
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
                immediate.push(streaming::RawStreamingChoice::ToolCallDelta {
                    id: key,
                    content: streaming::ToolCallDeltaContent::Name(func.name),
                });
            }
            ItemChunkKind::OutputItemDone(message) => {
                // Any completed item ends the block it carried; a text delta
                // arriving afterwards belongs to a (re)opened block.
                self.current_text_item = None;
                self.push_output_item_done(
                    message.item,
                    output_index,
                    &mut immediate,
                    options.emits_completed_tool_calls_immediately(),
                );
            }
            // Text and refusal deltas are the same visible-text stream: a
            // refusal is the assistant's message for that turn, and both
            // (re)open the item's text block before their fragment.
            ItemChunkKind::OutputTextDelta(DeltaTextChunk { delta, .. })
            | ItemChunkKind::RefusalDelta(DeltaTextChunk { delta, .. }) => {
                self.start_text_item(&outer_item_id, &mut immediate);
                immediate.push(streaming::RawStreamingChoice::Message(delta));
            }
            // Summary and raw-reasoning deltas differ only in which wire
            // event carries them; both are fragments of the output item's
            // reasoning block and accumulate under its slot identity.
            ItemChunkKind::ReasoningSummaryTextDelta(SummaryTextChunk { delta, .. })
            | ItemChunkKind::ReasoningTextDelta(DeltaTextChunkWithItemId { delta, .. }) => {
                // Reasoning interleaving text closes the open text block
                // downstream (`PartsAccumulator::reasoning_delta`); forget the
                // open message item so a later delta for the *same* item
                // re-emits `TextStart {id}` and reactivates its block instead
                // of silently opening a boundary-minted sibling (#2258 P2).
                self.current_text_item = None;
                let id = self.reasoning_slot_key(output_index, outer_item_id.as_deref());
                immediate.push(streaming::RawStreamingChoice::ReasoningDelta {
                    id,
                    provider_id: outer_item_id
                        .clone()
                        .and_then(crate::streaming::WireId::new),
                    reasoning: delta,
                });
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
                immediate.push(streaming::RawStreamingChoice::ToolCallDelta {
                    id: key,
                    content: streaming::ToolCallDeltaContent::Delta(delta.delta),
                });
            }
            _ => {}
        }

        immediate
    }

    pub(crate) fn record_response_chunk(
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
                    let mut end = slot.end_event(streaming::UnparseableToolInput::Drop);
                    end.call_id = self.pending_call_ids.remove(&index);
                    self.tool_calls
                        .push(streaming::RawStreamingChoice::ToolInputEnd(end));
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
        immediate: &mut Vec<StreamingRawChoice>,
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
                    None => crate::streaming::StreamPartId::wire(func.id.clone()),
                };
                let mut end = streaming::ToolInputEnd::new(
                    item_id.clone(),
                    streaming::UnparseableToolInput::Drop,
                );
                end.name = Some(func.name);
                // The finalized call reports the authoritative wire id even
                // when assembly keyed on a minted slot identity (the
                // accumulator honors the override).
                end.tool_id = crate::streaming::WireId::new(func.id.clone());
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
                            immediate.push(streaming::RawStreamingChoice::ToolCallDelta {
                                id: item_id,
                                content: streaming::ToolCallDeltaContent::Delta(
                                    func.arguments.as_str().to_owned(),
                                ),
                            });
                        }
                    }
                }
                end.call_id = Some(func.call_id);
                let end = streaming::RawStreamingChoice::ToolInputEnd(end);

                if emit_completed_tool_calls_immediately {
                    immediate.push(end);
                } else {
                    self.tool_calls.push(end);
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
                let provider_id = crate::streaming::WireId::new(id.clone());
                let key = self
                    .reasoning_slots
                    .remove(&output_index)
                    .unwrap_or(crate::streaming::StreamPartId::wire(id));
                immediate.extend(reasoning_end_from_done_item(
                    &key,
                    provider_id.as_ref(),
                    summary,
                    content,
                    encrypted_content,
                ));
            }
            Output::Message(message) => {
                immediate.push(streaming::RawStreamingChoice::MessageId(message.id));
            }
            // An unmodeled output item (e.g. a hosted-tool result such as
            // `web_search_call`) arriving on `response.output_item.done`. Surface
            // the raw item to stream consumers, mirroring how the non-streaming
            // decode preserves it on `CompletionResponse.output`.
            Output::Unknown(value) => {
                immediate.push(streaming::RawStreamingChoice::Unknown(value.into()));
            }
        }
    }

    /// Drain the buffered fully-delivered tool calls without finishing the
    /// stream. The errored-terminal path flushes these before the error and
    /// must not produce a terminal record.
    pub(crate) fn take_tool_calls(&mut self) -> Vec<StreamingRawChoice> {
        std::mem::take(&mut self.tool_calls)
    }

    pub(crate) fn finish(mut self) -> Vec<StreamingRawChoice> {
        let mut choices = Vec::new();
        choices.append(&mut self.tool_calls);
        // Only a genuine terminal event (`response.completed` or
        // `response.incomplete`) counts as the provider ending the turn; a
        // stream that ended without one was truncated,
        // and a synthesized terminal record would present the partial turn as
        // a successful, default-usage completion.
        if !self.saw_terminal {
            return choices;
        }
        choices.push(RawStreamingChoice::FinalResponse(
            StreamingCompletionResponse {
                usage: self.final_usage,
                // Stamped by the transport layer.
                provider_request_id: None,
                reasoning_metadata: self.reasoning_metadata,
                reasoning_context: self.reasoning_context,
                status: self.status,
                incomplete_details: self.incomplete_details,
                message_id: self.message_id,
                response_id: self.response_id,
                model: self.model,
            },
        ));
        choices
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

pub(crate) fn raw_choices_from_sse_body(
    body: &str,
    initial_usage: ResponsesUsage,
) -> Result<Vec<StreamingRawChoice>, CompletionError> {
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
    run_wire_buffered(frames, ResponsesAdapter::buffered(initial_usage))
}

pub(crate) async fn completion_response_from_sse_body(
    provider: &str,
    body: &str,
    raw_response: CompletionResponse,
) -> Result<completion::CompletionResponse, CompletionError> {
    let raw_choices = raw_choices_from_sse_body(
        body,
        raw_response
            .usage
            .clone()
            .unwrap_or_else(ResponsesUsage::new),
    )?;
    completion_response_from_raw_choices(provider, raw_choices, &raw_response)
        .await?
        .ok_or_else(|| CompletionError::ResponseError("Response contained no parts".to_owned()))
}

/// Replay accumulated raw choices through [`normalize_responses_stream`] and
/// merge the result with the parsed terminal response body.
///
/// The replayed stream is authoritative where it reported something; the
/// terminal body fills any gap it left (usage, message ID, finish reason,
/// model). Returns `Ok(None)` when the replay produced no content, leaving the
/// caller to decide how to fall back.
pub(crate) async fn completion_response_from_raw_choices(
    provider: &str,
    raw_choices: Vec<StreamingRawChoice>,
    raw_response: &CompletionResponse,
) -> Result<Option<completion::CompletionResponse>, CompletionError> {
    let stream = futures::stream::iter(
        raw_choices
            .into_iter()
            .map(Ok::<_, CompletionError>)
            .collect::<Vec<_>>(),
    );
    let mut stream = normalize_responses_stream(provider, Box::pin(stream));

    while let Some(item) = stream.next().await {
        item?;
    }

    if choice_is_empty(&stream.choice) {
        return Ok(None);
    }

    // Merge per content kind: the replayed choice is authoritative for what it
    // carried (reasoning, tool calls, streamed text), but some backends emit
    // message text only in the terminal body while streaming other kinds as
    // deltas. A replay with no message text takes the body's message content;
    // everything replayed is kept.
    let mut choice = std::mem::take(&mut stream.choice);
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
    let usage = terminal
        .as_ref()
        .map(|terminal| terminal.usage)
        .unwrap_or_else(|| usage_from_raw_response(raw_response));
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

/// Open a Responses SSE stream whose terminal record stays provider-native.
///
/// Pass the result through [`normalize_responses_stream`] to obtain the
/// normalized stream that [`completion::CompletionModel::stream`] returns.
pub(crate) fn raw_stream_from_event_source<HttpClient, RequestBody>(
    event_source: GenericEventSource<HttpClient, RequestBody>,
    span: tracing::Span,
) -> streaming::RawStreamingResult<StreamingCompletionResponse>
where
    HttpClient: HttpClientExt + Clone + 'static,
    RequestBody: Into<bytes::Bytes> + Clone + WasmCompatSend + 'static,
{
    raw_stream_from_event_source_with_options(event_source, span, ResponsesStreamOptions::strict())
}

pub(crate) fn raw_stream_from_event_source_with_options<HttpClient, RequestBody>(
    event_source: GenericEventSource<HttpClient, RequestBody>,
    span: tracing::Span,
    options: ResponsesStreamOptions,
) -> streaming::RawStreamingResult<StreamingCompletionResponse>
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
        ResponsesAdapter::live(options),
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
    fn live(options: ResponsesStreamOptions) -> Self {
        Self {
            accumulator: RawChoiceAccumulator::new(ResponsesUsage::new()),
            options,
            repair_envelopes: false,
            finished: false,
        }
    }

    fn buffered(initial_usage: ResponsesUsage) -> Self {
        Self {
            accumulator: RawChoiceAccumulator::new(initial_usage),
            options: ResponsesStreamOptions::strict(),
            repair_envelopes: true,
            finished: false,
        }
    }
}

impl WireAdapter for ResponsesAdapter {
    type Frame = WireFrame;
    type Event = ResponsesFrameEvent;
    type Response = StreamingCompletionResponse;

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

    fn interpret(&mut self, event: ResponsesFrameEvent, out: &mut AdapterOutput<Self::Response>) {
        if self.finished {
            return;
        }

        match event.chunk {
            StreamingCompletionChunk::Delta(chunk) => {
                out.extend(
                    self.accumulator
                        .decode_item_chunk(chunk, self.options)
                        .into_iter()
                        .map(Ok),
                );
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
                    out.extend(self.accumulator.take_tool_calls().into_iter().map(Ok));
                    out.push(Err(error));
                    self.finished = true;
                }
            }
        }
    }

    fn finish(&mut self, out: &mut AdapterOutput<Self::Response>) {
        let accumulator = std::mem::replace(
            &mut self.accumulator,
            RawChoiceAccumulator::new(ResponsesUsage::new()),
        );
        let final_usage = accumulator.final_usage.clone();

        // Flush buffered tool calls, then the terminal record when a genuine
        // terminal event arrived; EOF without one is truncation and the
        // accumulator withholds the record (deferral, never synthesis).
        out.extend(accumulator.finish().into_iter().map(Ok));

        let span = tracing::Span::current();
        span.record("gen_ai.usage.input_tokens", final_usage.input_tokens);
        span.record("gen_ai.usage.output_tokens", final_usage.output_tokens);
        let cached_tokens = final_usage
            .input_tokens_details
            .as_ref()
            .map(|d| d.cached_tokens)
            .unwrap_or(0);
        span.record("gen_ai.usage.cache_read.input_tokens", cached_tokens);
    }

    fn flush_before_terminal_error(&mut self, out: &mut AdapterOutput<Self::Response>) {
        // Tool calls the provider fully delivered are content: they flush
        // before the terminal error reaches the consumer.
        out.extend(self.accumulator.take_tool_calls().into_iter().map(Ok));
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
    /// Open a stream whose terminal record stays provider-native.
    ///
    /// This is the escape hatch for Responses-API terminal fields rig does not
    /// normalize. It shares the request builder, transport, telemetry, and
    /// error handling with
    /// [`CompletionModel::stream`](completion::CompletionModel::stream), which
    /// calls it and normalizes the terminal record — one network request either
    /// way.
    pub async fn raw_stream(
        &self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<streaming::RawStreamingResult<StreamingCompletionResponse>, CompletionError> {
        let system_instructions = completion_request.preamble.clone();
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
        let stream = raw_stream_from_event_source_with_options(event_source, span, options);
        Ok(
            crate::providers::internal::sse_transport::stamp_terminal_request_id(
                stream,
                request_id_slot,
                Ext::REQUEST_ID_HEADER,
                |response, id| response.provider_request_id = Some(id),
            ),
        )
    }

    pub(crate) async fn stream(
        &self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse, CompletionError> {
        let raw = self.raw_stream(completion_request).await?;

        Ok(normalize_responses_stream(Ext::PROVIDER_NAME, raw))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ContentPartChunkPart, ItemChunk, ItemChunkKind, RawChoiceAccumulator,
        ResponsesStreamOptions, StreamingCompletionChunk, classify_responses_frame,
        raw_choices_from_sse_body, reasoning_end_from_done_item,
    };
    use crate::completion::CompletionModel;
    use crate::message::ReasoningContent;
    use crate::providers::internal::openai_chat_completions_compatible::test_support::{
        sse_bytes_from_data_lines, sse_bytes_from_json_events,
    };
    use crate::providers::internal::wire::WireEvent;
    use crate::providers::openai::responses_api::{
        AdditionalParameters, CompletionResponse, IncompleteDetailsReason, OutputTokensDetails,
        ReasoningSummary, ResponseError, ResponseObject, ResponseStatus, ResponsesUsage,
    };
    use crate::streaming::{RawStreamingChoice, StreamedAssistantContent};
    use crate::test_utils::MockStreamingClient;
    use crate::{client::CompletionClient, providers::openai};
    use futures::StreamExt;
    use serde_json::{self, json};

    #[test]
    fn classify_known_event_decodes() {
        let frame = json!({
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "delta": "hi",
        })
        .to_string();
        assert!(matches!(
            classify_responses_frame(&frame),
            WireEvent::Known(StreamingCompletionChunk::Delta(_))
        ));
    }

    #[test]
    fn classify_unknown_event_type_is_unknown() {
        let frame = json!({
            "type": "response.web_search_call.searching",
            "output_index": 0,
            "sequence_number": 1,
        })
        .to_string();
        assert!(matches!(
            classify_responses_frame(&frame),
            WireEvent::Unknown { event_type, .. } if event_type == "response.web_search_call.searching"
        ));
    }

    /// #2258 G4: `response.reasoning_text.done` terminates every raw-reasoning
    /// block on all three Responses surfaces. It used to be absent from the
    /// known-event set, so each block logged a spurious "unknown event" warn
    /// and passed through as `Unknown`.
    ///
    /// Both halves of the fix are asserted here, because either alone is a
    /// regression: the tag must be KNOWN (no `Unknown`), and `ItemChunkKind`
    /// must carry a variant for it (no `Corrupt`, which is what naming the tag
    /// without the variant would have produced — strictly worse than the warn).
    ///
    /// No recorded cassette contains this event; the wire shape is the
    /// Responses spec's, so this unit test is the pin.
    #[test]
    fn classify_reasoning_text_done_is_known_and_decodes() {
        let frame = json!({
            "type": "response.reasoning_text.done",
            "item_id": "rs_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 7,
            "text": "the model's raw chain of thought",
        })
        .to_string();

        let event = classify_responses_frame(&frame);
        assert!(
            !matches!(event, WireEvent::Unknown { .. }),
            "the tag must be in the known-event set: {event:?}"
        );
        assert!(
            !matches!(event, WireEvent::Corrupt(_)),
            "a known tag with no matching ItemChunkKind variant decodes to Corrupt, which the \
             driver surfaces as an in-band Err — worse than the warn it replaced: {event:?}"
        );
        assert!(matches!(
            event,
            WireEvent::Known(StreamingCompletionChunk::Delta(chunk))
                if matches!(chunk.data, ItemChunkKind::ReasoningTextDone(_))
        ));
    }

    /// The done event restates text the deltas already streamed, so it must be
    /// a no-op: replaying it would double every raw-reasoning block.
    #[test]
    fn reasoning_text_done_emits_nothing() {
        let mut accumulator = RawChoiceAccumulator::new(ResponsesUsage::new());
        let chunk: ItemChunk = serde_json::from_value(json!({
            "type": "response.reasoning_text.done",
            "item_id": "rs_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 7,
            "text": "the model's raw chain of thought",
        }))
        .expect("reasoning text done event should deserialize");

        let emitted = accumulator.decode_item_chunk(chunk, ResponsesStreamOptions::strict());
        assert!(
            emitted.is_empty(),
            "the done restatement must not re-emit the reasoning text: {emitted:?}"
        );
    }

    #[test]
    fn classify_invalid_json_is_corrupt() {
        assert!(matches!(
            classify_responses_frame("{not json"),
            WireEvent::Corrupt(_)
        ));
    }

    #[test]
    fn classify_known_event_with_defective_payload_is_corrupt() {
        let frame = json!({
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "delta": 42,
        })
        .to_string();
        assert!(matches!(
            classify_responses_frame(&frame),
            WireEvent::Corrupt(_)
        ));
    }

    // The P2 probe shape from `rig-2257-code-review-findings-34ee8ba5.md`: a
    // known part tag whose payload is schema-defective must classify as
    // `Corrupt`, not slide into the unknown-part catch-all.
    #[test]
    fn classify_defective_known_content_part_is_corrupt() {
        let frame = json!({
            "type": "response.content_part.added",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "part": {"type": "output_text", "text": 42},
        })
        .to_string();
        assert!(matches!(
            classify_responses_frame(&frame),
            WireEvent::Corrupt(_)
        ));
    }

    #[test]
    fn content_part_known_tag_decodes() {
        let part: ContentPartChunkPart =
            serde_json::from_value(json!({"type": "output_text", "text": "hi"})).unwrap();
        assert!(matches!(part, ContentPartChunkPart::OutputText { text } if text == "hi"));
    }

    #[test]
    fn content_part_known_tag_with_defective_payload_errors() {
        let result = serde_json::from_value::<ContentPartChunkPart>(
            json!({"type": "output_text", "text": 42}),
        );
        assert!(result.is_err());
        let result = serde_json::from_value::<ContentPartChunkPart>(
            json!({"type": "summary_text", "text": 42}),
        );
        assert!(result.is_err());
    }

    // A non-string `type` is a data-level defect of the tagged shape, never a
    // skippable unknown part (#2258 F8).
    #[test]
    fn content_part_non_string_type_errors() {
        let result =
            serde_json::from_value::<ContentPartChunkPart>(json!({"type": 42, "text": "hi"}));
        assert!(result.is_err());
        let result =
            serde_json::from_value::<ContentPartChunkPart>(json!({"type": null, "text": "hi"}));
        assert!(result.is_err());
    }

    // Pins the documented duplicate-key edge (#2258 F8): `serde_json::Value`
    // keeps the last duplicate key, so the hand dispatch resolves on the LAST
    // `type` — unlike a derived internally-tagged enum, which takes the first.
    #[test]
    fn content_part_duplicate_type_key_dispatches_on_the_last_occurrence() {
        let part: ContentPartChunkPart =
            serde_json::from_str(r#"{"type":"bogus","type":"output_text","text":"hi"}"#).unwrap();
        assert!(matches!(part, ContentPartChunkPart::OutputText { text } if text == "hi"));
    }

    // `refusal` and `reasoning_text` part tags are not in the modeled set:
    // they must stay skippable no-ops (the content arrives via the
    // corresponding delta events), round-tripping the value verbatim.
    #[test]
    fn content_part_unknown_tag_is_preserved_verbatim() {
        let wire = json!({"type": "refusal", "refusal": "no"});
        let part: ContentPartChunkPart = serde_json::from_value(wire.clone()).unwrap();
        let ContentPartChunkPart::Unknown(value) = &part else {
            panic!("unmodeled part tag must fall back to Unknown");
        };
        assert_eq!(value, &wire);
        assert_eq!(serde_json::to_value(&part).unwrap(), wire);
    }

    fn sample_response(status: ResponseStatus) -> CompletionResponse {
        CompletionResponse {
            id: "resp_123".to_string(),
            object: ResponseObject::Response,
            provider_request_id: None,
            created_at: 0,
            status,
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            model: "gpt-5.4".to_string(),
            provider_reasoning: None,
            reasoning_metadata: None,
            reasoning_context: None,
            usage: None,
            output: Vec::new(),
            tools: Vec::new(),
            additional_parameters: AdditionalParameters::default(),
        }
    }

    async fn first_error_from_event(
        event: serde_json::Value,
    ) -> crate::completion::CompletionError {
        let client = openai::Client::builder()
            .http_client(MockStreamingClient {
                sse_bytes: sse_bytes_from_json_events(&[event]),
            })
            .api_key("test-key")
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-5.4");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");

        stream
            .next()
            .await
            .expect("stream should yield an item")
            .expect_err("stream should surface a provider error")
    }

    /// The provider-native terminal record, as `raw_stream` exposes it.
    async fn final_response_from_event(
        event: serde_json::Value,
    ) -> super::StreamingCompletionResponse {
        let client = openai::Client::builder()
            .http_client(MockStreamingClient {
                sse_bytes: sse_bytes_from_json_events(&[event]),
            })
            .api_key("test-key")
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-5.4");
        let request = model.completion_request("hello").build();
        let mut stream = model
            .raw_stream(request)
            .await
            .expect("stream should start");

        while let Some(item) = stream.next().await {
            if let RawStreamingChoice::FinalResponse(response) =
                item.expect("completed stream should not error")
            {
                return response;
            }
        }

        panic!("stream should yield a final response");
    }

    /// The normalized terminal record, as `stream` exposes it.
    async fn stream_final_from_event(event: serde_json::Value) -> crate::streaming::StreamFinal {
        let client = openai::Client::builder()
            .http_client(MockStreamingClient {
                sse_bytes: sse_bytes_from_json_events(&[event]),
            })
            .api_key("test-key")
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-5.4");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");

        while let Some(item) = stream.next().await {
            if let StreamedAssistantContent::Final(response) =
                item.expect("completed stream should not error")
            {
                return response;
            }
        }

        panic!("stream should yield a final response");
    }

    #[test]
    fn parse_sse_completion_body_preserves_error_payloads() {
        let mut response = sample_response(ResponseStatus::Failed);
        response.error = Some(ResponseError {
            code: "server_error".to_string(),
            message: "response failed".to_string(),
        });
        let events = [
            json!({
                "type": "response.failed",
                "sequence_number": 1,
                "response": response,
            }),
            json!({
                "type": "error",
                "error": {
                    "message": "boom",
                    "code": "server_error",
                    "type": "server_error"
                }
            }),
        ];

        for event in events {
            let payload = serde_json::to_string(&event).expect("event should serialize");
            let body = format!("data: {payload}\n");
            let err = super::parse_sse_completion_body(&body, "ChatGPT")
                .expect_err("error payload should surface as provider response");

            assert!(matches!(
                err,
                crate::completion::CompletionError::ProviderResponse(_)
            ));
            assert_eq!(err.provider_response_status(), None);
            assert_eq!(err.provider_response_body(), Some(payload.as_str()));
        }
    }

    #[test]
    fn reasoning_done_item_fuses_summary_content_and_encrypted_into_one_end() {
        let summary = vec![
            ReasoningSummary::SummaryText {
                text: "step 1".to_string(),
            },
            ReasoningSummary::SummaryText {
                text: "step 2".to_string(),
            },
        ];
        let content = vec!["private reasoning".to_string()];
        let end = reasoning_end_from_done_item(
            &crate::streaming::StreamPartId::wire("rs_1"),
            crate::streaming::WireId::new("rs_1").as_ref(),
            summary,
            content,
            Some("enc_blob".to_string()),
        );

        // ONE end event carrying every block in wire field order — never a
        // choice per block, which made siblings under one `rs_*` id.
        let Some(RawStreamingChoice::ReasoningEnd {
            id,
            reasoning: Some(reasoning),
            signature: None,
            wire_sent: true,
        }) = end
        else {
            panic!("expected one wire-sent ReasoningEnd restatement");
        };
        assert_eq!(id, crate::streaming::StreamPartId::wire("rs_1"));
        assert_eq!(reasoning.id.as_deref(), Some("rs_1"));
        assert_eq!(
            reasoning.content,
            vec![
                ReasoningContent::Summary("step 1".to_string()),
                ReasoningContent::Summary("step 2".to_string()),
                ReasoningContent::Text {
                    text: "private reasoning".to_string(),
                    signature: None,
                },
                ReasoningContent::Encrypted("enc_blob".to_string()),
            ]
        );
    }

    #[test]
    fn reasoning_output_item_done_emits_reasoning_text_content() {
        let body = format!(
            "data: {}\n",
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "sequence_number": 1,
                "item": {
                    "type": "reasoning",
                    "id": "rs_text_1",
                    "summary": [],
                    "content": [{ "type": "reasoning_text", "text": "visible reasoning" }],
                    "status": "completed"
                },
            })
        );

        let choices = raw_choices_from_sse_body(&body, ResponsesUsage::new())
            .expect("sse body should decode");

        // The done item arrives as one wire-sent end restatement whose
        // single block is the reasoning text.
        assert!(matches!(
            choices.first(),
            Some(RawStreamingChoice::ReasoningEnd {
                id,
                reasoning: Some(reasoning),
                wire_sent: true,
                ..
            }) if id == &crate::streaming::StreamPartId::wire("rs_text_1")
                && reasoning.content
                    == vec![ReasoningContent::Text {
                        text: "visible reasoning".to_string(),
                        signature: None,
                    }]
        ));
    }

    /// Envelope-less replay shape (ChatGPT bodies): an id-less summary
    /// delta mints an Output-kind key, the done item restates the whole
    /// block under the SAME adopted minted key, and visible text follows.
    /// The driver's boundary law must treat the same-key whole block as a
    /// close — this exact body used to abort every debug build
    /// (sequence-law O1, Responses variant).
    #[test]
    fn envelope_less_reasoning_then_text_decodes_without_violation() {
        let body = format!(
            "data: {}\ndata: {}\ndata: {}\n",
            json!({
                "type": "response.reasoning_summary_text.delta",
                "output_index": 0,
                "summary_index": 0,
                "sequence_number": 1,
                "delta": "thinking",
            }),
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "sequence_number": 2,
                "item": {
                    "type": "reasoning",
                    "id": "",
                    "summary": [{ "type": "summary_text", "text": "thinking, complete" }],
                    "status": "completed"
                },
            }),
            json!({
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 1,
                "content_index": 0,
                "sequence_number": 3,
                "delta": "the answer",
            }),
        );

        let choices = raw_choices_from_sse_body(&body, ResponsesUsage::new())
            .expect("sse body should decode without a sequence-law violation");
        assert!(choices.iter().any(
            |choice| matches!(choice, RawStreamingChoice::Message(text) if text == "the answer")
        ));
    }

    #[test]
    fn reasoning_text_delta_emits_reasoning_delta() {
        let body = format!(
            "data: {}\n",
            json!({
                "type": "response.reasoning_text.delta",
                "item_id": "rs_delta_1",
                "output_index": 0,
                "content_index": 0,
                "sequence_number": 1,
                "delta": "thinking",
            })
        );

        let choices = raw_choices_from_sse_body(&body, ResponsesUsage::new())
            .expect("sse body should decode");

        assert!(matches!(
            choices.first(),
            Some(RawStreamingChoice::ReasoningDelta { id, provider_id: _, reasoning })
                if id == &crate::streaming::StreamPartId::wire("rs_delta_1") && reasoning == "thinking"
        ));
    }

    #[test]
    fn unknown_output_item_surfaces_as_raw_unknown_choice() {
        // A hosted-tool item (web_search_call) arriving on
        // `response.output_item.done` must surface to stream consumers as
        // `RawStreamingChoice::Unknown` carrying the verbatim item, mirroring how
        // the non-streaming decode preserves it on `CompletionResponse.output`.
        let item = json!({
            "type": "web_search_call",
            "id": "ws_001",
            "status": "completed",
            "action": { "type": "search", "queries": ["rig framework"] },
        });
        let body = format!(
            "data: {}\n",
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "sequence_number": 1,
                "item": item,
            })
        );

        let choices = raw_choices_from_sse_body(&body, ResponsesUsage::new())
            .expect("sse body should decode");

        let unknown = choices.iter().find_map(|choice| match choice {
            RawStreamingChoice::Unknown(value) => Some(value),
            _ => None,
        });
        assert_eq!(
            unknown,
            Some(&item.clone().into()),
            "the raw web_search_call item should reach the consumer verbatim",
        );
    }

    #[test]
    fn reasoning_done_item_without_encrypted_emits_summary_only() {
        let summary = vec![ReasoningSummary::SummaryText {
            text: "only summary".to_string(),
        }];
        let end = reasoning_end_from_done_item(
            &crate::streaming::StreamPartId::wire("rs_2"),
            crate::streaming::WireId::new("rs_2").as_ref(),
            summary,
            Vec::new(),
            None,
        );

        let Some(RawStreamingChoice::ReasoningEnd {
            id,
            reasoning: Some(reasoning),
            ..
        }) = end
        else {
            panic!("expected one ReasoningEnd restatement");
        };
        assert_eq!(id, crate::streaming::StreamPartId::wire("rs_2"));
        assert_eq!(
            reasoning.content,
            vec![ReasoningContent::Summary("only summary".to_string())]
        );
    }

    #[test]
    fn empty_encrypted_reasoning_is_not_emitted() {
        let content = vec!["visible reasoning".to_string()];

        let end = reasoning_end_from_done_item(
            &crate::streaming::StreamPartId::wire("rs_1"),
            crate::streaming::WireId::new("rs_1").as_ref(),
            Vec::new(),
            content,
            Some(String::new()),
        );

        let Some(RawStreamingChoice::ReasoningEnd {
            reasoning: Some(reasoning),
            ..
        }) = end
        else {
            panic!("expected one ReasoningEnd restatement");
        };
        assert_eq!(
            reasoning.content,
            vec![ReasoningContent::Text {
                text: "visible reasoning".to_string(),
                signature: None,
            }],
            "an empty encrypted payload contributes no block"
        );

        // An entirely empty done item says nothing at the boundary.
        assert!(
            reasoning_end_from_done_item(
                &crate::streaming::StreamPartId::wire("rs_1"),
                crate::streaming::WireId::new("rs_1").as_ref(),
                Vec::new(),
                Vec::new(),
                Some(String::new()),
            )
            .is_none()
        );
    }

    #[test]
    fn content_part_added_deserializes_snake_case_part_type() {
        let chunk: StreamingCompletionChunk = serde_json::from_value(json!({
            "type": "response.content_part.added",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 3,
            "part": {
                "type": "output_text",
                "text": "hello"
            }
        }))
        .expect("content part event should deserialize");

        assert!(matches!(
            chunk,
            StreamingCompletionChunk::Delta(chunk)
                if matches!(
                    chunk.data,
                    ItemChunkKind::ContentPartAdded(_)
                )
        ));
    }

    #[test]
    fn content_part_done_deserializes_snake_case_part_type() {
        let chunk: StreamingCompletionChunk = serde_json::from_value(json!({
            "type": "response.content_part.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 4,
            "part": {
                "type": "summary_text",
                "text": "done"
            }
        }))
        .expect("content part done event should deserialize");

        assert!(matches!(
            chunk,
            StreamingCompletionChunk::Delta(chunk)
                if matches!(
                    chunk.data,
                    ItemChunkKind::ContentPartDone(_)
                )
        ));
    }

    #[test]
    fn reasoning_summary_part_added_deserializes_snake_case_part_type() {
        let chunk: StreamingCompletionChunk = serde_json::from_value(json!({
            "type": "response.reasoning_summary_part.added",
            "item_id": "rs_1",
            "output_index": 0,
            "summary_index": 0,
            "sequence_number": 5,
            "part": {
                "type": "summary_text",
                "text": "step 1"
            }
        }))
        .expect("reasoning summary part event should deserialize");

        assert!(matches!(
            chunk,
            StreamingCompletionChunk::Delta(chunk)
                if matches!(
                    chunk.data,
                    ItemChunkKind::ReasoningSummaryPartAdded(_)
                )
        ));
    }

    #[test]
    fn reasoning_summary_part_done_deserializes_snake_case_part_type() {
        let chunk: StreamingCompletionChunk = serde_json::from_value(json!({
            "type": "response.reasoning_summary_part.done",
            "item_id": "rs_1",
            "output_index": 0,
            "summary_index": 0,
            "sequence_number": 6,
            "part": {
                "type": "summary_text",
                "text": "step 2"
            }
        }))
        .expect("reasoning summary part done event should deserialize");

        assert!(matches!(
            chunk,
            StreamingCompletionChunk::Delta(chunk)
                if matches!(
                    chunk.data,
                    ItemChunkKind::ReasoningSummaryPartDone(_)
                )
        ));
    }

    #[tokio::test]
    async fn response_failed_chunk_surfaces_provider_error_without_empty_code_prefix() {
        let mut response = sample_response(ResponseStatus::Failed);
        response.error = Some(ResponseError {
            code: String::new(),
            message: "maximum context length exceeded".to_string(),
        });

        let event = json!({
            "type": "response.failed",
            "sequence_number": 1,
            "response": response,
        });

        let err = first_error_from_event(event).await;

        assert!(matches!(
            err,
            crate::completion::CompletionError::ProviderResponse(_)
        ));
        assert_eq!(err.provider_response_status(), None);
        assert!(err.provider_response_body().is_some_and(|body| {
            body.contains("response.failed") && body.contains("maximum context length exceeded")
        }));
    }

    #[tokio::test]
    async fn response_failed_chunk_surfaces_provider_error_with_code_prefix() {
        let mut response = sample_response(ResponseStatus::Failed);
        response.error = Some(ResponseError {
            code: "context_length_exceeded".to_string(),
            message: "maximum context length exceeded".to_string(),
        });

        let event = json!({
            "type": "response.failed",
            "sequence_number": 1,
            "response": response,
        });

        let err = first_error_from_event(event).await;

        assert!(matches!(
            err,
            crate::completion::CompletionError::ProviderResponse(_)
        ));
        assert_eq!(err.provider_response_status(), None);
        assert!(err.provider_response_body().is_some_and(|body| {
            body.contains("response.failed")
                && body.contains("context_length_exceeded")
                && body.contains("maximum context length exceeded")
        }));
    }

    #[tokio::test]
    async fn response_incomplete_chunk_is_a_successful_terminal_with_mapped_finish_reason() {
        let text_delta = json!({
            "type": "response.output_text.delta",
            "content_index": 0,
            "delta": "partial",
            "item_id": "msg_incomplete_1",
            "output_index": 0,
            "sequence_number": 1,
        });

        let mut response = sample_response(ResponseStatus::Incomplete);
        response.incomplete_details = Some(IncompleteDetailsReason {
            reason: "max_output_tokens".to_string(),
        });
        response.usage = Some(ResponsesUsage {
            input_tokens: 10,
            input_tokens_details: None,
            output_tokens: 5,
            output_tokens_details: Some(OutputTokensDetails {
                reasoning_tokens: 0,
            }),
            total_tokens: 15,
        });

        let incomplete = json!({
            "type": "response.incomplete",
            "sequence_number": 2,
            "response": response,
        });

        let client = openai::Client::builder()
            .http_client(MockStreamingClient {
                sse_bytes: sse_bytes_from_json_events(&[text_delta, incomplete]),
            })
            .api_key("test-key")
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-5.4");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");

        let mut text = String::new();
        let mut final_response = None;
        while let Some(item) = stream.next().await {
            match item.expect("incomplete stream should not error") {
                StreamedAssistantContent::Text(delta) => text.push_str(&delta.text),
                StreamedAssistantContent::Final(response) => final_response = Some(response),
                _ => {}
            }
        }

        // The partial output survives, and the terminal record maps the
        // incomplete status to the same finish reason as the unary path.
        assert_eq!(text, "partial");
        let final_response = final_response.expect("stream should yield a final response");
        assert_eq!(
            final_response.finish_reason,
            Some(crate::completion::FinishReason::Length)
        );
        assert_eq!(final_response.usage.input_tokens, 10);
        assert_eq!(final_response.usage.output_tokens, 5);
        assert_eq!(final_response.usage.total_tokens, 15);
    }

    /// A multi-block reasoning done item (summaries + `encrypted_content`)
    /// aggregates as exactly ONE reasoning part carrying every block in wire
    /// order — never sibling parts sharing one `rs_*` id, which would replay
    /// as duplicate reasoning input items carrying the identical id on the
    /// next request.
    #[tokio::test]
    async fn multi_block_reasoning_done_item_yields_one_part() {
        let reasoning_done = json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 1,
            "item": {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [
                    {"type": "summary_text", "text": "step 1"},
                    {"type": "summary_text", "text": "step 2"}
                ],
                "content": [],
                "encrypted_content": "enc_blob"
            }
        });
        let completed = json!({
            "type": "response.completed",
            "sequence_number": 2,
            "response": sample_response(ResponseStatus::Completed),
        });

        let client = openai::Client::builder()
            .http_client(MockStreamingClient {
                sse_bytes: sse_bytes_from_json_events(&[reasoning_done, completed]),
            })
            .api_key("test-key")
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-5.4");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");

        let mut completed_reasoning = Vec::new();
        while let Some(item) = stream.next().await {
            if let StreamedAssistantContent::Reasoning { reasoning, .. } =
                item.expect("stream items should be ok")
            {
                completed_reasoning.push(reasoning);
            }
        }

        assert_eq!(
            completed_reasoning.len(),
            1,
            "one done item must complete exactly one reasoning part, got {completed_reasoning:?}"
        );
        let reasoning = completed_reasoning.first().expect("one part");
        assert_eq!(reasoning.id.as_deref(), Some("rs_1"));
        assert_eq!(
            reasoning.content,
            vec![
                ReasoningContent::Summary("step 1".to_string()),
                ReasoningContent::Summary("step 2".to_string()),
                ReasoningContent::Encrypted("enc_blob".to_string()),
            ],
            "every block survives, in wire order, inside the one part"
        );

        // The aggregated choice replays as exactly one reasoning input item.
        let choice = stream.choice;
        let reasoning_parts = choice
            .iter()
            .filter(|content| matches!(content, crate::message::AssistantContent::Reasoning(_)))
            .count();
        assert_eq!(
            reasoning_parts, 1,
            "history must carry one reasoning part per rs_* id, got {choice:?}"
        );
    }

    /// A `response.failed` after a fully-delivered tool call: the tool call is
    /// content and flushes first, the terminal error follows, and nothing
    /// (least of all a terminal record) comes after it.
    #[tokio::test]
    async fn response_failed_flushes_delivered_tool_calls_before_the_error() {
        let tool_call_done = json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 1,
            "item": {
                "type": "function_call",
                "id": "fc_123",
                "arguments": "{}",
                "call_id": "call_123",
                "name": "example_tool",
                "status": "completed"
            }
        });

        let mut response = sample_response(ResponseStatus::Failed);
        response.error = Some(ResponseError {
            code: "server_error".to_string(),
            message: "response stream failed".to_string(),
        });

        let failed = json!({
            "type": "response.failed",
            "sequence_number": 2,
            "response": response,
        });

        let client = openai::Client::builder()
            .http_client(MockStreamingClient {
                sse_bytes: sse_bytes_from_json_events(&[tool_call_done, failed]),
            })
            .api_key("test-key")
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-5.4");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");

        let tool_call = match stream
            .next()
            .await
            .expect("stream should yield the flushed tool call")
            .expect("the flushed tool call must precede the terminal error")
        {
            StreamedAssistantContent::ToolCall { tool_call, .. } => tool_call,
            other => panic!("expected the flushed tool call first, got {other:?}"),
        };
        // The correlator drives rig's id; the item id rides on `provider`.
        assert_eq!(tool_call.id, "call_123");
        let provider = tool_call.provider.as_ref().expect("provider ids are kept");
        assert_eq!(provider.call_id, "call_123");
        assert_eq!(provider.item_id.as_deref(), Some("fc_123"));
        assert_eq!(tool_call.function.name, "example_tool");

        let err = stream
            .next()
            .await
            .expect("stream should yield an item")
            .expect_err("stream should surface a provider error");
        assert!(matches!(
            err,
            crate::completion::CompletionError::ProviderResponse(_)
        ));
        assert_eq!(err.provider_response_status(), None);
        assert!(err.provider_response_body().is_some_and(|body| {
            body.contains("response.failed") && body.contains("response stream failed")
        }));
        assert!(
            stream.next().await.is_none(),
            "stream should terminate immediately after the terminal error"
        );
        assert!(stream.response.is_none());
    }

    /// Same ordering for a transport failure: fully-delivered tool call, then
    /// the error, then the end — with no terminal record.
    #[tokio::test]
    async fn transport_error_flushes_delivered_tool_calls_before_the_error() {
        use crate::http_client::sse::GenericEventSource;
        use crate::test_utils::SequencedStreamingHttpClient;

        let tool_call_done = json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 1,
            "item": {
                "type": "function_call",
                "id": "fc_123",
                "arguments": "{}",
                "call_id": "call_123",
                "name": "example_tool",
                "status": "completed"
            }
        });
        let chunks = vec![
            Ok(sse_bytes_from_data_lines([tool_call_done.to_string()])),
            Err(crate::http_client::Error::InvalidStatusCodeWithMessage(
                http::StatusCode::BAD_GATEWAY,
                r#"{"error":{"message":"upstream unavailable"}}"#.to_string(),
            )),
        ];
        let client = SequencedStreamingHttpClient::new(chunks);
        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/responses")
            .body(Vec::new())
            .expect("request should build");
        let event_source = GenericEventSource::new(client, req);
        let mut stream = super::normalize_responses_stream(
            "openai",
            super::raw_stream_from_event_source(event_source, tracing::Span::none()),
        );

        match stream
            .next()
            .await
            .expect("stream should yield the flushed tool call")
            .expect("the flushed tool call must precede the transport error")
        {
            StreamedAssistantContent::ToolCall { tool_call, .. } => {
                assert_eq!(tool_call.id, "call_123");
                let provider = tool_call.provider.as_ref().expect("provider ids are kept");
                assert_eq!(provider.item_id.as_deref(), Some("fc_123"));
            }
            other => panic!("expected the flushed tool call first, got {other:?}"),
        }

        let err = stream
            .next()
            .await
            .expect("stream should yield the transport error")
            .expect_err("the transport failure must reach the consumer");
        assert_eq!(
            err.provider_response_status(),
            Some(http::StatusCode::BAD_GATEWAY)
        );

        assert!(
            stream.next().await.is_none(),
            "nothing may follow the terminal error"
        );
        assert!(stream.response.is_none());
    }

    /// A known terminal event with a data-level defect (malformed `usage`) is
    /// a corrupt frame, not silent truncation: the error surfaces and, since
    /// the terminal itself failed to parse, no terminal record is emitted.
    #[tokio::test]
    async fn known_terminal_with_malformed_usage_surfaces_error_without_terminal() {
        let mut event = json!({
            "type": "response.completed",
            "sequence_number": 1,
            "response": sample_response(ResponseStatus::Completed),
        });
        event["response"]["usage"] = json!("banana");

        let client = openai::Client::builder()
            .http_client(MockStreamingClient {
                sse_bytes: sse_bytes_from_json_events(&[event]),
            })
            .api_key("test-key")
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-5.4");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");

        let mut saw_error = false;
        let mut saw_final = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(StreamedAssistantContent::Final(_)) => saw_final = true,
                Ok(other) => panic!("unexpected stream item: {other:?}"),
                Err(err) => {
                    assert!(
                        matches!(err, crate::completion::CompletionError::JsonError(_)),
                        "expected a parse error item, got {err:?}"
                    );
                    saw_error = true;
                }
            }
        }

        assert!(saw_error, "the corrupt terminal must surface as an error");
        assert!(
            !saw_final,
            "a terminal that failed to parse must not produce a terminal record"
        );
        assert!(stream.response.is_none());
    }

    /// An invented event type stays skippable for forward compatibility; a
    /// later genuine terminal still completes the stream.
    #[tokio::test]
    async fn unknown_event_type_is_skipped_and_stream_completes() {
        let unknown = json!({
            "type": "response.rocket_launch",
            "payload": { "count": 3 }
        });
        let completed = json!({
            "type": "response.completed",
            "sequence_number": 2,
            "response": sample_response(ResponseStatus::Completed),
        });

        let client = openai::Client::builder()
            .http_client(MockStreamingClient {
                sse_bytes: sse_bytes_from_json_events(&[unknown, completed]),
            })
            .api_key("test-key")
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-5.4");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");

        let mut saw_final = false;
        while let Some(item) = stream.next().await {
            if let StreamedAssistantContent::Final(_) =
                item.expect("unknown event types must not surface as errors")
            {
                saw_final = true;
            }
        }
        assert!(
            saw_final,
            "the genuine terminal must still complete the stream"
        );
    }

    #[tokio::test]
    async fn refusal_content_part_frames_are_no_ops_and_refusal_text_streams() {
        // A refusal turn emits `response.content_part.added/.done` with a
        // `refusal` part — a shape outside the modeled text parts — followed
        // by the refusal text via `response.refusal.delta`. The part frames
        // must parse as no-ops (never error items); the deltas carry the
        // content.
        let part_added = json!({
            "type": "response.content_part.added",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "part": { "type": "refusal", "refusal": "" }
        });
        let refusal_delta = json!({
            "type": "response.refusal.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 2,
            "delta": "I can't help with that."
        });
        let part_done = json!({
            "type": "response.content_part.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 3,
            "part": { "type": "refusal", "refusal": "I can't help with that." }
        });
        let reasoning_part = json!({
            "type": "response.content_part.added",
            "item_id": "rs_1",
            "output_index": 1,
            "content_index": 0,
            "sequence_number": 4,
            "part": { "type": "reasoning_text", "text": "" }
        });
        let completed = json!({
            "type": "response.completed",
            "sequence_number": 5,
            "response": sample_response(ResponseStatus::Completed),
        });

        let client = openai::Client::builder()
            .http_client(MockStreamingClient {
                sse_bytes: sse_bytes_from_json_events(&[
                    part_added,
                    refusal_delta,
                    part_done,
                    reasoning_part,
                    completed,
                ]),
            })
            .api_key("test-key")
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-5.4");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");

        let mut texts = Vec::new();
        let mut saw_final = false;
        while let Some(item) = stream.next().await {
            match item.expect("content-part frames must not surface as errors") {
                StreamedAssistantContent::Text(text) => texts.push(text.text),
                StreamedAssistantContent::Final(_) => saw_final = true,
                _ => {}
            }
        }

        assert_eq!(texts, ["I can't help with that."]);
        assert!(saw_final, "the terminal must still arrive");
    }

    #[tokio::test]
    async fn truncated_stream_does_not_synthesize_a_terminal_record() {
        use crate::providers::internal::openai_chat_completions_compatible::test_support::sse_bytes_from_json_events;
        use crate::test_utils::MockStreamingClient;

        // Deltas then EOF without `response.completed`: the accumulator's
        // `saw_terminal` gate must withhold the terminal record rather than
        // present the truncated turn as a successful completion.
        let deltas = [
            json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "sequence_number": 1,
                "delta": "hel"
            }),
            json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "sequence_number": 2,
                "delta": "lo"
            }),
        ];

        let client = openai::Client::builder()
            .http_client(MockStreamingClient {
                sse_bytes: sse_bytes_from_json_events(&deltas),
            })
            .api_key("test-key")
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-5.4");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");

        let mut texts = Vec::new();
        let mut saw_terminal = false;
        while let Some(item) = stream.next().await {
            match item.expect("stream item should be Ok") {
                StreamedAssistantContent::Text(text) => texts.push(text.text),
                StreamedAssistantContent::Final(_) => saw_terminal = true,
                _ => {}
            }
        }

        assert_eq!(texts, ["hel", "lo"]);
        assert!(
            !saw_terminal,
            "EOF without response.completed must not synthesize a terminal record"
        );
        assert!(stream.response.is_none());
    }

    #[tokio::test]
    async fn streaming_error_event_preserves_full_payload_in_live_loop() {
        use crate::providers::internal::openai_chat_completions_compatible::test_support::sse_bytes_from_json_events;
        use crate::test_utils::MockStreamingClient;

        let payload = json!({
            "type": "error",
            "error": {
                "message": "boom",
                "code": "server_error",
                "type": "server_error"
            }
        });

        let client = openai::Client::builder()
            .http_client(MockStreamingClient {
                sse_bytes: sse_bytes_from_json_events(&[payload]),
            })
            .api_key("test-key")
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-5.4");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");

        let err = stream
            .next()
            .await
            .expect("stream should yield an item")
            .expect_err("stream should surface a provider response error");
        assert_eq!(err.provider_response_status(), None);
        assert!(
            err.provider_response_body().is_some_and(|body| {
                body.contains("\"type\":\"error\"") && body.contains("boom")
            })
        );
        assert!(
            stream.next().await.is_none(),
            "stream should terminate after error event"
        );
    }

    #[tokio::test]
    async fn streaming_http_non_success_preserves_status_and_body() {
        use crate::http_client::sse::GenericEventSource;
        use crate::test_utils::HttpErrorStreamingClient;

        let body = r#"{"error":{"message":"quota exceeded"}}"#;
        let client = HttpErrorStreamingClient::new(http::StatusCode::TOO_MANY_REQUESTS, body);
        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/responses")
            .body(Vec::new())
            .expect("request should build");
        let event_source = GenericEventSource::new(client, req);
        let span = tracing::Span::none();
        let mut stream = super::normalize_responses_stream(
            "openai",
            super::raw_stream_from_event_source(event_source, span),
        );

        let err = stream
            .next()
            .await
            .expect("stream should yield transport error")
            .expect_err("HTTP non-success should surface as a stream error");
        assert_eq!(
            err.provider_response_status(),
            Some(http::StatusCode::TOO_MANY_REQUESTS)
        );
        assert_eq!(err.provider_response_body(), Some(body));
        assert_eq!(
            err.provider_response_json().expect("valid JSON body"),
            Some(serde_json::json!({"error": {"message": "quota exceeded"}}))
        );
        assert!(
            stream.next().await.is_none(),
            "stream should terminate after HTTP non-success"
        );
    }

    /// The buffered unary path has no stream to carry error items, so a
    /// corrupt known frame fails the whole decode — even when a valid terminal
    /// follows — instead of returning a silently partial completion.
    #[test]
    fn corrupt_known_frame_fails_the_buffered_body() {
        let corrupt = json!({
            "type": "response.output_text.delta",
            "delta": 42
        });
        let completed = json!({
            "type": "response.completed",
            "sequence_number": 2,
            "response": sample_response(ResponseStatus::Completed),
        });
        let body = format!("data: {corrupt}\ndata: {completed}\n");

        let err = raw_choices_from_sse_body(&body, ResponsesUsage::new())
            .expect_err("a corrupt known frame must fail the buffered decode");
        assert!(
            err.to_string().contains("response.output_text.delta"),
            "the error should name the malformed event, got: {err}"
        );

        // Syntactically invalid JSON fails too.
        let body = format!("data: {{not json\ndata: {completed}\n");
        raw_choices_from_sse_body(&body, ResponsesUsage::new())
            .expect_err("invalid JSON must fail the buffered decode");

        // Unknown event types stay skippable.
        let unknown = json!({ "type": "response.rocket_launch", "count": 3 });
        let body = format!("data: {unknown}\ndata: {completed}\n");
        let choices = raw_choices_from_sse_body(&body, ResponsesUsage::new())
            .expect("unknown event types must stay skippable");
        assert!(
            choices
                .iter()
                .any(|choice| matches!(choice, RawStreamingChoice::FinalResponse(_))),
            "the genuine terminal must still be recorded"
        );
    }

    /// Envelope-less frames (ChatGPT's replayed bodies) are repaired and fed
    /// through the same typed interpreter as the live loop, so the buffered
    /// path agrees with the live path's semantics.
    #[test]
    fn envelope_less_frames_repair_onto_the_shared_interpreter() {
        let completed = json!({
            "type": "response.completed",
            "response": sample_response(ResponseStatus::Completed),
        });

        // A ChatGPT-style text delta with no envelope bookkeeping fields.
        let body = format!(
            "data: {}\ndata: {completed}\n",
            json!({ "type": "response.output_text.delta", "delta": "hi" })
        );
        let choices = raw_choices_from_sse_body(&body, ResponsesUsage::new())
            .expect("an envelope-less delta must repair and decode");
        assert!(
            choices
                .iter()
                .any(|choice| matches!(choice, RawStreamingChoice::Message(text) if text == "hi"))
        );

        // Live-path parity, pinned: a function-call-arguments delta with no
        // `item_id` is keyed by the minted slot identity (the repair injects
        // `output_index: 0`), matching the live loop — it must flow into
        // assembly instead of vanishing.
        let body = format!(
            "data: {}\ndata: {completed}\n",
            json!({ "type": "response.function_call_arguments.delta", "delta": "{}" })
        );
        let choices = raw_choices_from_sse_body(&body, ResponsesUsage::new())
            .expect("an id-less args delta must repair and decode");
        assert!(choices.iter().any(|choice| matches!(
            choice,
            RawStreamingChoice::ToolCallDelta { id, .. } if id == &crate::streaming::MintKind::Output.for_wire_index(0)
        )));

        // Live-path parity, pinned: an envelope-less bookkeeping event whose
        // data is intact (`.done` events) is a no-op, not an error as the old
        // salvage made it.
        let body = format!(
            "data: {}\ndata: {completed}\n",
            json!({ "type": "response.output_text.done", "text": "hi" })
        );
        let choices = raw_choices_from_sse_body(&body, ResponsesUsage::new())
            .expect("an envelope-less done event must repair to the live no-op");
        assert!(
            choices
                .iter()
                .any(|choice| matches!(choice, RawStreamingChoice::FinalResponse(_)))
        );

        // An envelope-less reasoning summary delta keys by the repaired
        // `output_index`, matching the live derivation.
        let body = format!(
            "data: {}\ndata: {completed}\n",
            json!({ "type": "response.reasoning_summary_text.delta", "delta": "think" })
        );
        let choices = raw_choices_from_sse_body(&body, ResponsesUsage::new())
            .expect("an envelope-less summary delta must repair and decode");
        assert!(choices.iter().any(|choice| matches!(
            choice,
            RawStreamingChoice::ReasoningDelta { id, provider_id: _, reasoning }
                if id == &crate::streaming::MintKind::Output.for_wire_index(0) && reasoning == "think"
        )));
    }

    /// The `max_output_tokens`-mid-tool-call shape: `arguments_delta`
    /// frames stream partial JSON and the done item restates the same
    /// truncated bytes (unparseable). Re-emitting the restatement as
    /// another raw delta put the partial JSON in the buffer TWICE — a
    /// delta-reassembling consumer rendered it twice and the bytes were
    /// double-charged against the accumulation bound. Fragments seen →
    /// the buffer already holds the bytes; only a fragment-less done item
    /// (pure replay of a truncated restatement) still routes its raw
    /// string through the buffer at all (#2258 P3).
    #[test]
    fn an_unparseable_restatement_is_not_reemitted_over_streamed_fragments() {
        let delta = json!({
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_1",
            "output_index": 0,
            "sequence_number": 1,
            "delta": "{\"x\":481",
        });
        let done = json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 2,
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "add",
                "arguments": "{\"x\":481",
                "status": "incomplete"
            },
        });
        let body = format!(
            "data: {delta}
data: {done}
"
        );

        let choices = raw_choices_from_sse_body(&body, ResponsesUsage::new())
            .expect("the truncated shape must decode");
        let raw_fragments: Vec<&str> = choices
            .iter()
            .filter_map(|choice| match choice {
                RawStreamingChoice::ToolCallDelta {
                    content: crate::streaming::ToolCallDeltaContent::Delta(fragment),
                    ..
                } => Some(fragment.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            raw_fragments,
            vec!["{\"x\":481"],
            "the streamed fragment is buffered once; the restatement adds nothing"
        );
    }

    /// The pure-replay half of the same policy: a truncated restatement
    /// with NO preceding fragments must still reach the buffer (else the
    /// bytes never arrive and the truncation policy has nothing to judge).
    #[test]
    fn a_fragmentless_unparseable_restatement_still_reaches_the_buffer() {
        let done = json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 1,
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "add",
                "arguments": "{\"x\":481",
                "status": "incomplete"
            },
        });
        let body = format!(
            "data: {done}
"
        );

        let choices = raw_choices_from_sse_body(&body, ResponsesUsage::new())
            .expect("the replayed truncated shape must decode");
        let raw_fragments = choices
            .iter()
            .filter(|choice| {
                matches!(
                    choice,
                    RawStreamingChoice::ToolCallDelta {
                        content: crate::streaming::ToolCallDeltaContent::Delta(_),
                        ..
                    }
                )
            })
            .count();
        assert_eq!(raw_fragments, 1, "the raw bytes must reach the buffer once");
    }

    /// A slot mixing id-bearing and id-less reasoning frames (gateways and
    /// ChatGPT's envelope-less replay bodies omit the id on a subset of a
    /// slot's events) must key every frame — and the done item — by ONE
    /// slot identity, the same discipline `tool_slots` applies. Per-event
    /// resolution split the slot into `Wire("rs_1")` and `Minted(Output, 0)`,
    /// and the done item superseded only one of them: the other survived as
    /// an orphaned partial part carrying the same provider id.
    #[tokio::test]
    async fn mixed_id_and_id_less_reasoning_frames_share_one_slot_key() {
        let with_id = json!({
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_1",
            "output_index": 0,
            "summary_index": 0,
            "sequence_number": 1,
            "delta": "s1 ",
        });
        let id_less = json!({
            "type": "response.reasoning_summary_text.delta",
            "output_index": 0,
            "summary_index": 0,
            "sequence_number": 2,
            "delta": "s2",
        });
        let done = json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 3,
            "item": {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "s1 s2"}],
                "content": [],
                "status": "completed",
            },
        });
        let completed = json!({
            "type": "response.completed",
            "response": sample_response(ResponseStatus::Completed),
        });
        let body = format!("data: {with_id}\ndata: {id_less}\ndata: {done}\ndata: {completed}\n");

        let raw_choices = raw_choices_from_sse_body(&body, ResponsesUsage::new())
            .expect("the mixed slot must decode");
        let mut keys = std::collections::HashSet::new();
        for choice in &raw_choices {
            match choice {
                RawStreamingChoice::ReasoningDelta { id, .. } => {
                    keys.insert(id.clone());
                }
                RawStreamingChoice::ReasoningEnd { id, .. } => {
                    keys.insert(id.clone());
                }
                _ => {}
            }
        }
        assert_eq!(
            keys.len(),
            1,
            "one slot, one assembly key — got {keys:?} across {raw_choices:?}"
        );

        let raw_response = sample_response(ResponseStatus::Completed);
        let response =
            super::completion_response_from_raw_choices("openai", raw_choices, &raw_response)
                .await
                .expect("the mixed slot should normalize")
                .expect("a reasoning-bearing stream is not empty");
        let reasoning_parts = response
            .choice
            .iter()
            .filter(|content| matches!(content, crate::completion::AssistantContent::Reasoning(_)))
            .count();
        assert_eq!(
            reasoning_parts, 1,
            "the done item supersedes the one delta-built part; nothing orphans"
        );
    }

    /// #2258 F3: an id-less reasoning delta is keyed by the minted
    /// `output-{index}` identity, and the slot's `output_item.done` full block
    /// (which always carries the real `rs_*` id) must adopt that minted
    /// identity — otherwise the restated summary appends beside the
    /// delta-built part and duplicates it. This is the ChatGPT envelope-less
    /// replay shape: the repair injects `output_index: 0` into the delta while
    /// the done item arrives envelope-full.
    #[tokio::test]
    async fn envelope_less_reasoning_deltas_are_superseded_by_their_done_item() {
        let delta = json!({ "type": "response.reasoning_summary_text.delta", "delta": "think" });
        let done = json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 2,
            "item": {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "think"}],
                "content": [],
                "status": "completed",
            },
        });
        let completed = json!({
            "type": "response.completed",
            "response": sample_response(ResponseStatus::Completed),
        });
        let body = format!("data: {delta}\ndata: {done}\ndata: {completed}\n");

        let raw_choices = raw_choices_from_sse_body(&body, ResponsesUsage::new())
            .expect("the envelope-less reasoning replay must decode");
        // The done item's restatement shares the minted per-slot identity.
        assert!(raw_choices.iter().any(|choice| matches!(
            choice,
            RawStreamingChoice::ReasoningEnd { id, reasoning: Some(_), .. }
                if id == &crate::streaming::MintKind::Output.for_wire_index(0)
        )));

        let raw_response = sample_response(ResponseStatus::Completed);
        let response =
            super::completion_response_from_raw_choices("chatgpt", raw_choices, &raw_response)
                .await
                .expect("replay should normalize")
                .expect("a reasoning-bearing replay is not empty");

        let reasoning: Vec<_> = response
            .choice
            .iter()
            .filter_map(|content| match content {
                crate::completion::AssistantContent::Reasoning(reasoning) => Some(reasoning),
                _ => None,
            })
            .collect();
        assert_eq!(
            reasoning.len(),
            1,
            "deltas and their full block must collapse to one reasoning item: {reasoning:?}"
        );
        let occurrences = reasoning
            .iter()
            .flat_map(|item| item.content.iter())
            .filter(|content| match content {
                ReasoningContent::Summary(text) | ReasoningContent::Text { text, .. } => {
                    text.contains("think")
                }
                _ => false,
            })
            .count();
        assert_eq!(
            occurrences, 1,
            "the restated summary must supersede its deltas, not duplicate them"
        );
    }

    /// #2258 P2: text deltas for one message item interleaved with reasoning
    /// must aggregate as ONE text part. Interleaving reasoning closes the open
    /// text block downstream, so the adapter must re-emit `TextStart` with the
    /// same item id when the item's text resumes — the accumulator's keyed
    /// reactivation then reopens the block instead of minting a sibling.
    #[tokio::test]
    async fn same_item_text_resumes_as_one_part_across_interleaved_reasoning() {
        let events = [
            json!({
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "sequence_number": 1,
                "delta": "hello "
            }),
            json!({
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs_2",
                "output_index": 1,
                "summary_index": 0,
                "sequence_number": 2,
                "delta": "because"
            }),
            json!({
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "sequence_number": 3,
                "delta": "world"
            }),
            json!({
                "type": "response.completed",
                "sequence_number": 4,
                "response": sample_response(ResponseStatus::Completed),
            }),
        ];
        let body = events
            .iter()
            .map(|event| format!("data: {event}\n"))
            .collect::<String>();

        let raw_choices = raw_choices_from_sse_body(&body, ResponsesUsage::new())
            .expect("the interleaved stream must decode");
        // The resumed item re-announces its block: two `TextStart { msg_1 }`.
        let starts = raw_choices
            .iter()
            .filter(|choice| {
                matches!(
                    choice,
                    RawStreamingChoice::TextStart { id, .. } if id == &crate::streaming::StreamPartId::wire("msg_1")
                )
            })
            .count();
        assert_eq!(
            starts, 2,
            "returning to the same item must re-emit its TextStart: {raw_choices:?}"
        );

        let raw_response = sample_response(ResponseStatus::Completed);
        let response =
            super::completion_response_from_raw_choices("openai", raw_choices, &raw_response)
                .await
                .expect("replay should normalize")
                .expect("a text-bearing replay is not empty");
        let texts: Vec<_> = response
            .choice
            .iter()
            .filter_map(|content| match content {
                crate::completion::AssistantContent::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            texts,
            ["hello world"],
            "same-item text must aggregate as one part around the reasoning"
        );
        assert!(
            response.choice.iter().any(|content| matches!(
                content,
                crate::completion::AssistantContent::Reasoning(_)
            )),
            "the interleaved reasoning must survive"
        );
    }

    /// #2258 P3: two parallel function calls whose events all lack `fc_*` ids
    /// must not share the `""` assembly key — each slot gets a minted
    /// `output-{index}` identity shared by its added/delta/done events, so two
    /// distinct calls assemble.
    /// A slot whose `added` event carries a real `fc_*` id but whose later
    /// args delta arrives id-less must keep ONE assembly key: slot-scoped
    /// identity (the bridge) makes event-scoped key-splitting
    /// unrepresentable, and the finalized call reports the wire id.
    #[tokio::test]
    async fn mixed_id_and_id_less_events_share_one_slot_key() {
        let events = [
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "sequence_number": 1,
                "item": {
                    "type": "function_call",
                    "id": "fc_real",
                    "call_id": "call_a",
                    "name": "tool_a",
                    "arguments": "",
                    "status": "in_progress",
                },
            }),
            // Id-less delta for the same slot: must resolve to the slot's
            // established key, not mint a second identity.
            json!({
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "sequence_number": 2,
                "delta": "{\"x\":1}"
            }),
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "sequence_number": 3,
                "item": {
                    "type": "function_call",
                    "id": "fc_real",
                    "call_id": "call_a",
                    "name": "tool_a",
                    "arguments": "{\"x\":1}",
                    "status": "completed",
                },
            }),
            json!({
                "type": "response.completed",
                "sequence_number": 4,
                "response": sample_response(ResponseStatus::Completed),
            }),
        ];
        let body = events
            .iter()
            .map(|event| format!("data: {event}\n"))
            .collect::<String>();

        let raw_choices = raw_choices_from_sse_body(&body, ResponsesUsage::new())
            .expect("the mixed-id stream must decode");

        // Every tool event (name delta, args delta, input end) carries the
        // slot's single key — no fragment dangles under a second identity.
        let mut keys: Vec<crate::streaming::StreamPartId> = raw_choices
            .iter()
            .filter_map(|choice| match choice {
                RawStreamingChoice::ToolCallDelta { id, .. } => Some(id.clone()),
                RawStreamingChoice::ToolInputEnd(end) => Some(end.id.clone()),
                _ => None,
            })
            .collect();
        keys.dedup();
        assert_eq!(
            keys,
            [crate::streaming::StreamPartId::wire("fc_real")],
            "one slot, one assembly key"
        );

        let raw_response = sample_response(ResponseStatus::Completed);
        let response =
            super::completion_response_from_raw_choices("openai", raw_choices, &raw_response)
                .await
                .expect("replay should normalize")
                .expect("a tool-bearing replay is not empty");
        let call = response
            .choice
            .iter()
            .find_map(|content| match content {
                crate::completion::AssistantContent::ToolCall(call) => Some(call.clone()),
                _ => None,
            })
            .expect("the call finalizes");
        assert_eq!(call.function.name, "tool_a");
        assert_eq!(call.function.arguments, serde_json::json!({"x": 1}));
    }

    #[tokio::test]
    async fn parallel_id_less_function_calls_assemble_distinctly() {
        let call_item = |name: &str, call_id: &str, arguments: &str| {
            json!({
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
                "status": "completed",
            })
        };
        let events = [
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "sequence_number": 1,
                "item": call_item("tool_a", "call_a", ""),
            }),
            json!({
                "type": "response.output_item.added",
                "output_index": 1,
                "sequence_number": 2,
                "item": call_item("tool_b", "call_b", ""),
            }),
            json!({
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "sequence_number": 3,
                "delta": "{\"x\":1}"
            }),
            json!({
                "type": "response.function_call_arguments.delta",
                "output_index": 1,
                "sequence_number": 4,
                "delta": "{\"y\":2}"
            }),
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "sequence_number": 5,
                "item": call_item("tool_a", "call_a", "{\"x\":1}"),
            }),
            json!({
                "type": "response.output_item.done",
                "output_index": 1,
                "sequence_number": 6,
                "item": call_item("tool_b", "call_b", "{\"y\":2}"),
            }),
            json!({
                "type": "response.completed",
                "sequence_number": 7,
                "response": sample_response(ResponseStatus::Completed),
            }),
        ];
        let body = events
            .iter()
            .map(|event| format!("data: {event}\n"))
            .collect::<String>();

        let raw_choices = raw_choices_from_sse_body(&body, ResponsesUsage::new())
            .expect("the id-less parallel-call stream must decode");
        let raw_response = sample_response(ResponseStatus::Completed);
        let response =
            super::completion_response_from_raw_choices("openai", raw_choices, &raw_response)
                .await
                .expect("replay should normalize")
                .expect("a tool-bearing replay is not empty");

        let mut calls: Vec<_> = response
            .choice
            .iter()
            .filter_map(|content| match content {
                crate::completion::AssistantContent::ToolCall(call) => Some((
                    call.function.name.clone(),
                    call.function.arguments.to_string(),
                )),
                _ => None,
            })
            .collect();
        calls.sort();
        assert_eq!(
            calls,
            [
                ("tool_a".to_owned(), json!({"x": 1}).to_string()),
                ("tool_b".to_owned(), json!({"y": 2}).to_string()),
            ],
            "each id-less slot must assemble its own call"
        );
    }

    /// A lost `output_item.done` frame followed by a healthy
    /// `response.completed` must not discard the call as truncation: the
    /// provider proved the turn ended, so the still-open slot closes at the
    /// terminal and finalizes from its streamed fragments — with the full
    /// dual-wire identity the added event announced. The same
    /// terminal-drain the sibling adapters ship (Interactions at
    /// `interaction.completed`, chat-compat at `finish_reason`).
    #[tokio::test]
    async fn a_lost_done_frame_does_not_discard_a_provider_completed_call() {
        let events = [
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "sequence_number": 1,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_abc",
                    "name": "get_weather",
                    "arguments": "",
                    "status": "in_progress",
                },
            }),
            json!({
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "sequence_number": 2,
                "delta": "{\"city\":\"Paris\"}"
            }),
            // The output_item.done frame is lost; the terminal still arrives.
            json!({
                "type": "response.completed",
                "sequence_number": 3,
                "response": sample_response(ResponseStatus::Completed),
            }),
        ];
        let body = events
            .iter()
            .map(|event| format!("data: {event}\n"))
            .collect::<String>();

        let raw_choices = raw_choices_from_sse_body(&body, ResponsesUsage::new())
            .expect("the stream must decode");
        let raw_response = sample_response(ResponseStatus::Completed);
        let response =
            super::completion_response_from_raw_choices("openai", raw_choices, &raw_response)
                .await
                .expect("replay should normalize")
                .expect("a tool-bearing replay is not empty");

        let calls: Vec<_> = response
            .choice
            .iter()
            .filter_map(|content| match content {
                crate::completion::AssistantContent::ToolCall(call) => Some(call),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 1, "the provider-completed call must survive");
        let call = calls[0];
        assert_eq!(call.function.name, "get_weather");
        assert_eq!(call.function.arguments, json!({"city": "Paris"}));
        let provider = call.provider.as_ref().expect("the wire issued ids");
        assert_eq!(provider.call_id, "call_abc");
        assert_eq!(provider.item_id.as_deref(), Some("fc_1"));
    }

    /// #2258 P3: id-less argument fragments must surface as deltas (keyed by
    /// the minted slot identity) rather than vanish; when the stream truncates
    /// before the authoritative `output_item.done` restatement, the settled
    /// truncation policy still applies — partial arguments never fabricate a
    /// call.
    #[tokio::test]
    async fn id_less_args_deltas_surface_and_truncation_fabricates_no_call() {
        let events = [
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "sequence_number": 1,
                "item": {
                    "type": "function_call",
                    "call_id": "call_a",
                    "name": "tool_a",
                    "arguments": "",
                    "status": "in_progress",
                },
            }),
            json!({
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "sequence_number": 2,
                "delta": "{\"loc\":"
            }),
        ];
        let body = events
            .iter()
            .map(|event| format!("data: {event}\n"))
            .collect::<String>();

        let raw_choices = raw_choices_from_sse_body(&body, ResponsesUsage::new())
            .expect("the truncated id-less stream must decode");
        // The fragment flowed into assembly under the minted identity.
        assert!(
            raw_choices.iter().any(|choice| matches!(
                choice,
                RawStreamingChoice::ToolCallDelta {
                    id,
                    content: crate::streaming::ToolCallDeltaContent::Delta(delta),
                } if id == &crate::streaming::MintKind::Output.for_wire_index(0) && delta == "{\"loc\":"
            )),
            "the id-less args fragment must surface as a delta: {raw_choices:?}"
        );

        // No done restatement arrived: the truncation policy withholds the
        // call rather than fabricating one from partial arguments.
        let raw_response = sample_response(ResponseStatus::Completed);
        let response =
            super::completion_response_from_raw_choices("openai", raw_choices, &raw_response)
                .await
                .expect("replay should normalize");
        assert!(
            response.is_none(),
            "partial arguments must not fabricate a call: {response:?}"
        );
    }

    #[test]
    fn refusal_content_part_frames_do_not_fail_the_buffered_body() {
        // The ChatGPT buffered route replays recorded SSE bodies; a refusal
        // turn's `content_part` frames (an unmodeled `refusal` part) must not
        // fail the whole completion — the refusal text arrives via the
        // modeled `response.refusal.delta`.
        let part_added = json!({
            "type": "response.content_part.added",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "part": { "type": "refusal", "refusal": "" }
        });
        let refusal_delta = json!({
            "type": "response.refusal.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 2,
            "delta": "no"
        });
        let completed = json!({
            "type": "response.completed",
            "sequence_number": 3,
            "response": sample_response(ResponseStatus::Completed),
        });
        let body = format!(
            "data: {part_added}
data: {refusal_delta}
data: {completed}
"
        );

        let choices = raw_choices_from_sse_body(&body, ResponsesUsage::new())
            .expect("refusal content-part frames must not fail the buffered decode");
        assert!(
            choices
                .iter()
                .any(|choice| matches!(choice, RawStreamingChoice::Message(text) if text == "no")),
            "the refusal text must be delivered"
        );
    }

    /// The replayed choice keeps its reasoning/tool calls, but message text
    /// present only in the terminal body's `output` must merge in when no text
    /// deltas were streamed (websocket replays hit exactly this quadrant).
    #[tokio::test]
    async fn terminal_body_message_text_merges_into_reasoning_only_replay() {
        use crate::providers::openai::responses_api::Output;

        let raw_choices = vec![RawStreamingChoice::ReasoningDelta {
            provider_id: crate::streaming::WireId::new("rs_1"),
            id: crate::streaming::StreamPartId::wire("rs_1"),
            reasoning: "thinking".to_string(),
        }];

        let mut raw_response = sample_response(ResponseStatus::Completed);
        raw_response.output = vec![
            serde_json::from_value::<Output>(json!({
                "type": "message",
                "id": "msg_body_1",
                "status": "completed",
                "role": "assistant",
                "content": [{ "type": "output_text", "annotations": [], "text": "full answer" }]
            }))
            .expect("output message should deserialize"),
        ];

        let response =
            super::completion_response_from_raw_choices("openai", raw_choices, &raw_response)
                .await
                .expect("replay should normalize")
                .expect("a reasoning-bearing replay is not empty");

        let text: String = response
            .choice
            .iter()
            .filter_map(|content| match content {
                crate::completion::AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "full answer");
        assert!(
            response.choice.iter().any(|content| matches!(
                content,
                crate::completion::AssistantContent::Reasoning(_)
            )),
            "the replayed reasoning must be kept"
        );
        assert_eq!(response.message_id.as_deref(), Some("msg_body_1"));
    }

    #[test]
    fn streaming_error_event_preserves_full_payload() {
        let payload = r#"{"type":"error","error":{"message":"boom","code":"server_error","type":"server_error"}}"#;
        let body = format!("data: {payload}\n");

        let err = super::raw_choices_from_sse_body(&body, super::ResponsesUsage::new())
            .expect_err("error event should surface as a provider response error");

        assert_eq!(err.provider_response_status(), None);
        assert_eq!(err.provider_response_body(), Some(payload));
        let json = err
            .provider_response_json()
            .expect("raw body should be valid JSON")
            .expect("parsed JSON should be present");
        assert_eq!(json["error"]["code"], "server_error");
    }

    #[tokio::test]
    async fn streaming_non_http_transport_error_stays_provider_error() {
        use crate::http_client::sse::GenericEventSource;
        use crate::test_utils::SequencedStreamingHttpClient;

        let chunks = vec![Err(crate::http_client::Error::InvalidContentType(
            http::HeaderValue::from_static("application/json"),
        ))];
        let client = SequencedStreamingHttpClient::new(chunks);
        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/responses")
            .body(Vec::new())
            .expect("request should build");
        let event_source = GenericEventSource::new(client, req);
        let span = tracing::Span::none();
        let mut stream = super::normalize_responses_stream(
            "openai",
            super::raw_stream_from_event_source(event_source, span),
        );

        let err = stream
            .next()
            .await
            .expect("stream should yield transport error")
            .expect_err("non-HTTP transport failure should surface as provider error");
        assert_eq!(
            err.to_string(),
            "ProviderError: Invalid content type was returned: \"application/json\""
        );
        assert!(matches!(
            err,
            crate::completion::CompletionError::ProviderError(_)
        ));
        // Rig-generated transport diagnostics are not provider response bodies.
        assert_eq!(err.provider_response_body(), None);
        assert_eq!(err.provider_response_status(), None);
    }

    #[tokio::test]
    async fn response_completed_chunk_populates_final_usage() {
        let mut response = sample_response(ResponseStatus::Completed);
        response.usage = Some(ResponsesUsage {
            input_tokens: 10,
            input_tokens_details: None,
            output_tokens: 5,
            output_tokens_details: Some(OutputTokensDetails {
                reasoning_tokens: 0,
            }),
            total_tokens: 15,
        });

        let event = json!({
            "type": "response.completed",
            "sequence_number": 1,
            "response": response,
        });

        let usage = final_response_from_event(event).await.usage;
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[tokio::test]
    async fn response_completed_chunk_populates_reasoning_metadata_and_context() {
        let response = sample_response(ResponseStatus::Completed);
        let mut event = json!({
            "type": "response.completed",
            "sequence_number": 1,
            "response": response,
        });
        let metadata = json!({
            "context": "all_turns",
            "effort": "ultra",
            "summary": null,
            "future_control": true
        });
        event["response"]["reasoning"] = metadata.clone();

        let response = final_response_from_event(event).await;
        assert_eq!(response.reasoning_context.as_deref(), Some("all_turns"));
        assert_eq!(response.reasoning_metadata.as_ref(), metadata.as_object());
    }

    #[tokio::test]
    async fn terminal_record_normalizes_into_the_stream_final() {
        let mut response = sample_response(ResponseStatus::Completed);
        response.usage = Some(ResponsesUsage {
            input_tokens: 10,
            input_tokens_details: None,
            output_tokens: 5,
            output_tokens_details: None,
            total_tokens: 15,
        });

        let mut event = json!({
            "type": "response.completed",
            "sequence_number": 1,
            "response": response,
        });
        event["response"]["output"] = json!([{
            "type": "message",
            "id": "msg_stream_1",
            "status": "completed",
            "role": "assistant",
            "content": [{ "type": "output_text", "annotations": [], "text": "hi" }]
        }]);

        let final_response = stream_final_from_event(event).await;

        assert_eq!(final_response.provider, "openai");
        assert_eq!(final_response.model.as_deref(), Some("gpt-5.4"));
        // The assistant message ID (`msg_...`), never the response ID
        // (`resp_123`) that the same event carries.
        assert_eq!(final_response.message_id.as_deref(), Some("msg_stream_1"));
        assert_eq!(
            final_response.finish_reason,
            Some(crate::completion::FinishReason::Stop)
        );
        assert_eq!(final_response.usage.input_tokens, 10);
        assert_eq!(final_response.usage.output_tokens, 5);
        assert_eq!(final_response.usage.total_tokens, 15);
    }

    #[tokio::test]
    async fn terminal_record_reports_tool_calls_when_the_stream_called_a_tool() {
        let tool_call_done = json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 1,
            "item": {
                "type": "function_call",
                "id": "fc_123",
                "arguments": "{}",
                "call_id": "call_123",
                "name": "example_tool",
                "status": "completed"
            }
        });
        let completed = json!({
            "type": "response.completed",
            "sequence_number": 2,
            "response": sample_response(ResponseStatus::Completed),
        });

        let client = openai::Client::builder()
            .http_client(MockStreamingClient {
                sse_bytes: sse_bytes_from_json_events(&[tool_call_done, completed]),
            })
            .api_key("test-key")
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-5.4");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");

        let mut final_response = None;
        while let Some(item) = stream.next().await {
            if let StreamedAssistantContent::Final(response) =
                item.expect("completed stream should not error")
            {
                final_response = Some(response);
            }
        }

        // `completed` is reconciled up to `ToolCalls` by `normalize_stream`,
        // using the call the stream actually emitted.
        assert_eq!(
            final_response
                .expect("stream should yield a final response")
                .finish_reason,
            Some(crate::completion::FinishReason::ToolCalls)
        );
    }

    #[test]
    fn terminal_record_preserves_an_unknown_incomplete_reason() {
        let response = super::StreamingCompletionResponse {
            status: Some(ResponseStatus::Incomplete),
            incomplete_details: Some(IncompleteDetailsReason {
                reason: "MAX_TOOL_CALLS".to_string(),
            }),
            model: Some("gpt-5.4".to_string()),
            message_id: Some("msg_1".to_string()),
            ..super::StreamingCompletionResponse::new(ResponsesUsage::new())
        };

        let final_response = crate::streaming::StreamFinal::from(("openai", response));

        assert_eq!(
            final_response.finish_reason,
            Some(crate::completion::FinishReason::Other(
                "MAX_TOOL_CALLS".to_string()
            ))
        );
        assert_eq!(final_response.message_id.as_deref(), Some("msg_1"));
        assert_eq!(final_response.model.as_deref(), Some("gpt-5.4"));
    }

    #[tokio::test]
    async fn done_sentinel_is_ignored_without_debug_parse_noise() {
        use std::io::{self, Write};
        use std::sync::{Arc, Mutex};

        #[derive(Clone)]
        struct SharedWriter(Arc<Mutex<Vec<u8>>>);

        impl Write for SharedWriter {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.0
                    .lock()
                    .expect("log buffer mutex should not be poisoned")
                    .extend_from_slice(buf);
                Ok(buf.len())
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let mut response = sample_response(ResponseStatus::Completed);
        response.usage = Some(ResponsesUsage {
            input_tokens: 4,
            input_tokens_details: None,
            output_tokens: 2,
            output_tokens_details: Some(OutputTokensDetails {
                reasoning_tokens: 0,
            }),
            total_tokens: 6,
        });

        // Scoped-subscriber tests must not run concurrently; see
        // `test_utils::scoped_tracing_subscriber_guard`.
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
        let captured = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_ansi(false)
            .without_time()
            .with_writer({
                let captured = captured.clone();
                move || SharedWriter(captured.clone())
            })
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);

        let client = openai::Client::builder()
            .http_client(MockStreamingClient {
                sse_bytes: bytes::Bytes::from(format!(
                    "data: {}\n\ndata: [DONE]\n\n",
                    serde_json::to_string(&json!({
                        "type": "response.completed",
                        "sequence_number": 1,
                        "response": response,
                    }))
                    .expect("response event should serialize")
                )),
            })
            .api_key("test-key")
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-5.4");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");

        let mut final_usage = None;
        while let Some(item) = stream.next().await {
            if let StreamedAssistantContent::Final(response) =
                item.expect("stream should complete successfully")
            {
                final_usage = Some(response.usage);
            }
        }

        let usage = final_usage.expect("expected final response");
        assert_eq!(usage.input_tokens, 4);
        assert_eq!(usage.output_tokens, 2);
        assert_eq!(usage.total_tokens, 6);

        let logs = String::from_utf8(
            captured
                .lock()
                .expect("log buffer mutex should not be poisoned")
                .clone(),
        )
        .expect("captured logs should be valid UTF-8");
        assert!(
            !logs.contains("Couldn't deserialize SSE data as StreamingCompletionChunk"),
            "expected [DONE] to bypass the parse-failure debug path, logs were: {logs}"
        );
    }

    #[tokio::test]
    async fn malformed_frame_surfaces_error_and_stream_still_completes() {
        let delta = json!({
            "type": "response.output_text.delta",
            "content_index": 0,
            "delta": "hello",
            "item_id": "msg_1",
            "logprobs": [],
            "output_index": 0,
            "sequence_number": 1
        });
        let completed = json!({
            "type": "response.completed",
            "sequence_number": 2,
            "response": sample_response(ResponseStatus::Completed),
        });
        let http_client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines([
                delta.to_string(),
                "{not valid json".to_string(),
                completed.to_string(),
            ]),
        };
        let client = openai::Client::builder()
            .http_client(http_client)
            .api_key("test-key")
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-5.4");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");

        let mut text = String::new();
        let mut saw_error = false;
        let mut terminal = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(StreamedAssistantContent::Text(chunk)) => text.push_str(&chunk.text),
                Ok(StreamedAssistantContent::Final(final_response)) => {
                    terminal = Some(final_response)
                }
                Ok(other) => panic!("unexpected stream item: {other:?}"),
                Err(err) => {
                    assert!(
                        matches!(err, crate::completion::CompletionError::JsonError(_)),
                        "expected a JSON parse error item, got {err:?}"
                    );
                    saw_error = true;
                }
            }
        }

        // The malformed frame is surfaced as an error item, and the content
        // and genuine terminal on either side of it both still arrive.
        assert_eq!(text, "hello");
        assert!(saw_error, "malformed frame should surface an error item");
        assert!(
            terminal.is_some(),
            "stream should still emit its terminal record"
        );
    }
}
