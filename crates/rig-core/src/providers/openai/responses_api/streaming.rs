//! Responses frame classification, event decoding, and terminal metadata.
//!
//! ```
//! use rig_core::providers::openai::responses_api::streaming::ResponsesDecoder;
//! let decoder = ResponsesDecoder::new("openai");
//! ```

use crate::error::ProviderError;
use crate::operation::{
    CallFragment, Completion, Finish, IfMalformed, ReasoningPart, Seal, TextPart,
};
use crate::providers::internal::wire;
use crate::providers::openai::responses_api::{
    IncompleteDetailsReason, ReasoningSummary, ResponseStatus, ResponsesUsage,
};
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

/// What the terminal response event says about the turn, before it maps to
/// the reply's end.
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
}

impl StreamingCompletionResponse {
    /// Create a terminal record carrying only usage; the remaining metadata is
    /// filled in from the terminal `response.completed` event as it arrives.
    pub fn new(usage: Option<ResponsesUsage>) -> Self {
        Self {
            usage,
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

/// The provider's end of the reply, from the Responses API's terminal
/// record, and the issuer of its reasoning for a gateway relaying an
/// upstream's.
///
/// The provider descriptor name is an input for the same reason it is on the
/// unary conversion: ChatGPT and Copilot stream this exact wire shape, so a
/// baked-in `"openai"` would mislabel them. The finish reason is left exactly
/// as the provider reported it; the fold reconciles it with the tool calls
/// the reply carried.
fn finish_of(
    provider: &str,
    upstream_reasoning_issuer: bool,
    response: StreamingCompletionResponse,
) -> (Finish, Option<String>) {
    let issuer = upstream_reasoning_issuer
        .then_some(response.model.as_deref())
        .flatten()
        .map(|model| crate::providers::openai::wire::upstream_reasoning_issuer(provider, model));
    let finish_reason = response
        .status
        .as_ref()
        .and_then(|status| super::map_finish_reason(status, response.incomplete_details.as_ref()));
    let finish = Finish {
        usage: crate::completion::Usage::from(&response),
        reason: finish_reason,
        message_id: response.message_id,
        response_id: response.response_id,
        model: response.model,
    };
    (finish, issuer)
}

/// Combine summaries, content, and encrypted data into one reasoning restatement.
/// Preserve `provider_id` and wire field order. Return `None` for empty content;
/// the caller must close the existing block with the returned restatement.
pub(crate) fn reasoning_from_done_item(
    provider_id: Option<&str>,
    summary: Vec<ReasoningSummary>,
    content: Vec<String>,
    encrypted_content: Option<String>,
    signature: Option<String>,
) -> Option<crate::message::Reasoning> {
    // Same builder as the unary decode, so the restatement and the
    // non-streaming conversion of one item cannot drift.
    let blocks = super::reasoning_content_blocks(summary, content, encrypted_content, signature);

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

/// Classify a tagged Responses event with strict decoding for known event types.
/// Callers must handle `error` and WebSocket `response.done` separately.
#[doc(hidden)]
pub fn classify_responses_frame(data: &str) -> WireEvent<StreamingCompletionChunk> {
    wire::classify_tagged_frame(data, "type", is_known_responses_event_type)
}

/// The assistant message ID (`msg_...`) a terminal response object carries,
/// which is deliberately not the response's own `resp_...` id.
fn message_id_from_response(response: &CompletionResponse) -> Option<String> {
    response.output.iter().find_map(|item| match item {
        Output::Message(message) => Some(message.id.clone()),
        _ => None,
    })
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
    Whole(Box<CompletionResponse>),
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

/// The OpenAI Responses wire's decoder: one state machine for the SSE
/// stream, the unary body and the websocket session.
pub struct ResponsesDecoder<'id> {
    /// Stable descriptor name the reply is attributed to: ChatGPT and
    /// Copilot stream this exact wire shape, so it is an input rather than
    /// a baked-in `"openai"`.
    provider: String,
    /// The terminal event's response object: the provider's own document
    /// of the turn, and the response's `raw`.
    document: Option<serde_json::Value>,
    /// Whether to repair absent envelope indices before retrying classification.
    /// Selected by dialect, independently of unary or streaming mode.
    repair_envelopes: bool,
    /// Whether reasoning belongs to the upstream model's family rather than
    /// to `provider`, a gateway ([`crate::providers::openai::wire::upstream_reasoning_issuer`]).
    upstream_reasoning_issuer: bool,
    /// The terminal record under assembly: what the terminal event says
    /// about the turn, filled in as the stream reports it.
    terminal: StreamingCompletionResponse,
    /// The text part of each message item. A text or refusal delta carrying
    /// another `item_id` extends that item's part, so two `message` output
    /// items aggregate as two distinct text parts instead of concatenating.
    texts: std::collections::HashMap<String, TextPart<'id>>,
    /// The message item whose text part fragments without an `item_id`
    /// extend (ChatGPT's envelope-less replays).
    current_text_item: Option<String>,
    /// The part fragments without any item id extend.
    anonymous_text: Option<TextPart<'id>>,
    /// The message items whose visible text a delta already delivered, and
    /// whether any fragment arrived that could not be attributed to one.
    /// The terminal restates the whole turn's output, so its message text
    /// is published only where no delta delivered it.
    delta_text_items: std::collections::HashSet<String>,
    /// Output slots with delivered text. Slot tracking prevents duplicate terminal
    /// text when a gateway changes item IDs between deltas and restatements.
    delta_text_slots: std::collections::HashSet<u64>,
    unattributed_text_delta: bool,
    /// The message items, by id and output slot, whose content-part extras
    /// are on their text part. Every snapshot restates them, so only the
    /// first attaches.
    extras_items: std::collections::HashSet<String>,
    extras_slots: std::collections::HashSet<u64>,
    /// The reasoning part of each output slot, fixed by its first fragment.
    reasoning: std::collections::HashMap<u64, ReasoningPart<'id>>,
}

impl<'id> ResponsesDecoder<'id> {
    /// A decoder for one reply of `provider`'s Responses endpoint.
    pub fn new(provider: &str) -> Self {
        Self {
            provider: provider.to_owned(),
            document: None,
            repair_envelopes: false,
            upstream_reasoning_issuer: false,
            terminal: StreamingCompletionResponse::new(None),
            texts: std::collections::HashMap::new(),
            current_text_item: None,
            anonymous_text: None,
            delta_text_items: std::collections::HashSet::new(),
            delta_text_slots: std::collections::HashSet::new(),
            unattributed_text_delta: false,
            extras_items: std::collections::HashSet::new(),
            extras_slots: std::collections::HashSet::new(),
            reasoning: std::collections::HashMap::new(),
        }
    }

    /// Salvage replayed frames that omit their envelope bookkeeping.
    pub fn with_envelope_repair(mut self) -> Self {
        self.repair_envelopes = true;
        self
    }

    /// Record reasoning as issued by the upstream model's family, for a
    /// gateway that relays each upstream's own reasoning state.
    pub fn with_upstream_reasoning_issuer(mut self) -> Self {
        self.upstream_reasoning_issuer = true;
        self
    }

    /// Seed the terminal's usage for a replayed body whose frames may not
    /// carry one (the unary Responses body's own `usage`).
    pub fn with_initial_usage(mut self, usage: Option<ResponsesUsage>) -> Self {
        self.terminal.usage = usage;
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
            wire::classify_marker_keyed_frame::<CompletionResponse>(data, WHOLE_BODY_MARKERS)
                .map(|response| ResponsesEvent::Whole(Box::new(response)))
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

    /// The text part a fragment of the message item `item_id` extends.
    fn text_part(
        &mut self,
        item_id: Option<&str>,
        out: &mut Out<'id, Completion>,
    ) -> &TextPart<'id> {
        let item_id = item_id
            .filter(|id| !id.is_empty())
            .map(str::to_owned)
            .or_else(|| self.current_text_item.clone());
        match item_id {
            Some(item_id) => {
                self.current_text_item = Some(item_id.clone());
                self.texts.entry(item_id).or_insert_with(|| out.text())
            }
            None => self.anonymous_text.get_or_insert_with(|| out.text()),
        }
    }

    /// Record that a delta delivered the visible text of a message item.
    ///
    /// The output slot is always recorded; the item id is recorded on top
    /// of it, because a delta the wire did not attribute extends whichever
    /// text part is current and is credited to that item, and with no part
    /// current there is nothing to attribute it to at all.
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

    /// Publish one message item's visible text as the fragments that built
    /// it, recording what it delivered so a terminal restating the same
    /// item merges nothing.
    fn publish_message_text(
        &mut self,
        output_index: u64,
        message: &super::OutputMessage,
        out: &mut Out<'id, Completion>,
    ) {
        if !message.content.is_empty() {
            self.note_text_delta(output_index, Some(&message.id));
        }
        self.note_extras(output_index, &message.id);
        for content in message.content.iter().cloned() {
            let mut text = super::text_block(content);
            super::stamp_phase(&mut text, message.phase.as_deref());
            let part = self.text_part(Some(&message.id), out);
            out.push_text(part, &text.text);
            if let Some(additional_params) = text.additional_params {
                out.text_params(part, additional_params);
            }
        }
    }

    /// Record that the message item at `output_index` has its extras on its
    /// text part.
    fn note_extras(&mut self, output_index: u64, item_id: &str) {
        self.extras_slots.insert(output_index);
        if !item_id.is_empty() {
            self.extras_items.insert(item_id.to_owned());
        }
    }

    /// Attach a message item's content-part extras, such as its citation
    /// annotations, to the text part its deltas built, in content-part order.
    /// Nothing attaches when a snapshot already did or no delta built a part
    /// for the item: text stated only by a snapshot publishes its extras
    /// with it.
    fn attach_message_extras(
        &mut self,
        output_index: u64,
        message: &super::OutputMessage,
        out: &mut Out<'id, Completion>,
    ) {
        if self.extras_slots.contains(&output_index) || self.extras_items.contains(&message.id) {
            return;
        }
        let Some(part) = self.texts.get(&message.id) else {
            return;
        };
        let extras = message
            .content
            .iter()
            .cloned()
            .filter_map(|content| super::text_block(content).additional_params)
            .reduce(|mut extras, next| {
                extras.merge(next);
                extras
            });
        if let Some(extras) = extras {
            out.text_params(part, extras);
        }
        self.note_extras(output_index, &message.id);
    }

    /// Publish nonempty terminal message content when no delta delivered it,
    /// and otherwise only the extras no `output_item.done` attached.
    /// Match by output position, item ID, or the unattributed-delta safeguard.
    fn merge_terminal_body_text(
        &mut self,
        response: &CompletionResponse,
        out: &mut Out<'id, Completion>,
    ) {
        // The item's position in `output[]` IS the `output_index` its
        // stream events carried, which is how a restatement is matched to
        // the deltas that already delivered it.
        for (output_index, item) in response.output.iter().enumerate() {
            let output_index = output_index as u64;
            let Output::Message(message) = item else {
                continue;
            };
            if message.content.is_empty() {
                continue;
            }
            if self.delta_delivered_text(output_index, &message.id) {
                self.attach_message_extras(output_index, message, out);
            } else {
                self.publish_message_text(output_index, message, out);
            }
        }
    }

    /// Write one output-item event into the reply.
    fn decode_item_chunk(
        &mut self,
        chunk: ItemChunk,
        out: &mut Out<'id, Completion>,
    ) -> Result<(), ProviderError> {
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
                // A function call interleaving a message item: a later
                // fragment without an item id cannot belong to that item.
                self.current_text_item = None;
                // Without a call_id the item id cannot become a fabricated
                // tool-result correlator: rig issues the id.
                out.call_fragment(
                    output_index as usize,
                    CallFragment {
                        id: Some(func.call_id.as_str()),
                        item_id: (!func.call_id.is_empty()).then_some(func.id.as_str()),
                        name: Some(func.name.as_str()),
                        ..CallFragment::default()
                    },
                )?;
            }
            ItemChunkKind::OutputItemDone(message) => {
                // Any completed item ends the one it carried; a fragment
                // arriving afterwards names its own item.
                self.current_text_item = None;
                self.push_output_item_done(message.item, output_index, out)?;
            }
            // Text and refusal deltas are the same visible-text stream: a
            // refusal is the assistant's message for that turn.
            ItemChunkKind::OutputTextDelta(DeltaTextChunk { delta, .. })
            | ItemChunkKind::RefusalDelta(DeltaTextChunk { delta, .. }) => {
                self.note_text_delta(output_index, outer_item_id.as_deref());
                let part = self.text_part(outer_item_id.as_deref(), out);
                out.push_text(part, &delta);
            }
            // Summary and raw-reasoning deltas differ only in which wire
            // event carries them; both are fragments of the output item's
            // reasoning part.
            ItemChunkKind::ReasoningSummaryTextDelta(SummaryTextChunk { delta, .. })
            | ItemChunkKind::ReasoningTextDelta(DeltaTextChunkWithItemId { delta, .. }) => {
                self.current_text_item = None;
                let part = self
                    .reasoning
                    .entry(output_index)
                    .or_insert_with(|| out.reasoning());
                out.push_reasoning(part, &delta);
            }
            ItemChunkKind::FunctionCallArgsDelta(delta) => {
                self.current_text_item = None;
                out.call_fragment(
                    output_index as usize,
                    CallFragment {
                        arguments: Some(delta.delta.as_str()),
                        ..CallFragment::default()
                    },
                )?;
            }
            _ => {}
        }
        Ok(())
    }

    fn push_output_item_done(
        &mut self,
        item: Output,
        output_index: u64,
        out: &mut Out<'id, Completion>,
    ) -> Result<(), ProviderError> {
        match item {
            Output::FunctionCall(func) => {
                let index = output_index as usize;
                let streamed = out.pending_has_arguments(index);
                // The done item restates the call: its name and ids win. An
                // empty call_id means no provider identity; the fc_* item id
                // is never substituted, since replay requires a real call_id.
                out.call_fragment(
                    index,
                    CallFragment {
                        id: Some(func.call_id.as_str()),
                        item_id: (!func.call_id.is_empty()).then_some(func.id.as_str()),
                        name: Some(func.name.as_str()),
                        ..CallFragment::default()
                    },
                )?;
                match func.arguments.parse() {
                    // Parsed restatements win.
                    Ok(arguments) => out.announce_pending(index, arguments),
                    // A raw restatement is buffered only when no fragment
                    // preceded it, so its bytes are not stated twice.
                    Err(_) if !streamed => out.call_fragment(
                        index,
                        CallFragment {
                            arguments: Some(func.arguments.as_str()),
                            ..CallFragment::default()
                        },
                    )?,
                    Err(_) => {}
                }
                // The done item completes the call.
                out.close_pending(index, IfMalformed::Drop)?;
            }
            Output::Reasoning {
                id,
                summary,
                content,
                encrypted_content,
                signature,
                ..
            } => {
                let provider_id = (!id.is_empty()).then_some(id);
                let part = self.reasoning.remove(&output_index);
                let restated = reasoning_from_done_item(
                    provider_id.as_deref(),
                    summary,
                    content,
                    encrypted_content,
                    signature,
                );
                match (part, restated) {
                    // The restatement supersedes the part's fragments.
                    (Some(part), restated) => out.close_reasoning(
                        part,
                        Seal {
                            id: provider_id,
                            restated,
                            ..Seal::default()
                        },
                    ),
                    (None, Some(restated)) => out.reasoning_block(restated),
                    // A contentless identified item is still replay state.
                    (None, None) => {
                        if let Some(id) = provider_id {
                            out.reasoning_block(crate::message::Reasoning {
                                id: Some(id),
                                content: Vec::new(),
                            });
                        }
                    }
                }
            }
            Output::Message(message) => {
                self.attach_message_extras(output_index, &message, out);
                if !message.id.is_empty() {
                    out.message_id(message.id);
                }
            }
            // An unmodeled output item (e.g. a hosted-tool result such as
            // `web_search_call`): surfaced raw to the consumer, as the
            // non-streaming decode preserves it on `CompletionResponse.output`.
            Output::Unknown(value) => {
                out.unknown(value.into());
            }
            // A compaction item: surfaced raw like an unmodeled item so a
            // stateless consumer can capture it from the stream.
            Output::Compaction(fields) => {
                let mut map = fields;
                map.insert(
                    "type".to_string(),
                    serde_json::Value::String("compaction".to_string()),
                );
                out.unknown(serde_json::Value::Object(map).into());
            }
        }
        Ok(())
    }

    /// Record a terminal event's facts: the text no delta delivered, the
    /// extras no item snapshot attached, how the turn ended, which model
    /// answered, and which assistant message (`msg_...`, not the response's
    /// `resp_...`) carried the output.
    fn record_terminal(&mut self, response: CompletionResponse, out: &mut Out<'id, Completion>) {
        self.document = serde_json::to_value(&response).ok();
        // The terminal restates the whole turn, so the message text no delta
        // delivered is published here: a gateway that states its answer only
        // in the terminal body still lands it in the choice, and one that
        // streamed the text first does not state it twice.
        self.merge_terminal_body_text(&response, out);
        if let Some(message_id) = message_id_from_response(&response) {
            self.terminal.message_id = Some(message_id);
        }
        if !response.id.is_empty() {
            self.terminal.response_id = Some(response.id.clone());
        }
        if !response.model.is_empty() {
            self.terminal.model = Some(response.model.clone());
        }
        self.terminal.status = Some(response.status);
        if response.incomplete_details.is_some() {
            self.terminal.incomplete_details = response.incomplete_details;
        }
        if response.usage.is_some() {
            self.terminal.usage = response.usage;
        }
        if response.reasoning_metadata.is_some() {
            self.terminal.reasoning_metadata = response.reasoning_metadata;
        }
        if response.reasoning_context.is_some() {
            self.terminal.reasoning_context = response.reasoning_context;
        }
    }

    /// The provider ended the turn: close the calls whose done event never
    /// came (their incomplete arguments drop), then end the reply.
    fn end(&mut self, mut out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        for index in out.pending_calls() {
            out.close_pending(index, IfMalformed::Drop)?;
        }
        if let Some(document) = self.document.take() {
            out.raw(document);
        }
        let terminal =
            std::mem::replace(&mut self.terminal, StreamingCompletionResponse::new(None));
        let (finish, issuer) = finish_of(&self.provider, self.upstream_reasoning_issuer, terminal);
        if let Some(issuer) = issuer {
            out.issued_by(issuer);
        }
        Ok(out.end(finish))
    }

    /// Replay a whole body as the items the stream sends, then end with the
    /// terminal the body itself is. Structured reasoning suppresses the
    /// top-level reasoning display string.
    fn replay_whole_response(
        &mut self,
        response: CompletionResponse,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
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
            out.reasoning_block(crate::message::Reasoning::new(reasoning));
        }

        for (output_index, item) in response.output.iter().cloned().enumerate() {
            let output_index = output_index as u64;
            if let Output::Message(message) = &item {
                self.publish_message_text(output_index, message, &mut out);
            }
            // Immediate publication keeps the output's order.
            self.push_output_item_done(item, output_index, &mut out)?;
        }
        self.record_terminal(response, &mut out);
        self.end(out)
    }
}

impl<'id> Decoder<'id, Completion> for ResponsesDecoder<'id> {
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
                self.decode_item_chunk(chunk, &mut out)?;
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
                    // usage are kept, and the status maps to the finish
                    // reason as on the unary path.
                    ResponseChunkKind::ResponseCompleted
                    | ResponseChunkKind::ResponseIncomplete => {
                        self.record_terminal(response, &mut out);
                        self.end(out)
                    }
                    ResponseChunkKind::ResponseFailed => {
                        Err(crate::error::ProviderError::from_provider_body(&raw))
                    }
                    ResponseChunkKind::ResponseCreated | ResponseChunkKind::ResponseInProgress => {
                        Ok(Flow::More)
                    }
                }
            }
            // The unary reply is the same turn stated at once.
            ResponsesEvent::Whole(response) => self.replay_whole_response(*response, out),
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
