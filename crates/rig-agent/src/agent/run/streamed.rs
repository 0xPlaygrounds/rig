//! Streamed-turn assembly for [`AgentRun`](super::AgentRun).
//!
//! A streamed model turn arrives as incremental [`StreamedAssistantContent`]
//! items. [`StreamedTurnAssembler`] is the sans-IO accumulator that turns that
//! item stream into the same canonical complete turn the non-streaming path
//! feeds the machine — while telling the driver what to forward to its
//! consumer and surfacing invalid tool calls the moment they appear, so a
//! driver can stop paying for a doomed provider stream early.
//!
//! The protocol, paired with the streamed entry points on
//! [`AgentRun`](super::AgentRun):
//!
//! 1. On [`AgentRunStep::CallModel`](super::AgentRunStep::CallModel), open a
//!    provider stream and create one assembler per turn with the tool names
//!    advertised for that turn.
//! 2. Feed every stream item to [`StreamedTurnAssembler::ingest`] and act on
//!    the returned [`StreamedTurnEvent`]s: forward items to the consumer, and
//!    on [`StreamedTurnEvent::InvalidToolCall`] consult
//!    [`AgentRun::resolve_streamed_invalid_tool_call`](super::AgentRun::resolve_streamed_invalid_tool_call) —
//!    [`StreamedResolution::Repaired`] continues the same stream via
//!    [`StreamedTurnAssembler::resolve_pending_invalid`];
//!    [`StreamedResolution::TurnAbandoned`] means drain the provider stream
//!    for usage and re-enter
//!    [`AgentRun::next_step`](super::AgentRun::next_step).
//! 3. When the provider stream ends, call [`StreamedTurnAssembler::finish`]
//!    and feed the result to
//!    [`AgentRun::streamed_turn`](super::AgentRun::streamed_turn); the run
//!    then proceeds exactly like a non-streamed one
//!    ([`CallTools`](super::AgentRunStep::CallTools) /
//!    [`Done`](super::AgentRunStep::Done)).
//!
//! [`crate::streaming::StreamingPrompt::stream_prompt`] drives this protocol
//! internally; hand-driven runs can use it to stream any
//! [`AgentRun`](super::AgentRun).

use std::collections::{BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

use rig_core::completion::FinishReason;
use rig_core::message::{
    AssistantContent, Reasoning, ToolCall, ToolFunction, ToolResult, non_empty,
};

use crate::{
    agent::prompt_request::{TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER, tool_result_message},
    completion::{CompletionError, Message, Usage},
    json_utils,
    streaming::{StreamedAssistantContent, ToolCallDeltaContent},
};

/// Assemble assistant content in canonical replay order: reasoning blocks,
/// then text, then trailing items (tool calls, images). Maps its inputs 1:1,
/// so the result is empty exactly when every input is.
pub(crate) fn ordered_assistant_content(
    reasoning_items: impl IntoIterator<Item = Reasoning>,
    text_items: impl IntoIterator<Item = AssistantContent>,
    trailing_items: impl IntoIterator<Item = AssistantContent>,
) -> Vec<AssistantContent> {
    let mut content_items = reasoning_items
        .into_iter()
        .map(AssistantContent::Reasoning)
        .collect::<Vec<_>>();
    content_items.extend(text_items);
    content_items.extend(trailing_items);
    content_items
}

/// [`ordered_assistant_content`], as an `Option` for slots where an empty
/// assembly means "no message".
pub(crate) fn ordered_streaming_assistant_content(
    reasoning_items: impl IntoIterator<Item = Reasoning>,
    text_items: impl IntoIterator<Item = AssistantContent>,
    trailing_items: impl IntoIterator<Item = AssistantContent>,
) -> Option<Vec<AssistantContent>> {
    non_empty(ordered_assistant_content(
        reasoning_items,
        text_items,
        trailing_items,
    ))
}

/// Whether a [`StreamedAssistantContent::Unknown`] payload is rig assistant
/// content, so excluding it from assembly loses transcript content.
///
/// The predicate is the decoder itself — a payload that parses as a tagged
/// [`AssistantContent`] block (`toolcall`/`reasoning`/`image` today, every
/// future variant automatically) is a replayed assistant block, not a
/// stream-item shape: the untagged stream variants carry different keys, so
/// it lands in `Unknown` and its content would silently vanish. A dropped
/// tool call additionally desyncs the turn — no pending call, no result.
///
/// Well-formed text does not reach this path: the tolerant block decode
/// ignores unknown keys, so a tagged text block or a text item with stray
/// sibling keys (0.41's flatten shape) decodes as
/// `StreamedAssistantContent::Text` and its text is *assembled*, with only
/// the stray keys dropped. The one way a text-carrying item can still land
/// in `Unknown` is a *malformed known field* — a non-object
/// `additional_params` fails the strict decode — and that item carries real
/// text, so it counts too. Anything else in `Unknown` is a provider-native
/// unmodeled item and stays quiet.
///
/// The whole outcome space is pinned by the decode-outcome matrix test
/// (`decode_outcome_matrix_is_total_and_no_shape_is_silent`): assembled,
/// excluded-and-counted, or excluded-quiet — no shape is silent.
fn unknown_payload_loses_assistant_content(payload: &serde_json::Value) -> bool {
    // `&Value` is itself a `Deserializer`, so the probe allocates nothing —
    // this runs on every `Unknown` item, and provider-native payloads can be
    // large and frequent.
    if AssistantContent::deserialize(payload).is_ok() {
        return true;
    }
    // A string `text` alongside an `additional_params` key: a text item
    // whose params were malformed enough to fail even the tolerant decode.
    // Its text is real transcript content.
    payload
        .get("text")
        .is_some_and(serde_json::Value::is_string)
        && payload.get("additional_params").is_some()
}

pub(crate) fn assistant_text_items_from_choice(
    choice: &[AssistantContent],
) -> Vec<AssistantContent> {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => (!text.text.is_empty()
                || text.additional_params.is_some())
            .then(|| AssistantContent::Text(text.clone())),
            _ => None,
        })
        .collect()
}

/// One invalid tool call surfaced mid-stream, awaiting a resolution from
/// [`AgentRun::resolve_streamed_invalid_tool_call`](super::AgentRun::resolve_streamed_invalid_tool_call).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StreamedInvalidToolCall {
    /// The rejected tool call. For a name delta this is a diagnostic call
    /// assembled from the streamed name and any buffered argument deltas.
    pub tool_call: ToolCall,
    /// Rig-generated identifier correlating this call's stream items.
    pub internal_call_id: String,
    /// Raw argument payload for diagnostics, when available.
    pub args: Option<String>,
    /// Executable Rig tools advertised to the provider for this turn.
    pub executable_tool_names: BTreeSet<String>,
    /// Tools allowed by the active tool choice for this turn.
    pub allowed_tool_names: BTreeSet<String>,
}

/// Snapshot of a streamed turn at the moment an invalid tool call appeared.
/// Used by the machine to build diagnostics and rollback messages from
/// exactly what the model has produced so far.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PartialStreamedTurn {
    /// Provider-assigned assistant message ID, when already known.
    pub message_id: Option<String>,
    /// Aggregated assistant text, when any text was streamed this turn.
    pub text: Option<String>,
    /// Accumulated reasoning, with any pending unsigned delta text assembled
    /// into a block.
    pub reasoning: Vec<Reasoning>,
    /// Tool calls already validated (or repaired) this turn.
    pub pending_tool_calls: Vec<ToolCall>,
}

impl PartialStreamedTurn {
    /// The assistant message representing this partial turn, in canonical
    /// order, including `current_tool_call` when provided. `None` when the
    /// turn has produced no representable content.
    pub(crate) fn assistant_message(&self, current_tool_call: Option<ToolCall>) -> Option<Message> {
        let text_items = match &self.text {
            Some(text) if !text.is_empty() => vec![AssistantContent::text(text.clone())],
            _ => Vec::new(),
        };
        let mut tool_items = self
            .pending_tool_calls
            .iter()
            .cloned()
            .map(AssistantContent::ToolCall)
            .collect::<Vec<_>>();
        if let Some(tool_call) = current_tool_call {
            tool_items.push(AssistantContent::ToolCall(tool_call));
        }

        let content = ordered_streaming_assistant_content(
            self.reasoning.iter().cloned(),
            text_items,
            tool_items,
        )?;
        Some(Message::Assistant {
            id: self.message_id.clone(),
            content,
        })
    }

    /// Rollback messages for a retried or skipped streamed turn: the partial
    /// assistant turn plus a user message carrying `feedback` for the invalid
    /// call and a synthetic "not executed" result for each validated peer.
    pub(crate) fn rollback_messages(
        &self,
        invalid_tool_call: ToolCall,
        feedback: String,
    ) -> Option<(Message, Message)> {
        // Every call — the invalid one and each validated peer — already
        // carries a unique, non-empty `ToolCallId` (minted at the provider
        // boundary when the wire issued none), so both sides of this
        // fabricated transcript pair correlate by id with no local minting
        // and no peer left holding an empty sentinel.
        let assistant_message = self.assistant_message(Some(invalid_tool_call.clone()))?;

        let mut retry_results = self
            .pending_tool_calls
            .iter()
            .map(|tool_call| {
                tool_result_message(
                    tool_call.id.clone(),
                    tool_call.provider.clone(),
                    tool_call.function.name.clone(),
                    TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER.to_string(),
                )
            })
            .collect::<Vec<_>>();
        retry_results.push(tool_result_message(
            invalid_tool_call.id,
            invalid_tool_call.provider,
            invalid_tool_call.function.name,
            feedback,
        ));

        // `retry_results` is non-empty: the invalid call's own feedback result
        // was just pushed unconditionally.
        let user_message = Message::User {
            content: retry_results,
        };

        Some((assistant_message, user_message))
    }
}

/// The assembled streamed turn, fed to
/// [`AgentRun::streamed_turn`](super::AgentRun::streamed_turn).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StreamedTurn {
    /// Provider-assigned assistant message ID, when available.
    pub message_id: Option<String>,
    /// The assistant content to record in history: canonical
    /// (reasoning → text → tool calls) when the turn produced reasoning or
    /// tool calls, otherwise the provider's aggregated choice as-is.
    pub choice: Vec<AssistantContent>,
    /// Executable Rig tools advertised to the provider for this turn.
    pub executable_tool_names: BTreeSet<String>,
    /// Tools allowed by the active tool choice for this turn.
    pub allowed_tool_names: BTreeSet<String>,
    /// `(tool_call_id, internal_call_id)` pairs for this turn's tool calls,
    /// in emission order. Carried into the run state so a resumed process
    /// keeps the IDs consumers already saw in tool-call deltas.
    #[serde(default)]
    pub internal_call_ids: Vec<(String, String)>,
    /// Why the provider stopped generating this turn, when it reported a
    /// reason — the streamed analogue of [`ModelTurn::finish_reason`], so a
    /// driver that feeds turns through `streamed_turn` records the same
    /// terminal reason the blocking surface does (rig#2322).
    ///
    /// [`ModelTurn::finish_reason`]: super::ModelTurn::finish_reason
    #[serde(default)]
    pub finish_reason: Option<FinishReason>,
}

/// What the machine decided about a mid-stream invalid tool call.
///
/// Deliberately exhaustive: a driver must handle every resolution, so adding
/// a variant is a breaking change by design.
#[derive(Debug)]
pub enum StreamedResolution {
    /// The tool name was repaired. Apply it via
    /// [`StreamedTurnAssembler::resolve_pending_invalid`] and keep consuming
    /// the provider stream.
    Repaired {
        /// The validated replacement tool name.
        tool_name: String,
    },
    /// The turn was rolled back (retry) or the call skipped; corrective
    /// messages are already in the history. Drain the provider stream for
    /// usage, record the completion call, then call
    /// [`AgentRun::next_step`](super::AgentRun::next_step).
    TurnAbandoned {
        /// For a skipped call, the synthetic tool result to surface to the
        /// consumer stream. Boxed: the result dwarfs the other variant.
        skipped_tool_result: Option<Box<ToolResult>>,
    },
}

/// What a driver must do with one ingested stream item.
///
/// Deliberately exhaustive: a driver must handle every event, so adding a
/// variant is a breaking change by design.
#[derive(Debug, Clone)]
pub enum StreamedTurnEvent {
    /// Forward the ingested item to the consumer as-is (text, reasoning, or
    /// reasoning deltas, after accumulation).
    EmitIngested,
    /// Forward this tool-call delta. Argument deltas buffered while the tool
    /// name awaited validation are replayed through this event.
    EmitToolCallDelta {
        /// Rig-generated identifier correlating this call's stream items.
        internal_call_id: String,
        /// The (possibly repaired) name or argument delta.
        content: ToolCallDeltaContent,
    },
    /// The model emitted an unknown or disallowed tool call. Resolve it via
    /// [`AgentRun::resolve_streamed_invalid_tool_call`](super::AgentRun::resolve_streamed_invalid_tool_call),
    /// then apply the outcome with
    /// [`StreamedTurnAssembler::resolve_pending_invalid`].
    InvalidToolCall(Box<StreamedInvalidToolCall>),
    /// The provider supplied its typed final payload. Record its usage (see
    /// [`AgentRun::record_streamed_completion_call`](super::AgentRun::record_streamed_completion_call));
    /// this does not establish that the provider stream reached EOF. When
    /// `emit_final` is set, the turn streamed text and the driver should buffer
    /// the final item until EOF finalizes the turn.
    Completed {
        /// Provider-reported usage for this call. Zero-valued usage means the
        /// provider reported no usage metrics.
        usage: Usage,
        /// Whether the ingested final item should be forwarded to the
        /// consumer (set when the turn streamed text).
        emit_final: bool,
        /// Why the provider stopped generating, when it reported a reason.
        ///
        /// Previously dropped here: the assembler read `usage` and `saw_text`
        /// off the terminal record and discarded the rest, so a turn truncated
        /// at the output-token limit reached the driver indistinguishable from
        /// one that simply stopped (rig#2322).
        finish_reason: Option<FinishReason>,
    },
}

#[derive(Default)]
struct ToolCallDeltaState {
    name_validated: bool,
    buffered_arguments: Vec<String>,
}

/// One reasoning part of the turn, in first-arrival order. A part opens as
/// delta text keyed by the stream's rig-generated correlator and is
/// superseded in place when a completed block restating the same part
/// arrives; a completed block matching no open part occupies its own slot.
struct ReasoningPart {
    correlator: Option<String>,
    provider_id: Option<String>,
    state: ReasoningPartState,
}

#[derive(Clone)]
enum ReasoningPartState {
    /// Delta text accumulated so far for a part with no completed block.
    Pending(String),
    /// The authoritative completed block (may carry signatures or encrypted
    /// content the deltas lacked).
    Completed(Reasoning),
}

/// Assemble one part's reasoning: a completed block as-is, a non-empty pending
/// delta buffer as its own block carrying only the part's provider-issued id.
fn reasoning_from_part(
    state: ReasoningPartState,
    provider_id: Option<String>,
) -> Option<Reasoning> {
    match state {
        ReasoningPartState::Completed(reasoning) => Some(reasoning),
        ReasoningPartState::Pending(text) if !text.is_empty() => {
            let mut assembled = Reasoning::new(&text);
            if let Some(id) = provider_id {
                assembled = assembled.with_id(id);
            }
            Some(assembled)
        }
        ReasoningPartState::Pending(_) => None,
    }
}

enum PendingInvalid {
    /// A complete tool call with a disallowed name.
    FullCall {
        tool_call: Box<ToolCall>,
        internal_call_id: String,
    },
    /// A streamed tool-name delta with a disallowed name.
    NameDelta { internal_call_id: String },
}

/// Sans-IO accumulator that assembles one streamed model turn. See the
/// [module docs](self) for the driving protocol.
pub struct StreamedTurnAssembler {
    executable_tool_names: BTreeSet<String>,
    allowed_tool_names: BTreeSet<String>,
    text: String,
    saw_text: bool,
    reasoning_parts: Vec<ReasoningPart>,
    pending_tool_calls: Vec<(ToolCall, String)>,
    delta_states: HashMap<String, ToolCallDeltaState>,
    pending_invalid: Option<PendingInvalid>,
    /// Terminal reason from this turn's provider final record, retained so
    /// [`Self::finish`] can carry it onto the [`StreamedTurn`] (rig#2322).
    finish_reason: Option<FinishReason>,
    /// Replayed assistant blocks excluded from assembly this turn (see
    /// [`unknown_payload_loses_assistant_content`]): counted per item,
    /// surfaced as one warning when the guard drops.
    excluded_assistant_content: ExclusionCount,
}

/// Count of replayed assistant blocks excluded from assembly in one turn.
///
/// The loudness contract lives on this guard's `Drop`, so it holds on
/// *every* termination path — `finish`, stream errors, hook cancellation,
/// abandonment, truncation — exactly once, and zero exclusions stay silent.
/// A dedicated one-field guard (not a `Drop` impl on the assembler itself)
/// keeps the assembler's fields freely movable.
#[derive(Default)]
struct ExclusionCount(usize);

impl Drop for ExclusionCount {
    fn drop(&mut self) {
        if self.0 > 0 {
            tracing::warn!(
                excluded = self.0,
                "stream items matching rig's tagged assistant-content \
                 serialization were excluded from the assembled assistant \
                 message — replayed assistant blocks are not stream-item \
                 shapes, and their content is lost from assembled history"
            );
        }
    }
}

impl StreamedTurnAssembler {
    /// Create an assembler for one streamed turn with the tool names
    /// advertised to the provider for that turn.
    pub fn new(
        executable_tool_names: BTreeSet<String>,
        allowed_tool_names: BTreeSet<String>,
    ) -> Self {
        Self {
            executable_tool_names,
            allowed_tool_names,
            text: String::new(),
            saw_text: false,
            reasoning_parts: Vec::new(),
            pending_tool_calls: Vec::new(),
            delta_states: HashMap::new(),
            pending_invalid: None,
            finish_reason: None,
            excluded_assistant_content: ExclusionCount::default(),
        }
    }

    /// Replayed assistant blocks excluded from assembly so far this turn.
    /// Zero on well-formed provider streams; non-zero means transcript
    /// content was lost (one warning summarizes the count at
    /// [`Self::finish`]).
    pub fn excluded_assistant_content(&self) -> usize {
        self.excluded_assistant_content.0
    }

    /// Aggregated assistant text streamed so far this turn (empty until the
    /// first text delta).
    pub fn aggregated_text(&self) -> &str {
        &self.text
    }

    /// Reasoning text accumulated for the currently pending part identified by
    /// `correlator`.
    ///
    /// Completed parts are deliberately skipped: a later delta may reuse a
    /// correlator after a completed restatement, in which case ingestion opens
    /// a new pending part and this returns that new part's aggregate.
    pub fn aggregated_reasoning(&self, correlator: &str) -> Option<&str> {
        self.reasoning_parts.iter().find_map(|part| {
            match (&part.state, part.correlator.as_deref()) {
                (ReasoningPartState::Pending(text), Some(id)) if id == correlator => {
                    Some(text.as_str())
                }
                _ => None,
            }
        })
    }

    /// Normalize the provider aggregate into the content committed for this
    /// turn. The reasoning is supplied by the caller: the finish path drains
    /// its parts by value ([`Self::drain_reasoning`]) instead of cloning
    /// them, while the partial-turn surface assembles borrowed
    /// ([`Self::assembled_reasoning`]) — agreement between the two is pinned
    /// by `canonical_choice_and_partial_turn_agree_on_multi_part_reasoning`.
    fn canonical_choice_with(
        &self,
        reasoning: Vec<Reasoning>,
        provider_choice: &[AssistantContent],
    ) -> Vec<AssistantContent> {
        if !self.pending_tool_calls.is_empty() || !reasoning.is_empty() {
            let text_items = assistant_text_items_from_choice(provider_choice);
            let tool_items = self
                .pending_tool_calls
                .iter()
                .map(|(tool_call, _)| AssistantContent::ToolCall(tool_call.clone()))
                .collect::<Vec<_>>();
            // Infallible on purpose: the enclosing guard makes at least one
            // input non-empty and the assembly maps its inputs 1:1, so there
            // is no empty case to fall back from.
            ordered_assistant_content(reasoning, text_items, tool_items)
        } else {
            provider_choice.to_vec()
        }
    }

    /// Record a completed reasoning block. It supersedes the same part —
    /// matched by the stream correlator first, regardless of whether that
    /// part is still pending or already completed (the stream restates one
    /// correlator per part, so a later same-correlator completion is the
    /// same part's authoritative whole, e.g. a signed restatement after an
    /// unsigned close), then by the durable provider id for pending parts —
    /// because the completed block restates the delta text plus payloads
    /// (signatures, encrypted content) the deltas lacked. A block matching
    /// no part by correlator merges with an earlier completed block sharing
    /// its provider id, else occupies a new slot; unmatched pending buffers
    /// are never dropped (a delta-only visible part and a completed
    /// encrypted block can coexist in one stream).
    fn ingest_completed_reasoning(&mut self, reasoning: &Reasoning, correlator: &str) {
        // An exact correlator match IS the part, whatever its state:
        // replace wholesale (pydantic-ai's replace-part semantics — the
        // completed block always carries the whole content, the
        // accumulator having merged signatures before yielding). Checked
        // before the provider-id fallbacks so a signed restatement can
        // never double-extend its own part. Failing that, the block
        // supersedes a pending part sharing its durable provider id.
        let replace_at = self
            .reasoning_parts
            .iter()
            .position(|part| part.correlator.as_deref() == Some(correlator))
            .or_else(|| {
                self.reasoning_parts.iter().position(|part| {
                    matches!(part.state, ReasoningPartState::Pending(_))
                        && matches!(
                            (&part.provider_id, &reasoning.id),
                            (Some(pending_id), Some(incoming_id)) if pending_id == incoming_id
                        )
                })
            });
        if let Some(part) = replace_at.and_then(|index| self.reasoning_parts.get_mut(index)) {
            if reasoning.id.is_some() {
                part.provider_id = reasoning.id.clone();
            }
            part.state = ReasoningPartState::Completed(reasoning.clone());
            return;
        }

        // Completed blocks sharing a provider-issued id extend one
        // another (the multi-part same-id reasoning item shape).
        let extends = self.reasoning_parts.iter_mut().rev().find(|part| {
            matches!(part.state, ReasoningPartState::Completed(_))
                && matches!(
                    (&part.provider_id, &reasoning.id),
                    (Some(existing_id), Some(incoming_id)) if existing_id == incoming_id
                )
        });
        if let Some(part) = extends {
            if let ReasoningPartState::Completed(existing) = &mut part.state {
                existing.content.extend(reasoning.content.clone());
            }
            return;
        }

        self.reasoning_parts.push(ReasoningPart {
            correlator: Some(correlator.to_owned()),
            provider_id: reasoning.id.clone(),
            state: ReasoningPartState::Completed(reasoning.clone()),
        });
    }

    /// The turn's reasoning in first-arrival order: completed blocks as-is,
    /// non-empty pending delta buffers each assembled into their own block
    /// carrying only the part's provider-issued id.
    fn assembled_reasoning(&self) -> Vec<Reasoning> {
        self.reasoning_parts
            .iter()
            .filter_map(|part| reasoning_from_part(part.state.clone(), part.provider_id.clone()))
            .collect()
    }

    /// [`Self::assembled_reasoning`], consuming the parts — the finish path
    /// owns the assembler, and reasoning blocks can carry large encrypted
    /// payloads that should move rather than clone.
    fn drain_reasoning(&mut self) -> Vec<Reasoning> {
        std::mem::take(&mut self.reasoning_parts)
            .into_iter()
            .filter_map(|part| reasoning_from_part(part.state, part.provider_id))
            .collect()
    }

    /// Ingest one provider stream item and return what the driver must do.
    ///
    /// # Errors
    /// Returns an error when the provider stream is inconsistent (argument
    /// deltas finishing without a validated tool name) or when an invalid
    /// tool call is still awaiting resolution.
    pub fn ingest(
        &mut self,
        item: &StreamedAssistantContent,
    ) -> Result<Vec<StreamedTurnEvent>, CompletionError> {
        if self.pending_invalid.is_some() {
            return Err(CompletionError::ResponseError(
                "streamed turn ingested while an invalid tool call awaits resolution".to_string(),
            ));
        }

        match item {
            StreamedAssistantContent::Text(text) => {
                if !self.saw_text {
                    self.text.clear();
                    self.saw_text = true;
                }
                self.text.push_str(&text.text);
                Ok(vec![StreamedTurnEvent::EmitIngested])
            }
            StreamedAssistantContent::Reasoning { reasoning, id } => {
                self.ingest_completed_reasoning(reasoning, id);
                Ok(vec![StreamedTurnEvent::EmitIngested])
            }
            StreamedAssistantContent::ReasoningDelta {
                id,
                reasoning,
                provider_id,
            } => {
                // Deltas lack signatures/encrypted content that full blocks
                // carry; mixing them into completed reasoning causes
                // providers like Anthropic to reject with "signature required",
                // so each part's text is kept aside, keyed by the stream
                // correlator, until its completed block (if any) supersedes
                // it. Only the provider-issued id may become the assembled
                // block's durable id — the public correlator is rig-generated
                // and must never enter history.
                let index = self
                    .reasoning_parts
                    .iter()
                    .position(|part| {
                        part.correlator.as_deref() == Some(id.as_str())
                            && matches!(part.state, ReasoningPartState::Pending(_))
                    })
                    .unwrap_or_else(|| {
                        self.reasoning_parts.push(ReasoningPart {
                            correlator: Some(id.clone()),
                            provider_id: None,
                            state: ReasoningPartState::Pending(String::new()),
                        });
                        self.reasoning_parts.len() - 1
                    });
                if let Some(part) = self.reasoning_parts.get_mut(index) {
                    if let ReasoningPartState::Pending(text) = &mut part.state {
                        text.push_str(reasoning);
                    }
                    if part.provider_id.is_none() {
                        part.provider_id = provider_id.clone();
                    }
                }
                Ok(vec![StreamedTurnEvent::EmitIngested])
            }
            StreamedAssistantContent::ToolCall {
                tool_call,
                internal_call_id,
            } => {
                if !self.allowed_tool_names.contains(&tool_call.function.name) {
                    return Ok(self.surface_invalid_call(
                        tool_call.clone(),
                        internal_call_id.clone(),
                        Some(json_utils::serialize_json_value(
                            &tool_call.function.arguments,
                        )),
                        PendingInvalid::FullCall {
                            tool_call: Box::new(tool_call.clone()),
                            internal_call_id: internal_call_id.clone(),
                        },
                    ));
                }

                self.pending_tool_calls
                    .push((tool_call.clone(), internal_call_id.clone()));
                Ok(Vec::new())
            }
            StreamedAssistantContent::ToolCallDelta {
                internal_call_id,
                content,
            } => {
                let key = internal_call_id.clone();
                match content {
                    ToolCallDeltaContent::Name(name) => {
                        if !self.allowed_tool_names.contains(name) {
                            let buffered_args = self
                                .delta_states
                                .get(&key)
                                .map(|state| state.buffered_arguments.join(""))
                                .unwrap_or_default();
                            let tool_call =
                                self.name_delta_diagnostic_tool_call(name, &buffered_args);
                            return Ok(self.surface_invalid_call(
                                tool_call,
                                internal_call_id.clone(),
                                Some(buffered_args),
                                PendingInvalid::NameDelta {
                                    internal_call_id: internal_call_id.clone(),
                                },
                            ));
                        }

                        Ok(self.validate_delta_name(&key, name.clone()))
                    }
                    ToolCallDeltaContent::Delta(arguments) => {
                        let state = self.delta_states.entry(key.clone()).or_default();
                        if state.name_validated {
                            Ok(vec![StreamedTurnEvent::EmitToolCallDelta {
                                internal_call_id: internal_call_id.clone(),
                                content: ToolCallDeltaContent::Delta(arguments.clone()),
                            }])
                        } else {
                            state.buffered_arguments.push(arguments.clone());
                            Ok(Vec::new())
                        }
                    }
                }
            }
            StreamedAssistantContent::Final(final_response) => {
                if let Some(err) = self.pending_delta_error() {
                    return Err(err);
                }

                let usage = final_response.usage;
                let emit_final = self.saw_text;
                self.saw_text = false;
                // `normalize_stream` has already reconciled this against the
                // tool calls actually seen (see `StreamFinal::finish_reason`),
                // so it is consumed as-is and never re-reconciled here.
                let finish_reason = final_response.finish_reason.clone();
                self.finish_reason = finish_reason.clone();
                Ok(vec![StreamedTurnEvent::Completed {
                    usage,
                    emit_final,
                    finish_reason,
                }])
            }
            StreamedAssistantContent::Unknown(payload) => {
                // Unmodeled provider item (e.g. a hosted-tool result): forward it
                // to the consumer but do not fold it into the accumulated
                // assistant message — there is no `AssistantContent::Unknown`, and
                // it must not perturb text/tool-call/reasoning accumulation.
                //
                // The exclusion loses transcript content when the payload is
                // rig assistant content (a replayed tagged block, not a
                // stream-item shape). Counted here — text deltas arrive
                // per-token, so per-item warns could flood the log — and
                // surfaced as one warning at turn end; the payload itself
                // stays redacted.
                if unknown_payload_loses_assistant_content(payload.value()) {
                    self.excluded_assistant_content.0 += 1;
                    tracing::debug!(
                        excluded = self.excluded_assistant_content.0,
                        "stream item is a replayed assistant block, not a \
                         stream-item shape; excluded from assembly"
                    );
                }
                Ok(vec![StreamedTurnEvent::EmitIngested])
            }
        }
    }

    /// Apply the machine's resolution for the invalid tool call surfaced by
    /// the last [`StreamedTurnEvent::InvalidToolCall`]. For a repaired name
    /// this returns the deltas to forward (the repaired name plus any
    /// buffered argument deltas).
    pub fn resolve_pending_invalid(
        &mut self,
        resolution: &StreamedResolution,
    ) -> Vec<StreamedTurnEvent> {
        let Some(pending) = self.pending_invalid.take() else {
            return Vec::new();
        };

        match (resolution, pending) {
            (
                StreamedResolution::Repaired { tool_name },
                PendingInvalid::FullCall {
                    mut tool_call,
                    internal_call_id,
                },
            ) => {
                tool_call.function.name = tool_name.clone();
                self.pending_tool_calls.push((*tool_call, internal_call_id));
                Vec::new()
            }
            (
                StreamedResolution::Repaired { tool_name },
                PendingInvalid::NameDelta { internal_call_id },
            ) => self.validate_delta_name(&internal_call_id, tool_name.clone()),
            (
                StreamedResolution::TurnAbandoned { .. },
                PendingInvalid::NameDelta { internal_call_id },
            ) => {
                // The abandoned call's buffered state must not trip the
                // pending-delta consistency check while usage is drained.
                self.delta_states.remove(&internal_call_id);
                Vec::new()
            }
            (StreamedResolution::TurnAbandoned { .. }, PendingInvalid::FullCall { .. }) => {
                Vec::new()
            }
        }
    }

    /// Error when argument deltas were buffered for a tool call whose name
    /// never validated — a provider-stream consistency violation.
    pub fn pending_delta_error(&self) -> Option<CompletionError> {
        self.delta_states
            .iter()
            .find(|(_, state)| !state.name_validated && !state.buffered_arguments.is_empty())
            .map(|(internal_call_id, state)| {
                CompletionError::ResponseError(format!(
                    "streamed tool call arguments received before a validated tool name for internal_call_id `{internal_call_id}` ({} buffered argument delta(s))",
                    state.buffered_arguments.len()
                ))
            })
    }

    /// Snapshot of the turn so far, for diagnostics and rollback messages.
    pub fn partial_turn(&self, message_id: Option<String>) -> PartialStreamedTurn {
        let reasoning = self.assembled_reasoning();

        PartialStreamedTurn {
            message_id,
            text: self.saw_text.then(|| self.text.clone()),
            reasoning,
            pending_tool_calls: self
                .pending_tool_calls
                .iter()
                .map(|(tool_call, _)| tool_call.clone())
                .collect(),
        }
    }

    /// Assemble the completed turn. `final_choice` is the provider's
    /// aggregated choice for the turn
    /// ([`crate::streaming::StreamingCompletionResponse::choice`]).
    pub fn finish(
        mut self,
        message_id: Option<String>,
        final_choice: &[AssistantContent],
    ) -> StreamedTurn {
        let reasoning = self.drain_reasoning();
        let choice = self.canonical_choice_with(reasoning, final_choice);
        let internal_call_ids: Vec<(String, String)> = self
            .pending_tool_calls
            .iter()
            .map(|(tool_call, internal_call_id)| {
                (tool_call.id.as_str().to_owned(), internal_call_id.clone())
            })
            .collect();

        StreamedTurn {
            message_id,
            choice,
            executable_tool_names: self.executable_tool_names,
            allowed_tool_names: self.allowed_tool_names,
            internal_call_ids,
            finish_reason: self.finish_reason.take(),
        }
    }

    /// Park resolution on `pending` and surface the rejected call to the
    /// caller as an [`StreamedTurnEvent::InvalidToolCall`].
    fn surface_invalid_call(
        &mut self,
        tool_call: ToolCall,
        internal_call_id: String,
        args: Option<String>,
        pending: PendingInvalid,
    ) -> Vec<StreamedTurnEvent> {
        let invalid = StreamedInvalidToolCall {
            tool_call,
            internal_call_id,
            args,
            executable_tool_names: self.executable_tool_names.clone(),
            allowed_tool_names: self.allowed_tool_names.clone(),
        };
        self.pending_invalid = Some(pending);
        vec![StreamedTurnEvent::InvalidToolCall(Box::new(invalid))]
    }

    fn name_delta_diagnostic_tool_call(&self, name: &str, buffered_args: &str) -> ToolCall {
        let diagnostic_args = if buffered_args.trim().is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_str(buffered_args).unwrap_or(serde_json::Value::Null)
        };
        // Diagnostic only: the durable provider id is unknown at delta
        // time, and no stream-internal key may surface — so the call mints
        // its correlation handle and `provider` stays `None` (hooks
        // faithfully observe that no provider id exists). The same minted
        // id correlates the retry transcript pair in `rollback_messages`.
        ToolCall::new(
            rig_core::message::ToolCallId::mint(),
            ToolFunction::new(name.to_string(), diagnostic_args),
        )
    }

    fn validate_delta_name(&mut self, key: &str, name: String) -> Vec<StreamedTurnEvent> {
        let state = self.delta_states.entry(key.to_owned()).or_default();
        state.name_validated = true;
        let buffered_arguments = std::mem::take(&mut state.buffered_arguments);

        let mut events = vec![StreamedTurnEvent::EmitToolCallDelta {
            internal_call_id: key.to_owned(),
            content: ToolCallDeltaContent::Name(name),
        }];
        events.extend(buffered_arguments.into_iter().map(|arguments| {
            StreamedTurnEvent::EmitToolCallDelta {
                internal_call_id: key.to_owned(),
                content: ToolCallDeltaContent::Delta(arguments),
            }
        }));
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::hook::InvalidToolCallAction;
    use crate::agent::run::{AgentRun, AgentRunStep};
    use crate::completion::PromptError;
    use crate::test_utils::mock_final;
    use rig_core::message::{Text, ToolResultContent, UserContent};
    use serde_json::json;

    fn tool_names(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|name| (*name).to_string()).collect()
    }

    fn assembler() -> StreamedTurnAssembler {
        StreamedTurnAssembler::new(tool_names(&["add"]), tool_names(&["add"]))
    }

    fn text_item(text: &str) -> StreamedAssistantContent {
        StreamedAssistantContent::Text(Text::new(text.to_string()))
    }

    fn tool_call(id: &str, name: &str) -> ToolCall {
        // The provider-boundary shape: the wire id becomes both the durable
        // id and the provider correlator.
        ToolCall::from_wire(id, ToolFunction::new(name.to_string(), json!({"x": 1})))
    }

    fn tool_call_item(id: &str, name: &str) -> StreamedAssistantContent {
        StreamedAssistantContent::ToolCall {
            tool_call: tool_call(id, name),
            internal_call_id: format!("internal_{id}"),
        }
    }

    fn final_item() -> StreamedAssistantContent {
        StreamedAssistantContent::Final(mock_final(Usage::new()))
    }

    fn name_delta(id: &str, name: &str) -> StreamedAssistantContent {
        StreamedAssistantContent::ToolCallDelta {
            internal_call_id: format!("internal_{id}"),
            content: ToolCallDeltaContent::Name(name.to_string()),
        }
    }

    fn args_delta(id: &str, arguments: &str) -> StreamedAssistantContent {
        StreamedAssistantContent::ToolCallDelta {
            internal_call_id: format!("internal_{id}"),
            content: ToolCallDeltaContent::Delta(arguments.to_string()),
        }
    }

    fn expect_invalid(events: Vec<StreamedTurnEvent>) -> StreamedInvalidToolCall {
        match events.into_iter().next() {
            Some(StreamedTurnEvent::InvalidToolCall(invalid)) => *invalid,
            other => panic!("expected InvalidToolCall, got {other:?}"),
        }
    }

    #[test]
    fn text_accumulates_and_emits() {
        let mut asm = assembler();
        let events = asm
            .ingest(&text_item("hel"))
            .expect("ingest should succeed");
        assert!(matches!(
            events.as_slice(),
            [StreamedTurnEvent::EmitIngested]
        ));
        asm.ingest(&text_item("lo")).expect("ingest should succeed");
        assert_eq!(asm.aggregated_text(), "hello");
    }

    #[test]
    fn unknown_item_emits_to_consumer_without_touching_accumulation() {
        let mut asm = assembler();
        asm.ingest(&text_item("answer"))
            .expect("ingest text should succeed");

        let events = asm
            .ingest(&StreamedAssistantContent::Unknown(
                json!({ "type": "web_search_call", "id": "ws_1" }).into(),
            ))
            .expect("ingest unknown should succeed");

        // The unmodeled item is forwarded to the consumer ...
        assert!(matches!(
            events.as_slice(),
            [StreamedTurnEvent::EmitIngested]
        ));
        // ... but perturbs no accumulation state used to build the assistant message.
        assert_eq!(asm.aggregated_text(), "answer");
    }

    /// The decode-outcome contract, as a total matrix: every stream-item
    /// payload has exactly one of three outcomes — assembled,
    /// excluded-and-counted (one warning at turn end), or excluded-quiet
    /// (provider-native unmodeled) — and no shape is silent. `expected` is a
    /// wildcard-free match, so a new shape class cannot compile without a
    /// mandated outcome, and the coverage assert below fails until it also
    /// has a fixture.
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum ShapeClass {
        WellFormedText,
        UnknownKeyedText,
        TaggedText,
        TaggedRigBlock,
        MalformedParamsText,
        /// A provider-native frame that happens to carry a string `text`
        /// key (e.g. an annotation event). Tolerance folds its text into
        /// the message — the documented noise tradeoff: never losing real
        /// text outranks occasionally ingesting a frame's caption.
        ProviderNativeTextCarrying,
        ProviderNativeUnmodeled,
    }

    #[derive(Debug, PartialEq)]
    enum ExpectedOutcome {
        Assembled { text: &'static str },
        ExcludedAndCounted,
        ExcludedQuiet,
    }

    /// The matrix's outcome column. No wildcard arm — the compiler is the
    /// missing-cell error.
    fn expected(shape: ShapeClass) -> ExpectedOutcome {
        match shape {
            ShapeClass::WellFormedText
            | ShapeClass::UnknownKeyedText
            | ShapeClass::TaggedText
            | ShapeClass::ProviderNativeTextCarrying => ExpectedOutcome::Assembled { text: "hi" },
            ShapeClass::TaggedRigBlock | ShapeClass::MalformedParamsText => {
                ExpectedOutcome::ExcludedAndCounted
            }
            ShapeClass::ProviderNativeUnmodeled => ExpectedOutcome::ExcludedQuiet,
        }
    }

    /// The matrix's fixture rows. Every shape class appears at least once
    /// (pinned by the coverage assert in the test); classes with several
    /// wire spellings carry one fixture per spelling.
    fn decode_matrix_cases() -> Vec<(ShapeClass, serde_json::Value)> {
        vec![
            (ShapeClass::WellFormedText, json!({"text": "hi"})),
            (
                ShapeClass::UnknownKeyedText,
                json!({"text": "hi", "citations": ["stray"], "future": 1}),
            ),
            (
                ShapeClass::TaggedText,
                json!({"type": "text", "text": "hi"}),
            ),
            (
                ShapeClass::TaggedRigBlock,
                json!({"type": "toolcall", "id": "call_1",
                       "function": {"name": "add", "arguments": {}}}),
            ),
            (
                ShapeClass::TaggedRigBlock,
                json!({"type": "reasoning", "id": null, "content": []}),
            ),
            (
                ShapeClass::TaggedRigBlock,
                json!({"type": "image", "data": {"type": "base64", "value": "aGk="}}),
            ),
            (
                ShapeClass::MalformedParamsText,
                json!({"text": "hi", "additional_params": []}),
            ),
            (
                ShapeClass::MalformedParamsText,
                json!({"type": "text", "text": "hi", "additional_params": []}),
            ),
            (
                ShapeClass::ProviderNativeUnmodeled,
                json!({"type": "web_search_call", "id": "ws_1"}),
            ),
            (
                ShapeClass::ProviderNativeTextCarrying,
                json!({"type": "output_text.annotation", "text": "hi"}),
            ),
            (ShapeClass::ProviderNativeUnmodeled, json!({"text": 42})),
        ]
    }

    #[test]
    fn decode_outcome_matrix_is_total_and_no_shape_is_silent() {
        let cases = decode_matrix_cases();
        // Vacuity floor: an emptied fixture table must fail loudly, not
        // pass by checking nothing.
        assert!(!cases.is_empty(), "decode_matrix_cases returned no rows");
        // Coverage: every shape class has at least one fixture. Extend
        // `witnesses` (and `decode_matrix_cases`) when adding a variant —
        // `expected` already refuses to compile without a classification.
        let witnesses = [
            ShapeClass::WellFormedText,
            ShapeClass::UnknownKeyedText,
            ShapeClass::TaggedText,
            ShapeClass::TaggedRigBlock,
            ShapeClass::MalformedParamsText,
            ShapeClass::ProviderNativeTextCarrying,
            ShapeClass::ProviderNativeUnmodeled,
        ];
        for shape in witnesses {
            assert!(
                cases.iter().any(|(case_shape, _)| *case_shape == shape),
                "no fixture for {shape:?} — add a row to decode_matrix_cases"
            );
        }

        for (shape, payload) in cases {
            let item = serde_json::from_value::<StreamedAssistantContent>(payload.clone())
                .expect("stream-item decode is tolerant and must not fail");
            let mut asm = assembler();
            match expected(shape) {
                ExpectedOutcome::Assembled { text } => {
                    assert!(
                        matches!(&item, StreamedAssistantContent::Text(t) if t.text == text),
                        "{shape:?} must decode as stream text: {payload}"
                    );
                    asm.ingest(&item).expect("ingest");
                    assert_eq!(asm.aggregated_text(), text, "{shape:?}: {payload}");
                    assert_eq!(
                        asm.excluded_assistant_content(),
                        0,
                        "{shape:?} must not count as excluded: {payload}"
                    );
                }
                ExpectedOutcome::ExcludedAndCounted => {
                    assert!(
                        matches!(&item, StreamedAssistantContent::Unknown(_)),
                        "{shape:?} must decode Unknown: {payload}"
                    );
                    asm.ingest(&item).expect("ingest");
                    assert_eq!(asm.aggregated_text(), "", "{shape:?}: {payload}");
                    assert_eq!(
                        asm.excluded_assistant_content(),
                        1,
                        "{shape:?} loses assistant content and must be counted: {payload}"
                    );
                }
                ExpectedOutcome::ExcludedQuiet => {
                    assert!(
                        matches!(&item, StreamedAssistantContent::Unknown(_)),
                        "{shape:?} must decode Unknown: {payload}"
                    );
                    asm.ingest(&item).expect("ingest");
                    assert_eq!(asm.aggregated_text(), "", "{shape:?}: {payload}");
                    assert_eq!(
                        asm.excluded_assistant_content(),
                        0,
                        "{shape:?} is provider-native and must stay quiet: {payload}"
                    );
                }
            }
        }
    }
    #[test]
    fn choice_text_items_judge_annotation_by_presence() {
        // `AdditionalParams` is non-empty by construction — an empty carrier
        // is unrepresentable (`try_from_value(json!({}))` yields `None`) —
        // so plain `is_some()` is the whole annotation rule and live and
        // restored classification agree by type.
        let unannotated = AssistantContent::Text(Text {
            text: String::new(),
            additional_params: rig_core::message::AdditionalParams::try_from_value(json!({}))
                .expect("object params"),
        });
        assert!(assistant_text_items_from_choice(&[unannotated]).is_empty());

        // A genuinely annotated empty block is content and survives.
        let annotated = AssistantContent::Text(Text {
            text: String::new(),
            additional_params: rig_core::message::AdditionalParams::try_from_value(
                json!({"citations": [1]}),
            )
            .expect("object params"),
        });
        assert_eq!(assistant_text_items_from_choice(&[annotated]).len(), 1);
    }

    #[test]
    fn argument_deltas_buffer_until_name_validates() {
        let mut asm = assembler();

        let events = asm
            .ingest(&args_delta("tc_1", "{\"x\""))
            .expect("ingest should succeed");
        assert!(events.is_empty(), "arguments must buffer before the name");

        let events = asm
            .ingest(&name_delta("tc_1", "add"))
            .expect("ingest should succeed");
        let contents: Vec<_> = events
            .iter()
            .map(|event| match event {
                StreamedTurnEvent::EmitToolCallDelta { content, .. } => content.clone(),
                other => panic!("expected EmitToolCallDelta, got {other:?}"),
            })
            .collect();
        assert_eq!(
            contents,
            vec![
                ToolCallDeltaContent::Name("add".to_string()),
                ToolCallDeltaContent::Delta("{\"x\"".to_string()),
            ]
        );

        // Subsequent argument deltas now pass straight through.
        let events = asm
            .ingest(&args_delta("tc_1", ":1}"))
            .expect("ingest should succeed");
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn buffered_arguments_without_validated_name_error_at_final() {
        let mut asm = assembler();
        asm.ingest(&args_delta("tc_1", "{\"x\":1}"))
            .expect("ingest should succeed");

        assert!(asm.pending_delta_error().is_some());
        assert!(asm.ingest(&final_item()).is_err());
    }

    #[test]
    fn finish_orders_reasoning_text_then_tool_calls() {
        let mut asm = assembler();
        asm.ingest(&StreamedAssistantContent::ReasoningDelta {
            id: "corr_1".to_string(),
            provider_id: Some("rs_1".to_string()),
            reasoning: "think".to_string(),
        })
        .expect("ingest should succeed");
        asm.ingest(&tool_call_item("tc_1", "add"))
            .expect("ingest should succeed");

        // Provider aggregation order differs deliberately.
        let final_choice = vec![
            AssistantContent::text("answer"),
            AssistantContent::ToolCall(tool_call("tc_1", "add")),
        ];

        let turn = asm.finish(Some("msg_1".to_string()), &final_choice);
        let kinds: Vec<&'static str> = turn
            .choice
            .iter()
            .map(|item| match item {
                AssistantContent::Reasoning(_) => "reasoning",
                AssistantContent::Text(_) => "text",
                AssistantContent::ToolCall(_) => "tool_call",
                _ => "other",
            })
            .collect();
        assert_eq!(kinds, vec!["reasoning", "text", "tool_call"]);
    }

    fn reasoning_delta(
        correlator: &str,
        provider_id: Option<&str>,
        text: &str,
    ) -> StreamedAssistantContent {
        StreamedAssistantContent::ReasoningDelta {
            id: correlator.to_string(),
            provider_id: provider_id.map(str::to_string),
            reasoning: text.to_string(),
        }
    }

    fn completed_reasoning(
        correlator: &str,
        provider_id: Option<&str>,
        text: &str,
        signature: Option<&str>,
    ) -> StreamedAssistantContent {
        let mut reasoning = Reasoning::new_with_signature(text, signature.map(str::to_string));
        if let Some(provider_id) = provider_id {
            reasoning = reasoning.with_id(provider_id.to_string());
        }
        StreamedAssistantContent::Reasoning {
            reasoning,
            id: correlator.to_string(),
        }
    }

    fn assembled_reasoning_of(asm: &StreamedTurnAssembler) -> Vec<Reasoning> {
        asm.partial_turn(None).reasoning
    }

    #[test]
    fn aggregated_reasoning_delta_is_scoped_to_each_interleaved_part() {
        let mut asm = assembler();
        asm.ingest(&reasoning_delta("corr_a", None, "first "))
            .expect("ingest");
        assert_eq!(asm.aggregated_reasoning("corr_a"), Some("first "));

        asm.ingest(&reasoning_delta("corr_b", Some("rs_b"), "second"))
            .expect("ingest");
        assert_eq!(asm.aggregated_reasoning("corr_b"), Some("second"));

        asm.ingest(&reasoning_delta("corr_a", Some("rs_a"), "part"))
            .expect("ingest");
        assert_eq!(asm.aggregated_reasoning("corr_a"), Some("first part"));
        assert_eq!(asm.aggregated_reasoning("corr_b"), Some("second"));
        assert_eq!(asm.aggregated_reasoning("missing"), None);

        let reasoning = assembled_reasoning_of(&asm);
        assert_eq!(reasoning[0].id.as_deref(), Some("rs_a"));
        assert_eq!(reasoning[1].id.as_deref(), Some("rs_b"));
    }

    #[test]
    fn aggregated_reasoning_delta_uses_a_new_pending_part_after_completion() {
        let mut asm = assembler();
        asm.ingest(&reasoning_delta("corr_a", Some("rs_a"), "old"))
            .expect("ingest");
        asm.ingest(&completed_reasoning(
            "corr_a",
            Some("rs_a"),
            "old",
            Some("sig"),
        ))
        .expect("ingest");
        assert_eq!(asm.aggregated_reasoning("corr_a"), None);

        asm.ingest(&reasoning_delta("corr_a", Some("rs_new"), "new"))
            .expect("ingest");
        assert_eq!(asm.aggregated_reasoning("corr_a"), Some("new"));
    }

    #[test]
    fn interleaved_delta_parts_stay_distinct_in_arrival_order() {
        let mut asm = assembler();
        asm.ingest(&reasoning_delta("corr_a", None, "first "))
            .expect("ingest");
        asm.ingest(&reasoning_delta("corr_a", None, "part"))
            .expect("ingest");
        asm.ingest(&tool_call_item("tc_1", "add")).expect("ingest");
        asm.ingest(&reasoning_delta("corr_b", None, "second part"))
            .expect("ingest");

        let reasoning = assembled_reasoning_of(&asm);
        assert_eq!(
            reasoning.len(),
            2,
            "two parts must not merge: {reasoning:?}"
        );
        assert!(matches!(
            reasoning[0].content.first(),
            Some(rig_core::message::ReasoningContent::Text { text, .. }) if text == "first part"
        ));
        assert!(matches!(
            reasoning[1].content.first(),
            Some(rig_core::message::ReasoningContent::Text { text, .. }) if text == "second part"
        ));
    }

    #[test]
    fn delta_only_part_survives_alongside_a_completed_block() {
        // The openrouter shape: visible chain-of-thought streams as deltas
        // whose synthesized end stays silent, while an encrypted block
        // arrives completed. Both must reach history, deltas first.
        let mut asm = assembler();
        asm.ingest(&reasoning_delta("corr_cot", None, "visible thoughts"))
            .expect("ingest");
        asm.ingest(&completed_reasoning(
            "corr_enc",
            Some("rd_1"),
            "encrypted payload",
            Some("sig"),
        ))
        .expect("ingest");

        let reasoning = assembled_reasoning_of(&asm);
        assert_eq!(
            reasoning.len(),
            2,
            "the visible chain of thought must not be dropped: {reasoning:?}"
        );
        assert!(matches!(
            reasoning[0].content.first(),
            Some(rig_core::message::ReasoningContent::Text { text, .. })
                if text == "visible thoughts"
        ));
        assert_eq!(reasoning[0].id, None);
        assert_eq!(reasoning[1].id.as_deref(), Some("rd_1"));
    }

    /// A later completion restating the SAME correlator is the same part's
    /// authoritative whole (the unsigned-close-then-signed-restatement
    /// shape): it replaces the completed slot, never appends a duplicate.
    #[test]
    fn a_same_correlator_completion_replaces_the_completed_part() {
        let mut asm = assembler();
        asm.ingest(&reasoning_delta("corr_a", None, "think"))
            .expect("ingest");
        asm.ingest(&completed_reasoning("corr_a", None, "think", None))
            .expect("ingest");
        asm.ingest(&completed_reasoning("corr_a", None, "think", Some("sig")))
            .expect("ingest");

        let reasoning = assembled_reasoning_of(&asm);
        assert_eq!(
            reasoning.len(),
            1,
            "one part per correlator, signed restatement replaces: {reasoning:?}"
        );
        assert!(matches!(
            reasoning[0].content.first(),
            Some(rig_core::message::ReasoningContent::Text { text, signature: Some(sig) })
                if text == "think" && sig == "sig"
        ));
    }

    /// Same shape with a provider id: the exact-correlator match must win
    /// BEFORE the shared-provider-id extend fallback, or the signed
    /// restatement doubles its own text.
    #[test]
    fn a_same_correlator_completion_with_a_provider_id_does_not_double_extend() {
        let mut asm = assembler();
        asm.ingest(&reasoning_delta("corr_a", Some("rs_1"), "think"))
            .expect("ingest");
        asm.ingest(&completed_reasoning("corr_a", Some("rs_1"), "think", None))
            .expect("ingest");
        asm.ingest(&completed_reasoning(
            "corr_a",
            Some("rs_1"),
            "think",
            Some("sig"),
        ))
        .expect("ingest");

        let reasoning = assembled_reasoning_of(&asm);
        assert_eq!(reasoning.len(), 1, "{reasoning:?}");
        assert_eq!(
            reasoning[0].content.len(),
            1,
            "the restatement must replace, not extend: {reasoning:?}"
        );
    }

    #[test]
    fn completed_block_supersedes_its_deltas_by_correlator() {
        let mut asm = assembler();
        asm.ingest(&reasoning_delta("corr_a", None, "streamed text"))
            .expect("ingest");
        asm.ingest(&completed_reasoning(
            "corr_a",
            None,
            "streamed text",
            Some("sig_1"),
        ))
        .expect("ingest");

        let reasoning = assembled_reasoning_of(&asm);
        assert_eq!(
            reasoning.len(),
            1,
            "the completed block replaces its own deltas: {reasoning:?}"
        );
        assert!(matches!(
            reasoning[0].content.first(),
            Some(rig_core::message::ReasoningContent::Text { text, signature: Some(sig) })
                if text == "streamed text" && sig == "sig_1"
        ));
    }

    #[test]
    fn completed_block_supersedes_its_deltas_by_provider_id() {
        let mut asm = assembler();
        asm.ingest(&reasoning_delta("corr_a", Some("rs_1"), "streamed text"))
            .expect("ingest");
        // A completed restatement whose correlator does not match (e.g. a
        // whole-block event minted its own) still supersedes via the
        // durable provider handle.
        asm.ingest(&completed_reasoning(
            "corr_other",
            Some("rs_1"),
            "restated text",
            None,
        ))
        .expect("ingest");

        let reasoning = assembled_reasoning_of(&asm);
        assert_eq!(reasoning.len(), 1, "{reasoning:?}");
        assert!(matches!(
            reasoning[0].content.first(),
            Some(rig_core::message::ReasoningContent::Text { text, .. }) if text == "restated text"
        ));
    }

    #[test]
    fn completed_blocks_sharing_a_provider_id_extend_one_part() {
        let mut asm = assembler();
        asm.ingest(&completed_reasoning(
            "corr_1",
            Some("rs_1"),
            "step-1",
            Some("sig-1"),
        ))
        .expect("ingest");
        asm.ingest(&completed_reasoning(
            "corr_2",
            Some("rs_1"),
            "step-2",
            Some("sig-2"),
        ))
        .expect("ingest");
        asm.ingest(&completed_reasoning("corr_3", Some("rs_2"), "other", None))
            .expect("ingest");

        let reasoning = assembled_reasoning_of(&asm);
        assert_eq!(reasoning.len(), 2, "{reasoning:?}");
        assert_eq!(reasoning[0].id.as_deref(), Some("rs_1"));
        assert_eq!(reasoning[0].content.len(), 2);
        assert_eq!(reasoning[1].id.as_deref(), Some("rs_2"));
    }

    #[test]
    fn completed_blocks_without_ids_stay_separate_parts() {
        let mut asm = assembler();
        asm.ingest(&completed_reasoning("corr_1", None, "first", None))
            .expect("ingest");
        asm.ingest(&completed_reasoning("corr_2", None, "second", None))
            .expect("ingest");

        let reasoning = assembled_reasoning_of(&asm);
        assert_eq!(
            reasoning.len(),
            2,
            "id-less blocks never merge: {reasoning:?}"
        );
    }

    #[test]
    fn each_delta_part_keeps_its_own_provider_id() {
        let mut asm = assembler();
        asm.ingest(&reasoning_delta("corr_a", Some("rs_a"), "alpha"))
            .expect("ingest");
        asm.ingest(&reasoning_delta("corr_b", Some("rs_b"), "beta"))
            .expect("ingest");

        let reasoning = assembled_reasoning_of(&asm);
        assert_eq!(reasoning.len(), 2, "{reasoning:?}");
        assert_eq!(reasoning[0].id.as_deref(), Some("rs_a"));
        assert_eq!(reasoning[1].id.as_deref(), Some("rs_b"));
    }

    #[test]
    fn canonical_choice_and_partial_turn_agree_on_multi_part_reasoning() {
        let mut asm = assembler();
        asm.ingest(&reasoning_delta("corr_a", None, "visible"))
            .expect("ingest");
        asm.ingest(&completed_reasoning(
            "corr_b",
            Some("rd_1"),
            "enc",
            Some("sig"),
        ))
        .expect("ingest");

        let partial = asm.partial_turn(None).reasoning;
        let final_choice = vec![AssistantContent::text("")];
        let turn = asm.finish(None, &final_choice);
        let finished: Vec<Reasoning> = turn
            .choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::Reasoning(reasoning) => Some(reasoning.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(partial, finished, "partial and finished assembly agree");
        assert_eq!(finished.len(), 2);
    }

    #[test]
    fn finish_passes_raw_choice_through_for_plain_text_turns() {
        let mut asm = assembler();
        asm.ingest(&text_item("hi")).expect("ingest should succeed");

        let final_choice = vec![AssistantContent::text("hi")];
        let turn = asm.finish(None, &final_choice);
        assert_eq!(
            serde_json::to_value(&turn.choice).expect("serialize"),
            serde_json::to_value(&final_choice).expect("serialize"),
        );
    }

    #[test]
    fn streamed_run_completes_a_tool_roundtrip() {
        let mut run = AgentRun::new("add things").max_turns(2);

        // Turn 1: the model streams one tool call.
        let AgentRunStep::CallModel { .. } = run.next_step().expect("next_step") else {
            panic!("expected CallModel");
        };
        let mut asm = assembler();
        assert!(
            asm.ingest(&tool_call_item("tc_1", "add"))
                .expect("ingest should succeed")
                .is_empty()
        );
        let usage = Usage {
            input_tokens: 5,
            output_tokens: 7,
            total_tokens: 12,
            ..Usage::new()
        };
        run.record_streamed_completion_call(
            usage,
            rig_core::completion::ResponseIdentity::default(),
            None,
        )
        .expect("record should succeed");
        let final_choice = vec![AssistantContent::ToolCall(tool_call("tc_1", "add"))];
        run.streamed_turn(asm.finish(Some("msg_1".to_string()), &final_choice))
            .expect("streamed_turn should succeed");

        let AgentRunStep::CallTools { calls } = run.next_step().expect("next_step") else {
            panic!("expected CallTools");
        };
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].internal_call_id.as_deref(), Some("internal_tc_1"));
        run.tool_results(vec![UserContent::tool_result(
            "tc_1",
            "add",
            vec![ToolResultContent::text("2")],
        )])
        .expect("tool_results should succeed");

        // Turn 2: plain text finishes the run.
        let AgentRunStep::CallModel { .. } = run.next_step().expect("next_step") else {
            panic!("expected CallModel");
        };
        let asm = assembler();
        run.record_streamed_completion_call(
            Usage::new(),
            rig_core::completion::ResponseIdentity::default(),
            None,
        )
        .expect("record should succeed");
        let final_choice = vec![AssistantContent::text("done")];
        run.streamed_turn(asm.finish(None, &final_choice))
            .expect("streamed_turn should succeed");

        let AgentRunStep::Done(response) = run.next_step().expect("next_step") else {
            panic!("expected Done");
        };
        assert_eq!(response.output, "done");
        assert_eq!(response.usage, usage);
        assert_eq!(response.completion_calls.len(), 2);
        assert_eq!(response.completion_calls[0].usage, usage);
        assert_eq!(response.completion_calls[1].usage, Usage::new());
        // prompt, assistant tool call, tool result, final assistant text
        assert_eq!(
            response
                .messages
                .expect("messages should be recorded")
                .len(),
            4
        );
    }

    #[test]
    fn streamed_invalid_tool_call_retry_rolls_back_with_partial_turn() {
        let mut run = AgentRun::new("use the tool")
            .max_turns(2)
            .max_invalid_tool_call_retries(1);
        run.next_step().expect("next_step");

        let mut asm = assembler();
        asm.ingest(&text_item("thinking ")).expect("ingest");
        let invalid = expect_invalid(
            asm.ingest(&tool_call_item("tc_1", "default_api"))
                .expect("ingest should succeed"),
        );
        let partial = asm.partial_turn(Some("msg_1".to_string()));
        assert_eq!(partial.text.as_deref(), Some("thinking "));

        let context = run.streamed_invalid_tool_call_context(&partial, &invalid);
        assert!(context.is_streaming);
        assert_eq!(context.tool_name, "default_api");
        assert_eq!(context.internal_call_id.as_deref(), Some("internal_tc_1"));

        let resolution = run
            .resolve_streamed_invalid_tool_call(
                &partial,
                &invalid,
                InvalidToolCallAction::retry("use add instead"),
            )
            .expect("retry should be accepted");
        assert!(matches!(
            resolution,
            StreamedResolution::TurnAbandoned {
                skipped_tool_result: None
            }
        ));
        asm.resolve_pending_invalid(&resolution);

        // Usage from the drained stream is recorded after the rollback.
        run.record_streamed_completion_call(
            Usage::new(),
            rig_core::completion::ResponseIdentity::default(),
            None,
        )
        .expect("record after rollback should succeed");

        // The rollback appended the partial assistant turn and feedback.
        assert_eq!(run.messages().len(), 3);
        let AgentRunStep::CallModel { turn, .. } = run.next_step().expect("next_step") else {
            panic!("expected CallModel retry");
        };
        assert_eq!(turn, 2);
    }

    #[test]
    fn streamed_invalid_tool_call_stop_leaves_run_terminal() {
        let mut run = AgentRun::new("use the tool");
        run.next_step().expect("next_step");

        let mut asm = assembler();
        let invalid = expect_invalid(
            asm.ingest(&tool_call_item("tc_1", "default_api"))
                .expect("ingest should succeed"),
        );
        let partial = asm.partial_turn(Some("msg_1".to_string()));

        let err = run
            .resolve_streamed_invalid_tool_call(
                &partial,
                &invalid,
                InvalidToolCallAction::stop("operator stop"),
            )
            .expect_err("stop should cancel the run");
        assert!(matches!(
            err,
            PromptError::PromptCancelled { reason, .. } if reason == "operator stop"
        ));

        let err = run
            .next_step()
            .expect_err("a stopped streamed run must remain terminal");
        assert!(matches!(
            err,
            PromptError::PromptCancelled { reason, .. }
                if reason.contains("next_step called after the run already failed")
        ));
    }

    #[test]
    fn streamed_invalid_tool_call_retry_cannot_emit_call_past_total_budget() {
        let mut run = AgentRun::new("use the tool")
            .max_turns(1)
            .max_invalid_tool_call_retries(1);
        run.next_step().expect("initial model call");

        let mut asm = assembler();
        let invalid = expect_invalid(
            asm.ingest(&tool_call_item("tc_1", "default_api"))
                .expect("ingest should succeed"),
        );
        let partial = asm.partial_turn(Some("msg_1".to_string()));
        let resolution = run
            .resolve_streamed_invalid_tool_call(
                &partial,
                &invalid,
                InvalidToolCallAction::retry("use add instead"),
            )
            .expect("retry resolution should be accepted");
        assert!(matches!(
            resolution,
            StreamedResolution::TurnAbandoned {
                skipped_tool_result: None
            }
        ));
        run.record_streamed_completion_call(
            Usage::new(),
            rig_core::completion::ResponseIdentity::default(),
            None,
        )
        .expect("completion call should be recorded");
        assert_eq!(run.completion_calls().len(), 1);

        let err = run
            .next_step()
            .expect_err("retry must not emit a second model call");
        assert!(matches!(
            err,
            PromptError::MaxTurnsError { max_turns: 1, .. }
        ));
        assert_eq!(run.turn(), 1);
    }

    #[test]
    fn streamed_invalid_tool_call_skip_returns_synthetic_result() {
        let mut run = AgentRun::new("use the tool").max_turns(2);
        run.next_step().expect("next_step");

        let mut asm = assembler();
        let invalid = expect_invalid(
            asm.ingest(&tool_call_item("tc_1", "default_api"))
                .expect("ingest should succeed"),
        );
        let partial = asm.partial_turn(None);

        let resolution = run
            .resolve_streamed_invalid_tool_call(
                &partial,
                &invalid,
                InvalidToolCallAction::skip("not available"),
            )
            .expect("skip should be accepted");
        let StreamedResolution::TurnAbandoned {
            skipped_tool_result: Some(tool_result),
        } = &resolution
        else {
            panic!("expected skipped tool result");
        };
        assert_eq!(tool_result.call, "tc_1");
    }

    #[test]
    fn streamed_invalid_name_delta_repair_replays_buffered_arguments() {
        let mut run = AgentRun::new("use the tool").max_turns(2);
        run.next_step().expect("next_step");

        let mut asm = assembler();
        asm.ingest(&args_delta("tc_1", "{\"x\":1}"))
            .expect("ingest should succeed");
        let invalid = expect_invalid(
            asm.ingest(&name_delta("tc_1", "default_api"))
                .expect("ingest should succeed"),
        );
        assert_eq!(invalid.args.as_deref(), Some("{\"x\":1}"));

        let partial = asm.partial_turn(None);
        let resolution = run
            .resolve_streamed_invalid_tool_call(
                &partial,
                &invalid,
                InvalidToolCallAction::repair("add"),
            )
            .expect("repair should be accepted");
        assert!(matches!(
            resolution,
            StreamedResolution::Repaired { ref tool_name } if tool_name == "add"
        ));

        let events = asm.resolve_pending_invalid(&resolution);
        let contents: Vec<_> = events
            .iter()
            .map(|event| match event {
                StreamedTurnEvent::EmitToolCallDelta { content, .. } => content.clone(),
                other => panic!("expected EmitToolCallDelta, got {other:?}"),
            })
            .collect();
        assert_eq!(
            contents,
            vec![
                ToolCallDeltaContent::Name("add".to_string()),
                ToolCallDeltaContent::Delta("{\"x\":1}".to_string()),
            ]
        );
    }

    #[test]
    fn streamed_turn_rejects_unknown_tool_calls_fail_fast() {
        let mut run = AgentRun::new("use the tool");
        run.next_step().expect("next_step");

        let turn = StreamedTurn {
            message_id: None,
            choice: vec![AssistantContent::ToolCall(tool_call("tc_1", "unknown"))],
            executable_tool_names: tool_names(&["add"]),
            allowed_tool_names: tool_names(&["add"]),
            internal_call_ids: Vec::new(),
            finish_reason: None,
        };
        let err = run
            .streamed_turn(turn)
            .expect_err("unknown tool should fail fast");
        assert!(matches!(
            err,
            PromptError::UnknownToolCall { tool_name, .. } if tool_name == "unknown"
        ));
    }

    #[test]
    fn streamed_completion_call_record_requires_a_model_call() {
        // A fresh run has emitted no CallModel: recording must be rejected
        // even though the machine is in its initial PreparingRequest state.
        let mut run = AgentRun::new("hello");
        let err = run
            .record_streamed_completion_call(
                Usage::new(),
                rig_core::completion::ResponseIdentity::default(),
                None,
            )
            .expect_err("recording before any model call must be rejected");
        assert!(matches!(err, PromptError::PromptCancelled { .. }));

        // The run stays drivable.
        run.next_step().expect("next_step should still succeed");
        run.record_streamed_completion_call(
            Usage::new(),
            rig_core::completion::ResponseIdentity::default(),
            None,
        )
        .expect("recording during a pending model call succeeds");
    }

    #[test]
    fn duplicate_tool_call_ids_keep_distinct_internal_ids_through_the_run() {
        let mut run = AgentRun::new("do both").max_turns(2);
        run.next_step().expect("next_step");

        let mut asm = assembler();
        asm.ingest(&StreamedAssistantContent::ToolCall {
            tool_call: tool_call("tc_1", "add"),
            internal_call_id: "internal_a".to_string(),
        })
        .expect("ingest should succeed");
        asm.ingest(&StreamedAssistantContent::ToolCall {
            tool_call: tool_call("tc_1", "add"),
            internal_call_id: "internal_b".to_string(),
        })
        .expect("ingest should succeed");
        run.record_streamed_completion_call(
            Usage::new(),
            rig_core::completion::ResponseIdentity::default(),
            None,
        )
        .expect("record should succeed");

        let final_choice = vec![
            AssistantContent::ToolCall(tool_call("tc_1", "add")),
            AssistantContent::ToolCall(tool_call("tc_1", "add")),
        ];
        run.streamed_turn(asm.finish(None, &final_choice))
            .expect("streamed_turn should succeed");

        // The internal IDs survive in the run state itself: a serde round
        // trip must keep both calls distinguishable.
        let serialized = serde_json::to_string(&run).expect("serialize");
        let mut restored: AgentRun = serde_json::from_str(&serialized).expect("deserialize");
        let AgentRunStep::CallTools { calls } = restored.next_step().expect("next_step") else {
            panic!("expected CallTools");
        };
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].internal_call_id.as_deref(), Some("internal_a"));
        assert_eq!(calls[1].internal_call_id.as_deref(), Some("internal_b"));
    }

    #[test]
    fn streamed_turn_records_the_completion_call_when_the_driver_did_not() {
        let mut run = AgentRun::new("hello");
        run.next_step().expect("next_step");

        let asm = assembler();
        let final_choice = vec![AssistantContent::text("done")];
        run.streamed_turn(asm.finish(None, &final_choice))
            .expect("streamed_turn should succeed");

        // Exactly one CompletionCall per model call, even without an explicit
        // record; usage is simply unreported.
        assert_eq!(run.completion_calls().len(), 1);
        assert_eq!(run.completion_calls()[0].usage, Usage::new());
    }

    #[test]
    fn streamed_completion_call_is_recorded_once_per_turn() {
        let mut run = AgentRun::new("hello");
        run.next_step().expect("next_step");

        run.record_streamed_completion_call(
            Usage::new(),
            rig_core::completion::ResponseIdentity::default(),
            None,
        )
        .expect("first record succeeds");
        let err = run
            .record_streamed_completion_call(
                Usage::new(),
                rig_core::completion::ResponseIdentity::default(),
                None,
            )
            .expect_err("second record for the same turn must be rejected");
        assert!(matches!(err, PromptError::PromptCancelled { .. }));
        assert_eq!(run.completion_calls().len(), 1);
    }

    #[test]
    fn streamed_run_serde_round_trips_while_tools_pend() {
        let mut run = AgentRun::new("add things").max_turns(2);
        run.next_step().expect("next_step");

        let mut asm = assembler();
        asm.ingest(&tool_call_item("tc_1", "add"))
            .expect("ingest should succeed");
        run.record_streamed_completion_call(
            Usage::new(),
            rig_core::completion::ResponseIdentity::default(),
            None,
        )
        .expect("record should succeed");
        let final_choice = vec![AssistantContent::ToolCall(tool_call("tc_1", "add"))];
        run.streamed_turn(asm.finish(None, &final_choice))
            .expect("streamed_turn should succeed");
        run.next_step().expect("CallTools step");

        let serialized = serde_json::to_string(&run).expect("serialize mid-run");
        let mut restored: AgentRun =
            serde_json::from_str(&serialized).expect("deserialize mid-run");
        restored
            .tool_results(vec![UserContent::tool_result(
                "tc_1",
                "add",
                vec![ToolResultContent::text("2")],
            )])
            .expect("tool_results should succeed");
        assert!(matches!(
            restored.next_step().expect("next turn"),
            AgentRunStep::CallModel { turn: 2, .. }
        ));
    }
}
