//! Streamed-turn assembly for [`AgentRun`](super::AgentRun).
//!
//! A streamed model turn arrives as incremental [`StreamEvent`]s. [`StreamedTurnAssembler`] is the sans-IO accumulator that turns that
//! item stream into the same canonical complete turn the non-streaming path
//! feeds the machine — while telling the driver what to forward to its
//! consumer and surfacing invalid tool calls the moment they appear, so a
//! driver can stop paying for a doomed provider stream early.
//!
//! The protocol, paired with the streamed entry points on
//! `AgentRun`:
//!
//! 1. On `AgentRunStep::CallModel`, open a
//!    provider stream and create one assembler per turn with the tool names
//!    advertised for that turn.
//! 2. Feed every stream item to [`StreamedTurnAssembler::ingest`] and act on
//!    the returned [`StreamedTurnEvent`]s: forward items to the consumer, and
//!    on [`StreamedTurnEvent::InvalidToolCall`] consult
//!    `AgentRun::resolve_streamed_invalid_tool_call` —
//!    [`StreamedResolution::Repaired`] continues the same stream via
//!    [`StreamedTurnAssembler::resolve_pending_invalid`];
//!    [`StreamedResolution::TurnAbandoned`] means drain the provider stream
//!    for usage and re-enter
//!    `AgentRun::next_step`.
//! 3. When the provider stream ends, call [`StreamedTurnAssembler::finish`]
//!    and feed the result to
//!    `AgentRun::streamed_turn`; the run
//!    then proceeds exactly like a non-streamed one
//!    (`CallTools` /
//!    `Done`).
//!
//! `AgentRunner::stream` drives this protocol
//! internally; hand-driven runs can use it to stream any
//! `AgentRun`.

use std::collections::{BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

use rig_core::completion::FinishReason;
use rig_core::message::{
    AssistantContent, Reasoning, ToolCall, ToolFunction, ToolResult, non_empty,
};
use rig_core::streaming::BlockId;

use super::transcript::{TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER, tool_result_message};
use rig_core::completion::{CompletionError, Message, Usage};
use rig_core::json_utils;
use rig_core::streaming::{BlockClose, BlockKind, Delta, StreamEvent};

/// Assemble assistant content in canonical replay order: reasoning blocks,
/// then text, then trailing items (tool calls, images). Maps its inputs 1:1,
/// so the result is empty exactly when every input is.
pub fn ordered_assistant_content(
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
pub fn ordered_streaming_assistant_content(
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

/// Whether a [`StreamEvent::Unknown`] payload is rig assistant
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
/// a text delta and its text is *assembled*, with only
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

pub fn assistant_text_items_from_choice(choice: &[AssistantContent]) -> Vec<AssistantContent> {
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
/// `AgentRun::resolve_streamed_invalid_tool_call`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamedInvalidToolCall {
    /// The rejected tool call. For a name delta this is a diagnostic call
    /// assembled from the streamed name and any buffered argument deltas.
    pub tool_call: ToolCall,
    /// Rig-generated identifier correlating this call's stream items.
    pub block_id: BlockId,
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
    pub fn assistant_message(&self, current_tool_call: Option<ToolCall>) -> Option<Message> {
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
    pub fn rollback_messages(
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
/// `AgentRun::streamed_turn`.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// `(tool_call_id, block_id)` pairs for this turn's tool calls,
    /// in emission order. Carried into the run state so a resumed process
    /// keeps the IDs consumers already saw in tool-call deltas.
    #[serde(default)]
    pub block_ids: Vec<(String, BlockId)>,
    /// Why the provider stopped generating this turn, when it reported a
    /// reason — the streamed analogue of `ModelTurn::finish_reason`, so a
    /// driver that feeds turns through `streamed_turn` records the same
    /// terminal reason the blocking surface does (rig#2322).
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
    /// `AgentRun::next_step`.
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamedTurnEvent {
    /// Forward the ingested item to the consumer as-is (text, reasoning, or
    /// reasoning deltas, after accumulation).
    EmitIngested,
    /// Forward this tool-call delta. Argument deltas buffered while the tool
    /// name awaited validation are replayed through this event.
    EmitToolCallDelta {
        /// The block this call streams under.
        block_id: BlockId,
        /// The (possibly repaired) name or argument delta.
        delta: Delta,
    },
    /// The model emitted an unknown or disallowed tool call. Resolve it via
    /// `AgentRun::resolve_streamed_invalid_tool_call`,
    /// then apply the outcome with
    /// [`StreamedTurnAssembler::resolve_pending_invalid`].
    InvalidToolCall(Box<StreamedInvalidToolCall>),
    /// The provider supplied its typed final payload. Record its usage (see
    /// `AgentRun::record_streamed_completion_call`);
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

#[derive(Default, Clone, Serialize, Deserialize)]
struct ToolCallDeltaState {
    name_validated: bool,
    buffered_arguments: Vec<String>,
}

/// One reasoning part of the turn, in first-arrival order. A part opens as
/// delta text keyed by the stream's block id and is
/// superseded in place when a completed block restating the same part
/// arrives; a completed block matching no open part occupies its own slot.
#[derive(Clone, Serialize, Deserialize)]
struct ReasoningPart {
    correlator: Option<BlockId>,
    provider_id: Option<String>,
    state: ReasoningPartState,
}

#[derive(Clone, Serialize, Deserialize)]
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

#[derive(Clone, Serialize, Deserialize)]
enum PendingInvalid {
    /// A complete tool call with a disallowed name.
    FullCall {
        tool_call: Box<ToolCall>,
        block_id: BlockId,
    },
    /// A streamed tool-name delta with a disallowed name.
    NameDelta { block_id: BlockId },
}

/// Sans-IO accumulator that assembles one streamed model turn. See the
/// [module docs](self) for the driving protocol.
///
/// `Clone + Serialize + Deserialize`, like `AgentRun`: a
/// mid-stream assembler can be persisted and resumed (same caveats — no
/// cross-version format stability). Dropping one mid-turn is a normal
/// cancellation path and is silent unless replayed assistant content was
/// excluded from assembly, which warns once on drop.
#[derive(Clone, Serialize, Deserialize)]
pub struct StreamedTurnAssembler {
    executable_tool_names: BTreeSet<String>,
    allowed_tool_names: BTreeSet<String>,
    text: String,
    saw_text: bool,
    reasoning_parts: Vec<ReasoningPart>,
    pending_tool_calls: Vec<(ToolCall, BlockId)>,
    delta_states: HashMap<BlockId, ToolCallDeltaState>,
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
/// `Clone` copies the count: each lineage owns its exclusions and warns on
/// its own drop. Serde carries the count so a persisted mid-stream assembler
/// resumes with its loudness contract intact.
#[derive(Default, Clone, Serialize, Deserialize)]
#[serde(transparent)]
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
    /// The provider-issued id of the reasoning part identified by
    /// `correlator`, when its block start carried one.
    pub fn reasoning_provider_id(&self, correlator: &BlockId) -> Option<&str> {
        self.reasoning_parts
            .iter()
            .find(|part| part.correlator.as_ref() == Some(correlator))
            .and_then(|part| part.provider_id.as_deref())
    }

    pub fn aggregated_reasoning(&self, correlator: &BlockId) -> Option<&str> {
        self.reasoning_parts
            .iter()
            .find_map(|part| match (&part.state, part.correlator.as_ref()) {
                (ReasoningPartState::Pending(text), Some(id)) if id == correlator => {
                    Some(text.as_str())
                }
                _ => None,
            })
    }

    /// Normalize the provider aggregate into the content committed for this
    /// turn. The reasoning is supplied by the caller: the finish path drains
    /// its parts by value ([`Self::drain_reasoning`]) instead of cloning
    /// them, while the partial-turn surface assembles borrowed
    /// ([`Self::assembled_reasoning`]) — agreement between the two is pinned
    /// by `canonical_choice_and_partial_turn_agree_on_multi_part_reasoning`.
    fn canonical_choice_with(
        pending_tool_calls: Vec<(ToolCall, BlockId)>,
        reasoning: Vec<Reasoning>,
        provider_choice: &[AssistantContent],
    ) -> Vec<AssistantContent> {
        if !pending_tool_calls.is_empty() || !reasoning.is_empty() {
            let text_items = assistant_text_items_from_choice(provider_choice);
            let tool_items = pending_tool_calls
                .into_iter()
                .map(|(tool_call, _)| AssistantContent::ToolCall(tool_call))
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
    fn ingest_completed_reasoning(&mut self, reasoning: &Reasoning, correlator: &BlockId) {
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
            .position(|part| part.correlator.as_ref() == Some(correlator))
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
                part.provider_id.clone_from(&reasoning.id);
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
            correlator: Some(correlator.clone()),
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
        item: &StreamEvent,
    ) -> Result<Vec<StreamedTurnEvent>, CompletionError> {
        if self.pending_invalid.is_some() {
            return Err(CompletionError::ResponseError(
                "streamed turn ingested while an invalid tool call awaits resolution".to_string(),
            ));
        }

        match item {
            StreamEvent::BlockDelta {
                delta: Delta::Text { text },
                ..
            } => {
                if !self.saw_text {
                    self.text.clear();
                    self.saw_text = true;
                }
                self.text.push_str(text);
                Ok(vec![StreamedTurnEvent::EmitIngested])
            }
            // Block bookkeeping the assembler does not fold: the message id
            // (read off the stream by the driver), text block boundaries and
            // metadata (the provider's aggregate carries them), a tool-call
            // block opening (its deltas open the assembly).
            StreamEvent::BlockStart {
                kind: BlockKind::Message | BlockKind::Text { .. } | BlockKind::ToolCall,
                ..
            }
            | StreamEvent::BlockDelta {
                delta: Delta::TextMeta { .. },
                ..
            }
            | StreamEvent::BlockEnd {
                end: BlockClose::Text,
                ..
            } => Ok(vec![StreamedTurnEvent::EmitIngested]),
            StreamEvent::BlockStart {
                id,
                kind: BlockKind::Reasoning { provider_id },
            } => {
                // The block's durable provider id arrives at its start; the
                // part opens here (or on its first delta) and keeps it.
                let pending = self.reasoning_parts.iter_mut().find(|part| {
                    part.correlator.as_ref() == Some(id)
                        && matches!(part.state, ReasoningPartState::Pending(_))
                });
                match pending {
                    Some(part) => {
                        if part.provider_id.is_none() {
                            part.provider_id.clone_from(provider_id);
                        }
                    }
                    None => self.reasoning_parts.push(ReasoningPart {
                        correlator: Some(id.clone()),
                        provider_id: provider_id.clone(),
                        state: ReasoningPartState::Pending(String::new()),
                    }),
                }
                Ok(vec![StreamedTurnEvent::EmitIngested])
            }
            StreamEvent::BlockEnd {
                id,
                end: BlockClose::Reasoning { .. },
                block,
            } => {
                // An authoritative close carries the completed block; a
                // synthesized silent boundary carries nothing to fold.
                if let Some(AssistantContent::Reasoning(reasoning)) = block {
                    self.ingest_completed_reasoning(reasoning, id);
                }
                Ok(vec![StreamedTurnEvent::EmitIngested])
            }
            StreamEvent::BlockDelta {
                id,
                delta: Delta::Reasoning { text: reasoning },
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
                        part.correlator.as_ref() == Some(id)
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
                if let Some(part) = self.reasoning_parts.get_mut(index)
                    && let ReasoningPartState::Pending(text) = &mut part.state
                {
                    text.push_str(reasoning);
                }
                Ok(vec![StreamedTurnEvent::EmitIngested])
            }
            StreamEvent::BlockEnd {
                id: block_id,
                end: BlockClose::ToolCall(_),
                block,
            } => {
                // The completed call is on the end event when the accumulator
                // finalized one; an end that dropped its call (never fully
                // arrived) assembles nothing. Neither is forwarded: the
                // driver reports the model's completed calls itself when the
                // turn commits.
                let Some(AssistantContent::ToolCall(tool_call)) = block else {
                    self.delta_states.remove(block_id);
                    return Ok(Vec::new());
                };
                if !self.allowed_tool_names.contains(&tool_call.function.name) {
                    return Ok(self.surface_invalid_call(
                        tool_call.clone(),
                        block_id.clone(),
                        Some(json_utils::serialize_json_value(
                            &tool_call.function.arguments,
                        )),
                        PendingInvalid::FullCall {
                            tool_call: Box::new(tool_call.clone()),
                            block_id: block_id.clone(),
                        },
                    ));
                }

                self.pending_tool_calls
                    .push((tool_call.clone(), block_id.clone()));
                Ok(Vec::new())
            }
            StreamEvent::BlockDelta {
                id: block_id,
                delta: delta @ (Delta::ToolName { .. } | Delta::ToolArguments { .. }),
            } => {
                let key = block_id.clone();
                match delta {
                    Delta::ToolName { name } => {
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
                                block_id.clone(),
                                Some(buffered_args),
                                PendingInvalid::NameDelta {
                                    block_id: block_id.clone(),
                                },
                            ));
                        }

                        Ok(self.validate_delta_name(key, name.clone()))
                    }
                    Delta::ToolArguments { arguments } => {
                        let state = self.delta_states.entry(key).or_default();
                        if state.name_validated {
                            Ok(vec![StreamedTurnEvent::EmitToolCallDelta {
                                block_id: block_id.clone(),
                                delta: Delta::ToolArguments {
                                    arguments: arguments.clone(),
                                },
                            }])
                        } else {
                            state.buffered_arguments.push(arguments.clone());
                            Ok(Vec::new())
                        }
                    }
                    Delta::Text { .. } | Delta::TextMeta { .. } | Delta::Reasoning { .. } => {
                        // Excluded by the arm's pattern.
                        Ok(vec![StreamedTurnEvent::EmitIngested])
                    }
                }
            }
            StreamEvent::Final(final_response) => {
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
                self.finish_reason.clone_from(&finish_reason);
                Ok(vec![StreamedTurnEvent::Completed {
                    usage,
                    emit_final,
                    finish_reason,
                }])
            }
            StreamEvent::Unknown(payload) => {
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
                    block_id,
                },
            ) => {
                tool_call.function.name.clone_from(tool_name);
                self.pending_tool_calls.push((*tool_call, block_id));
                Vec::new()
            }
            (
                StreamedResolution::Repaired { tool_name },
                PendingInvalid::NameDelta { block_id },
            ) => self.validate_delta_name(block_id, tool_name.clone()),
            (StreamedResolution::TurnAbandoned { .. }, PendingInvalid::NameDelta { block_id }) => {
                // The abandoned call's buffered state must not trip the
                // pending-delta consistency check while usage is drained.
                self.delta_states.remove(&block_id);
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
            .map(|(block_id, state)| {
                CompletionError::ResponseError(format!(
                    "streamed tool call arguments received before a validated tool name for block_id `{block_id}` ({} buffered argument delta(s))",
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
    /// (`StreamingCompletionResponse::choice`).
    pub fn finish(
        mut self,
        message_id: Option<String>,
        final_choice: &[AssistantContent],
    ) -> StreamedTurn {
        let reasoning = self.drain_reasoning();
        let pending_tool_calls = std::mem::take(&mut self.pending_tool_calls);
        let block_ids: Vec<(String, BlockId)> = pending_tool_calls
            .iter()
            .map(|(tool_call, block_id)| (tool_call.id.as_str().to_owned(), block_id.clone()))
            .collect();
        let choice = Self::canonical_choice_with(pending_tool_calls, reasoning, final_choice);

        StreamedTurn {
            message_id,
            choice,
            executable_tool_names: self.executable_tool_names,
            allowed_tool_names: self.allowed_tool_names,
            block_ids,
            finish_reason: self.finish_reason.take(),
        }
    }

    /// Park resolution on `pending` and surface the rejected call to the
    /// caller as an [`StreamedTurnEvent::InvalidToolCall`].
    fn surface_invalid_call(
        &mut self,
        tool_call: ToolCall,
        block_id: BlockId,
        args: Option<String>,
        pending: PendingInvalid,
    ) -> Vec<StreamedTurnEvent> {
        let invalid = StreamedInvalidToolCall {
            tool_call,
            block_id,
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

    fn validate_delta_name(&mut self, key: BlockId, name: String) -> Vec<StreamedTurnEvent> {
        let state = self.delta_states.entry(key.clone()).or_default();
        state.name_validated = true;
        let buffered_arguments = std::mem::take(&mut state.buffered_arguments);

        let mut events = vec![StreamedTurnEvent::EmitToolCallDelta {
            block_id: key.clone(),
            delta: Delta::ToolName { name },
        }];
        events.extend(buffered_arguments.into_iter().map(|arguments| {
            StreamedTurnEvent::EmitToolCallDelta {
                block_id: key.clone(),
                delta: Delta::ToolArguments { arguments },
            }
        }));
        events
    }
}

#[cfg(test)]
mod tests;
