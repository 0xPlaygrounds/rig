//! Streamed-turn assembly and early invalid-call diagnostics for [`super::AgentRun`].
//!
//! Drivers ingest events, resolve invalid calls before continuing, and finish at
//! stream EOF. Abandoned turns still require draining provider usage. The assembler
//! performs no I/O and returns forwarding instructions as [`StreamedTurnEvent`].
//!
//! ```
//! use rig_agent::run::streamed::StreamedTurnAssembler;
//! let assembler = StreamedTurnAssembler::new(Default::default(), Default::default());
//! assert!(assembler.aggregated_text().is_empty());
//! ```

use std::collections::{BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

use rig_core::completion::FinishReason;
use rig_core::error::ProviderError;
use rig_core::message::{
    AssistantContent, Reasoning, ToolCall, ToolFunction, ToolResult, non_empty,
};
use rig_core::streaming::BlockId;

use super::policy::InvalidToolCallReason;
use super::transcript::{TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER, tool_result_message};
use rig_core::completion::{Message, Usage};
use rig_core::json_utils;
use rig_core::streaming::{BlockClose, BlockKind, Delta, StreamEvent};

/// Canonical streamed content ordering: reasoning, text, then trailing items.
pub use rig_core::message::{canonical_streamed_choice, ordered_assistant_content};

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

/// Detect unknown payloads containing assistant content that assembly would lose:
/// tagged assistant blocks or text with malformed additional parameters.
fn unknown_payload_loses_assistant_content(payload: &serde_json::Value) -> bool {
    // Deserialize by reference to avoid cloning large unknown payloads.
    if AssistantContent::deserialize(payload).is_ok() {
        return true;
    }
    // Malformed metadata must not hide the loss of an otherwise valid text field.
    payload
        .get("text")
        .is_some_and(serde_json::Value::is_string)
        && payload.get("additional_params").is_some()
}

/// The text items of a choice, as the streamed surface reports them.
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
    /// assembled from the streamed name and any buffered argument deltas;
    /// for malformed arguments its `arguments` is `Null`; no object was
    /// ever parsed, and fabricating one would misrepresent the wire.
    pub tool_call: ToolCall,
    /// Rig-generated identifier correlating this call's stream items.
    pub block_id: BlockId,
    /// Raw argument payload for diagnostics, when available.
    pub args: Option<String>,
    /// Executable Rig tools advertised to the provider for this turn.
    pub executable_tool_names: BTreeSet<String>,
    /// Tools allowed by the active tool choice for this turn.
    pub allowed_tool_names: BTreeSet<String>,
    /// Why the call was rejected.
    pub reason: InvalidToolCallReason,
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
        // Preserve call IDs so synthetic results correlate with their diagnostic calls.
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
    pub block_ids: Vec<(rig_core::message::ToolCallId, BlockId)>,
    /// Provider-reported terminal reason for this turn, when available.
    pub finish_reason: Option<FinishReason>,
}

/// Resolution a driver must apply to a mid-stream invalid tool call.
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
        /// consumer stream.
        skipped_tool_result: Option<ToolResult>,
    },
    /// The invalid call is dropped and the turn goes on without it: the
    /// runner's `UnhandledInvalidToolCall::Ignore` on the streaming
    /// surface. Apply it via
    /// [`StreamedTurnAssembler::resolve_pending_invalid`] and keep consuming
    /// the provider stream; nothing of the call enters the run.
    Ignored,
}

/// Required driver action after ingesting a stream item.
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
    InvalidToolCall(StreamedInvalidToolCall),
    /// The provider supplied its typed final payload. Record its usage (see
    /// `AgentRun::record_streamed_completion_call`);
    /// this does not establish that the provider stream reached EOF. When
    /// `emit_final` is set, the turn streamed text and the driver should buffer
    /// the final item until EOF finalizes the turn.
    Completed {
        /// Provider-reported usage for this call. Usage whose counters are
        /// all `None` means the provider reported no usage metrics.
        usage: Usage,
        /// Whether the ingested final item should be forwarded to the
        /// consumer (set when the turn streamed text).
        emit_final: bool,
        /// Why the provider stopped generating, when reported.
        finish_reason: Option<FinishReason>,
        /// The provider's own terminal record, as carried on
        /// `StreamFinal::raw`: what the driver records with
        /// [`AgentRun::record_streamed_completion_call`](super::AgentRun::record_streamed_completion_call)
        /// so the call carries this attempt's payload.
        raw: serde_json::Value,
    },
}

#[derive(Default, Clone, Serialize, Deserialize)]
struct ToolCallDeltaState {
    name_validated: bool,
    buffered_arguments: Vec<String>,
    /// The block's call was ignored (`StreamedResolution::Ignored` on its
    /// name delta): its later deltas and its end are swallowed, and the
    /// state is kept so they have something to land on.
    ignored: bool,
}

/// One reasoning part of the turn, in first-arrival order. A part opens as
/// delta text keyed by the stream's block id and is
/// superseded in place when a completed block restating the same part
/// arrives; a completed block matching no open part occupies its own slot.
#[derive(Clone, Serialize, Deserialize)]
struct ReasoningPart {
    correlator: Option<BlockId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    aliases: Vec<BlockId>,
    provider_id: Option<String>,
    state: ReasoningPartState,
}

impl ReasoningPart {
    fn matches_key(&self, key: &BlockId) -> bool {
        self.correlator.as_ref() == Some(key) || self.aliases.contains(key)
    }
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

// Keep independently addressable blocks internally so a trailing signature
// replaces only its own content. Group completed provider-item parts only when
// exposing history, preserving the established grouping of distinct keys while
// keeping repeated-key siblings separate.
fn group_reasoning(parts: impl Iterator<Item = ReasoningPart>) -> Vec<Reasoning> {
    let mut groups: Vec<(Reasoning, Vec<BlockId>, bool)> = Vec::new();
    for part in parts {
        let completed = matches!(part.state, ReasoningPartState::Completed(_));
        let keys: Vec<_> = part.correlator.into_iter().chain(part.aliases).collect();
        let Some(reasoning) = reasoning_from_part(part.state, part.provider_id) else {
            continue;
        };
        if completed
            && reasoning.id.is_some()
            && let Some((existing, existing_keys, _)) = groups
                .iter_mut()
                .rev()
                .find(|(existing, _, completed)| *completed && existing.id == reasoning.id)
            && keys.iter().all(|key| !existing_keys.contains(key))
        {
            existing.content.extend(reasoning.content);
            existing_keys.extend(keys);
            continue;
        }
        groups.push((reasoning, keys, completed));
    }
    groups
        .into_iter()
        .map(|(reasoning, _, _)| reasoning)
        .collect()
}

#[derive(Clone, Serialize, Deserialize)]
enum PendingInvalid {
    /// A complete tool call with a disallowed name.
    FullCall {
        tool_call: ToolCall,
        block_id: BlockId,
    },
    /// A streamed tool-name delta with a disallowed name.
    NameDelta { block_id: BlockId },
    /// A complete tool call whose arguments were not JSON. The provider's
    /// accumulator already finalized the block; nothing is buffered here
    /// to replay, so the only resolutions are abandon or fail.
    MalformedArgs { tool_call: ToolCall },
}

/// Serializable accumulator for one streamed turn. Persisted state requires the
/// same library version; it does not itself resume a provider connection.
/// Drivers must resolve pending invalid calls before ingesting more events.
/// Dropping warns once if assistant content was excluded, otherwise stays silent.
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
    /// The ids of calls an [`StreamedResolution::Ignored`] dropped: they
    /// are still in the provider's own view of the turn (the stream's
    /// snapshot) and must not come back through it at [`Self::finish`].
    ignored_calls: Vec<rig_core::message::ToolCallId>,
    /// Terminal reason from this turn's provider final record, retained so
    /// [`Self::finish`] can carry it onto the [`StreamedTurn`].
    finish_reason: Option<FinishReason>,
    /// Replayed assistant blocks excluded from assembly this turn (see
    /// [`unknown_payload_loses_assistant_content`]): counted per item,
    /// surfaced as one warning when the guard drops.
    excluded_assistant_content: ExclusionCount,
}

/// Persisted count of excluded assistant blocks. Each clone warns once on drop
/// when nonzero, including cancellation and error paths. A separate drop guard
/// leaves the assembler's fields movable.
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
            ignored_calls: Vec::new(),
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

    /// Return the newest matching reasoning part's provider ID, or `None` if
    /// absent. Reused correlators refer to their latest pending or completed part.
    pub fn reasoning_provider_id(&self, correlator: &BlockId) -> Option<&str> {
        self.reasoning_parts
            .iter()
            .rev()
            .find(|part| part.matches_key(correlator))
            .and_then(|part| part.provider_id.as_deref())
    }

    /// The reasoning text accumulated so far under `correlator`.
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

    /// Combine accepted calls and reasoning with provider text and images in
    /// canonical order. Without calls or reasoning, preserve the provider choice.
    fn canonical_choice_with(
        pending_tool_calls: Vec<(ToolCall, BlockId)>,
        reasoning: Vec<Reasoning>,
        provider_choice: &[AssistantContent],
    ) -> Vec<AssistantContent> {
        if !pending_tool_calls.is_empty() || !reasoning.is_empty() {
            let parts = reasoning
                .into_iter()
                .map(AssistantContent::Reasoning)
                .chain(assistant_text_items_from_choice(provider_choice))
                .chain(
                    pending_tool_calls
                        .into_iter()
                        .map(|(tool_call, _)| AssistantContent::ToolCall(tool_call)),
                )
                .chain(
                    provider_choice
                        .iter()
                        .filter(|content| matches!(content, AssistantContent::Image(_)))
                        .cloned(),
                )
                .collect();
            canonical_streamed_choice(parts)
        } else {
            provider_choice.to_vec()
        }
    }

    /// A completed block replaces its pending deltas. After a key closes,
    /// an explicit whole block or a second signature opens a sibling; only
    /// trailing metadata for an unsigned close updates that completed part.
    /// The normalized end retains this distinction in its original payload.
    fn ingest_completed_reasoning(
        &mut self,
        reasoning: &Reasoning,
        correlator: &BlockId,
        restatement: bool,
    ) {
        let replace_at = self
            .reasoning_parts
            .iter()
            .rposition(|part| {
                part.matches_key(correlator) && matches!(part.state, ReasoningPartState::Pending(_))
            })
            .or_else(|| {
                // A known block key is more specific than a shared provider
                // item id. Its next whole block must not consume a different
                // key's still-pending content.
                if self
                    .reasoning_parts
                    .iter()
                    .any(|part| part.matches_key(correlator))
                {
                    return None;
                }
                self.reasoning_parts.iter().rposition(|part| {
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
            if !part.matches_key(correlator) {
                part.aliases.push(correlator.clone());
            }
            part.state = ReasoningPartState::Completed(reasoning.clone());
            return;
        }

        if let Some(part) = self
            .reasoning_parts
            .iter_mut()
            .rev()
            .find(|part| part.matches_key(correlator))
        {
            let signed = matches!(&part.state, ReasoningPartState::Completed(existing)
                if existing.content.iter().any(|content| matches!(content,
                    rig_core::message::ReasoningContent::Text { signature: Some(_), .. })));
            if !restatement && !signed {
                if reasoning.id.is_some() {
                    part.provider_id.clone_from(&reasoning.id);
                }
                part.state = ReasoningPartState::Completed(reasoning.clone());
                return;
            }
            // Keep this completed sibling separately addressable. History
            // grouping recognizes key reuse and preserves both parts.
            self.reasoning_parts.push(ReasoningPart {
                aliases: Vec::new(),
                correlator: Some(correlator.clone()),
                provider_id: reasoning.id.clone(),
                state: ReasoningPartState::Completed(reasoning.clone()),
            });
            return;
        }

        self.reasoning_parts.push(ReasoningPart {
            aliases: Vec::new(),
            correlator: Some(correlator.clone()),
            provider_id: reasoning.id.clone(),
            state: ReasoningPartState::Completed(reasoning.clone()),
        });
    }

    /// The turn's reasoning in first-arrival order. Completed parts from
    /// distinct keys sharing a provider item are grouped; repeated-key siblings
    /// and pending delta buffers retain their separate slots.
    fn assembled_reasoning(&self) -> Vec<Reasoning> {
        group_reasoning(self.reasoning_parts.iter().cloned())
    }

    /// Consume reasoning parts in canonical grouping, avoiding copies of large
    /// encrypted payloads.
    fn drain_reasoning(&mut self) -> Vec<Reasoning> {
        group_reasoning(std::mem::take(&mut self.reasoning_parts).into_iter())
    }

    /// Ingest one provider stream item and return what the driver must do.
    ///
    /// # Errors
    /// Returns an error when the provider stream is inconsistent (argument
    /// deltas finishing without a validated tool name) or when an invalid
    /// tool call is still awaiting resolution.
    pub fn ingest(&mut self, item: &StreamEvent) -> Result<Vec<StreamedTurnEvent>, ProviderError> {
        if self.pending_invalid.is_some() {
            return Err(ProviderError::Response(
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
            // The driver and provider aggregate retain message identity and text metadata.
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
                let pending = self.reasoning_parts.iter_mut().find(|part| {
                    part.matches_key(id) && matches!(part.state, ReasoningPartState::Pending(_))
                });
                match pending {
                    Some(part) => {
                        if part.provider_id.is_none() {
                            part.provider_id.clone_from(provider_id);
                        }
                    }
                    None => self.reasoning_parts.push(ReasoningPart {
                        aliases: Vec::new(),
                        correlator: Some(id.clone()),
                        provider_id: provider_id.clone(),
                        state: ReasoningPartState::Pending(String::new()),
                    }),
                }
                Ok(vec![StreamedTurnEvent::EmitIngested])
            }
            StreamEvent::BlockEnd {
                id,
                end:
                    BlockClose::Reasoning {
                        reasoning: restatement,
                        ..
                    },
                block,
            } => {
                if let Some(AssistantContent::Reasoning(reasoning)) = block {
                    self.ingest_completed_reasoning(reasoning, id, restatement.is_some());
                } else if let Some(part) = self.reasoning_parts.iter_mut().rev().find(|part| {
                    part.matches_key(id) && matches!(part.state, ReasoningPartState::Pending(_))
                }) && let ReasoningPartState::Pending(text) = &part.state
                {
                    // A synthesized silent close publishes no whole block,
                    // but still ends this part. Later deltas reopen the key;
                    // they must not extend or overwrite the completed text.
                    let content = if text.is_empty() {
                        Vec::new()
                    } else {
                        Reasoning::new(text).content
                    };
                    part.state = ReasoningPartState::Completed(Reasoning {
                        provider: None,
                        id: part.provider_id.clone(),
                        content,
                    });
                }
                Ok(vec![StreamedTurnEvent::EmitIngested])
            }
            StreamEvent::BlockDelta {
                id,
                delta: Delta::Reasoning { text: reasoning },
            } => {
                // Keep unsigned deltas separate until an authoritative completed block
                // supplies metadata. Only provider IDs, not correlators, enter history.
                let index = self
                    .reasoning_parts
                    .iter()
                    .position(|part| {
                        part.matches_key(id) && matches!(part.state, ReasoningPartState::Pending(_))
                    })
                    .unwrap_or_else(|| {
                        self.reasoning_parts.push(ReasoningPart {
                            aliases: Vec::new(),
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
                // The driver emits completed calls at turn commit, not at block end.
                if self
                    .delta_states
                    .get(block_id)
                    .is_some_and(|state| state.ignored)
                {
                    // The final durable ID may differ from the name delta's block key;
                    // retain it so provider snapshots cannot restore an ignored call.
                    if let Some(AssistantContent::ToolCall(call)) = block {
                        self.ignored_calls.push(call.id.clone());
                    }
                    self.delta_states.remove(block_id);
                    return Ok(Vec::new());
                }
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
                            tool_call: tool_call.clone(),
                            block_id: block_id.clone(),
                        },
                        InvalidToolCallReason::UnknownTool,
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
                if self
                    .delta_states
                    .get(&key)
                    .is_some_and(|state| state.ignored)
                {
                    return Ok(Vec::new());
                }
                match delta {
                    Delta::ToolName { name } => {
                        if !self.allowed_tool_names.contains(name) {
                            let buffered_args = self
                                .delta_states
                                .get(&key)
                                .map(|state| state.buffered_arguments.join(""))
                                .unwrap_or_default();
                            let tool_call =
                                self.name_delta_diagnostic_tool_call(&key, name, &buffered_args);
                            return Ok(self.surface_invalid_call(
                                tool_call,
                                block_id.clone(),
                                Some(buffered_args),
                                PendingInvalid::NameDelta {
                                    block_id: block_id.clone(),
                                },
                                InvalidToolCallReason::UnknownTool,
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
                // `StreamingCompletionResponse` has already reconciled this
                // against the tool calls the accumulator actually saw (see
                // `FinishReason::reconcile_with_output`), so it is consumed
                // as-is and never re-reconciled here.
                let finish_reason = final_response.finish_reason.clone();
                self.finish_reason.clone_from(&finish_reason);
                Ok(vec![StreamedTurnEvent::Completed {
                    usage,
                    emit_final,
                    finish_reason,
                    raw: final_response.raw.clone(),
                }])
            }
            StreamEvent::Unknown(payload) => {
                // Unknown items have no assistant-content representation. Count lost
                // assistant blocks for one warning without exposing payloads or flooding logs.
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
                self.pending_tool_calls.push((tool_call, block_id));
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
            (
                StreamedResolution::TurnAbandoned { .. },
                PendingInvalid::FullCall { .. } | PendingInvalid::MalformedArgs { .. },
            ) => Vec::new(),
            // Repair is rejected upstream for malformed arguments (the run
            // fails closed); reaching here would be a protocol violation, so
            // the call is simply not resurrected.
            (StreamedResolution::Repaired { .. }, PendingInvalid::MalformedArgs { .. }) => {
                Vec::new()
            }
            (StreamedResolution::Ignored, PendingInvalid::MalformedArgs { tool_call }) => {
                self.ignored_calls.push(tool_call.id.clone());
                Vec::new()
            }
            (StreamedResolution::Ignored, PendingInvalid::FullCall { tool_call, .. }) => {
                self.ignored_calls.push(tool_call.id.clone());
                Vec::new()
            }
            (StreamedResolution::Ignored, PendingInvalid::NameDelta { block_id }) => {
                // Retain a tombstone so later deltas cannot buffer or resurrect this call.
                self.ignored_calls
                    .push(rig_core::message::ToolCallId::from_block(&block_id));
                let state = self.delta_states.entry(block_id).or_default();
                state.buffered_arguments.clear();
                state.name_validated = false;
                state.ignored = true;
                Vec::new()
            }
        }
    }

    /// Error when argument deltas were buffered for a tool call whose name
    /// never validated, indicating an inconsistent provider stream.
    pub fn pending_delta_error(&self) -> Option<ProviderError> {
        self.delta_states
            .iter()
            .find(|(_, state)| !state.name_validated && !state.buffered_arguments.is_empty())
            .map(|(block_id, state)| {
                ProviderError::Response(format!(
                    "streamed tool call arguments received before a validated tool name for block_id `{block_id}` ({} buffered argument delta(s))",
                    state.buffered_arguments.len()
                ))
            })
    }

    /// Snapshot of the turn so far, for diagnostics and rollback messages.
    /// Its reasoning records `issuer` as in [`Self::finish`].
    pub fn partial_turn(
        &self,
        message_id: Option<String>,
        issuer: Option<&str>,
    ) -> PartialStreamedTurn {
        let reasoning = self
            .assembled_reasoning()
            .into_iter()
            .map(|reasoning| match (&reasoning.provider, issuer) {
                (None, Some(issuer)) => reasoning.with_provider(issuer),
                _ => reasoning,
            })
            .collect();

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
    /// (`StreamingCompletionResponse::choice`) and `issuer` the service its
    /// reasoning came from (`StreamingCompletionResponse::reasoning_issuer`),
    /// recorded on every reasoning part that names none. With no issuer the
    /// reasoning keeps unknown provenance, which every request replays.
    pub fn finish(
        mut self,
        message_id: Option<String>,
        final_choice: &[AssistantContent],
        issuer: Option<&str>,
    ) -> StreamedTurn {
        let reasoning = self.drain_reasoning();
        let pending_tool_calls = std::mem::take(&mut self.pending_tool_calls);
        let block_ids: Vec<(rig_core::message::ToolCallId, BlockId)> = pending_tool_calls
            .iter()
            .map(|(tool_call, block_id)| (tool_call.id.clone(), block_id.clone()))
            .collect();
        // An ignored call is dropped from the turn: the provider's own view
        // of the turn still carries it, and the turn is re-validated.
        let ignored = std::mem::take(&mut self.ignored_calls);
        let provider_choice: Vec<AssistantContent> = final_choice
            .iter()
            .filter(|content| match content {
                AssistantContent::ToolCall(call) => !ignored.contains(&call.id),
                AssistantContent::Text(_)
                | AssistantContent::Reasoning(_)
                | AssistantContent::Image(_) => true,
            })
            .cloned()
            .collect();
        let choice = Self::canonical_choice_with(pending_tool_calls, reasoning, &provider_choice);
        let choice = match issuer {
            Some(issuer) => rig_core::streaming::stamp_reasoning(choice, issuer),
            None => choice,
        };

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
        reason: InvalidToolCallReason,
    ) -> Vec<StreamedTurnEvent> {
        let invalid = StreamedInvalidToolCall {
            tool_call,
            block_id,
            args,
            executable_tool_names: self.executable_tool_names.clone(),
            allowed_tool_names: self.allowed_tool_names.clone(),
            reason,
        };
        self.pending_invalid = Some(pending);
        vec![StreamedTurnEvent::InvalidToolCall(invalid)]
    }

    /// Surface malformed completed arguments for resolution. Retains raw argument
    /// text and call identity, uses `Null` parsed arguments, and clears pending
    /// delta bookkeeping because the provider accumulator already closed the block.
    pub fn surface_malformed_input(
        &mut self,
        detail: &rig_core::error::MalformedToolInput,
    ) -> Vec<StreamedTurnEvent> {
        let tool_call = ToolCall {
            id: detail.id.clone(),
            provider: detail.provider.clone(),
            function: rig_core::message::ToolFunction {
                name: detail.name.clone(),
                arguments: serde_json::Value::Null,
            },
            signature: None,
            additional_params: None,
        };
        let block_id = detail
            .provider
            .as_ref()
            .map(|provider| BlockId::wire(provider.call_id.as_str()))
            .or_else(|| detail.id.generated().cloned())
            .unwrap_or_else(|| BlockId::wire(detail.id.wire_hint().as_ref()));
        // The accumulator's block is closed; any delta bookkeeping this
        // assembler kept for it must not trip the pending-delta check.
        self.delta_states.remove(&block_id);
        self.surface_invalid_call(
            tool_call.clone(),
            block_id,
            Some(detail.raw.clone()),
            PendingInvalid::MalformedArgs { tool_call },
            InvalidToolCallReason::MalformedArguments {
                error: detail.error.clone(),
            },
        )
    }

    fn name_delta_diagnostic_tool_call(
        &self,
        key: &BlockId,
        name: &str,
        buffered_args: &str,
    ) -> ToolCall {
        let diagnostic_args = if buffered_args.trim().is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_str(buffered_args).unwrap_or(serde_json::Value::Null)
        };
        // Provider identity is not known yet; use the block's deterministic ID
        // to correlate diagnostics and rollback results without inventing a provider ID.
        ToolCall::new(
            rig_core::message::ToolCallId::from_block(key),
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
