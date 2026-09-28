//! Streamed-turn assembly and early invalid-call diagnostics for [`super::AgentRun`].
//!
//! Drivers ingest a completion stream's items, resolve invalid calls before
//! continuing, and finish with the response the stream folded into. The
//! assembler performs no I/O and returns forwarding instructions as
//! [`StreamedTurnEvent`].
//!
//! ```
//! use rig_agent::run::streamed::StreamedTurnAssembler;
//! let assembler = StreamedTurnAssembler::new(Default::default(), Default::default());
//! assert!(assembler.aggregated_text().is_empty());
//! ```

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use rig_core::NonEmpty;
use rig_core::completion::{FinishReason, Message};
use rig_core::error::ProviderError;
use rig_core::json_utils;
use rig_core::message::{
    AssistantContent, CallId, Reasoning, Sealed, ToolCall, ToolFunction, ToolName, ToolResult,
};
use rig_core::streaming::{Item, PartKind, StreamEvent};

use super::policy::InvalidToolCallReason;
use super::transcript::{TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER, tool_result_message};

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

/// One invalid tool call surfaced mid-stream, awaiting a resolution from
/// `AgentRun::resolve_streamed_invalid_tool_call`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamedInvalidToolCall {
    /// The rejected tool call. For malformed arguments its `arguments` is
    /// `Null`: no object was ever parsed, and fabricating one would
    /// misrepresent the wire.
    pub tool_call: ToolCall,
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
    /// The reasoning parts that ended so far, sealed to their issuer.
    pub reasoning: Vec<Sealed<Reasoning>>,
    /// Tool calls already validated (or repaired) this turn.
    pub pending_tool_calls: Vec<ToolCall>,
}

impl PartialStreamedTurn {
    /// The assistant message representing this partial turn (reasoning, then
    /// text, then calls), including `current_tool_call` when provided.
    /// `None` when the turn has produced no representable content.
    pub fn assistant_message(&self, current_tool_call: Option<ToolCall>) -> Option<Message> {
        let content = self
            .reasoning
            .iter()
            .cloned()
            .map(AssistantContent::Reasoning)
            .chain(
                self.text
                    .as_ref()
                    .filter(|text| !text.is_empty())
                    .map(|text| AssistantContent::text(text.clone())),
            )
            .chain(
                self.pending_tool_calls
                    .iter()
                    .cloned()
                    .chain(current_tool_call)
                    .map(AssistantContent::ToolCall),
            )
            .collect();
        Some(Message::Assistant {
            id: self.message_id.clone(),
            content: NonEmpty::from_vec(content).ok()?,
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
                    tool_call.function.name.clone(),
                    TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER.to_string(),
                )
            })
            .collect::<Vec<_>>();
        retry_results.push(tool_result_message(
            invalid_tool_call.id,
            invalid_tool_call.function.name,
            feedback,
        ));

        let user_message = Message::User {
            content: NonEmpty::from_vec(retry_results).ok()?,
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
    /// The assistant content to record in history, in the order its parts
    /// started, with ignored calls left out and repaired calls renamed.
    pub choice: Vec<AssistantContent>,
    /// Executable Rig tools advertised to the provider for this turn.
    pub executable_tool_names: BTreeSet<String>,
    /// Tools allowed by the active tool choice for this turn.
    pub allowed_tool_names: BTreeSet<String>,
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
    /// messages are already in the history. Finish the provider stream for
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
    /// Forward the ingested item to the consumer as-is.
    EmitIngested,
    /// Hold the ingested item back: it belongs to a tool call that is not
    /// validated until its end.
    HoldToolCall,
    /// Forward the held items of the call that just ended, then its end
    /// carrying `call` (the call as validated, with a repaired name).
    EmitToolCall {
        /// The validated call.
        call: ToolCall,
    },
    /// The model emitted an unknown or disallowed tool call, or one whose
    /// arguments are not JSON. Resolve it via
    /// `AgentRun::resolve_streamed_invalid_tool_call`, then apply the
    /// outcome with [`StreamedTurnAssembler::resolve_pending_invalid`].
    InvalidToolCall(StreamedInvalidToolCall),
}

#[derive(Clone, Serialize, Deserialize)]
enum PendingInvalid {
    /// A complete tool call with a disallowed name.
    Call { tool_call: ToolCall },
    /// A tool call whose arguments were not JSON. The reply ended with it,
    /// so the only resolutions are abandon or fail.
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
    /// Reasoning text per part, by the part's position.
    reasoning: BTreeMap<usize, String>,
    pending_tool_calls: Vec<ToolCall>,
    pending_invalid: Option<PendingInvalid>,
    /// Calls an [`StreamedResolution::Ignored`] dropped: the stream's
    /// response still carries them, and [`Self::finish`] leaves them out.
    ignored_calls: Vec<CallId>,
    /// Calls a [`StreamedResolution::Repaired`] renamed, applied to the
    /// stream's response at [`Self::finish`].
    repaired_calls: Vec<(CallId, ToolName)>,
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
            reasoning: BTreeMap::new(),
            pending_tool_calls: Vec::new(),
            pending_invalid: None,
            ignored_calls: Vec::new(),
            repaired_calls: Vec::new(),
            excluded_assistant_content: ExclusionCount::default(),
        }
    }

    /// Replayed assistant blocks excluded from assembly so far this turn.
    /// Zero on well-formed provider streams; non-zero means transcript
    /// content was lost (one warning summarizes the count).
    pub fn excluded_assistant_content(&self) -> usize {
        self.excluded_assistant_content.0
    }

    /// Aggregated assistant text streamed so far this turn.
    pub fn aggregated_text(&self) -> &str {
        &self.text
    }

    /// The reasoning text accumulated so far for the part at `index`.
    pub fn aggregated_reasoning(&self, index: usize) -> Option<&str> {
        self.reasoning.get(&index).map(String::as_str)
    }

    /// Ingest one provider stream item and return what the driver must do.
    ///
    /// # Errors
    /// Returns an error when an invalid tool call is still awaiting
    /// resolution.
    pub fn ingest(
        &mut self,
        item: &Item<StreamEvent>,
    ) -> Result<Vec<StreamedTurnEvent>, ProviderError> {
        if self.pending_invalid.is_some() {
            return Err(ProviderError::Response(
                "streamed turn ingested while an invalid tool call awaits resolution".to_string(),
            ));
        }
        let event = match item {
            Item::Event(event) => event,
            Item::Unknown(payload) => {
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
                return Ok(vec![StreamedTurnEvent::EmitIngested]);
            }
        };
        match event {
            StreamEvent::Text { text, .. } => {
                self.text.push_str(text);
                Ok(vec![StreamedTurnEvent::EmitIngested])
            }
            StreamEvent::Reasoning { part, text } => {
                self.reasoning
                    .entry(part.index())
                    .or_default()
                    .push_str(text);
                Ok(vec![StreamedTurnEvent::EmitIngested])
            }
            StreamEvent::Start {
                kind: PartKind::ToolCall,
                ..
            }
            | StreamEvent::Arguments { .. } => Ok(vec![StreamedTurnEvent::HoldToolCall]),
            StreamEvent::End {
                content: AssistantContent::ToolCall(tool_call),
                ..
            } => {
                if !self
                    .allowed_tool_names
                    .contains(tool_call.function.name.as_str())
                {
                    return Ok(self.surface_invalid_call(
                        tool_call.clone(),
                        Some(json_utils::serialize_json_value(
                            &tool_call.function.arguments,
                        )),
                        PendingInvalid::Call {
                            tool_call: tool_call.clone(),
                        },
                        InvalidToolCallReason::UnknownTool,
                    ));
                }
                self.pending_tool_calls.push(tool_call.clone());
                Ok(vec![StreamedTurnEvent::EmitToolCall {
                    call: tool_call.clone(),
                }])
            }
            StreamEvent::Start { .. } | StreamEvent::End { .. } => {
                Ok(vec![StreamedTurnEvent::EmitIngested])
            }
        }
    }

    /// Apply the machine's resolution for the invalid tool call surfaced by
    /// the last [`StreamedTurnEvent::InvalidToolCall`]. A repaired call is
    /// returned to forward under its new name.
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
                PendingInvalid::Call { mut tool_call },
            ) => {
                if let Ok(tool_name) = ToolName::new(tool_name.clone()) {
                    self.repaired_calls
                        .push((tool_call.id.clone(), tool_name.clone()));
                    tool_call.function.name = tool_name;
                }
                self.pending_tool_calls.push(tool_call.clone());
                vec![StreamedTurnEvent::EmitToolCall { call: tool_call }]
            }
            // Repair is rejected upstream for malformed arguments (the run
            // fails closed); reaching here would be a protocol violation, so
            // the call is simply not resurrected.
            (StreamedResolution::Repaired { .. }, PendingInvalid::MalformedArgs { .. })
            | (StreamedResolution::TurnAbandoned { .. }, _) => Vec::new(),
            (
                StreamedResolution::Ignored,
                PendingInvalid::Call { tool_call } | PendingInvalid::MalformedArgs { tool_call },
            ) => {
                self.ignored_calls.push(tool_call.id);
                Vec::new()
            }
        }
    }

    /// Snapshot of the turn so far, for diagnostics and rollback messages.
    /// `ended` is what the stream folded so far; its reasoning parts are the
    /// snapshot's.
    pub fn partial_turn(
        &self,
        message_id: Option<String>,
        ended: &[AssistantContent],
    ) -> PartialStreamedTurn {
        PartialStreamedTurn {
            message_id,
            text: (!self.text.is_empty()).then(|| self.text.clone()),
            reasoning: ended
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Reasoning(reasoning) => Some(reasoning.clone()),
                    _ => None,
                })
                .collect(),
            pending_tool_calls: self.pending_tool_calls.clone(),
        }
    }

    /// Assemble the completed turn from the response the stream folded
    /// into: its choice in start order, with the calls this turn ignored
    /// left out and the ones it repaired renamed.
    pub fn finish(
        self,
        message_id: Option<String>,
        response: &rig_core::completion::CompletionResponse,
    ) -> StreamedTurn {
        let choice = response
            .choice
            .iter()
            .filter(|content| match content {
                AssistantContent::ToolCall(call) => !self.ignored_calls.contains(&call.id),
                AssistantContent::Text(_)
                | AssistantContent::Reasoning(_)
                | AssistantContent::Image(_) => true,
            })
            .cloned()
            .map(|content| match content {
                AssistantContent::ToolCall(mut call) => {
                    if let Some((_, name)) =
                        self.repaired_calls.iter().find(|(id, _)| *id == call.id)
                    {
                        call.function.name = name.clone();
                    }
                    AssistantContent::ToolCall(call)
                }
                other => other,
            })
            .collect();

        StreamedTurn {
            message_id,
            choice,
            executable_tool_names: self.executable_tool_names.clone(),
            allowed_tool_names: self.allowed_tool_names.clone(),
            finish_reason: response.finish_reason(),
        }
    }

    /// Park resolution on `pending` and surface the rejected call to the
    /// caller as an [`StreamedTurnEvent::InvalidToolCall`].
    fn surface_invalid_call(
        &mut self,
        tool_call: ToolCall,
        args: Option<String>,
        pending: PendingInvalid,
        reason: InvalidToolCallReason,
    ) -> Vec<StreamedTurnEvent> {
        let invalid = StreamedInvalidToolCall {
            tool_call,
            args,
            executable_tool_names: self.executable_tool_names.clone(),
            allowed_tool_names: self.allowed_tool_names.clone(),
            reason,
        };
        self.pending_invalid = Some(pending);
        vec![StreamedTurnEvent::InvalidToolCall(invalid)]
    }

    /// Surface a call whose arguments were not JSON, the error that ended
    /// the reply, for resolution. Retains raw argument text and call
    /// identity, with `Null` parsed arguments.
    pub fn surface_malformed_input(
        &mut self,
        detail: &rig_core::error::MalformedToolInput,
    ) -> Vec<StreamedTurnEvent> {
        let Ok(name) = ToolName::new(detail.name.clone()) else {
            return Vec::new();
        };
        let tool_call = ToolCall::new(
            detail.id.clone(),
            ToolFunction::new(name, serde_json::Value::Null),
        );
        self.surface_invalid_call(
            tool_call.clone(),
            Some(detail.raw.clone()),
            PendingInvalid::MalformedArgs { tool_call },
            InvalidToolCallReason::MalformedArguments {
                error: detail.error.clone(),
            },
        )
    }
}

#[cfg(test)]
mod tests;
