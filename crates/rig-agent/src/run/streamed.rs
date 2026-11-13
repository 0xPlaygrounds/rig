//! Streamed-turn assembly and early invalid-call diagnostics for [`super::AgentRun`].
//!
//! Drivers ingest a completion stream's items, resolve invalid calls before
//! continuing, and finish with the response the stream folded into. The
//! assembler performs no I/O and returns forwarding instructions as
//! [`StreamedTurnEvent`].
//!
//! ```
//! use rig_agent::run::streamed::StreamedTurnAssembler;
//! let assembler = StreamedTurnAssembler::new(Default::default());
//! assert_eq!(assembler.excluded_assistant_content(), 0);
//! ```

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use rig_core::completion::{FinishReason, Message};
use rig_core::error::ProviderError;
use rig_core::message::{AssistantContent, AssistantMessage, ToolCall, ToolName};
use rig_core::streaming::{Item, PartKind, StreamEvent};
use rig_core::tool::feedback::{not_executed, tool_result_message};

use super::policy::TurnPolicy;

/// Detect unknown payloads containing assistant content that assembly would lose:
/// tagged assistant blocks or text with a malformed provider item.
fn unknown_payload_loses_assistant_content(payload: &serde_json::Value) -> bool {
    // Deserialize by reference to avoid cloning large unknown payloads.
    if AssistantContent::deserialize(payload).is_ok() {
        return true;
    }
    // Malformed metadata must not hide the loss of an otherwise valid text field.
    payload
        .get("text")
        .is_some_and(serde_json::Value::is_string)
        && payload.get("native").is_some()
}

/// One invalid tool call surfaced mid-stream, awaiting a resolution from
/// `AgentRun::resolve_streamed_invalid_tool_call`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamedInvalidToolCall {
    /// The rejected tool call.
    pub tool_call: ToolCall,
    /// Raw argument payload for diagnostics, when available.
    pub args: Option<String>,
    /// The tool policy of the turn the call was judged under.
    pub policy: TurnPolicy,
}

/// Snapshot of a streamed turn at the moment an invalid tool call appeared.
/// Used by the machine to build diagnostics and rollback messages from
/// exactly what the model has produced so far.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialStreamedTurn {
    /// The turn the reply began, with no content: its origin, so a rolled
    /// back turn is known to come from its model. A reply cut before its end
    /// keeps no provider item, so the rolled back turn replays canonically.
    #[serde(default)]
    pub head: AssistantMessage,
    /// The parts that arrived so far, in the order they started, text still
    /// streaming included.
    pub content: Vec<AssistantContent>,
    /// Tool calls already validated (or repaired) this turn.
    pub pending_tool_calls: Vec<ToolCall>,
}

impl PartialStreamedTurn {
    /// The assistant message representing this partial turn, in arrival
    /// order: each validated call as validated, `current_tool_call` in its
    /// own place, and any other call left out. A call keeps a provider item
    /// only where the partial reply kept it, which a reply cut before its
    /// end never does. `None` when the turn has produced no representable
    /// content.
    pub fn assistant_message(&self, current_tool_call: Option<ToolCall>) -> Option<Message> {
        let (content, _) = self.content_with(current_tool_call);
        if content.is_empty() {
            return None;
        }
        Some(Message::Assistant(self.head.clone().with_content(content)))
    }

    /// The content of [`Self::assistant_message`], and the position in it
    /// of `current_tool_call`. Each call part takes the first call still
    /// unmatched with its id, so calls sharing an id pair in arrival order.
    fn content_with(
        &self,
        current_tool_call: Option<ToolCall>,
    ) -> (Vec<AssistantContent>, Option<usize>) {
        let current_slot = self.pending_tool_calls.len();
        let mut calls: Vec<(usize, ToolCall)> = self
            .pending_tool_calls
            .iter()
            .cloned()
            .chain(current_tool_call)
            .enumerate()
            .collect();
        let mut content = Vec::new();
        let mut current = None;
        let mut push_call = |content: &mut Vec<AssistantContent>,
                             (slot, call): (usize, ToolCall)| {
            if slot == current_slot {
                current = Some(content.len());
            }
            content.push(AssistantContent::ToolCall(call));
        };
        for part in &self.content {
            match part {
                AssistantContent::ToolCall(call) => {
                    let Some(at) = calls.iter().position(|(_, kept)| kept.id == call.id) else {
                        continue;
                    };
                    let (slot, mut kept) = calls.remove(at);
                    kept.native.clone_from(&call.native);
                    push_call(&mut content, (slot, kept));
                }
                part if !part.is_blank() => content.push(part.clone()),
                _ => {}
            }
        }
        for (slot, mut call) in calls {
            call.native = None;
            push_call(&mut content, (slot, call));
        }
        (content, current)
    }

    /// Rollback messages for a retried or skipped streamed turn: the partial
    /// assistant turn plus a user message carrying `feedback` for the invalid
    /// call and a synthetic "not executed" result for each validated peer.
    pub fn rollback_messages(
        &self,
        invalid_tool_call: ToolCall,
        feedback: String,
    ) -> Option<(Message, Message)> {
        // Answered by position, since a peer may share the invalid call's id.
        let (content, invalid_at) = self.content_with(Some(invalid_tool_call));
        let results = content
            .iter()
            .enumerate()
            .filter_map(|(at, part)| match part {
                AssistantContent::ToolCall(call) if Some(at) == invalid_at => {
                    Some(tool_result_message(
                        call.id.clone(),
                        call.function.name.clone(),
                        feedback.clone(),
                    ))
                }
                AssistantContent::ToolCall(call) => Some(not_executed(call)),
                _ => None,
            })
            .collect();
        let assistant_message = Message::Assistant(self.head.clone().with_content(content));
        Some((assistant_message, Message::User { content: results }))
    }
}

/// The assembled streamed turn, fed to
/// `AgentRun::streamed_turn`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamedTurn {
    /// The turn's origin, stop and provider message, without content.
    pub head: AssistantMessage,
    /// The assistant content to record in history, in the order its parts
    /// started, with ignored calls left out and repaired calls renamed.
    pub choice: Vec<AssistantContent>,
    /// The tool policy the turn was prepared with.
    pub policy: TurnPolicy,
    /// Provider-reported terminal reason for this turn, when available.
    pub finish_reason: Option<FinishReason>,
    /// The positions in [`Self::choice`] of the calls the reply was cut off
    /// in the middle of
    /// ([`CompletionResponse::cut_off_calls`](rig_core::completion::CompletionResponse::cut_off_calls)).
    /// Each is answered as cut off and never run, whatever its name.
    pub cut_off: Vec<usize>,
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
    /// messages, the skip's tool results among them, are already in the
    /// history, where [`project`](crate::run::project) reads them. Finish the
    /// provider stream for usage, record the completion call, then call
    /// `AgentRun::next_step`.
    TurnAbandoned,
    /// The invalid call is dropped and the turn goes on without it: the
    /// run's `UnhandledInvalidToolCall::Ignore` on the streaming
    /// surface. Apply it via
    /// [`StreamedTurnAssembler::resolve_pending_invalid`] and keep consuming
    /// the provider stream; nothing of the call enters the run.
    Ignored,
}

/// Required driver action after ingesting a stream item.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum StreamedTurnEvent {
    /// Forward the ingested item to the consumer as-is.
    EmitIngested,
    /// Forward the ingested argument fragment of a call to an allowed tool,
    /// once the tool-call delta hooks let it through. The call is not
    /// executable until its end.
    EmitToolCallDelta,
    /// Hold the ingested item back: it belongs to a call naming a tool the
    /// turn does not allow, which is resolved at its end.
    HoldToolCall,
    /// Forward the held items of the call that just ended, then its end
    /// carrying `call` (the call as validated, with a repaired name).
    EmitToolCall {
        /// The validated call.
        call: ToolCall,
    },
    /// The model emitted an unknown or disallowed tool call. Resolve it via
    /// `AgentRun::resolve_streamed_invalid_tool_call`, then apply the
    /// outcome with [`StreamedTurnAssembler::resolve_pending_invalid`].
    InvalidToolCall(StreamedInvalidToolCall),
}

/// A tool call still streaming: the tool it named as it started, and
/// whether its items stream live.
#[derive(Clone, Serialize, Deserialize)]
struct OpenCall {
    name: Option<ToolName>,
    live: bool,
}

/// A complete tool call with a disallowed name, awaiting resolution.
#[derive(Clone, Serialize, Deserialize)]
struct PendingInvalid {
    tool_call: ToolCall,
}

/// A tool call that ended this turn and was settled without abandoning it.
#[derive(Clone, Serialize, Deserialize)]
struct EndedCall {
    /// The call as validated, with a repaired name.
    call: ToolCall,
    /// Whether a [`StreamedResolution::Ignored`] dropped it.
    ignored: bool,
}

/// Serializable accumulator for one streamed turn. Persisted state requires the
/// same library version; it does not itself resume a provider connection.
/// Drivers must resolve pending invalid calls before ingesting more events.
/// Dropping warns once if assistant content was excluded, otherwise stays silent.
#[derive(Clone, Serialize, Deserialize)]
pub struct StreamedTurnAssembler {
    policy: TurnPolicy,
    /// Tool calls still streaming, by the part's position.
    #[serde(default)]
    open_calls: BTreeMap<usize, OpenCall>,
    /// The calls that ended, in the order they ended. The stream's
    /// response still carries ignored calls and original names;
    /// [`Self::resolved`] applies these to it.
    ended_calls: Vec<EndedCall>,
    pending_invalid: Option<PendingInvalid>,
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
    /// Create an assembler for one streamed turn with the policy the turn
    /// was prepared with.
    pub fn new(policy: TurnPolicy) -> Self {
        Self {
            policy,
            open_calls: BTreeMap::new(),
            ended_calls: Vec::new(),
            pending_invalid: None,
            excluded_assistant_content: ExclusionCount::default(),
        }
    }

    /// Replayed assistant blocks excluded from assembly so far this turn.
    /// Zero on well-formed provider streams; non-zero means transcript
    /// content was lost (one warning summarizes the count).
    pub fn excluded_assistant_content(&self) -> usize {
        self.excluded_assistant_content.0
    }

    /// The tool the streaming call at part `index` named as it started.
    pub fn streaming_tool_name(&self, index: usize) -> Option<&ToolName> {
        self.open_calls
            .get(&index)
            .and_then(|call| call.name.as_ref())
    }

    /// Ingest one provider stream item and return what the driver must do.
    ///
    /// # Errors
    /// Returns an error when an invalid tool call is still awaiting
    /// resolution.
    pub fn ingest(&mut self, item: &Item<StreamEvent>) -> Result<StreamedTurnEvent, ProviderError> {
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
                return Ok(StreamedTurnEvent::EmitIngested);
            }
        };
        match event {
            StreamEvent::Text { .. } | StreamEvent::Reasoning { .. } => {
                Ok(StreamedTurnEvent::EmitIngested)
            }
            StreamEvent::Start {
                part,
                kind: PartKind::ToolCall,
                name,
            } => {
                let live = name
                    .as_ref()
                    .is_some_and(|name| self.policy.allows(name.as_str()));
                let name = name.clone();
                self.open_calls
                    .insert(part.index(), OpenCall { name, live });
                Ok(if live {
                    StreamedTurnEvent::EmitIngested
                } else {
                    StreamedTurnEvent::HoldToolCall
                })
            }
            StreamEvent::Arguments { part, .. } => {
                let live = self
                    .open_calls
                    .get(&part.index())
                    .is_some_and(|call| call.live);
                Ok(if live {
                    StreamedTurnEvent::EmitToolCallDelta
                } else {
                    StreamedTurnEvent::HoldToolCall
                })
            }
            StreamEvent::End {
                part,
                content: AssistantContent::ToolCall(tool_call),
            } => {
                self.open_calls.remove(&part.index());
                if !self.policy.allows(tool_call.function.name.as_str()) {
                    return Ok(self.surface_invalid_call(
                        tool_call.clone(),
                        Some(tool_call.function.arguments_json()),
                    ));
                }
                self.ended_calls.push(EndedCall {
                    call: tool_call.clone(),
                    ignored: false,
                });
                Ok(StreamedTurnEvent::EmitToolCall {
                    call: tool_call.clone(),
                })
            }
            StreamEvent::Start { .. } | StreamEvent::End { .. } => {
                Ok(StreamedTurnEvent::EmitIngested)
            }
        }
    }

    /// Apply the machine's resolution for the invalid tool call surfaced by
    /// the last [`StreamedTurnEvent::InvalidToolCall`]. A repaired call is
    /// returned to forward under its new name.
    pub fn resolve_pending_invalid(
        &mut self,
        resolution: &StreamedResolution,
    ) -> Option<StreamedTurnEvent> {
        let pending = self.pending_invalid.take()?;

        let PendingInvalid { mut tool_call } = pending;
        match resolution {
            StreamedResolution::Repaired { tool_name } => {
                if let Ok(tool_name) = ToolName::new(tool_name.clone()) {
                    tool_call.function.name = tool_name;
                }
                self.ended_calls.push(EndedCall {
                    call: tool_call.clone(),
                    ignored: false,
                });
                Some(StreamedTurnEvent::EmitToolCall { call: tool_call })
            }
            StreamedResolution::TurnAbandoned => None,
            StreamedResolution::Ignored => {
                self.ended_calls.push(EndedCall {
                    call: tool_call,
                    ignored: true,
                });
                None
            }
        }
    }

    /// Keep the invalid tool call surfaced by the last
    /// [`StreamedTurnEvent::InvalidToolCall`] in the turn as it is, for a
    /// call the reply was cut off in (one of its response's
    /// [`cut_off_calls`](rig_core::completion::CompletionResponse::cut_off_calls)).
    /// The run answers it as cut off, so it is not resolved as invalid. The
    /// call is returned to forward like any ended call.
    pub fn keep_pending_cut_off(&mut self) -> Option<StreamedTurnEvent> {
        let PendingInvalid { tool_call } = self.pending_invalid.take()?;
        self.ended_calls.push(EndedCall {
            call: tool_call.clone(),
            ignored: false,
        });
        Some(StreamedTurnEvent::EmitToolCall { call: tool_call })
    }

    /// Snapshot of the turn so far, for diagnostics and rollback messages:
    /// `partial` is what the stream folded so far
    /// ([`Streamed::partial`](rig_core::streaming::Streamed::partial)).
    pub fn partial_turn(
        &self,
        partial: &rig_core::completion::CompletionResponse,
    ) -> PartialStreamedTurn {
        PartialStreamedTurn {
            head: partial.continued(Vec::new()),
            content: self.resolved(&partial.choice),
            pending_tool_calls: self
                .ended_calls
                .iter()
                .filter(|ended| !ended.ignored)
                .map(|ended| ended.call.clone())
                .collect(),
        }
    }

    /// Assemble the completed turn from the response the stream folded
    /// into: its choice in start order, with the calls this turn ignored
    /// left out and the ones it repaired renamed.
    pub fn finish(self, response: &rig_core::completion::CompletionResponse) -> StreamedTurn {
        let choice = self.resolved(&response.choice);
        let cut_off = choice
            .iter()
            .enumerate()
            .filter_map(|(at, content)| match content {
                AssistantContent::ToolCall(call)
                    if response.cut_off_calls().any(|cut| cut.id == call.id) =>
                {
                    Some(at)
                }
                _ => None,
            })
            .collect();
        StreamedTurn {
            head: response.head(),
            choice,
            policy: self.policy,
            finish_reason: response.finish_reason(),
            cut_off,
        }
    }

    /// `choice` with the calls this turn ignored left out and the ones it
    /// repaired renamed. Each call part takes the first ended call still
    /// unmatched with its id, so calls sharing an id pair in the order they
    /// ended, which is the order a reply restated whole lists them in.
    fn resolved(&self, choice: &[AssistantContent]) -> Vec<AssistantContent> {
        let mut ended: Vec<&EndedCall> = self.ended_calls.iter().collect();
        choice
            .iter()
            .filter_map(|content| {
                let AssistantContent::ToolCall(call) = content else {
                    return Some(content.clone());
                };
                let Some(at) = ended.iter().position(|ended| ended.call.id == call.id) else {
                    return Some(content.clone());
                };
                let ended = ended.remove(at);
                (!ended.ignored).then(|| {
                    let mut call = call.clone();
                    call.function.name.clone_from(&ended.call.function.name);
                    AssistantContent::ToolCall(call)
                })
            })
            .collect()
    }

    /// Park resolution on `tool_call` and surface it to the caller as an
    /// [`StreamedTurnEvent::InvalidToolCall`].
    fn surface_invalid_call(
        &mut self,
        tool_call: ToolCall,
        args: Option<String>,
    ) -> StreamedTurnEvent {
        let invalid = StreamedInvalidToolCall {
            tool_call: tool_call.clone(),
            args,
            policy: self.policy.clone(),
        };
        self.pending_invalid = Some(PendingInvalid { tool_call });
        StreamedTurnEvent::InvalidToolCall(invalid)
    }
}

#[cfg(test)]
mod tests;
