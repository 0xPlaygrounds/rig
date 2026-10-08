//! The tool calls of one assistant turn and the answers fed for them.

use rig_core::message::{Message, ToolCall, UserContent};
use serde::{Deserialize, Serialize};

use super::calls::{ExecCall, MalformedCall, PendingToolCall};

/// What one tool call of the current batch requires, decided at batch creation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) enum ToolSlot {
    /// The driver executes the call.
    Execute(ToolCall),
    /// The call's arguments are not a JSON object; it is answered, never run.
    Malformed(ToolCall),
    /// The call's result content is settled.
    Answered(UserContent),
}

impl ToolSlot {
    /// The single owner of a call's classification: a preresolved result
    /// answers it, unparsed arguments make it malformed, and anything else is
    /// executable.
    pub(super) fn classify(tool_call: ToolCall, preresolved: Option<UserContent>) -> Self {
        match preresolved {
            Some(content) => Self::Answered(content),
            None if tool_call.function.invalid_arguments.is_some() => Self::Malformed(tool_call),
            None => Self::Execute(tool_call),
        }
    }
}

/// One assistant turn's tool calls, each answered once its result is settled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct ToolBatch {
    pub(super) slots: Vec<ToolSlot>,
}

impl ToolBatch {
    /// The calls of turn `turn`'s batch a driver must still answer, in call order.
    pub(super) fn pending(&self, turn: usize) -> Vec<PendingToolCall> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| match slot {
                ToolSlot::Execute(tool_call) => Some(PendingToolCall::Execute(ExecCall {
                    turn,
                    index,
                    tool_call: tool_call.clone(),
                })),
                ToolSlot::Malformed(tool_call) => Some(PendingToolCall::Malformed(MalformedCall {
                    turn,
                    index,
                    tool_call: tool_call.clone(),
                })),
                ToolSlot::Answered(_) => None,
            })
            .collect()
    }

    /// The results as one user message in call order, once every call is answered.
    pub(super) fn complete(self) -> Result<Message, Self> {
        if self
            .slots
            .iter()
            .all(|slot| matches!(slot, ToolSlot::Answered(_)))
        {
            Ok(self.interrupt())
        } else {
            Err(self)
        }
    }

    /// The results so far in call order, every unanswered call closed by
    /// [`close_pending`](rig_core::transcript::close_pending).
    pub(super) fn interrupt(&self) -> Message {
        let open = self.slots.iter().filter_map(|slot| match slot {
            ToolSlot::Execute(tool_call) | ToolSlot::Malformed(tool_call) => Some(tool_call),
            ToolSlot::Answered(_) => None,
        });
        let mut closed = match rig_core::transcript::close_pending(open) {
            Message::User { content } => content.into_iter(),
            _ => Vec::new().into_iter(),
        };
        let content = self
            .slots
            .iter()
            .filter_map(|slot| match slot {
                ToolSlot::Answered(content) => Some(content.clone()),
                _ => closed.next(),
            })
            .collect();
        Message::User { content }
    }
}
