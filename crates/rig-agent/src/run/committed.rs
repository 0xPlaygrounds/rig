//! The run's append-only history, and the one projection of its tool calls
//! and results that the engine and a foreign host both stream.

use rig_core::completion::Message;
use rig_core::message::{AssistantContent, ToolCall, ToolResult, UserContent};
use serde::{Deserialize, Serialize};

use crate::agent::MultiTurnStreamItem;

/// The run's history: the prompt and every message a turn commits after it.
/// Only appends; the single in-place edit is the not-yet-sent prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct CommittedLog(Vec<Message>);

impl CommittedLog {
    pub(crate) fn new(prompt: Message) -> Self {
        Self(vec![prompt])
    }

    pub(crate) fn commit(&mut self, message: Message) {
        self.0.push(message);
    }

    pub(crate) fn commit_all(&mut self, messages: impl IntoIterator<Item = Message>) {
        self.0.extend(messages);
    }

    /// Replace the prompt while it is the only message, before any turn
    /// committed anything after it. Returns whether it was replaced.
    pub(crate) fn replace_unstarted_prompt(&mut self, prompt: Message) -> bool {
        match self.0.as_mut_slice() {
            [slot] => {
                *slot = prompt;
                true
            }
            _ => false,
        }
    }
}

impl std::ops::Deref for CommittedLog {
    type Target = [Message];

    fn deref(&self) -> &[Message] {
        &self.0
    }
}

/// A tool call or result committed to a run's history, in commit order.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum CommittedItem<'a> {
    /// An assistant tool call.
    ToolCall(&'a ToolCall),
    /// A user tool result answering a call.
    ToolResult(&'a ToolResult),
}

/// Every tool call and tool result in `committed`, in message order: the
/// calls of each assistant message and the results of each user message.
/// Text, reasoning and the prompt's other content yield nothing.
///
/// Pure and sans-IO. A driver keeps a cursor, starting at
/// [`AgentRun::messages`](super::AgentRun::messages)`().len()` so a resumed
/// run does not replay, and after each protocol call projects the messages
/// past it and moves it to the end. Outside a pending
/// [`CallTools`](super::AgentRunStep::CallTools) batch every projected call
/// has a projected result: a final output-tool turn is committed without its
/// call, and an output reprompt commits the call with its feedback result.
///
/// ```
/// use rig_agent::run::{AgentRun, CommittedItem, project};
/// let run = AgentRun::new("hi");
/// let cursor = run.messages().len();
/// // ... drive the run one protocol call ...
/// for item in project(&run.messages()[cursor..]) {
///     if let CommittedItem::ToolResult(result) = item {
///         println!("result for {}", result.call);
///     }
/// }
/// ```
pub fn project(committed: &[Message]) -> impl Iterator<Item = CommittedItem<'_>> {
    committed.iter().flat_map(|message| {
        let (calls, results): (&[AssistantContent], &[UserContent]) = match message {
            Message::Assistant(assistant) => (&assistant.content, &[]),
            Message::User { content } => (&[], content),
            Message::System { .. } => (&[], &[]),
        };
        let calls = calls.iter().filter_map(|item| match item {
            AssistantContent::ToolCall(call) => Some(CommittedItem::ToolCall(call)),
            _ => None,
        });
        let results = results.iter().filter_map(|item| match item {
            UserContent::ToolResult(result) => Some(CommittedItem::ToolResult(result)),
            _ => None,
        });
        calls.chain(results)
    })
}

/// The stream's tool items for `committed`, the only place they are built.
///
/// `executed` is the settled batch's effective calls, by call position:
/// `Some` for a call whose body ran, which announces it with
/// [`ToolExecutionCommitted`](MultiTurnStreamItem::ToolExecutionCommitted)
/// right before its result. It is empty when no batch settled. A batch whose
/// committed results do not line up with it is refused with the number of
/// results that were committed.
pub(crate) fn committed_stream_items(
    committed: &[Message],
    executed: &[Option<ToolCall>],
) -> Result<Vec<MultiTurnStreamItem>, usize> {
    let mut items = Vec::new();
    let mut results = 0;
    for item in project(committed) {
        match item {
            CommittedItem::ToolCall(tool_call) => items.push(MultiTurnStreamItem::ToolCall {
                tool_call: tool_call.clone(),
            }),
            CommittedItem::ToolResult(tool_result) => {
                if let Some(Some(tool_call)) = executed.get(results) {
                    items.push(MultiTurnStreamItem::ToolExecutionCommitted {
                        tool_call: tool_call.clone(),
                    });
                }
                results += 1;
                items.push(MultiTurnStreamItem::ToolResult {
                    tool_result: tool_result.clone(),
                });
            }
        }
    }
    let aligned = executed.is_empty() || executed.len() == results;
    aligned.then_some(items).ok_or(results)
}

#[cfg(test)]
mod tests;
