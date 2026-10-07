//! The run's append-only history, and the one pure projection of its tool
//! calls and results that the engine and a foreign host both stream.

use rig_core::completion::Message;
use rig_core::message::{AssistantContent, ToolCall, ToolResult, UserContent};
use serde::{Deserialize, Serialize};

/// The run's history: the prompt and every message a turn commits after it.
/// Only appends; the single in-place edit is the not-yet-sent prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct CommittedLog(Vec<Message>);

impl CommittedLog {
    pub(crate) fn new(prompt: Message) -> Self {
        Self(vec![prompt])
    }

    pub(crate) fn commit(&mut self, messages: impl IntoIterator<Item = Message>) {
        self.0.extend(messages);
    }

    /// The prompt while it is the only message, before any turn committed
    /// anything after it.
    pub(crate) fn unstarted_prompt(&mut self) -> Option<&mut Message> {
        match self.0.as_mut_slice() {
            [slot] => Some(slot),
            _ => None,
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

/// Every tool call and tool result in `committed`, in message order; all
/// other content yields nothing.
///
/// Pure and sans-IO. A driver keeps a cursor, starting at
/// [`AgentRun::projection_start`](super::AgentRun::projection_start), and
/// after each protocol call projects the messages past it and moves it to the
/// end. Outside a pending
/// [`CallTools`](super::AgentRunStep::CallTools) batch every projected call
/// has a projected result: a final output-tool turn is committed without its
/// call, and an output reprompt commits the call with its feedback result.
///
/// ```
/// use rig_agent::run::{AgentRun, CommittedItem, project};
/// let run = AgentRun::new("hi");
/// let cursor = run.projection_start(); // then drive one protocol call
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
        calls.chain(results.iter().filter_map(|item| match item {
            UserContent::ToolResult(result) => Some(CommittedItem::ToolResult(result)),
            _ => None,
        }))
    })
}

#[cfg(test)]
mod tests;
