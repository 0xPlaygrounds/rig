//! Subagents: agents that start, message and wait on other agents, built
//! only on the kernel's public API: open tools, spawned agents
//! ([`SpawnedBy`]), [`Deliver`] with an [`Origin`], [`TurnEnded`],
//! [`Restored`] and saved components. Leave [`SubagentsPlugin`] out of the
//! app, or replace it, and there are no subagents.
//!
//! `task` spawns a child agent with its own model, reasoning setting and
//! tools, and answers at once with its id; `message` sends a child or a
//! peer a request, or an agent that asked the caller its report; `wait`
//! keeps the caller's turn open until an agent sends it a message. Each
//! request stays one of the owing agent's [`OpenRequests`], saved with it,
//! until it reports to the asker; a restart answers each as interrupted.
//! Subagents see the app's [`PromptSection`]s like every agent, and are
//! told their role in sections of their own ([`SectionOf`]), not in their
//! saved system prompt, which is their parent's.
//!
//! A subagent's [`Lifecycle`] is [`Working`] while a turn runs,
//! [`WaitingOn`] an agent that owes it a report when its turn ended, and
//! [`Finished`] with its answer when nothing it asked is open; inserting
//! [`Finished`] reports the answer on every open request. So a subagent
//! waiting on a peer never reports its "waiting" text, and its later answer
//! reaches its asker. The reports of the `task` calls of one reply are
//! notes ([`DeliveryMode::Note`]) but the last, which carries the parent
//! on. A request or `wait` that would make agents wait on each other is
//! refused.
//!
//! [`Working`]: Lifecycle::Working
//! [`WaitingOn`]: Lifecycle::WaitingOn
//! [`Finished`]: Lifecycle::Finished

use rig_core::effect::EffectId;

use rig_harness::prelude::*;

mod lifecycle;
#[cfg(feature = "tui")]
mod looks;
mod tools;

pub use tools::{MAX_DEPTH, MESSAGE, TASK, WAIT};

/// Adds the `task`, `message` and `wait` tools, the subagents' lifecycle
/// and the reports that answer their requests; with the `tui` feature (on
/// by default), also how the terminal view draws the tools' calls.
#[derive(Default)]
pub struct SubagentsPlugin;

impl Plugin for SubagentsPlugin {
    fn build(&self, app: &mut App) {
        tools::add(app);
        lifecycle::add(app);
        #[cfg(feature = "tui")]
        looks::add(app);
    }
}

/// On a subagent: its task's short title, which names it.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub struct Subtask(pub String);

/// On a subagent: it may `message` and `wait` for its siblings that have
/// [`Peers`] too, and they for it. A `task` with `peers` set inserts it.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub struct Peers;

/// On an agent: the requests other agents sent it, oldest first, until it
/// reports on them. Each gets exactly one report, to its asker.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub struct OpenRequests(pub Vec<Request>);

/// A request an agent owes a report on.
#[derive(Reflect, Clone, Debug, PartialEq, Eq)]
pub struct Request {
    /// The call that asked, which the report names.
    pub id: RequestId,
    /// The agent that asked, which the report goes to.
    pub asker: AgentId,
    /// For a `task` reported with the other `task` calls of one reply:
    /// that reply's model call. Not saved: a restart answers every open
    /// request at once.
    #[reflect(ignore)]
    pub batch: Option<EffectId>,
}

/// On a subagent: where it stands with the requests it owes. It never
/// changes in place; each insert is a step. A restored agent has none
/// until its next turn.
#[derive(Component, Reflect, Clone, Debug, PartialEq, Eq)]
#[component(immutable)]
#[reflect(Component, Clone, Debug)]
pub enum Lifecycle {
    /// A turn of it runs, or its task is about to start one.
    Working,
    /// Its turn ended while this agent owes it a report, which carries it
    /// on; it reports nothing meanwhile.
    WaitingOn(Entity),
    /// Its turn ended with this answer and nothing it asked is open: the
    /// answer is its report on every request it owes.
    Finished(String),
}

/// On an open `wait` call: the agent whose message it waits for.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship(relationship_target = AwaitedBy)]
pub struct Awaits(pub Entity);

/// On an agent: the open `wait` calls waiting for a message from it.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship_target(relationship = Awaits)]
pub struct AwaitedBy(Vec<Entity>);

/// A call's answer.
fn answered(text: String) -> ToolOutput {
    ToolOutput(ToolResult::success(text.into()))
}

/// A call's refusal.
fn refused(why: String) -> ToolOutput {
    ToolOutput(ToolResult::failed(ToolExecutionError::other(why)))
}

#[cfg(test)]
mod tests;
