//! Subagents: agents that start, message and wait on other agents, built
//! only on the kernel's public API: open tools, an effect of its own
//! ([`SpawnSubagent`]), spawned agents ([`SpawnedBy`]), messages
//! ([`Incoming`]) with an [`Author`], [`TurnOutcome`], [`Restored`] work and
//! saved components. Leave [`SubagentsPlugin`] out of the app, or replace
//! it, and there are no subagents.
//!
//! - `task` spawns a child with its own model, reasoning setting and tools,
//!   and answers with its id; `message` sends a child, a peer or an agent
//!   that asked the caller a request or a report; `wait` keeps the
//!   caller's turn open until an agent messages it.
//! - Each request stays in the owing agent's [`OpenRequests`] (saved) until
//!   it reports to the asker with a message marked [`Report`]. A restart
//!   reports open requests as stopped.
//! - Stopping an agent's turn stops its subagents' turns too
//!   ([`StopTurn`] follows [`SpawnedBy`]). A subagent stopped so reports
//!   why as a note from this plugin ([`DeliveryMode::Inform`]) that starts
//!   no turn of the agent that asked: its model reads it with its next
//!   turn. One whose turn a plugin ends, such as for a budget, reports as
//!   any other.
//! - A subagent's [`Lifecycle`] says whether it works, waits on an agent
//!   that owes it a report, or finished. Its final answer is its report.
//!   Of several `task` reports from one reply, only the last carries the
//!   parent on; the others are Inform messages.
//! - With the default `tui` feature, each agent's status line says its
//!   name and the agents at work; `AgentsPlugin`'s `/agents` shows one.
//! - Subagents are told their role in prompt sections of their own
//!   ([`SectionOf`]). A request or `wait` that would make agents wait on
//!   each other is refused.
//!
//! Every agent is spawned through a [`SpawnSubagent`] effect, so other
//! plugins shape spawning with ordinary stage systems, as below.
//! [`SubagentPolicy`] caps nesting; a plugin spawns its own
//! [`Role::Helper`] the same way. Reports are messages like any other, so a
//! plugin edits or takes them over as it would any message.
//!
//! ```
//! use rig_harness::prelude::*;
//! use rig_basics::subagents::{Role, SpawnSubagent, SpawnedAgent, SubagentSystems};
//!
//! /// Subagents (not helpers) get only the read-only tools.
//! fn read_only(
//!     mut requests: Query<&mut SpawnSubagent, Added<Admitted>>,
//!     tools: Query<(&ToolDef, &Footprint)>,
//! ) {
//!     let reading = tools.iter().filter(|(_, footprint)| footprint.0.reads_only());
//!     let reading: Vec<String> = reading.map(|(def, _)| def.0.name.to_string()).collect();
//!     for mut request in requests.iter_mut().filter(|request| request.role == Role::Task) {
//!         request.tools = Some(reading.clone());
//!     }
//! }
//!
//! /// An agent nothing spawned starts one subagent, no more.
//! fn one_at_a_time(
//!     mut requests: Query<
//!         (&ForAgent, &mut Decision),
//!         (ReadyToStart, PluginEffects, With<SpawnSubagent>, Without<Ask>),
//!     >,
//!     roots: Query<&Spawned, Without<SpawnedBy>>,
//! ) {
//!     for (ForAgent(parent), mut decision) in &mut requests {
//!         if roots.get(*parent).is_ok_and(|spawned| spawned.iter().next().is_some()) {
//!             decision.decide(Decider::plugin("one-at-a-time"), Outcome::refused("one at a time"));
//!         }
//!     }
//! }
//!
//! /// Tells each new agent where the project's notes are.
//! fn with_notes(spawned: Query<&SpawnedAgent, ReadyToSettle>, mut commands: Commands) {
//!     for agent in &spawned {
//!         let notes = PromptSection::new(PromptSection::ORDER_PROJECT, "notes", "See NOTES.md.");
//!         commands.spawn((notes, SectionOf(agent.0)));
//!     }
//! }
//!
//! let mut app = App::new();
//! app.add_systems(AgentPass, (
//!     read_only.in_set(Stage::Prepare),
//!     one_at_a_time.in_set(Stage::Gate).after(SubagentSystems::Gate),
//!     with_notes.in_set(Stage::Review),
//! ));
//! ```

use rig_core::effect::EffectId;

use rig_harness::prelude::*;
use serde::{Deserialize, Serialize};

use crate::approval::ApprovalPlugin;

#[cfg(feature = "tui")]
mod agents;
mod lifecycle;
#[cfg(feature = "tui")]
mod looks;
mod spawn;
mod tools;

pub use spawn::{Role, SUBAGENT, SpawnSubagent, SpawnedAgent, SubagentPolicy};
pub use tools::{MESSAGE, TASK, WAIT};

/// The name the subagent tools decide effects and write notes as
/// ([`Decider::plugin`], [`Author::plugin`]), in effect logs and
/// conversations: the module's.
pub const ORIGIN: &str = "subagents";

/// The plugin's stage system sets.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SubagentSystems {
    /// In Gate: refuses a request deeper than [`SubagentPolicy`] allows.
    Gate,
    /// After every gate, before approvals and [`GateSystems::Fallback`]:
    /// after a restart, stops the restored turn of an agent that owed
    /// reports, so they report as stopped. To let that work carry on, a
    /// plugin claims it in Gate (hold, decide, supersede, or end the turn);
    /// removing `Interrupted` alone is not enough.
    StopRestored,
}

/// Adds the `task`, `message` and `wait` tools, spawning the agents
/// [`SpawnSubagent`] asks for, the subagents' lifecycle and the reports
/// that answer their requests; with the `tui` feature (on by default),
/// also how the terminal view draws the tools' calls.
#[derive(Default)]
pub struct SubagentsPlugin;

impl Plugin for SubagentsPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<ApprovalPlugin>() {
            app.add_plugins(ApprovalPlugin);
        }
        app.configure_sets(AgentPass, SubagentSystems::Gate.in_set(Stage::Gate))
            .keeps_saved::<Subtask>()
            .keeps_saved::<Helper>()
            .keeps_saved::<Peers>()
            .keeps_saved::<OpenRequests>();
        spawn::add(app);
        tools::add(app);
        lifecycle::add(app);
        #[cfg(feature = "tui")]
        looks::add(app);
        #[cfg(feature = "tui")]
        agents::add(app);
    }
}

/// On a subagent: it runs a task, whose short title is its
/// [`AgentTitle`], which names it.
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub struct Subtask;

/// On a helper a plugin spawned ([`Role::Helper`]); its title is its
/// [`AgentTitle`], which names it.
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub struct Helper;

/// On a message: it is an agent's report on requests it was sent; its
/// author names the agent and the last request.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct Report;

/// On a subagent: it may `message` and `wait` for its siblings that have
/// [`Peers`] too, and they for it. A `task` with `peers` set inserts it.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub struct Peers;

/// On an agent: the requests other agents sent it, oldest first, until it
/// reports on them. Each gets exactly one report, to its asker.
/// Saved through serde, as it holds rig-core's [`RequestId`].
#[derive(Component, Reflect, Clone, Debug, Default, Serialize, Deserialize)]
#[reflect(
    opaque,
    Component,
    Saved,
    Default,
    Clone,
    Debug,
    Serialize,
    Deserialize
)]
#[serde(transparent)]
pub struct OpenRequests(pub Vec<Request>);

/// A request an agent owes a report on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    /// The call that asked, which the report names.
    pub id: RequestId,
    /// The agent that asked, which the report goes to.
    pub asker: AgentId,
    /// For a `task` reported with the other `task` calls of one reply:
    /// that reply's model call. Not saved: a restart answers every open
    /// request at once.
    #[serde(skip)]
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
    /// answer is its report on every request it owes. A rewrite of the
    /// session that changes it inserts it again, mapped.
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
fn answered(text: String) -> Outcome {
    ToolResult::success(text.into()).into()
}

/// A call's refusal.
fn refused(why: String) -> Outcome {
    ToolResult::failed(ToolExecutionError::other(why)).into()
}

#[cfg(test)]
mod tests;
