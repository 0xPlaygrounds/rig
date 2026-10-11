//! Subagents: agents that start, message and wait on other agents, built
//! only on the kernel's public API: open tools, an effect of its own
//! ([`SpawnSubagent`]), spawned agents ([`SpawnedBy`]), messages
//! ([`Incoming`]) with an [`Origin`], [`TurnOutcome`], [`Restored`] work and
//! saved components. Leave [`SubagentsPlugin`] out of the app, or replace
//! it, and there are no subagents.
//!
//! `task` asks for a child agent with its own model, reasoning setting and
//! tools, and answers with its id once it is spawned; `message` sends a
//! child or a peer a request, or an agent that asked the caller its report;
//! `wait` keeps the caller's turn open until an agent sends it a message.
//! Each request stays one of the owing agent's [`OpenRequests`], saved with
//! it, until it reports to the asker (a message marked [`Report`]); a
//! restart stops the work it was doing for them, which reports each as
//! interrupted. A subagent's [`Lifecycle`] says whether it works, waits on
//! an agent that owes it a report, or finished with its answer, which is
//! its report; the reports of the `task` calls of one reply are notes but
//! the last, which carries the parent on. Subagents are told their role in
//! prompt sections of their own ([`SectionOf`]), and a request or `wait`
//! that would make agents wait on each other is refused.
//!
//! Other plugins decide how agents are spawned with ordinary stage systems,
//! since every agent is spawned through a [`SpawnSubagent`] request, an
//! effect: a Prepare system edits it (a `task` call's own arguments are the
//! call's, the request's parent), a Gate system refuses it or holds it until,
//! say, a slot frees, and a Review system reads the agent spawned
//! ([`SpawnedAgent`]) to add components of its own. [`SubagentPolicy`] says
//! how deep agents nest. A plugin spawns a [`Role::Helper`] of its own the
//! same way, which reports to no one and `message` cannot reach. Reports
//! are messages like any to edit or take over, and a subagent whose turn a
//! plugin stops reports the reason and what it had said.
//!
//! ```
//! use rig_harness::prelude::*;
//! use rig_subagents::{Role, SpawnSubagent, SpawnedAgent};
//!
//! /// Subagents (not helpers) get only the read-only tools.
//! fn read_only(
//!     mut requests: Query<&mut SpawnSubagent, Added<Admitted>>,
//!     tools: Query<(&ToolDef, &Footprint)>,
//! ) {
//!     let reading = tools.iter().filter(|(_, footprint)| **footprint == Footprint::ReadOnly);
//!     let reading: Vec<String> = reading.map(|(def, _)| def.0.name.to_string()).collect();
//!     for mut request in requests.iter_mut().filter(|request| request.role == Role::Task) {
//!         request.tools = Some(reading.clone());
//!     }
//! }
//!
//! /// An agent nothing spawned starts one subagent, no more.
//! fn one_at_a_time(
//!     requests: Query<(Entity, &ForAgent), (With<SpawnSubagent>, Undecided)>,
//!     roots: Query<&Spawned, Without<SpawnedBy>>,
//!     mut commands: Commands,
//! ) {
//!     for (request, ForAgent(parent)) in &requests {
//!         if roots.get(*parent).is_ok_and(|spawned| spawned.iter().next().is_some()) {
//!             commands.entity(request).insert_if_new(Outcome::refused("one at a time"));
//!         }
//!     }
//! }
//!
//! /// Tells each new agent where the project's notes are.
//! fn with_notes(spawned: Query<&SpawnedAgent, Unreviewed>, mut commands: Commands) {
//!     for agent in &spawned {
//!         let notes = PromptSection::new(PromptSection::ORDER_PROJECT, "notes", "See NOTES.md.");
//!         commands.spawn((notes, SectionOf(agent.0)));
//!     }
//! }
//!
//! let mut app = App::new();
//! app.add_systems(AgentPass, (
//!     read_only.in_set(Stage::Prepare),
//!     one_at_a_time.in_set(Stage::Gate),
//!     with_notes.in_set(Stage::Review),
//! ));
//! ```

use rig_core::effect::EffectId;

use rig_harness::prelude::*;

mod lifecycle;
#[cfg(feature = "tui")]
mod looks;
mod spawn;
mod tools;

pub use spawn::{Role, SUBAGENT, SpawnSubagent, SpawnedAgent, SubagentPolicy};
pub use tools::{MESSAGE, TASK, WAIT};

/// Adds the `task`, `message` and `wait` tools, spawning the agents
/// [`SpawnSubagent`] asks for, the subagents' lifecycle and the reports
/// that answer their requests; with the `tui` feature (on by default),
/// also how the terminal view draws the tools' calls.
#[derive(Default)]
pub struct SubagentsPlugin;

impl Plugin for SubagentsPlugin {
    fn build(&self, app: &mut App) {
        spawn::add(app);
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

/// On a helper a plugin spawned ([`Role::Helper`]): its title, which names
/// it.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub struct Helper(pub String);

/// On a message: it is an agent's report on requests it was sent; its
/// origin names the agent and the last request.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct Report;

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
fn answered(text: String) -> Outcome {
    ToolResult::success(text.into()).into()
}

/// A call's refusal.
fn refused(why: String) -> Outcome {
    ToolResult::failed(ToolExecutionError::other(why)).into()
}

#[cfg(test)]
mod tests;
