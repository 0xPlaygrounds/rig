//! The agent the user talks to when no agent is named, such as the one a
//! front shows first and `--print` sends to: of the agents nothing
//! spawned, the first by id; and the agent the user looks at
//! ([`ShownAgent`]).

use rig_ecs::agent::{Agent, AgentId, SpawnedBy};
use rig_ecs::prelude::*;
use rig_memory::main_order;

/// The agents, each with whether it is a subagent, for [`main_agent`].
pub type MainAgentQuery<'w, 's> =
    Query<'w, 's, (Entity, &'static AgentId, Has<SpawnedBy>), With<Agent>>;

/// The agent the user talks to when none is named: the first one by
/// rig-memory's `main_order`, which a library program sorts saved agents by.
pub fn main_agent(agents: &MainAgentQuery) -> Option<Entity> {
    agents
        .iter()
        .min_by(|a, b| main_order(a.2, &a.1.0).cmp(&main_order(b.2, &b.1.0)))
        .map(|(entity, ..)| entity)
}

/// The agent the user looks at, which a picker's command runs for: the
/// front the user uses sets it, such as the terminal view showing a
/// subagent. In `First`, the harness sets it to the [`main_agent`]
/// while it names none, or one that is gone.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShownAgent(pub Option<Entity>);

/// Shows the main agent while [`ShownAgent`] names no agent.
pub(crate) fn keep_shown(mut shown: ResMut<ShownAgent>, agents: MainAgentQuery) {
    if shown.0.is_some_and(|agent| agents.contains(agent)) {
        return;
    }
    shown.set_if_neq(ShownAgent(main_agent(&agents)));
}
