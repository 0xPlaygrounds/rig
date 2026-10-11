//! The agent the user talks to when no agent is named, such as the one a
//! front shows first and `--print` sends to: of the agents nothing
//! spawned, the first by id; and the agent the user looks at
//! ([`Shown`]).

use rig_ecs::agent::{Agent, AgentId, SpawnedBy};
use rig_ecs::prelude::*;

/// The agents, each with whether it is a subagent, for [`primary`].
pub type PrimaryQuery<'w, 's> =
    Query<'w, 's, (Entity, &'static AgentId, Has<SpawnedBy>), With<Agent>>;

/// Where an agent sorts among the agents of a session: the ones the user
/// started first, each group by id. The first is the agent the user talks
/// to when none is named.
pub fn primary_order(spawned: bool, id: &AgentId) -> (bool, &str) {
    (spawned, id.0.as_str())
}

/// The agent the user talks to when none is named: the first one by
/// [`primary_order`].
pub fn primary(agents: &PrimaryQuery) -> Option<Entity> {
    agents
        .iter()
        .min_by(|a, b| primary_order(a.2, a.1).cmp(&primary_order(b.2, b.1)))
        .map(|(entity, ..)| entity)
}

/// The agent the user looks at, which a picker's command runs for: the
/// front the user uses sets it, such as the terminal view showing a
/// subagent. In `First`, the harness sets it to the [`primary`] agent
/// while it names none, or one that is gone.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Shown(pub Option<Entity>);

/// Shows the primary agent while [`Shown`] names no agent.
pub(crate) fn keep_shown(mut shown: ResMut<Shown>, agents: PrimaryQuery) {
    if shown.0.is_some_and(|agent| agents.contains(agent)) {
        return;
    }
    shown.set_if_neq(Shown(primary(&agents)));
}
