//! The agent the user talks to when no agent is named, such as the one a
//! front shows first and `--print` sends to: the session's
//! [`CurrentAgent`], always one nothing spawned; and the agent the user
//! looks at ([`ShownAgent`]). Showing an agent makes the top-level agent
//! it belongs to current, so `/fork`, rewind and `/agents` move it in any
//! front, and every front continues the branch the user showed last.
//!
//! ```
//! use rig_harness::prelude::*;
//!
//! /// Greets the agent the user talks to.
//! fn greet(agents: MainAgentQuery, mut notices: MessageWriter<Notice>) {
//!     if let Some(agent) = main_agent(&agents) {
//!         notices.write(Notice::info(agent, "hello"));
//!     }
//! }
//!
//! App::new().add_plugins(CurrentAgentPlugin).add_systems(Update, greet);
//! ```

use bevy_ecs::system::SystemParam;
use rig_ecs::SaveSystems;
use rig_ecs::agent::{Agent, AgentId, SpawnedBy};
use rig_ecs::prelude::*;

/// The id of the session's current agent, the top-level agent the user
/// talks to when none is named ([`main_agent`]); empty before one was
/// shown. Saved with the session under rig-memory's `CURRENT`
/// (`rig::current_agent`), which a library program reads.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Resource, Saved, Default, Clone, Debug, PartialEq)]
#[type_path = "rig"]
#[type_name = "current_agent"]
pub struct CurrentAgent(pub String);

/// What [`main_agent`] reads: the [`CurrentAgent`], if the app has one,
/// and the agents nothing spawned.
#[derive(SystemParam)]
pub struct MainAgentQuery<'w, 's> {
    current: Option<Res<'w, CurrentAgent>>,
    agents: Query<'w, 's, (Entity, &'static AgentId), (With<Agent>, Without<SpawnedBy>)>,
}

/// The agent the user talks to when none is named: the live agent
/// [`CurrentAgent`] names, else, in a session that names none yet, such as
/// a new one, the top-level agent first by id, as rig-memory's
/// `Session::main_agent` finds it.
pub fn main_agent(agents: &MainAgentQuery) -> Option<Entity> {
    let current = agents.current.as_ref().map(|current| current.0.as_str());
    let named = (agents.agents.iter()).find(|(_, id)| Some(id.0.as_str()) == current);
    named
        .or_else(|| agents.agents.iter().min_by_key(|(_, id)| &id.0))
        .map(|(entity, _)| entity)
}

/// The agent the user looks at, which a picker's command runs for: the
/// front the user uses sets it, such as the terminal view showing a
/// subagent. While it names no live agent, the harness sets it to the
/// [`main_agent`].
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShownAgent(pub Option<Entity>);

/// The [`CurrentAgent`] and the [`ShownAgent`], kept as the module says;
/// part of the harness.
pub struct CurrentAgentPlugin;

impl Plugin for CurrentAgentPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CurrentAgent>()
            .init_resource::<ShownAgent>()
            .add_systems(First, keep_shown)
            .add_systems(
                Last,
                follow_shown
                    .run_if(resource_changed::<ShownAgent>)
                    .in_set(CurrentAgentSystems),
            )
            .configure_sets(Last, CurrentAgentSystems.before(SaveSystems));
    }
}

/// Where the [`CurrentAgent`] follows the shown agent, in `Last`, before
/// the session saves it: a system there that reads the current agent runs
/// after it.
#[derive(SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CurrentAgentSystems;

/// Shows the main agent while [`ShownAgent`] names no live agent.
fn keep_shown(
    mut shown: ResMut<ShownAgent>,
    current: MainAgentQuery,
    agents: Query<(), With<Agent>>,
) {
    if shown.0.is_some_and(|agent| agents.contains(agent)) {
        return;
    }
    shown.set_if_neq(ShownAgent(main_agent(&current)));
}

/// Makes the top-level agent of the shown agent current.
fn follow_shown(
    shown: Res<ShownAgent>,
    parents: Query<&SpawnedBy>,
    ids: Query<&AgentId, With<Agent>>,
    mut current: ResMut<CurrentAgent>,
) {
    let Some(shown) = shown.0 else {
        return;
    };
    let root = parents.iter_ancestors(shown).last().unwrap_or(shown);
    if let Ok(id) = ids.get(root)
        && current.0 != id.0
    {
        current.0.clone_from(&id.0);
    }
}

#[cfg(test)]
mod tests;
