//! The separable core: agents, the agent loop, the dispatch path, the
//! registry and the session. Nothing here depends on a view.

pub mod agent;
pub mod dispatch;
pub mod models;
pub(crate) mod process;
pub mod registry;
pub mod reload;
pub mod save;
pub mod session;
pub mod turn;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

/// The phases of the agent loop in `Update`: finished work is collected
/// first, then new work starts.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AgentSet {
    /// Model and tool calls are polled and their results applied.
    Poll,
    /// Queued model calls and the next tool call of each agent start.
    Start,
}

/// Agent components, the loop, the dispatch path, the registry events and
/// the session's effect log. Spawns the first agent at startup.
#[derive(Default)]
pub struct CorePlugin;

impl Plugin for CorePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<agent::AgentId>()
            .register_type::<agent::Conversation>()
            .register_type::<agent::Model>()
            .register_type::<agent::Effort>()
            .register_type::<agent::SystemPrompt>()
            .register_type::<agent::ToolAccess>()
            .init_resource::<dispatch::Effects>()
            .configure_sets(Update, (AgentSet::Poll, AgentSet::Start).chain())
            .add_observer(agent::connect)
            .add_observer(turn::submit)
            .add_observer(turn::stop)
            .add_observer(registry::run_command)
            .add_systems(Startup, spawn_first_agent)
            .add_systems(
                Update,
                (
                    (turn::poll_model_calls, turn::poll_tool_calls)
                        .chain()
                        .in_set(AgentSet::Poll),
                    (turn::start_tool_calls, turn::start_model_calls)
                        .chain()
                        .in_set(AgentSet::Start),
                ),
            )
            .add_systems(Last, dispatch::flush_effects);
    }
}

/// Spawns one agent unless a plugin already did, and greets it.
fn spawn_first_agent(
    agents: Query<(), With<agent::Agent>>,
    session: Res<session::Session>,
    mut commands: Commands,
) {
    if !agents.is_empty() {
        return;
    }
    let agent = commands.spawn(agent::Agent).id();
    commands.trigger(registry::Notice::info(
        agent,
        "Type a message, or /help for commands. Pick a model with /model.",
    ));
    if !session.dir.is_dir() {
        commands.trigger(registry::Notice::error(
            agent,
            format!(
                "Cannot create the session directory {}; logs and effects are not saved.",
                session.dir.display()
            ),
        ));
    }
}
