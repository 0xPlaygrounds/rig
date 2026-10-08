//! The agent core: agents as entities, the turn loop, the one effect
//! dispatch path, the tool and command registries, and session saving. It
//! does not depend on any view.

pub mod agent;
pub mod commands;
pub mod effects;
pub mod models;
pub mod session;
pub mod tools;
pub mod turn;

use bevy_app::prelude::*;
use bevy_ecs::error::warn;
use bevy_ecs::prelude::*;
use bevy_log::info;

use agent::{Agent, Notice, PickRequest, TurnFinished};
use effects::Effects;
use session::SessionPaths;
use turn::AgentSystems;

/// Agents, their turn loop, effects, and the tool and command registries.
/// Spawns one agent at startup when the restored session has none.
pub struct AgentPlugin;

impl Plugin for AgentPlugin {
    fn build(&self, app: &mut App) {
        // A panicking or failing system, observer or command from a plugin
        // is logged instead of stopping the app.
        if app.get_error_handler().is_none() {
            app.set_error_handler(warn);
        }
        let log = app
            .world()
            .get_resource::<SessionPaths>()
            .map(SessionPaths::effects);
        let effects = Effects::continuing(log.as_deref());
        app.insert_resource(effects)
            .add_message::<Notice>()
            .add_message::<TurnFinished>()
            .add_message::<PickRequest>()
            .configure_sets(
                Update,
                (
                    AgentSystems::Start,
                    AgentSystems::Poll,
                    AgentSystems::Settle,
                )
                    .chain(),
            )
            .add_systems(Startup, spawn_first_agent)
            .add_systems(
                Update,
                (
                    turn::start_completions.in_set(AgentSystems::Start),
                    (turn::poll_model_calls, turn::poll_tool_calls).in_set(AgentSystems::Poll),
                    turn::settle_tools.in_set(AgentSystems::Settle),
                ),
            )
            .add_systems(Last, log_notices)
            .add_observer(turn::on_submit)
            .add_observer(turn::on_interrupt)
            .add_observer(turn::on_set_model)
            .add_observer(turn::on_set_effort);
    }
}

fn spawn_first_agent(agents: Query<(), With<Agent>>, mut commands: Commands) {
    if agents.is_empty() {
        commands.spawn((Name::new("agent"), Agent));
    }
}

fn log_notices(mut notices: MessageReader<Notice>) {
    for notice in notices.read() {
        info!("notice: {}", notice.0);
    }
}
