//! The agent core: agents as entities, the turn loop, the one effect
//! dispatch path, the tool and command registries, models, and session
//! saving. It depends on neither the host nor any view: the host fills in
//! what the core needs, such as [`save::SessionPaths`].

pub mod agent;
pub mod blocking;
pub mod commands;
pub mod effects;
pub mod models;
pub mod save;
pub mod tools;
pub mod turn;

use bevy_app::prelude::*;
use bevy_ecs::error::warn;
use bevy_ecs::prelude::*;
use bevy_log::{info, warn};

use agent::{Agent, AgentId, Notice, NoticeLevel, PickRequest, TurnFinished};
use effects::Effects;
use save::SessionPaths;
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
            .add_systems(Startup, (spawn_first_agent, describe_tools))
            .add_systems(
                Update,
                (
                    turn::start_completions.in_set(AgentSystems::Start),
                    (turn::poll_model_calls, turn::poll_tool_calls).in_set(AgentSystems::Poll),
                    turn::settle_tools.in_set(AgentSystems::Settle),
                ),
            )
            .add_systems(
                Last,
                (
                    log_agents,
                    turn::stop_turns_on_exit
                        .in_set(bevy_app::OnAppExitSystems)
                        .before(save::save_session)
                        .run_if(on_message::<AppExit>),
                ),
            )
            .add_observer(turn::on_submit)
            .add_observer(turn::on_interrupt)
            .add_observer(turn::on_set_model)
            .add_observer(turn::on_model_chosen)
            .add_observer(turn::on_set_effort);
    }
}

fn spawn_first_agent(agents: Query<(), With<Agent>>, mut commands: Commands) {
    if agents.is_empty() {
        commands.spawn((Name::new("agent"), Agent));
    }
}

/// Logs each notice, at its level, and each finished turn, with the stable
/// id of the agent they are about.
fn log_agents(
    mut notices: MessageReader<Notice>,
    mut finished: MessageReader<TurnFinished>,
    agents: Query<&AgentId>,
) {
    for notice in notices.read() {
        let agent = notice
            .agent
            .and_then(|agent| agents.get(agent).ok())
            .map_or("-", |id| id.0.as_str());
        match notice.level {
            NoticeLevel::Info => info!(agent, "notice: {}", notice.text),
            NoticeLevel::Error => warn!(agent, "notice: {}", notice.text),
        }
    }
    for turn in finished.read() {
        if let Ok(id) = agents.get(turn.agent) {
            info!(agent = %id.0, "turn finished");
        }
    }
}

/// Describes every registered tool in the effect log's header.
fn describe_tools(effects: Res<Effects>, tools: Query<&tools::ToolHandler>) {
    effects.describe(tools.iter().map(|tool| tool.0.descriptor()).collect());
}
