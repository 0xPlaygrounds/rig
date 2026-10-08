//! The agent core: agents as entities, the turn loop, subagents, the one
//! effect dispatch path, the tool and command registries, models, and
//! session saving. It depends on neither the host nor any view: the host fills in
//! what the core needs, such as [`save::SessionPaths`].

pub mod agent;
pub mod attach;
pub mod blocking;
pub mod calls;
pub mod commands;
pub mod compaction;
pub mod effects;
pub mod inbox;
pub mod models;
pub mod prompt;
pub mod recovery;
pub mod save;
pub mod subagents;
pub mod tools;
pub mod turn;
pub mod usage;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::{info, warn};

use agent::{Agent, AgentId, Notice, NoticeLevel, PickRequest, TurnFinished};
use calls::{Wake, poll_calls};
use compaction::Summary;
use effects::Effects;
use recovery::RetryDue;
use rig_core::message::ToolResult;
use save::SessionPaths;
use turn::{ModelReply, PollCalls};

/// Agents, their turn loop, effects, and the tool and command registries.
/// Spawns one agent at startup when the restored session has none. It sets
/// no error handler: that is the application's choice.
pub struct AgentPlugin;

impl Plugin for AgentPlugin {
    fn build(&self, app: &mut App) {
        let log = app
            .world()
            .get_resource::<SessionPaths>()
            .map(|paths| paths.effects());
        let effects = Effects::continuing(log.as_deref());
        app.insert_resource(effects)
            .init_resource::<Wake>()
            .add_message::<Notice>()
            .add_message::<TurnFinished>()
            .add_message::<PickRequest>()
            .add_message::<inbox::Recalled>()
            .add_systems(
                Startup,
                (spawn_first_agent, describe_tools, subagents::link_restored),
            )
            .add_systems(
                Update,
                (
                    poll_calls::<ModelReply>,
                    poll_calls::<ToolResult>,
                    poll_calls::<RetryDue>,
                    poll_calls::<Summary>,
                    turn::stream_partials,
                )
                    .in_set(PollCalls),
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
            .add_observer(turn::on_call_model)
            .add_observer(turn::on_model_done)
            .add_observer(turn::on_tool_done)
            .add_observer(turn::on_retry_due)
            .add_observer(turn::on_compact)
            .add_observer(turn::on_summarize)
            .add_observer(turn::on_summary_done)
            .add_observer(turn::on_retry)
            .add_observer(turn::on_interrupt)
            .add_observer(turn::on_turn_end)
            .add_observer(inbox::on_follow_up)
            .add_observer(inbox::recall_on_turn_end)
            .add_observer(subagents::answer_on_turn_end)
            .add_observer(subagents::stop_when_unassigned)
            .add_observer(turn::on_set_model)
            .add_observer(turn::on_model_chosen)
            .add_observer(turn::on_set_effort)
            .add_observer(usage::log_turn_spending);
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

/// Describes every registered tool in the effect log's header, by name.
fn describe_tools(effects: Res<Effects>, tools: Query<(&tools::ToolDef, &tools::ToolHandler)>) {
    let mut tools: Vec<_> = tools.iter().collect();
    tools.sort_by(|a, b| a.0.0.name.as_str().cmp(b.0.0.name.as_str()));
    effects.describe(
        tools
            .into_iter()
            .map(|(_, tool)| tool.0.descriptor())
            .collect(),
    );
}
