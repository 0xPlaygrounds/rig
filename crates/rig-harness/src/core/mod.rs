//! The agent core: agents as entities and the agents they spawn, the turn loop, the one
//! effect dispatch path, the tool and command
//! registries, models and `/login` sign-ins, and the session logs. It
//! depends on neither the host nor any view: the host fills in what the core
//! needs, such as [`journal::SessionPaths`].

pub mod agent;
pub mod attach;
pub mod blocking;
pub mod calls;
pub mod commands;
pub mod compaction;
pub mod effects;
pub mod inbox;
pub mod journal;
pub mod login;
pub mod models;
pub mod prompt;
pub mod recovery;
pub mod restore;
pub mod tools;
pub mod turn;
pub mod usage;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::{info, warn};

use agent::{Agent, AgentId, Notice, NoticeLevel, PickRequest};
use calls::{Done, Wake, poll_calls};
use compaction::Summary;
use effects::Effects;
use journal::{SessionLog, SessionPaths};
use recovery::RetryDue;
use rig_core::message::ToolResult;
use turn::{ModelReply, PollCalls};

/// Agents, their turn loop, effects, and the tool and command registries.
/// Spawns one agent at startup when the restored session has none. Its
/// [`SessionLog`] logs nothing until [`journal::JournalPlugin`] restored
/// the session. It sets no error handler: that is the application's choice.
pub struct AgentPlugin;

impl Plugin for AgentPlugin {
    fn build(&self, app: &mut App) {
        let paths = app.world().get_resource::<SessionPaths>().cloned();
        let effects_log = paths.as_ref().map(|paths| paths.effects());
        let effects = Effects::continuing(effects_log.as_deref());
        app.insert_resource(effects)
            .insert_resource(SessionLog::new(paths.map(|paths| paths.0)))
            .init_resource::<Wake>()
            .add_message::<Notice>()
            .add_message::<PickRequest>()
            .add_message::<inbox::Recalled>()
            .add_systems(Startup, (spawn_first_agent, describe_tools))
            .add_systems(
                Update,
                (
                    poll_calls::<ModelReply, Done<ModelReply>>,
                    poll_calls::<ToolResult, tools::ToolOutput>,
                    poll_calls::<RetryDue, Done<RetryDue>>,
                    poll_calls::<Summary, Done<Summary>>,
                    poll_calls::<login::SignedInResult, Done<login::SignedInResult>>,
                    login::show_login_prompts,
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
                        .run_if(on_message::<AppExit>),
                ),
            )
            .add_observer(inbox::on_deliver)
            .add_observer(commands::on_run_command)
            .add_observer(turn::on_call_model)
            .add_observer(turn::on_model_done)
            .add_observer(turn::on_tool_done)
            .add_observer(turn::on_retry_due)
            .add_observer(turn::on_compact)
            .add_observer(turn::on_summarize)
            .add_observer(turn::on_summary_done)
            .add_observer(turn::on_retry)
            .add_observer(turn::on_interrupt)
            .add_observer(login::on_sign_in)
            .add_observer(login::on_signed_in)
            .add_observer(login::on_sign_out)
            .add_observer(login::cancel_on_interrupt)
            .add_observer(turn::on_turn_despawn)
            .add_observer(inbox::recall_on_turn_end)
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

/// Logs each notice, at its level, with the stable id of the agent it is
/// about.
fn log_agents(mut notices: MessageReader<Notice>, agents: Query<&AgentId>) {
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
}

/// Describes every registered tool in the effect log's header, by name.
fn describe_tools(effects: Res<Effects>, tools: Query<&tools::ToolDef>) {
    let mut tools: Vec<_> = tools.iter().collect();
    tools.sort_by(|a, b| a.0.name.as_str().cmp(b.0.name.as_str()));
    effects.describe(tools.into_iter().map(tools::ToolDef::descriptor).collect());
}
