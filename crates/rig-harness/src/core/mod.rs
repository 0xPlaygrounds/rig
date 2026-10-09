//! The agent core: agents as entities and the agents they spawn, the turn loop, the one
//! effect dispatch path, the tool and command registries, models, and the
//! session logs. It
//! depends on neither the host nor any view, nor on the file system: the
//! app fills in what the core needs, such as the [`store::SessionStore`]
//! the session is kept in and the [`models::ModelConnector`].

pub mod agent;
pub mod calls;
pub mod commands;
pub mod compaction;
pub mod effects;
#[cfg(feature = "fs-journal")]
pub mod fs_journal;
pub mod inbox;
pub mod journal;
pub mod models;
pub mod prompt;
pub mod recovery;
pub mod restore;
pub mod store;
pub mod tools;
pub mod turn;
pub mod usage;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::{info, warn};

use agent::{Agent, AgentId, Notice, NoticeLevel};
use calls::{Done, Wake, poll_calls};
use compaction::{CompactionPolicy, Summary};
use effects::Effects;
use journal::SessionLog;
use recovery::RetryDue;
use rig_core::message::ToolResult;
use store::SessionStore;
use turn::{ModelReply, PollCalls};

/// The system in `Last`, on exit, that stops the running turns and leaves
/// them for the restart (see [`turn::Exiting`]). A system that logs what
/// the turns left runs after it.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StopTurns;

/// The system in `Last` that writes the frame's journal records and
/// effects to the [`SessionStore`]. A system that logs for the frame runs
/// before it.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WriteJournal;

/// Agents, their turn loop, effects, and the tool and command registries.
/// Spawns one agent at startup when the restored session has none. Its
/// [`SessionLog`] logs to the [`SessionStore`] inserted before it is
/// built, if any, and nothing until [`journal::JournalPlugin`] restored
/// the session. It sets no error handler: that is the application's choice.
pub struct AgentPlugin;

impl Plugin for AgentPlugin {
    fn build(&self, app: &mut App) {
        let store = app.world().get_resource::<SessionStore>().cloned();
        let effects = Effects::continuing(store.as_ref().map(|store| &*store.0));
        app.insert_resource(effects)
            .insert_resource(SessionLog::new(store.map(|store| store.0)))
            .init_resource::<Wake>()
            .init_resource::<models::ModelConnector>()
            .init_resource::<CompactionPolicy>()
            .add_message::<Notice>()
            .add_message::<inbox::Recalled>()
            .add_systems(Startup, (spawn_first_agent, describe_tools))
            .add_systems(
                Update,
                (
                    poll_calls::<ModelReply, Done<ModelReply>>,
                    poll_calls::<ToolResult, tools::ToolOutput>,
                    poll_calls::<RetryDue, Done<RetryDue>>,
                    poll_calls::<Summary, Done<Summary>>,
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
                        .in_set(StopTurns)
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
