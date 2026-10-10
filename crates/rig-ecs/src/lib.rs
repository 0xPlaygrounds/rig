//! rig-ecs: a Bevy agent runtime on rig-core. Agents are entities whose
//! components hold their conversation, model, reasoning setting, system
//! prompt and tool access; a running turn is an entity of its agent, and
//! its model and tool calls are entities of the turn that run on Bevy's
//! task pools. Every call goes through one recorded effect dispatch path.
//! Tools and commands are registered by plugins, and the session journal
//! is kept in the [`store::SessionStore`] the app inserts. Views read what
//! the agents do from [`activity`]. Time is Bevy's: a retried model call
//! waits on a delayed command, and plugins that animate or poll run on
//! `on_real_timer` with a [`calls::KeepAwake`], instead of threads of their
//! own.
//!
//! The runtime depends on no view and no file system: the app fills in
//! what it needs, such as the store and the [`models::ModelConnector`].
//! Features: `subagents` (default) adds the [`subagents::SubagentsPlugin`]
//! tools, and `fs-journal` a JSON-lines [`fs_journal::JsonlDirStore`],
//! native-only.
//!
//! ```no_run
//! use rig_ecs::prelude::*;
//! use rig_ecs::store::{MemoryStore, SessionStore};
//!
//! let mut app = App::new();
//! app.insert_resource(SessionStore::new(MemoryStore::default()))
//!     .add_plugins((AgentPlugin, JournalPlugin));
//! ```

pub mod activity;
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
pub mod restore;
pub mod store;
#[cfg(feature = "subagents")]
pub mod subagents;
pub mod tools;
pub mod turn;
pub mod usage;

/// What a plugin needs: Bevy's app, ECS, reflection and time preludes
/// with the time run conditions, the agent components and requests, and
/// the tool and command registries.
pub mod prelude {
    pub use bevy_app::prelude::*;
    pub use bevy_ecs::prelude::*;
    pub use bevy_reflect::prelude::*;
    pub use bevy_time::common_conditions::{on_real_timer, on_timer};
    pub use bevy_time::prelude::*;

    pub use crate::AgentPlugin;
    pub use crate::activity::{Activity, ActivitySystems, MessageFeed};
    pub use crate::agent::{
        ActiveTurn, Agent, AgentId, CallOf, Compact, Connection, Conversation, EffectParent,
        Effort, Interrupt, ModelChoice, Notice, NoticeLevel, Retry, SetEffort, SetModel,
        SettingsChosen, Spawned, SpawnedBy, SystemPrompt, ToolAccess, ToolCallRun, TurnEnded,
        TurnOf, TurnOutcome,
    };
    pub use crate::calls::{KeepAwake, Wake};
    pub use crate::commands::{AppCommandsExt, CommandArgs, RunCommand};
    pub use crate::compaction::Compacted;
    pub use crate::inbox::{
        Attachment, Deliver, DeliveryMode, Inbox, Origin, OriginKind, Recalled, RequestId,
    };
    pub use crate::journal::{AppSaveExt, JournalPlugin};
    pub use crate::prompt::{PromptSection, ToolRules};
    pub use crate::restore::Restored;
    pub use crate::tools::{AppToolsExt, Footprint, ToolCalled, ToolOptions, ToolOutput, failed};
    pub use crate::turn::{Backoff, Recovery};
    pub use crate::usage::{Spending, TurnSpending};
    pub use rig_core::tool::{PortableTool, Tool, ToolExecutionError, args_schema};
}

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::{info, warn};
use bevy_time::{Time, TimePlugin, Virtual};

use agent::{Agent, AgentId, Notice, NoticeLevel};
use calls::{Done, Wake, poll_calls, settle};
use compaction::{CompactionPolicy, Summary};
use effects::Effects;
use journal::SessionLog;
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
/// the session. It adds Bevy's `TimePlugin` unless the app has it, for the
/// clock a retried model call waits on, and then lets that clock count
/// frames up to [`calls::MAX_FRAME_GAP`] apart in full. It sets no error
/// handler: that is the application's choice.
pub struct AgentPlugin;

impl Plugin for AgentPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<TimePlugin>() {
            app.add_plugins(TimePlugin)
                .insert_resource(Time::<Virtual>::from_max_delta(calls::MAX_FRAME_GAP));
        }
        let store = app.world().get_resource::<SessionStore>().cloned();
        let effects = Effects::continuing(store.as_ref().map(|store| &*store.0));
        app.add_plugins(activity::ActivityPlugin)
            .insert_resource(effects)
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
                    poll_calls::<Summary, Done<Summary>>,
                    turn::stream_partials,
                )
                    .in_set(PollCalls),
            )
            .add_systems(
                Last,
                (
                    log_agents,
                    inbox::start_turns.before(settle).before(WriteJournal),
                    settle,
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
