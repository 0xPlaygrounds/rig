//! rig-ecs: a Bevy agent runtime on rig-core. Agents are entities whose
//! components hold their conversation, model, reasoning setting, system
//! prompt and tool access; a running turn is an entity of its agent, and
//! its model and tool calls are entities of the turn that run on Bevy's
//! task pools. Every call goes through one effect dispatch path, which a
//! plugin may record ([`effects`]). Tools and commands are registered by
//! plugins, and the session journal is kept in the
//! [`journal::SessionStore`] the app inserts. Time is Bevy's: a retried
//! model call waits on a delayed command, and plugins that animate or poll
//! run on `on_real_timer` with a [`calls::KeepAwake`], instead of threads
//! of their own.
//!
//! The runtime depends on no view and no file system: the app fills in
//! what it needs, such as the store (one of rig-cassette's
//! [`journal`](rig_cassette::journal) stores) and the [`model::Models`].
//!
//! ```no_run
//! use rig_cassette::journal::MemoryStore;
//! use rig_ecs::journal::SessionStore;
//! use rig_ecs::prelude::*;
//!
//! let mut app = App::new();
//! app.insert_resource(SessionStore::new(MemoryStore::default()))
//!     .add_plugins((AgentPlugin, JournalPlugin));
//! ```

pub mod agent;
pub mod calls;
pub mod commands;
pub mod effects;
pub mod inbox;
pub mod journal;
pub mod model;
pub mod prompt;
pub mod restore;
pub mod tools;
pub mod turn;

/// What a plugin needs: Bevy's app, ECS, reflection and time preludes
/// with the time run conditions and `Disabled`, the agent components and
/// requests, the tool and command registries, and rig-core's message types
/// (`message::Message` is the conversation's message; the prelude's own
/// `Message` is Bevy's message trait).
pub mod prelude {
    pub use bevy_app::prelude::*;
    pub use bevy_ecs::entity_disabling::Disabled;
    pub use bevy_ecs::prelude::*;
    pub use bevy_reflect::prelude::*;
    pub use bevy_time::common_conditions::{on_real_timer, on_timer};
    pub use bevy_time::prelude::*;

    pub use crate::AgentPlugin;
    pub use crate::agent::{
        ActiveTurn, Agent, AgentId, CallOf, Calls, Condensed, Conversation, EffectParent,
        Interrupt, LastUsage, Notice, NoticeLevel, Partial, Queued, Retry, Spawned, SpawnedBy,
        SystemPrompt, ToolAccess, ToolCallRun, TurnEnded, TurnOf, TurnOutcome,
    };
    pub use crate::calls::{Done, KeepAwake, PollCalls, Running, Wake};
    pub use crate::commands::{AppCommandsExt, CommandArgs, RunCommand, SlashCommand};
    pub use crate::inbox::{
        Attachment, Deliver, DeliveryMode, Inbox, Origin, OriginKind, Recalled, RequestId,
    };
    pub use crate::journal::{Commit, Committed, JournalPlugin, ReflectSaved, SessionRestored};
    pub use crate::model::{Connection, Effort, ModelChoice, Models, SetEffort, SetModel};
    pub use crate::prompt::{PromptSection, ToolRules};
    pub use crate::restore::Restored;
    pub use crate::tools::{AppToolsExt, Footprint, ToolCalled, ToolDef, ToolOptions, ToolOutput};
    pub use crate::turn::{
        Backoff, CallModel, ModelFailed, ModelReply, ModelRequest, PrepareRequest, Recovery,
    };
    pub use rig_core::message::{self, AssistantContent, AssistantMessage, ToolCall, UserContent};
    pub use rig_core::tool::{PortableTool, Tool, ToolExecutionError, ToolResult, args_schema};
}

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::SingleThreadedExecutor;
use bevy_time::{Time, TimePlugin, Virtual};

use agent::{Agent, Notice};
use calls::{PollCalls, Wake, poll_calls, settle};
use journal::SessionLog;

/// The system in `Last`, on exit, that stops the running turns and leaves
/// them for the restart (see [`turn::Exiting`]). It runs before
/// [`WriteJournal`], both in Bevy's `OnAppExitSystems`.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StopTurns;

/// The systems in `Last` that write the frame's journal records to the
/// [`journal::SessionStore`], and those a plugin adds to write its own logs, such as
/// the effect log: such a system says `.in_set(WriteJournal)` and nothing
/// else, and runs after the turns stopped on exit, so it writes what they
/// left too.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WriteJournal;

/// Agents, their turn loop, effects, and the tool and command registries.
/// Spawns one agent at startup when the restored session has none. Its
/// [`SessionLog`] logs to the app's [`journal::SessionStore`], if any, once
/// [`journal::JournalPlugin`] restored the session from it. It adds Bevy's `TimePlugin` unless the app has it, for the
/// clock a retried model call waits on, and then runs `First` and
/// `PreUpdate` on Bevy's single-threaded executor. It lets that clock count
/// frames up to [`calls::MAX_FRAME_GAP`] apart in full under any loop. It
/// sets no error handler: that is the application's choice.
pub struct AgentPlugin;

impl Plugin for AgentPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<TimePlugin>() {
            // An app without a clock of its own is a small headless one, not
            // a game: its `First` and `PreUpdate` hold little but the clock's
            // systems, too little for Bevy's multi-threaded executor, which
            // would wake the task pool for them every frame.
            let single = |schedule: &mut Schedule| {
                schedule.set_executor(SingleThreadedExecutor::new());
            };
            app.add_plugins(TimePlugin)
                .edit_schedule(First, single)
                .edit_schedule(PreUpdate, single);
        }
        app.init_resource::<effects::Effects>()
            .init_resource::<SessionLog>()
            .init_resource::<Wake>()
            .init_resource::<model::Models>()
            .add_message::<Notice>()
            .add_message::<journal::Committed>()
            .add_message::<inbox::Recalled>()
            .configure_sets(
                Last,
                (StopTurns, WriteJournal)
                    .chain()
                    .in_set(bevy_app::OnAppExitSystems),
            )
            .add_systems(Startup, spawn_first_agent)
            .add_systems(
                Update,
                (poll_calls, turn::stream_partials).in_set(PollCalls),
            )
            .add_systems(
                Last,
                (
                    inbox::start_turns.before(settle).before(WriteJournal),
                    settle,
                    turn::stop_turns_on_exit
                        .in_set(StopTurns)
                        .run_if(on_message::<AppExit>),
                ),
            )
            .add_observer(inbox::on_deliver)
            .add_observer(commands::on_run_command)
            .add_observer(turn::on_call_model)
            .add_observer(turn::on_model_done)
            .add_observer(turn::on_tool_done)
            .add_observer(turn::on_model_request)
            .add_observer(turn::on_retry)
            .add_observer(turn::on_interrupt)
            .add_observer(turn::on_turn_despawn)
            .add_observer(inbox::recall_on_turn_end)
            .add_observer(model::on_set_model)
            .add_observer(model::connect)
            .add_observer(model::check_effort)
            .add_observer(model::on_set_effort);
    }

    fn finish(&self, app: &mut App) {
        if let Some(mut time) = app.world_mut().get_resource_mut::<Time<Virtual>>() {
            time.set_max_delta(calls::MAX_FRAME_GAP);
        }
    }
}

fn spawn_first_agent(agents: Query<(), With<Agent>>, mut commands: Commands) {
    if agents.is_empty() {
        commands.spawn((Name::new("agent"), Agent));
    }
}
