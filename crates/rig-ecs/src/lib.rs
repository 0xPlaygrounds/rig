//! rig-ecs, the kernel of the rig coding agent: agents, their turns and
//! their calls are Bevy entities, and the calls run on Bevy's task pools.
//! Every call goes through one effect dispatch path, recorded as it settles
//! ([`effects`]); plugins register the tools. The kernel saves every change
//! of an agent's [`Conversation`](agent::Conversation) to the
//! [`session_log::SessionStore`] the app inserts. Time is Bevy's: a retried
//! model call waits on a delayed command, and plugins that animate or poll
//! run on `on_real_timer`, not on threads of their own.
//!
//! The kernel depends on no view and no file system: the app fills in what
//! it needs, such as the store (any [`rig_memory::Store`]) and the
//! [`model::Models`].
//!
//! ```no_run
//! use rig_ecs::session_log::SessionStore;
//! use rig_ecs::prelude::*;
//! use rig_memory::MemoryStore;
//!
//! let mut app = App::new();
//! app.insert_resource(SessionStore::new(MemoryStore::default()))
//!     .add_plugins((AgentPlugin, SessionLogPlugin));
//! ```

pub mod agent;
pub mod effects;
pub mod fork;
pub mod inbox;
pub mod model;
pub mod prompt;
pub mod requests;
pub mod restore;
pub mod retry;
pub mod session_log;
pub mod spawn;
pub mod stage;
pub mod tasks;
pub mod tools;
pub mod turn;

/// What a plugin needs: Bevy's app, ECS, reflection and time preludes (with
/// the time run conditions and `Disabled`), the kernel's components,
/// requests and stages, the tool registry, rig-core's conversation types and
/// rig-memory's store. `Message` is Bevy's message trait, a conversation's
/// message is `message::Message`, and `Conversation` is the agent's
/// component.
pub mod prelude {
    pub use bevy_app::prelude::*;
    pub use bevy_ecs::entity_disabling::Disabled;
    pub use bevy_ecs::prelude::*;
    pub use bevy_reflect::prelude::*;
    pub use bevy_time::common_conditions::{on_real_timer, on_timer};
    pub use bevy_time::prelude::*;

    pub use crate::AgentPlugin;
    pub use crate::agent::{
        ActiveTurn, Agent, AgentId, CacheKey, CallOf, Calls, ContextStale, Conversation,
        EffectParent, LastUsage, Notice, NoticeLevel, Partial, PartialCall, RunTools, Spawned,
        SpawnedBy, Stop, StopReached, StopTurn, SystemPrompt, ToolAccess, Turn, TurnOf,
        TurnOutcome,
    };
    pub use crate::fork::{AgentForked, ForkAgent, ForkError, ForkResult};
    pub use crate::inbox::{
        Attachment, Deliver, DeliveredIn, DeliveryMode, Incoming, MessagePart, MessageRecord,
        Recalled, Unsent,
    };
    pub use crate::model::{
        ChoiceKind, ChoiceTaken, Connection, ModelChoice, ModelRefused, Models, ReasoningSetting,
        SetModel, SetReasoning,
    };
    pub use crate::prompt::{AgentSections, PromptSection, SectionOf};
    pub use crate::requests::{RequestOf, Requests};
    pub use crate::restore::{CarryOn, Interrupted, Restored, SavedKeys};
    pub use crate::retry::Verdict;
    pub use crate::session_log::{
        ConversationChanged, ReflectSaved, Retire, SessionLogPlugin, SessionRestored,
    };
    pub use crate::spawn::{AgentSpec, Origin, SpawnRefused, Spawner, spawn_root};
    pub use crate::stage::{
        Admitted, AgentLoop, AgentPass, AgentWork, BetweenStages, Decision, Effect, EffectRecorded,
        EffectRequest, ForAgent, GateSystems, HoldOf, Holds, IfStopped, IncomingMessages,
        ModelCalls, Outcome, PluginEffects, PrepareSystems, Published, ReadyToSettle, ReadyToStart,
        ReviewSystems, Settled, Stage, Started, SupersededBy, Supersedes, ToolCalls, Waiting,
        despawn_open_children, follow_successor,
    };
    pub use crate::tasks::{Done, PollTasks, Running, Wake};
    pub use crate::tools::{
        AppToolsExt, CallHandler, DeclaredFootprint, Footprint, InReply, ToolCalled, ToolDef,
        ToolOptions, ToolRequest, WorldToolsExt,
    };
    pub use crate::turn::{
        Adapted, Backoff, DroppedOptions, Exiting, FirstStreamed, ModelRequest, Recovery,
        ReportedUsage, RetryVerdict, SentTo, SideCall, StreamDelta, StreamTiming, TurnCall,
    };
    pub use rig_core::conversation::{
        Aside, AsideId, Author, AuthorKind, CallRef, Change, ChangeRecord, Entry, EntryRef, Halt,
        MessageId, RequestId, STOPPED, Spliced, StopCause, TurnEnd, TurnRecord,
    };
    pub use rig_core::effect::Decider;
    pub use rig_core::message::{self, AssistantContent, AssistantMessage, ToolCall, UserContent};
    pub use rig_core::now_ms;
    pub use rig_core::streaming::{Part, PartKind, StreamEvent};
    pub use rig_core::tool::{
        Scope, Tool, ToolContext, ToolExecutionError, ToolResult, Touches, args_schema,
    };
    pub use rig_memory::{
        Files, LogWriteError, Repair, Rewrite, Rewritten, SavedAgent, Session, Stamp, Store,
        Strings, as_of, history,
    };
}

use bevy_app::MainScheduleOrder;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::SingleThreadedExecutor;
use bevy_ecs::schedule::{InternedScheduleLabel, ScheduleLabel};
use bevy_time::TimePlugin;

use agent::{Agent, Notice};
use session_log::{
    SaveConversations, SaveFrame, SavedOne, SessionLog, save_conversations, save_now,
};
use tasks::{PollTasks, Wake, poll_tasks, run_settle_frames};

/// The system in `Last`, on exit, that stops the running turns as the
/// user's stop does, for the [`Exiting`](turn::Exiting) reason, in Bevy's
/// `OnAppExitSystems`. What they leave is saved after `Last`.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StopTurnsOnExit;

/// The kernel's `Startup` steps, in order: [`StartupSystems::FirstAgent`]
/// spawns the first agent when the restored session has none, then
/// [`StartupSystems::Reconcile`] spawns what the restored agents left half
/// done as [`Restored`](restore::Restored) work, which starts their turns. A
/// system that needs the first agent before any restored turn starts goes
/// between them.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StartupSystems {
    /// Spawns the first agent when the restored session has none.
    FirstAgent,
    /// Spawns the restored agents' half-done work.
    Reconcile,
}

/// The frame's last schedule, the kernel's. It runs after `Last` and after
/// the kernel queued what the frame changed for the session log: every
/// conversation changed and every [`ReflectSaved`](session_log::ReflectSaved)
/// value, so whatever a plugin changes in `Last`, on the exit frame too, is
/// saved with the frame. It holds the [`WriteLogs`] systems and what reads
/// what they wrote; a system here that changes saved state is saved with
/// the next frame, and on the exit frame never.
#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Persist;

/// The systems in [`Persist`] that write the frame's log records to the
/// [`session_log::SessionStore`], and those a plugin adds to write its own
/// logs, such as the effect log: such a system goes in [`Persist`] and
/// says `.in_set(WriteLogs)`. A failed write stops the logging and sends a
/// [`Notice`] here, at the end of the frame, which a system in an earlier
/// schedule reads the next frame and, on the exit frame, never: on exit,
/// read [`SessionLog::failure`] in [`Persist`], `.after(WriteLogs)`.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WriteLogs;

/// The kernel: agents, their turn loop, effects and the tool registry, and
/// the saving of every conversation change (announced as
/// [`session_log::ConversationChanged`]).
///
/// It spawns one agent at startup when the restored session has none, and
/// logs to the app's [`session_log::SessionStore`] once
/// [`session_log::SessionLogPlugin`] restored the session. It adds Bevy's
/// `TimePlugin` unless the app has one, for the clock a retried call waits
/// on, and then runs `First`, `PreUpdate`, `Last` and [`Persist`]
/// single-threaded. It sets no error handler.
pub struct AgentPlugin;

impl Plugin for AgentPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<TimePlugin>() {
            // An app without a clock of its own is a small headless one, not
            // a game: its `First` and `PreUpdate` hold little but the clock's
            // systems, and its `Last` and `Persist` little but the session
            // log's, too little for Bevy's multi-threaded executor, which
            // would wake the task pool for them every frame (a frame that
            // saves a conversation paid more in thread handoffs than in
            // saving).
            let single = |schedule: &mut Schedule| {
                schedule.set_executor(SingleThreadedExecutor::new());
            };
            app.add_plugins(TimePlugin)
                .edit_schedule(First, single)
                .edit_schedule(PreUpdate, single)
                .edit_schedule(Last, single)
                .edit_schedule(Persist, single);
        }
        if let Some(mut order) = app.world_mut().get_resource_mut::<MainScheduleOrder>() {
            let at = |order: &MainScheduleOrder, label: InternedScheduleLabel| {
                order.labels.iter().position(|it| *it == label)
            };
            // A conversation changed before `Update` is saved before it runs.
            if let Some(update) = at(&order, Update.intern()) {
                order.labels.insert(update, SaveConversations.intern());
            }
            // Saved and written after every schedule a plugin runs systems in.
            if let Some(last) = at(&order, Last.intern()) {
                order.labels.insert(last + 1, SaveFrame.intern());
                order.labels.insert(last + 2, Persist.intern());
            }
        }
        app.init_resource::<effects::Effects>()
            .init_resource::<SessionLog>()
            .init_resource::<SavedOne>()
            .init_resource::<spawn::PendingIds>()
            .edit_schedule(SaveConversations, |schedule| {
                schedule.set_executor(SingleThreadedExecutor::new());
            })
            .edit_schedule(SaveFrame, |schedule| {
                schedule.set_executor(SingleThreadedExecutor::new());
            })
            .add_systems(SaveConversations, save_conversations)
            .add_systems(SaveFrame, save_now.map(drop))
            .init_resource::<Wake>()
            .init_resource::<model::Models>()
            .init_resource::<retry::RetryPolicy>()
            .init_resource::<tools::UnavailableCalls>()
            .add_message::<Notice>()
            .add_message::<session_log::ConversationChanged>()
            .add_message::<inbox::Recalled>()
            // Notices and commits are written by systems that never meet
            // on one effect, and their order within a pass means nothing.
            .allow_ambiguous_resource::<Messages<Notice>>()
            .allow_ambiguous_resource::<Messages<session_log::ConversationChanged>>()
            .configure_sets(Last, StopTurnsOnExit.in_set(bevy_app::OnAppExitSystems))
            .configure_sets(
                Startup,
                (StartupSystems::FirstAgent, StartupSystems::Reconcile).chain(),
            )
            .add_systems(
                Startup,
                spawn_first_agent.in_set(StartupSystems::FirstAgent),
            )
            // A reply that came in this frame has its calls started in it.
            .configure_sets(Update, PollTasks.before(stage::AgentLoop))
            .add_systems(Update, poll_tasks.in_set(PollTasks))
            .add_systems(
                PreUpdate,
                model::reconnect.run_if(resource_changed::<model::Models>),
            )
            .add_systems(
                Last,
                (
                    run_settle_frames,
                    turn::stop_turns_on_exit
                        .in_set(StopTurnsOnExit)
                        .run_if(on_message::<AppExit>),
                ),
            )
            .add_observer(session_log::retire)
            .add_observer(requests::end_with_settled)
            .add_observer(requests::refuse_ended)
            .add_observer(inbox::on_deliver)
            .add_observer(turn::on_stop_turn)
            .add_observer(turn::reach_once_gone)
            .add_observer(inbox::unsent_on_turn_end)
            .add_observer(inbox::unmark)
            .add_observer(model::on_set_model)
            .add_observer(model::connect)
            .add_observer(model::check_effort)
            .add_observer(model::on_set_reasoning);
        stage::build(app);
    }
}

fn spawn_first_agent(world: &mut World) {
    let mut agents = world.query_filtered::<(), With<Agent>>();
    if agents.iter(world).next().is_none() {
        spawn::spawn_root(world);
    }
}
