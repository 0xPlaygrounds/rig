//! rig-harness: a terminal coding agent built as a Bevy app on rig-core.
//!
//! An agent is an entity whose components hold its conversation, model,
//! reasoning setting, system prompt and tool access; an agent a plugin
//! spawns for another, such as a subagent of the built-in
//! [`SubagentsPlugin`](builtin::SubagentsPlugin), is one more,
//! [`SpawnedBy`](core::agent::SpawnedBy) that agent. A running turn is an
//! entity of its agent, and the turn's model calls and tool calls are
//! entities of the turn that run on Bevy's task pools; every one goes
//! through the one recorded dispatch path. Tools and
//! slash commands are registered by Bevy plugins, the built-in ones exactly
//! as a third-party plugin registers its own.
//!
//! [`RigHarnessPlugins`] is the agent app: the session, its mode (the
//! terminal view, or `--print` without one), the agent core, the session
//! logs, the project context, the launcher protocol, `/reload` and the session
//! commands (`/new`, `/resume`, `/name`). It adds none of Bevy's own
//! plugins, so it sits next to `DefaultPlugins` in a windowed app.
//! [`HeadlessPlugins`] is what a terminal app needs from Bevy instead: the
//! log, the task pools, a clean exit on signals, and a loop that sleeps
//! until there is work. The built-in tools and commands and the terminal
//! view are added on their own, as the `rig` launcher's generated
//! `main.rs` does from `plugins.toml`. The error handler is the
//! application's to set:
//!
//! ```no_run
//! use rig_harness::builtin::{BuiltinCommandsPlugin, BuiltinToolsPlugin, SubagentsPlugin};
//! use rig_harness::prelude::*;
//!
//! fn main() -> AppExit {
//!     App::new()
//!         .set_error_handler(rig_harness::error::warn)
//!         .add_plugins((RigHarnessPlugins, HeadlessPlugins))
//!         // With feature `tui`, `rig_harness::tui::TuiPlugin` adds the terminal view;
//!         .add_plugins((BuiltinToolsPlugin, BuiltinCommandsPlugin, SubagentsPlugin))
//!         .run()
//! }
//! ```

pub mod builtin;
pub mod core;
pub mod host;
#[cfg(feature = "tui")]
pub mod tui;

use bevy_app::{
    PluginGroup, PluginGroupBuilder, TaskPoolOptions, TaskPoolPlugin,
    TaskPoolThreadAssignmentPolicy,
};
use bevy_log::LogPlugin;

pub use bevy_app::{App, AppExit};
/// Bevy's error handlers, for [`App::set_error_handler`]: `warn` logs a
/// failing system, observer or command instead of stopping the app.
pub use bevy_ecs::error;

/// What a plugin needs: Bevy's app and ECS preludes, the agent components
/// and requests, and the tool and command registries.
pub mod prelude {
    pub use bevy_app::prelude::*;
    pub use bevy_ecs::prelude::*;
    pub use bevy_reflect::prelude::*;

    pub use crate::core::agent::{
        ActiveTurn, Agent, AgentId, CallOf, Compact, Connection, Conversation, EffectParent,
        Effort, Focus, Interrupt, ModelChoice, Notice, NoticeLevel, Retry, SetEffort, SetModel,
        Spawned, SpawnedBy, SystemPrompt, ToolAccess, ToolCallRun, TurnEnded, TurnOf, TurnOutcome,
        TurnRequest,
    };
    pub use crate::core::blocking::blocking;
    pub use crate::core::calls::Wake;
    pub use crate::core::commands::{AppCommandsExt, CommandArgs, RunCommand, send_input};
    pub use crate::core::compaction::Compacted;
    pub use crate::core::inbox::{
        Deliver, DeliveryMode, Inbox, Origin, OriginKind, Recalled, RequestId,
    };
    pub use crate::core::journal::ReflectSaved;
    pub use crate::core::prompt::{PromptSection, ToolRules};
    pub use crate::core::recovery::{Backoff, Recovery};
    pub use crate::core::tools::{AppToolsExt, Footprint, ToolCalled, ToolOptions, ToolOutput};
    pub use crate::core::usage::{Spending, TurnSpending};
    pub use crate::host::headless::RunMode;
    pub use crate::host::sessions::{SessionName, SwitchSession};
    pub use crate::{HeadlessPlugins, RigHarnessPlugins};
    pub use rig_core::tool::{PortableTool, Tool, ToolExecutionError};
}

/// What every rig-harness app has: the session and its [`RunMode`](host::headless::RunMode)
/// (with the print mode), the agent core and the session logs, the project
/// context in the system prompt (`AGENTS.md` and the environment, with `/context`), the
/// launcher protocol, `/reload` and `/new`, `/resume` and `/name`. The tools,
/// the commands other than `/reload` and the views are plugins of their own, so `plugins.toml`
/// lists the built-in ones like any other and can leave them out.
pub struct RigHarnessPlugins;

impl PluginGroup for RigHarnessPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(host::session::SessionPlugin)
            .add(host::headless::ModePlugin)
            .add(core::AgentPlugin)
            .add(core::journal::JournalPlugin)
            .add(host::context::ProjectContextPlugin)
            .add(host::launcher::LauncherPlugin)
            .add(host::reload::ReloadPlugin)
            .add(host::sessions::SessionsPlugin)
    }
}

/// What a terminal app takes from Bevy where a windowed one has
/// `DefaultPlugins`: the log written to the session, task pools sized for
/// an agent, a clean exit on SIGINT, SIGTERM and SIGHUP, and a loop that
/// sleeps until [`Wake`](core::calls::Wake)d. Added after
/// [`RigHarnessPlugins`], whose session the log writes to; a windowing plugin
/// added later sets its own runner in place of this loop.
pub struct HeadlessPlugins;

impl PluginGroup for HeadlessPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(LogPlugin {
                fmt_layer: host::session::log_layer,
                ..LogPlugin::default()
            })
            .add(TaskPoolPlugin {
                task_pool_options: task_pools(),
            })
            .add(host::signals::ExitOnSignalPlugin)
            .add(host::runner::RunnerPlugin)
    }
}

/// Bevy's pools sized for an agent rather than a game. Model calls stream
/// on the IO pool and tool calls run on the async compute pool, each with
/// 2 to 4 threads, so a few agents' calls run side by side; blocking tool
/// work runs on threads of its own through [`prelude::blocking`]. Bevy's
/// own systems keep the rest of the cores.
fn task_pools() -> TaskPoolOptions {
    let calls = || TaskPoolThreadAssignmentPolicy {
        min_threads: 2,
        max_threads: 4,
        percent: 0.25,
        on_thread_spawn: None,
        on_thread_destroy: None,
    };
    TaskPoolOptions {
        io: calls(),
        async_compute: calls(),
        ..TaskPoolOptions::default()
    }
}
