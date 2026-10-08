//! rig-code: a terminal coding agent built as a Bevy app on rig-core.
//!
//! An agent is an entity whose components hold its conversation, model,
//! reasoning setting, system prompt and tool access. A running turn is an
//! entity of its agent, and the turn's model calls and tool calls are
//! entities of the turn that run on Bevy's task pools; every one goes
//! through the one recorded dispatch path. Tools and
//! slash commands are registered by Bevy plugins, the built-in ones exactly
//! as a third-party plugin registers its own.
//!
//! [`RigCodePlugins`] is the agent app: the session, the agent core,
//! saving, the project context, the launcher protocol and `/reload`. It adds none of Bevy's own
//! plugins, so it sits next to `DefaultPlugins` in a windowed app.
//! [`HeadlessPlugins`] is what a terminal app needs from Bevy instead: the
//! log, the task pools, a clean exit on signals, and a loop that sleeps
//! until there is work. The built-in tools and commands and the terminal
//! view are added on their own, as the `rig` launcher's generated
//! `main.rs` does from `plugins.toml`. The error handler is the
//! application's to set:
//!
//! ```no_run
//! use rig_code::builtin::{BuiltinCommandsPlugin, BuiltinToolsPlugin};
//! use rig_code::prelude::*;
//!
//! fn main() -> AppExit {
//!     App::new()
//!         .set_error_handler(rig_code::error::warn)
//!         .add_plugins((RigCodePlugins, HeadlessPlugins))
//!         // With feature `tui`, `rig_code::tui::TuiPlugin` adds the terminal view.
//!         .add_plugins((BuiltinToolsPlugin, BuiltinCommandsPlugin))
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
        ActiveTurn, Agent, AgentId, Connection, Conversation, Effort, Interrupt, ModelChoice,
        Notice, NoticeLevel, Retry, SetEffort, SetModel, Submit, SystemPrompt, ToolAccess,
        TurnFinished, TurnOf,
    };
    pub use crate::core::blocking::blocking;
    pub use crate::core::calls::Wake;
    pub use crate::core::commands::{AppCommandsExt, CommandArgs};
    pub use crate::core::prompt::{PromptSection, ToolRules};
    pub use crate::core::recovery::{Backoff, Recovery};
    pub use crate::core::save::ReflectSaved;
    pub use crate::core::tools::AppToolsExt;
    pub use crate::core::usage::{Spending, TurnSpending};
    pub use crate::{HeadlessPlugins, RigCodePlugins};
    pub use rig_core::tool::{PortableTool, Tool, ToolExecutionError};
}

/// What every rig-code app has: the session, the agent core and saving,
/// the project context in the system prompt (`AGENTS.md` and the
/// environment, with `/context`), the launcher protocol and `/reload`. The tools, the commands other than
/// `/reload` and the views are plugins of their own, so `plugins.toml`
/// lists the built-in ones like any other and can leave them out.
pub struct RigCodePlugins;

impl PluginGroup for RigCodePlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(host::session::SessionPlugin)
            .add(core::AgentPlugin)
            .add(core::save::SavePlugin)
            .add(host::context::ProjectContextPlugin)
            .add(host::launcher::LauncherPlugin)
            .add(host::reload::ReloadPlugin)
    }
}

/// What a terminal app takes from Bevy where a windowed one has
/// `DefaultPlugins`: the log written to the session, task pools sized for
/// an agent, a clean exit on SIGINT, SIGTERM and SIGHUP, and a loop that
/// sleeps until [`Wake`](core::calls::Wake)d. Added after
/// [`RigCodePlugins`], whose session the log writes to; a windowing plugin
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
