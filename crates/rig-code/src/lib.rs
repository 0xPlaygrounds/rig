//! rig-code: a terminal coding agent built as a Bevy app on rig-core.
//!
//! An agent is an entity whose components hold its conversation, model,
//! reasoning setting, system prompt, tool access and status. Model calls and
//! tool calls are entities owned by their agent and run on Bevy's task
//! pools; every one goes through the one recorded dispatch path. Tools and
//! slash commands are registered by Bevy plugins, the built-in ones exactly
//! as a third-party plugin registers its own.
//!
//! [`RigCodePlugins`] is the part every agent app has. The loop runner, the
//! built-in tools and commands, and the terminal view are added on their
//! own, as the `rig` launcher's generated `main.rs` does from
//! `plugins.toml`:
//!
//! ```no_run
//! use rig_code::builtin::{BuiltinCommandsPlugin, BuiltinToolsPlugin};
//! use rig_code::prelude::*;
//!
//! fn main() -> AppExit {
//!     App::new()
//!         .add_plugins((RigCodePlugins, rig_code::runner()))
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

use std::time::Duration;

use bevy_app::{
    PluginGroup, PluginGroupBuilder, ScheduleRunnerPlugin, TaskPoolOptions, TaskPoolPlugin,
    TaskPoolThreadAssignmentPolicy,
};
use bevy_log::LogPlugin;

pub use bevy_app::{App, AppExit};

/// What a plugin needs: Bevy's app and ECS preludes, the agent components
/// and requests, and the tool and command registries.
pub mod prelude {
    pub use bevy_app::prelude::*;
    pub use bevy_ecs::prelude::*;
    pub use bevy_reflect::prelude::*;

    pub use crate::RigCodePlugins;
    pub use crate::core::agent::{
        Agent, AgentId, AgentStatus, Connection, Conversation, Effort, Interrupt, ModelChoice,
        Notice, NoticeLevel, SetEffort, SetModel, Submit, SystemPrompt, ToolAccess, TurnFinished,
    };
    pub use crate::core::blocking::blocking;
    pub use crate::core::commands::{AppCommandsExt, CommandArgs};
    pub use crate::core::save::ReflectSaved;
    pub use crate::core::tools::AppToolsExt;
    pub use rig_core::tool::{PortableTool, Tool, ToolExecutionError};
}

/// What every rig-code app has: the session and its log, Bevy's task
/// pools, the agent core and saving, the launcher protocol and `/reload`.
/// The tools, the commands other than `/reload`, the views and the
/// [`runner`] are plugins of their own, so `plugins.toml` lists the
/// built-in ones like any other and can leave them out.
pub struct RigCodePlugins;

impl PluginGroup for RigCodePlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(host::session::SessionPlugin)
            .add(LogPlugin {
                fmt_layer: host::session::log_layer,
                ..LogPlugin::default()
            })
            .add(TaskPoolPlugin {
                task_pool_options: task_pools(),
            })
            .add(core::AgentPlugin)
            .add(core::save::SavePlugin)
            .add(host::launcher::LauncherPlugin)
            .add(host::reload::ReloadPlugin)
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

/// The app's loop without a window: one frame every 16 ms. It is added
/// apart from [`RigCodePlugins`] and before the listed plugins, so a
/// windowing plugin added later sets its own runner in its place.
pub fn runner() -> ScheduleRunnerPlugin {
    ScheduleRunnerPlugin::run_loop(Duration::from_millis(16))
}
