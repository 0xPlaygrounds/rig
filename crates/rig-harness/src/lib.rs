//! rig-harness: a terminal coding agent, a Bevy app on [`rig_ecs`]'s agent
//! runtime and rig-core.
//!
//! An agent is an entity whose components hold its conversation, model,
//! reasoning setting, system prompt and tool access; an agent a plugin
//! spawns for another, such as a subagent of rig-ecs's
//! [`SubagentsPlugin`](rig_ecs::subagents::SubagentsPlugin), is one more,
//! [`SpawnedBy`](rig_ecs::agent::SpawnedBy) that agent. Tools and
//! slash commands are registered by Bevy plugins, the built-in ones exactly
//! as a third-party plugin registers its own, and a plugin re-arms its saved
//! obligations after a restart on [`Restored`](rig_ecs::restore::Restored).
//! This crate adds the terminal app around the runtime: the session
//! directory, the terminal view, `--print`, the launcher protocol, sign-in,
//! `@path` attachments and the built-in tools and commands.
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
//! use rig_harness::builtin::{BuiltinCommandsPlugin, BuiltinToolsPlugin};
//! use rig_harness::prelude::*;
//! use rig_harness::rig_ecs::subagents::SubagentsPlugin;
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
//!
//! # A window beside the terminal
//!
//! A plugin opens a Bevy window with [`windowed`]: Bevy's `DefaultPlugins`
//! without the log, task pools, signal handler and loop that
//! [`HeadlessPlugins`] already set. Its `WinitPlugin` then runs the app's
//! loop. In the plugin's `finish`, where winit's event loop exists, it
//! points the [`Wake`](rig_ecs::calls::Wake) at that loop, so agent
//! activity and terminal input wake the window's loop instead of a poll:
//!
//! ```ignore
//! use std::time::Duration;
//! use bevy::prelude::*;
//! use bevy::window::ExitCondition;
//! use bevy::winit::{EventLoopProxyWrapper, UpdateMode, WinitSettings, WinitUserEvent};
//! use rig_harness::prelude::{RunMode, Wake};
//!
//! #[derive(Default)]
//! pub struct DashboardPlugin;
//!
//! impl Plugin for DashboardPlugin {
//!     fn build(&self, app: &mut App) {
//!         if app.world().get_resource::<RunMode>().is_some_and(RunMode::is_headless) {
//!             return;
//!         }
//!         // Closing the window leaves the agent running.
//!         let window = WindowPlugin {
//!             exit_condition: ExitCondition::DontExit,
//!             ..default()
//!         };
//!         // Frames when woken, and at least every second as without a window.
//!         let mode = UpdateMode::reactive_low_power(Duration::from_secs(1));
//!         app.add_plugins(rig_harness::windowed(DefaultPlugins.set(window)))
//!             .insert_resource(WinitSettings { focused_mode: mode, unfocused_mode: mode });
//!     }
//!
//!     fn finish(&self, app: &mut App) {
//!         if let Some(proxy) = app.world().get_resource::<EventLoopProxyWrapper>() {
//!             let proxy = (**proxy).clone();
//!             app.insert_resource(Wake::new(move || {
//!                 proxy.send_event(WinitUserEvent::WakeUp).ok();
//!             }));
//!         }
//!     }
//! }
//! ```
//!
//! The plugin crate depends on `bevy` with the features it draws with, at
//! the version rig-harness uses.

pub mod attach;
pub mod builtin;
pub mod host;
#[cfg(feature = "tui")]
pub mod tui;
pub mod view;

use bevy_app::{
    PluginGroup, PluginGroupBuilder, ScheduleRunnerPlugin, TaskPoolOptions, TaskPoolPlugin,
    TaskPoolThreadAssignmentPolicy,
};
use bevy_log::LogPlugin;

pub use bevy_app::{App, AppExit};
/// Bevy's error handlers, for [`App::set_error_handler`]: `warn` logs a
/// failing system, observer or command instead of stopping the app.
pub use bevy_ecs::error;
/// The agent runtime this app is built on; `plugins.toml` names its
/// plugins through it, such as `rig_harness::rig_ecs::subagents::SubagentsPlugin`.
pub use rig_ecs;

/// What a plugin needs: rig-ecs's prelude (Bevy's app and ECS preludes,
/// the agent components and requests, and the tool and command
/// registries) and the app's session, views and plugin groups.
pub mod prelude {
    pub use rig_ecs::prelude::*;

    pub use crate::host::headless::RunMode;
    pub use crate::host::sessions::{SessionName, SwitchSession};
    pub use crate::view::{Focus, PickItem, PickRequest, send_input};
    pub use crate::{HeadlessPlugins, RigHarnessPlugins};
    pub use rig_tools::blocking;
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
            .add(view::ViewPlugin)
            .add(rig_ecs::AgentPlugin)
            .add(rig_ecs::journal::JournalPlugin)
            .add(host::compaction::CodingCompactionPlugin)
            .add(host::context::ProjectContextPlugin)
            .add(host::defaults::DefaultsPlugin)
            .add(host::launcher::LauncherPlugin)
            .add(host::reload::ReloadPlugin)
            .add(host::sessions::SessionsPlugin)
    }
}

/// What a terminal app takes from Bevy where a windowed one has
/// `DefaultPlugins`: the log written to the session, task pools sized for
/// an agent, a clean exit on SIGINT, SIGTERM and SIGHUP, and a loop that
/// sleeps until [`Wake`](rig_ecs::calls::Wake)d. Added after
/// [`RigHarnessPlugins`], whose session the log writes to; a windowing plugin
/// added later through [`windowed`] sets its own runner in place of this
/// loop (see the crate docs).
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
            .add(rig_ecs::runner::RunnerPlugin)
    }
}

/// `plugins`, such as Bevy's `DefaultPlugins`, without what
/// [`HeadlessPlugins`] already sets: the log, the task pools, the signal
/// handler and the loop. Its windowing plugin then runs the loop; the crate
/// docs show how a window plugin wakes it.
pub fn windowed(plugins: impl PluginGroup) -> PluginGroupBuilder {
    let mut plugins = plugins.build();
    if plugins.contains::<LogPlugin>() {
        plugins = plugins.disable::<LogPlugin>();
    }
    if plugins.contains::<TaskPoolPlugin>() {
        plugins = plugins.disable::<TaskPoolPlugin>();
    }
    if plugins.contains::<ScheduleRunnerPlugin>() {
        plugins = plugins.disable::<ScheduleRunnerPlugin>();
    }
    #[cfg(any(unix, windows))]
    if plugins.contains::<bevy_app::TerminalCtrlCHandlerPlugin>() {
        plugins = plugins.disable::<bevy_app::TerminalCtrlCHandlerPlugin>();
    }
    plugins
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
