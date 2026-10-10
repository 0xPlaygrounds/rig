//! rig-harness: a terminal coding agent, a Bevy app on [`rig_ecs`]'s agent
//! runtime and rig-core.
//!
//! An agent is an entity whose components hold its conversation, model,
//! reasoning setting, system prompt and tool access; an agent a plugin
//! spawns for another, such as a subagent of the
//! [`SubagentsPlugin`](plugins::subagents::SubagentsPlugin), is one more,
//! [`SpawnedBy`](rig_ecs::agent::SpawnedBy) that agent. Tools and
//! slash commands are registered by Bevy plugins, the built-in ones exactly
//! as a third-party plugin registers its own, and a plugin re-arms its saved
//! obligations after a restart on [`Restored`](rig_ecs::restore::Restored).
//! This crate adds the terminal app around the runtime: the session
//! directory, the launcher protocol, and the default plugins, such as the
//! terminal view, `--print`, sign-in and the built-in tools and commands.
//!
//! [`RigHarnessPlugins`] is what the binary needs to run and be
//! relaunched: the session, the run mode and what fronts share
//! ([`front`]), the agent core and the session logs, the launcher protocol
//! and `/reload`. It adds none of Bevy's own plugins, so it sits next to
//! `DefaultPlugins` in a windowed app. [`HeadlessPlugins`] is what a
//! terminal app needs from Bevy instead: Bevy's `MinimalPlugins` (task
//! pools, frame count, clock) with the log, a clean exit on signals, and a
//! loop that sleeps until there is work. Everything else is a plugin of
//! [`plugins`], listed in `plugins.toml` like any other and added by the
//! `rig` launcher's generated `main.rs` with [`load`]. The error handler is
//! the application's to set:
//!
//! ```no_run
//! use rig_harness::prelude::*;
//! use rig_harness::{load, plugins};
//!
//! fn main() -> AppExit {
//!     let mut app = App::new();
//!     app.set_error_handler(rig_harness::error::warn)
//!         .add_plugins((HeadlessPlugins, RigHarnessPlugins));
//!     load::<plugins::tools::BuiltinToolsPlugin>(&mut app, "rig-harness", "");
//!     load::<plugins::print::PrintPlugin>(&mut app, "rig-harness", "");
//!     app.run()
//! }
//! ```
//!
//! [`plugin_guide`] shows each kind of extension with an example: tools,
//! slash commands, tool renderers, terminal panels, turns, saved state,
//! timers, and a window beside the terminal ([`windowed`]). The `rig`
//! launcher makes a plugin crate with `rig plugin new <name>`.

pub mod front;
#[doc = include_str!("../PLUGINS.md")]
pub mod plugin_guide {}
pub mod host;
mod load;
pub mod plugins;
#[cfg(feature = "tui")]
pub mod tui;

pub use load::{Build, BuildKind, PluginSource, ProvidedBy, Provides, load};

use bevy::MinimalPlugins;
use bevy::diagnostic::FrameCountPlugin;
use bevy::time::TimePlugin;
use bevy_app::{
    Plugin, PluginGroup, PluginGroupBuilder, ScheduleRunnerPlugin, TaskPoolOptions, TaskPoolPlugin,
    TaskPoolThreadAssignmentPolicy,
};
use bevy_log::LogPlugin;

pub use bevy_app::{App, AppExit};
/// Bevy's error handlers, for [`App::set_error_handler`]: `warn` logs a
/// failing system, observer or command instead of stopping the app.
pub use bevy_ecs::error;
/// The rig-core this app is built on, for the conversation's messages
/// (`rig_harness::rig_core::message::Message`) and rig-core's tools.
pub use rig_core;
/// The agent runtime this app is built on.
pub use rig_ecs;

/// What a plugin needs, so a typical one imports only this: rig-ecs's
/// prelude (Bevy's app and ECS preludes, the agent components and
/// requests, and the tool and command registries), what agents do and
/// say, the app's session, views and plugin groups, and with feature
/// `tui` the terminal view's panels and tool renderers. ratatui itself is
/// [`tui::ratatui`].
pub mod prelude {
    pub use rig_ecs::prelude::*;

    pub use std::time::Duration;

    pub use crate::front::{Busy, Focus, Front, PickItem, PickRequest, RunMode, send_input};
    pub use crate::host::reload::ReloadStatus;
    pub use crate::plugins::activity::{
        Activity, ActivitySystems, FedMessage, MessageFeed, Preview, PreviewKind, Status,
        ToolActivity,
    };
    pub use crate::plugins::sessions::{SessionTitle, SwitchSession};
    pub use crate::plugins::usage::{Spending, TurnSpending};
    #[cfg(feature = "tui")]
    pub use crate::tui::ratatui::layout::Constraint;
    #[cfg(feature = "tui")]
    pub use crate::tui::{
        AppToolRenderersExt, Focused, PanelCanvas, Placement, RESULT_LINES, RequestRedraw,
        TuiPanel, TuiScreen, TuiSystems,
    };
    pub use crate::{
        Build, HeadlessPlugins, PluginSource, ProvidedBy, Provides, RigHarnessPlugins,
    };
    pub use rig_core::transcript::final_answer;
    pub use rig_ecs::agent::{PrimaryQuery, primary};
    pub use rig_tools::blocking;
}

/// What the binary needs to run and be relaunched: the session and its
/// log, the [`RunMode`](front::RunMode), the agent core and the session
/// logs, the launcher protocol and `/reload`. Everything else is a plugin `plugins.toml`
/// lists, which can leave it out.
pub struct RigHarnessPlugins;

impl PluginGroup for RigHarnessPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(host::session::SessionPlugin)
            .add(front::FrontPlugin)
            .add(rig_ecs::AgentPlugin)
            .add(rig_ecs::journal::JournalPlugin)
            .add(host::launcher::LauncherPlugin)
            .add(host::reload::ReloadPlugin)
    }
}

/// What a terminal app takes from Bevy where a windowed one has
/// `DefaultPlugins`: Bevy's `MinimalPlugins` with task pools sized for an
/// agent and the frame count and clock, the log written to the session, a
/// clean exit on SIGINT, SIGTERM and SIGHUP, and in place of Bevy's
/// `ScheduleRunnerPlugin` a loop that sleeps until
/// [`Wake`](rig_ecs::calls::Wake)d or until the clock's next deadline.
/// Added before [`RigHarnessPlugins`], so the agent core runs on its clock;
/// a windowing plugin added later through [`windowed`] sets its own runner
/// in place of this loop (see [`plugin_guide`]).
pub struct HeadlessPlugins;

impl PluginGroup for HeadlessPlugins {
    fn build(self) -> PluginGroupBuilder {
        MinimalPlugins
            .build()
            .set(TaskPoolPlugin {
                task_pool_options: task_pools(),
            })
            .disable::<ScheduleRunnerPlugin>()
            .add(LogPlugin {
                fmt_layer: host::session::log_layer,
                ..LogPlugin::default()
            })
            .add(host::signals::ExitOnSignalPlugin)
            .add(host::runner::RunnerPlugin)
    }
}

/// `plugins`, such as Bevy's `DefaultPlugins`, without what
/// [`HeadlessPlugins`] already sets: the log, the task pools, the frame
/// count, the clock, the signal handler and the loop. Its windowing plugin
/// then runs the loop; [`plugin_guide`] shows how a window plugin wakes it.
pub fn windowed(plugins: impl PluginGroup) -> PluginGroupBuilder {
    let plugins = without::<LogPlugin>(plugins.build());
    let plugins = without::<TaskPoolPlugin>(plugins);
    let plugins = without::<FrameCountPlugin>(plugins);
    let plugins = without::<TimePlugin>(plugins);
    let plugins = without::<ScheduleRunnerPlugin>(plugins);
    #[cfg(any(unix, windows))]
    let plugins = without::<bevy_app::TerminalCtrlCHandlerPlugin>(plugins);
    plugins
}

/// `plugins` with `P` disabled, if it has `P`.
fn without<P: Plugin>(plugins: PluginGroupBuilder) -> PluginGroupBuilder {
    if plugins.contains::<P>() {
        plugins.disable::<P>()
    } else {
        plugins
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
