//! rig-harness: the harness of the rig coding agent, a Bevy app on the
//! [`rig_ecs`] kernel. It holds what the binary needs to run and be
//! relaunched (the session, the `rig` launcher protocol, how the agent was
//! started: [`Invoked`](host::launcher::Invoked)) and the services every
//! plugin shares. Everything else, the terminal view, `--print`, `/reload`
//! and the built-in tools included, is a plugin crate in the repository's
//! `plugins/` folder, built only on this crate's public API.
//!
//! [`HeadlessPlugins`] is what a terminal app takes from Bevy, and
//! [`RigHarnessPlugins`] is the harness. The launcher's generated `main.rs`
//! adds the plugins `plugins.toml` lists with [`load`]. The error handler is
//! the application's to set:
//!
//! ```no_run
//! use rig_harness::load;
//! use rig_harness::prelude::*;
//!
//! #[derive(Default)]
//! struct HelloPlugin;
//!
//! impl Plugin for HelloPlugin {
//!     fn build(&self, _app: &mut App) {}
//! }
//!
//! fn main() -> AppExit {
//!     let mut app = App::new();
//!     app.set_error_handler(rig_harness::error::warn)
//!         .add_plugins((HeadlessPlugins, RigHarnessPlugins));
//!     load::<HelloPlugin>(&mut app, "rig-hello", "path plugins/hello");
//!     app.run()
//! }
//! ```
//!
//! [`plugin_guide`] is the guide to writing a plugin; `rig plugin new <name>`
//! makes one.

#[doc = include_str!("../PLUGINS.md")]
pub mod plugin_guide {}

/// Where the plugin guide ([`plugin_guide`]) is on disk, for an agent
/// that reads it: in this crate's own source, from a checkout or from
/// crates.io alike.
pub const PLUGIN_GUIDE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/PLUGINS.md");
pub mod agent_title;
pub mod ask;
pub mod clock;
pub mod commands;
pub mod describe;
pub mod host;
mod load;
pub mod main_agent;
pub mod per_call;
pub mod pick;
pub mod plugin_data;
pub mod progress;
pub mod secrets;
pub mod services;
pub mod settings;
pub mod successor;
pub mod timings;
pub mod watch;
pub mod workspace;

pub use describe::run;
pub use load::{
    AppKeepsExt, Keeps, PluginEntity, PluginSource, PluginSystems, ProvidedBy, Provides, SavedAs,
    check_stages_at_start, load, stage_conflicts,
};

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
/// The kernel this app is built on.
pub use rig_ecs;
/// The `rig` launcher protocol: the `RIG_HOME` layout, sessions, and the
/// environment and exit codes the launcher and the agent share.
pub use rig_harness_protocol as harness_protocol;
/// The store this app keeps its sessions in, for reading and saving a
/// session without restoring it (`rig_harness::rig_memory::DirStore`,
/// `load`, `save`) and for compaction's pure functions.
pub use rig_memory;
/// The tool library this app is built on, for the process runner
/// [`Processes`](services::Processes) hands out
/// (`rig_harness::rig_tools::process::Spawn`) and the file tools' rules
/// (`rig_harness::rig_tools::fs::place`).
pub use rig_tools;

/// What a plugin needs, so a typical one imports only this: the kernel's
/// prelude and the harness's services. A plugin crate's own types come from
/// that crate, such as the terminal view's panels from `rig-tui`.
pub mod prelude {
    pub use rig_ecs::prelude::*;

    pub use std::time::Duration;

    pub use bevy_app::OnAppExitSystems;

    pub use crate::agent_title::AgentTitle;
    pub use crate::ask::{Answer, Answerer, Answerers, Ask, AskSystems, Choice};
    pub use crate::clock::{Due, KeepAwake, WakeAt};
    pub use crate::commands::{
        AppCommandsExt, CommandArgs, CommandSystems, RunCommand, SlashCommand,
    };
    pub use crate::harness_protocol::{Home, StartReason};
    pub use crate::host::cli::{AppCliExt, Flag, Front};
    pub use crate::host::exit::{Exit, ExitReason, ExitRequest};
    pub use crate::host::launcher::{self, Invoked};
    pub use crate::host::session::{
        AppStart, DeleteSession, Logged, LoggedQueue, Scratch, SessionDeleteFailed, SessionDeleted,
        SessionPaths, StoreOpener, edit_stopped, rewrite_stopped,
    };
    pub use crate::main_agent::{MainAgentQuery, ShownAgent, main_agent};
    pub use crate::per_call::{
        AppPerCallExt, CallPlace, PerCall, PerCallSystems, PerCallTool, PerCallTools,
    };
    pub use crate::pick::{Chosen, Highlighted, Pick, PickItem, PickPreview, Picked, Picker};
    pub use crate::plugin_data::PluginData;
    pub use crate::progress::{CallOutput, ToolProgress};
    pub use crate::secrets::{Secret, Secrets};
    pub use crate::services::{Processes, ProjectRoot, RigHome, TokioRuntime, WorkDir};
    pub use crate::settings::{AppSettingsExt, Settings};
    pub use crate::successor::resend;
    pub use crate::timings::{PluginTimings, SystemTimings};
    pub use crate::watch::{FileChanges, FileWatch};
    pub use crate::workspace::ProjectWorkspace;
    pub use crate::{
        AppKeepsExt, HeadlessPlugins, Keeps, PluginEntity, PluginSource, PluginSystems, ProvidedBy,
        Provides, RigHarnessPlugins, SavedAs,
    };
    pub use rig_core::conversation::Note;
    pub use rig_tools::attach::{first_typed, typed};
    pub use rig_tools::{Spill, blocking};
}

/// The harness: the session and its log, the kernel
/// ([`rig_ecs::AgentPlugin`], [`rig_ecs::session_log::SessionLogPlugin`]),
/// the launcher protocol, and the services every agent uses: slash
/// commands, the clock the loop sleeps by, file watches, the questions and
/// pickers every front answers, and tool progress. Everything else is a
/// plugin `plugins.toml` lists, which can leave it out.
pub struct RigHarnessPlugins;

impl PluginGroup for RigHarnessPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(services::ServicesPlugin)
            .add(host::session::SessionPlugin)
            .add(rig_ecs::AgentPlugin)
            .add(rig_ecs::session_log::SessionLogPlugin)
            .add(commands::CommandPlugin)
            .add(host::launcher::LauncherPlugin)
            .add(host::exit::ExitPlugin)
            .add(clock::ClockPlugin)
            .add(watch::WatchPlugin)
            .add(ask::AskPlugin)
            .add(pick::PickPlugin)
            .add(progress::ProgressPlugin)
    }
}

/// What a terminal app takes from Bevy where a windowed one has
/// `DefaultPlugins`: Bevy's `MinimalPlugins` with task pools sized for an
/// agent and the frame count and clock, the log written to the session
/// with its warnings and errors passed on as data, a clean exit on SIGINT,
/// SIGTERM and SIGHUP, and in place of Bevy's `ScheduleRunnerPlugin` a
/// loop that runs the schedules on the main thread and sleeps until
/// [`Wake`](rig_ecs::tasks::Wake)d or until the clock's next deadline.
/// Added before [`RigHarnessPlugins`], so the kernel runs on its clock;
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
                custom_layer: host::session::log_events,
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
/// own systems keep the rest of the cores. [`HeadlessPlugins`] sets them; a
/// test app sets them too, so it runs its calls as the agent does.
pub fn task_pools() -> TaskPoolOptions {
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
