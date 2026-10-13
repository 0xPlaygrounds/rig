//! rig-harness: the core of a terminal coding agent, a Bevy app on
//! [`rig_ecs`]'s agent runtime and rig-core.
//!
//! An agent is an entity whose components hold its conversation, model,
//! reasoning setting, system prompt and tool access; an agent a plugin
//! spawns for another is one more, [`SpawnedBy`](rig_ecs::agent::SpawnedBy)
//! that agent. Tools and slash commands are registered by Bevy plugins, and
//! a plugin re-arms its saved obligations after a restart from its saved
//! components, and gates the work a restart carries on
//! ([`Restored`](rig_ecs::restore::Restored)) like any. This crate is only what the
//! terminal app needs around the runtime to run and be relaunched: the
//! session directory and the `rig` launcher protocol, with how the agent
//! was started ([`Invoked`](host::launcher::Invoked)). Every other part of
//! the agent, `/reload`, the terminal view, `--print`, sign-in and the
//! built-in tools and commands included, is a plugin crate in the
//! repository's `plugins/` folder, built only on this crate's public API.
//!
//! [`RigHarnessPlugins`] is what the binary needs to run and be
//! relaunched: the session, the agent core and the session logs, and the
//! launcher protocol. It adds none of Bevy's own plugins, so it sits next
//! to `DefaultPlugins` in a windowed app. [`HeadlessPlugins`] is what a
//! terminal app needs from Bevy instead: Bevy's `MinimalPlugins` (task
//! pools, frame count, clock) with the log, a clean exit on signals, and a
//! loop that sleeps until there is work. Everything else is a plugin listed
//! in `plugins.toml` and added by the `rig` launcher's generated `main.rs`
//! with [`load`]. The error handler is the application's to set:
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
//! [`plugin_guide`] states the rules every plugin follows (a hook is a
//! system in a stage; holds add up and the first answer wins; a failing
//! stage system closes its stage) and shows each kind of extension with an
//! example: tools and gating their calls, questions for the user, slash
//! commands, model requests, turn endings, a plugin's own effects,
//! settings and options, saved state, time, fronts, terminal panels, and a
//! window beside the terminal ([`windowed`]). The `rig` launcher makes a
//! plugin crate with `rig plugin new <name>`.

#[doc = include_str!("../PLUGINS.md")]
pub mod plugin_guide {}
pub mod ask;
pub mod checkpoint;
pub mod clock;
pub mod commands;
pub mod describe;
pub mod headless;
pub mod host;
mod load;
pub mod note;
pub mod per_call;
pub mod pick;
pub mod plugin_data;
pub mod preview;
pub mod primary;
pub mod progress;
pub mod secrets;
pub mod services;
pub mod settings;
pub mod successor;
pub mod timings;
pub mod watch;
pub mod workspace;

pub use describe::run;
pub use load::{PluginEntity, PluginSource, PluginSystems, ProvidedBy, Provides, load};

/// Where the plugin guide ([`plugin_guide`]) is on disk, for an agent
/// that reads it.
pub const PLUGIN_GUIDE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/PLUGINS.md");

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
/// The `rig` launcher protocol: the `RIG_HOME` layout, sessions, and the
/// environment and exit codes the launcher and the agent share.
pub use rig::harness_protocol;
/// The rig-core this app is built on, for the conversation's messages
/// (`rig_harness::rig_core::message::Message`) and rig-core's tools.
pub use rig_core;
/// The agent runtime this app is built on.
pub use rig_ecs;
/// The store this app keeps its sessions in, for reading, saving and
/// forking a session without restoring it (`rig_harness::rig_memory::DirStore`,
/// `load`, `save`) and for compaction's pure functions.
pub use rig_memory;
/// The tool library this app is built on, for the process runner
/// [`Processes`](services::Processes) hands out
/// (`rig_harness::rig_tools::process::Spawn`) and the file tools' rules
/// (`rig_harness::rig_tools::fs::place`).
pub use rig_tools;

/// What a plugin needs, so a typical one imports only this: rig-ecs's
/// prelude (Bevy's app and ECS preludes, the agent components and
/// requests, and the tool registry), slash commands, what agents say, the
/// app's session, its log's warnings and launcher, and the plugin groups.
/// A plugin crate's own types come from that crate, such as the terminal
/// view's panels and status line from `rig-tui`.
pub mod prelude {
    pub use rig_ecs::prelude::*;

    pub use std::time::Duration;

    pub use bevy_app::OnAppExitSystems;

    pub use crate::ask::{Answer, Answerer, Answerers, Ask, AskSystems, Choice};
    pub use crate::checkpoint::{CheckpointAlso, CheckpointPlugin, Checkpoints, TurnCheckpoint};
    pub use crate::clock::{Due, KeepAwake, WakeAt};
    pub use crate::commands::{
        AppCommandsExt, CommandArgs, CommandSystems, RunCommand, SlashCommand,
    };
    pub use crate::harness_protocol::Home;
    pub use crate::headless::{AppHeadlessExt, HeadlessEnded, HeadlessTurnEnded, KeepRunning};
    pub use crate::host::cli::{AppCliExt, Flag, Front};
    pub use crate::host::exit::{Exit, ExitReason, ExitRequest};
    pub use crate::host::launcher::{self, Invoked};
    pub use crate::host::session::{
        DeleteSession, FirstStart, ForkSession, ForkSessions, ForkedFrom, LogEvents, Logged,
        Scratch, SessionDeleteFailed, SessionDeleted, SessionForked, SessionPaths, SessionStores,
        first_typed, rewrite_stopped, typed,
    };
    pub use crate::note::Note;
    pub use crate::per_call::{AppPerCallExt, Place};
    pub use crate::pick::{Chosen, Pick, PickItem, Picked, Picker};
    pub use crate::plugin_data::PluginData;
    pub use crate::preview::{Preview, PreviewPlugin, PreviewSystems, Previewed};
    pub use crate::primary::{PrimaryQuery, Shown, primary};
    pub use crate::progress::{CallOutput, ToolProgress};
    pub use crate::secrets::{Secret, Secrets};
    pub use crate::services::{Processes, ProjectRoot, RigHome, TokioRuntime, WorkDir};
    pub use crate::settings::{AppSettingsExt, Settings};
    pub use crate::successor::{follow_successor, resend};
    pub use crate::timings::{PluginTimings, SystemTimings};
    pub use crate::watch::{FileChanges, FileWatch};
    pub use crate::workspace::ProjectWorkspace;
    pub use crate::{
        HeadlessPlugins, PluginEntity, PluginSource, PluginSystems, ProvidedBy, Provides,
        RigHarnessPlugins,
    };
    pub use rig_tools::{Spill, blocking};
}

/// What the binary needs to run and be relaunched, and the services every
/// plugin shares: the process's home, project, child processes and tokio
/// runtime ([`services`]), the session and its log, the agent core and the
/// session logs, the launcher protocol with the plugins' options and
/// fronts ([`host`]), slash commands ([`commands`]), why the app exits,
/// wall-clock wakes ([`clock`]),
/// file watches ([`watch`]), questions for the user ([`ask`]), pickers
/// with the agent the user looks at ([`pick`]), previews of what a tool
/// call would leave on disk ([`preview`]) and what every call that may
/// write anywhere changed ([`checkpoint`]).
/// Everything else is a plugin `plugins.toml` lists, which can leave it
/// out.
pub struct RigHarnessPlugins;

impl PluginGroup for RigHarnessPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(services::ServicesPlugin)
            .add(host::session::SessionPlugin)
            .add(rig_ecs::AgentPlugin)
            .add(rig_ecs::journal::JournalPlugin)
            .add(commands::CommandPlugin)
            .add(host::launcher::LauncherPlugin)
            .add(host::exit::ExitPlugin)
            .add(clock::ClockPlugin)
            .add(watch::WatchPlugin)
            .add(ask::AskPlugin)
            .add(pick::PickPlugin)
            .add(progress::ProgressPlugin)
            .add(preview::PreviewPlugin)
            .add(checkpoint::CheckpointPlugin)
    }
}

/// What a terminal app takes from Bevy where a windowed one has
/// `DefaultPlugins`: Bevy's `MinimalPlugins` with task pools sized for an
/// agent and the frame count and clock, the log written to the session
/// with its warnings and errors passed on as data, a clean exit on SIGINT,
/// SIGTERM and SIGHUP, and in place of Bevy's `ScheduleRunnerPlugin` a
/// loop that runs the schedules on the main thread and sleeps until
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
