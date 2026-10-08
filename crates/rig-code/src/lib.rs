//! A minimal coding agent built on Rig and Bevy.
//!
//! Each agent is an entity whose building blocks are components, and every
//! model or tool call in flight is an entity related to its agent. Tools and
//! slash commands are registered by Bevy plugins through [`ecs::RigAppExt`];
//! the built-in ones use the same calls. The terminal UI is one plugin over
//! those components.
//!
//! ```no_run
//! use rig_code::bevy::app::{App, AppExit};
//!
//! fn main() -> AppExit {
//!     App::new().add_plugins(rig_code::RigCodePlugins).run()
//! }
//! ```

use std::{fs::OpenOptions, time::Duration};

use bevy::{
    app::{
        App, PluginGroup, PluginGroupBuilder, ScheduleRunnerPlugin, TaskPoolOptions, TaskPoolPlugin,
    },
    log::{BoxedFmtLayer, LogPlugin, tracing_subscriber},
};

pub use bevy;

pub mod ecs;
pub mod tui;

/// The exit code that asks the launcher to start the freshly built binary.
/// It is `EX_TEMPFAIL`, which neither panics (101) nor signals produce.
pub const RELOAD_EXIT_CODE: u8 = 75;

/// Frame period of the app loop: the latency of terminal input.
const FRAME: Duration = Duration::from_millis(16);

/// Everything the agent needs: Bevy's task pools, file logging and a
/// sleeping run loop, then the agent core, the built-in tools and commands,
/// and the terminal UI.
pub struct RigCodePlugins;

impl PluginGroup for RigCodePlugins {
    fn build(self) -> PluginGroupBuilder {
        let mut task_pool_options = TaskPoolOptions::default();
        // Tools do blocking file and process IO on the IO pool.
        task_pool_options.io.min_threads = 2;
        task_pool_options.io.max_threads = 8;
        PluginGroupBuilder::start::<Self>()
            .add(TaskPoolPlugin { task_pool_options })
            .add(LogPlugin {
                fmt_layer: log_to_file,
                ..LogPlugin::default()
            })
            .add(ScheduleRunnerPlugin::run_loop(FRAME))
            .add(ecs::AgentPlugin)
            .add(ecs::tools::ToolsPlugin)
            .add(ecs::command::CommandsPlugin)
            .add(tui::TuiPlugin)
    }
}

/// Write logs to `agent.log` in the session directory, never to stderr.
fn log_to_file(_app: &mut App) -> Option<BoxedFmtLayer> {
    let layer = tracing_subscriber::fmt::Layer::default().with_ansi(false);
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(ecs::paths::log_file());
    Some(match file {
        Ok(file) => Box::new(layer.with_writer(std::sync::Mutex::new(file))),
        Err(_) => Box::new(layer.with_writer(std::io::sink)),
    })
}
