//! The base app: task pools, a frame loop that no view owns, logging to a
//! file, and error and panic handling that never print to the terminal.

use std::path::PathBuf;
use std::time::Duration;

use bevy::app::{ScheduleRunnerPlugin, TaskPoolPlugin};
use bevy::diagnostic::FrameCountPlugin;
use bevy::log::tracing_subscriber::fmt;
use bevy::log::{BoxedFmtLayer, LogPlugin};
use bevy::prelude::*;
use bevy::time::TimePlugin;

/// Where the agent keeps its sessions, logs and defaults.
#[derive(Resource, Clone)]
pub struct DataDir(pub PathBuf);

/// The data directory: `RIG_DATA_DIR`, or `$RIG_HOME/data`. `None` when
/// neither is set; the agent then refuses to start, so a run never writes
/// into a home directory it was not given.
pub(crate) fn data_dir() -> Option<PathBuf> {
    let set = |name| std::env::var_os(name).filter(|value| !value.is_empty());
    set("RIG_DATA_DIR")
        .map(PathBuf::from)
        .or_else(|| set("RIG_HOME").map(|home| PathBuf::from(home).join("data")))
}

/// The app every agent starts from, before any plugin of the list.
pub(crate) fn base_app(data: PathBuf) -> App {
    let mut app = App::new();
    app.insert_resource(DataDir(data));
    app.add_plugins((
        TaskPoolPlugin::default(),
        FrameCountPlugin,
        TimePlugin,
        // A windowing plugin added later replaces this runner.
        ScheduleRunnerPlugin::run_loop(Duration::from_millis(16)),
        LogPlugin {
            fmt_layer: log_to_file,
            ..default()
        },
    ));
    // A failing or panicking plugin system is logged, not fatal.
    app.set_error_handler(bevy::ecs::error::warn);
    std::panic::set_hook(Box::new(|info| {
        error!("{info}\n{}", std::backtrace::Backtrace::capture());
    }));
    app
}

/// A log over this size is moved aside when the agent starts.
const LOG_LIMIT: u64 = 8 * 1024 * 1024;

/// Sends formatted logs to `logs/agent.log` in the data directory, or
/// nowhere when the file cannot be opened: never to stderr.
fn log_to_file(app: &mut App) -> Option<BoxedFmtLayer> {
    let file = app.world().get_resource::<DataDir>().and_then(|data| {
        let logs = data.0.join("logs");
        std::fs::create_dir_all(&logs).ok()?;
        let log = logs.join("agent.log");
        // Keep the log bounded: start over, keeping one older log.
        if log.metadata().is_ok_and(|meta| meta.len() > LOG_LIMIT) {
            let _ = std::fs::rename(&log, logs.join("agent.log.old"));
        }
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log)
            .ok()
    });
    let layer = fmt::Layer::default().with_ansi(false);
    Some(match file {
        Some(file) => Box::new(layer.with_writer(std::sync::Mutex::new(file))),
        None => Box::new(layer.with_writer(std::io::sink)),
    })
}
