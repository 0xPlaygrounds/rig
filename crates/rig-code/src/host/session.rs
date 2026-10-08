//! Where this process keeps its session: the directory the launcher names,
//! the text log, and panics routed to that log instead of the terminal.

use std::fs::{self, OpenOptions};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use bevy_app::prelude::*;
use bevy_log::tracing_subscriber::fmt;
use bevy_log::{BoxedFmtLayer, error};

use crate::core::save::SessionPaths;

/// The session paths from `RIG_HOME` (default `$HOME/.rig`) and
/// `RIG_SESSION` (default a new id): `$RIG_HOME/sessions/$RIG_SESSION/`,
/// with the directory created.
pub fn paths_from_env() -> SessionPaths {
    let home = std::env::var_os("RIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".rig")))
        .unwrap_or_else(|| PathBuf::from(".rig"));
    // The launcher's own default and id format (`src/launcher` in the
    // `rig` crate); these apply when the agent runs without it.
    let id = std::env::var("RIG_SESSION").unwrap_or_else(|_| {
        let seconds = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs())
            .unwrap_or_default();
        format!("{seconds}-{}", std::process::id())
    });
    let dir = home.join("sessions").join(&id);
    // A directory that cannot be created shows up as a failed save.
    fs::create_dir_all(&dir).ok();
    SessionPaths { id, dir }
}

/// The session's text log.
pub fn log_path(paths: &SessionPaths) -> PathBuf {
    paths.dir.join("agent.log")
}

/// Inserts the [`SessionPaths`] from the environment and routes panics to
/// the log.
pub struct SessionPlugin;

impl Plugin for SessionPlugin {
    fn build(&self, app: &mut App) {
        std::panic::set_hook(Box::new(|info| {
            error!("{info}\n{}", std::backtrace::Backtrace::capture());
        }));
        app.insert_resource(paths_from_env());
    }
}

/// The `LogPlugin` formatter: plain text appended to the session's log, so
/// nothing is written to stderr.
pub fn log_layer(app: &mut App) -> Option<BoxedFmtLayer> {
    let file = app
        .world()
        .get_resource::<SessionPaths>()
        .and_then(|paths| {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_path(paths))
                .ok()
        });
    let layer = fmt::Layer::default().with_ansi(false);
    Some(match file {
        Some(file) => Box::new(layer.with_writer(Mutex::new(file))),
        None => Box::new(layer.with_writer(std::io::sink)),
    })
}
