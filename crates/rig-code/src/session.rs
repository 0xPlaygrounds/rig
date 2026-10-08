//! The session: its id, its directory, and the log file in it.
//!
//! Every session writes to `<data>/sessions/<id>/`. The data directory is
//! `RIG_DATA_DIR` when set, then `$RIG_HOME/data`, then the platform's data
//! directory under `rig`. Logs go to `rig-code.log` there, never to the
//! terminal the view draws on.

use std::{
    path::PathBuf,
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};

use bevy_app::{App, Plugin};
use bevy_ecs::prelude::*;
use bevy_log::{BoxedFmtLayer, tracing_subscriber};

/// The running session.
#[derive(Resource, Debug, Clone)]
pub struct Session {
    /// The session id: `RIG_SESSION` when set, else `<unix seconds>-<pid>`.
    pub id: String,
    /// The directory holding the session's files.
    pub dir: PathBuf,
}

impl Session {
    /// The session the environment names, with its directory created.
    pub fn from_env() -> Self {
        let id = std::env::var("RIG_SESSION").unwrap_or_else(|_| {
            let secs = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_secs());
            format!("{secs}-{}", std::process::id())
        });
        let dir = data_dir().join("sessions").join(&id);
        // A session that cannot create its directory still runs; its files
        // fail to open and are skipped.
        let _ = std::fs::create_dir_all(&dir);
        Self { id, dir }
    }
}

/// The directory for rig's binaries and sessions.
pub fn data_dir() -> PathBuf {
    let var = |name: &str| std::env::var_os(name).filter(|value| !value.is_empty());
    if let Some(dir) = var("RIG_DATA_DIR") {
        return dir.into();
    }
    if let Some(home) = var("RIG_HOME") {
        return PathBuf::from(home).join("data");
    }
    if let Some(data) = var("XDG_DATA_HOME") {
        return PathBuf::from(data).join("rig");
    }
    match var("HOME") {
        Some(home) => PathBuf::from(home).join(".local/share/rig"),
        None => PathBuf::from(".rig"),
    }
}

/// Inserts [`Session`]. Comes before `LogPlugin`, whose file layer reads it.
pub struct SessionPlugin;

impl Plugin for SessionPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Session::from_env());
    }
}

/// The `LogPlugin` formatter: plain text appended to the session's
/// `rig-code.log`, or nowhere when that file cannot be opened.
pub(crate) fn file_log_layer(app: &mut App) -> Option<BoxedFmtLayer> {
    let file = app.world().get_resource::<Session>().and_then(|session| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(session.dir.join("rig-code.log"))
            .ok()
    });
    let layer = tracing_subscriber::fmt::Layer::default().with_ansi(false);
    Some(match file {
        Some(file) => Box::new(layer.with_writer(Mutex::new(file))),
        None => Box::new(layer.with_writer(std::io::sink)),
    })
}

/// Route panics to the log instead of stderr. Bevy catches panics in
/// systems, observers and commands; this keeps their messages off the
/// terminal.
pub(crate) fn log_panics() {
    std::panic::set_hook(Box::new(|info| {
        bevy_log::error!("panic: {info}");
    }));
}
