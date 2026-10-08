//! Where the agent keeps its files, and the session they belong to. Logs and
//! panics go to the session's `agent.log`, never to the terminal.

use std::fs::{self, File, OpenOptions};
use std::hash::{BuildHasher, Hasher, RandomState};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::{BoxedFmtLayer, tracing_subscriber};

/// The agent's directories. With `RIG_HOME` set, all of them live under
/// it (`config`, `cache`, `data`); otherwise they follow the XDG base
/// directory variables, each with a `rig` subdirectory.
#[derive(Resource, Clone, Debug)]
pub struct Dirs {
    /// User-edited configuration.
    pub config: PathBuf,
    /// Rebuildable files, such as build output.
    pub cache: PathBuf,
    /// Sessions.
    pub data: PathBuf,
}

impl Dirs {
    /// The directories the environment selects.
    pub fn from_env() -> Self {
        if let Some(home) = std::env::var_os("RIG_HOME").filter(|home| !home.is_empty()) {
            let home = PathBuf::from(home);
            return Self {
                config: home.join("config"),
                cache: home.join("cache"),
                data: home.join("data"),
            };
        }
        let user = std::env::home_dir().unwrap_or_default();
        let base = |variable: &str, default: &str| {
            std::env::var_os(variable)
                .filter(|value| !value.is_empty())
                .map_or_else(|| user.join(default), PathBuf::from)
                .join("rig")
        };
        Self {
            config: base("XDG_CONFIG_HOME", ".config"),
            cache: base("XDG_CACHE_HOME", ".cache"),
            data: base("XDG_DATA_HOME", ".local/share"),
        }
    }
}

/// This run's session: `<data>/sessions/<id>/`, holding `agent.log` and
/// `effects.jsonl`.
#[derive(Resource, Clone, Debug)]
pub struct Session {
    /// The session id, unique per start.
    pub id: String,
    /// The session directory.
    pub dir: PathBuf,
}

impl Session {
    /// The log file.
    pub fn log_path(&self) -> PathBuf {
        self.dir.join("agent.log")
    }

    /// The effect log, one `EffectRecord` per line.
    pub fn effects_path(&self) -> PathBuf {
        self.dir.join("effects.jsonl")
    }
}

/// A random number, from the standard library's per-process hash seed.
pub(crate) fn random() -> u64 {
    RandomState::new().build_hasher().finish()
}

/// Inserts [`Dirs`] and a new [`Session`], creates the session directory
/// and sends panic messages to the session log.
pub(crate) fn open(app: &mut App) {
    let dirs = Dirs::from_env();
    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let id = format!("{started}-{:04x}", random() & 0xffff);
    let session = Session {
        dir: dirs.data.join("sessions").join(&id),
        id,
    };
    // A directory that cannot be created is reported in the view at startup.
    let _ = fs::create_dir_all(&session.dir);
    let log = session.log_path();
    std::panic::set_hook(Box::new(move |info| {
        let backtrace = std::backtrace::Backtrace::capture();
        if let Ok(mut file) = append(&log) {
            let _ = writeln!(file, "panic: {info}\n{backtrace}");
        }
    }));
    app.insert_resource(dirs).insert_resource(session);
}

/// Opens `path` for appending, creating it if needed.
pub(crate) fn append(path: &Path) -> std::io::Result<File> {
    OpenOptions::new().create(true).append(true).open(path)
}

/// The `LogPlugin` formatter: plain text into the session's `agent.log`, or
/// nowhere when the file cannot be opened.
pub(crate) fn log_layer(app: &mut App) -> Option<BoxedFmtLayer> {
    let file = app
        .world()
        .get_resource::<Session>()
        .and_then(|session| append(&session.log_path()).ok());
    let layer = tracing_subscriber::fmt::Layer::default().with_ansi(false);
    Some(match file {
        Some(file) => Box::new(layer.with_writer(Mutex::new(file))),
        None => Box::new(layer.with_writer(std::io::sink)),
    })
}
