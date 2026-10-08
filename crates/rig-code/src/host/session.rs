//! Where the agent keeps its files, and the session they belong to. Logs and
//! panics go to the session's `agent.log`, never to the terminal.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::{BoxedFmtLayer, tracing_subscriber};

use crate::core::{SessionDir, append, random};

/// The agent's data directory, which holds its sessions: `$RIG_HOME/data`
/// when `RIG_HOME` is set, else `$XDG_DATA_HOME/rig` or
/// `~/.local/share/rig`. The launcher uses the same rules.
#[derive(Resource, Clone, Debug)]
pub struct Dirs {
    /// Sessions and the resume file.
    pub data: PathBuf,
}

impl Dirs {
    /// The directory the environment selects.
    pub fn from_env() -> Self {
        let set = |variable: &str| std::env::var_os(variable).filter(|value| !value.is_empty());
        let data = match set("RIG_HOME") {
            Some(home) => PathBuf::from(home).join("data"),
            None => set("XDG_DATA_HOME")
                .map_or_else(
                    || {
                        std::env::home_dir()
                            .unwrap_or_default()
                            .join(".local/share")
                    },
                    PathBuf::from,
                )
                .join("rig"),
        };
        Self { data }
    }

    /// The file naming the session the next start restores. It is written
    /// before a reload exit and deleted once the restarted agent is ready.
    pub fn resume_path(&self) -> PathBuf {
        self.data.join("resume")
    }
}

/// Inserts [`Dirs`] and the [`SessionDir`]: the one the resume file names,
/// or a new one. Creates the session directory and sends panic messages to
/// the session log.
pub(crate) fn open(app: &mut App) {
    let dirs = Dirs::from_env();
    let resume = fs::read_to_string(dirs.resume_path())
        .ok()
        .map(|id| id.trim().to_owned())
        .filter(|id| {
            !id.is_empty()
                && id
                    .chars()
                    .all(|character| character.is_ascii_alphanumeric() || character == '-')
        });
    let resumed = resume.is_some();
    let id = resume.unwrap_or_else(|| {
        let started = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        format!("{started}-{:04x}", random() & 0xffff)
    });
    let session = SessionDir {
        dir: dirs.data.join("sessions").join(&id),
        id,
        resumed,
    };
    // A directory that cannot be created is reported in the view at startup.
    let _ = fs::create_dir_all(&session.dir);
    let log = log_path(&session);
    std::panic::set_hook(Box::new(move |info| {
        let backtrace = std::backtrace::Backtrace::capture();
        if let Ok(mut file) = append(&log) {
            let _ = writeln!(file, "panic: {info}\n{backtrace}");
        }
    }));
    app.insert_resource(dirs).insert_resource(session);
}

/// The session's log file.
fn log_path(session: &SessionDir) -> PathBuf {
    session.dir.join("agent.log")
}

/// The `LogPlugin` formatter: plain text into the session's `agent.log`, or
/// nowhere when the file cannot be opened.
pub(crate) fn log_layer(app: &mut App) -> Option<BoxedFmtLayer> {
    let file = app
        .world()
        .get_resource::<SessionDir>()
        .and_then(|session| append(&log_path(session)).ok());
    let layer = tracing_subscriber::fmt::Layer::default().with_ansi(false);
    Some(match file {
        Some(file) => Box::new(layer.with_writer(Mutex::new(file))),
        None => Box::new(layer.with_writer(std::io::sink)),
    })
}
