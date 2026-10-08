//! Where a session keeps its files.
//!
//! The launcher names the session directory in `RIG_CODE_SESSION_DIR`. A
//! binary started without it, such as `cargo run -p rig-code`, opens a new
//! session under `$RIG_HOME/sessions`, or under the user's data directory.

use std::{
    path::{Path, PathBuf},
    sync::OnceLock,
    time::{SystemTime, UNIX_EPOCH},
};

/// The session directory, created on first use. Every call returns the same
/// directory for the life of the process.
pub fn session_dir() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::var_os("RIG_CODE_SESSION_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| data_root().join("sessions").join(new_session_id()));
        // A directory that cannot be created surfaces as failed writes in the
        // files below it, which are reported where they happen.
        let _ = std::fs::create_dir_all(&dir);
        dir
    })
}

/// The log file every log line and stray stdout or stderr write goes to.
pub fn log_file() -> PathBuf {
    session_dir().join("agent.log")
}

/// The effect log of the session.
pub fn effects_file() -> PathBuf {
    session_dir().join("effects.jsonl")
}

/// The saved agents, restored at startup.
pub fn state_file() -> PathBuf {
    session_dir().join("state.json")
}

/// `$RIG_HOME`, else `$XDG_DATA_HOME/rig`, else `~/.local/share/rig`.
fn data_root() -> PathBuf {
    if let Some(home) = std::env::var_os("RIG_HOME").filter(|home| !home.is_empty()) {
        return PathBuf::from(home);
    }
    if let Some(data) = std::env::var_os("XDG_DATA_HOME").filter(|data| !data.is_empty()) {
        return PathBuf::from(data).join("rig");
    }
    std::env::home_dir()
        .unwrap_or_default()
        .join(".local/share/rig")
}

fn new_session_id() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or_default();
    format!("{seconds}-{}", std::process::id())
}
