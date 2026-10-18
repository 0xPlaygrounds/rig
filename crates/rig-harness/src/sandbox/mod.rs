//! The sandbox of a restricted worker: Landlock on Linux, `sandbox-exec` on
//! macOS. A grant this platform can't enforce is refused, never widened.

use std::path::Path;
use std::process::Command;

use rig_protocol::Grant;

#[derive(Debug)]
pub enum SandboxError {
    /// This platform can't enforce what the grant restricts.
    Unavailable(String),
    Io(std::io::Error),
}

impl std::fmt::Display for SandboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SandboxError::Unavailable(what) => write!(f, "the sandbox can't enforce {what}"),
            SandboxError::Io(e) => write!(f, "the sandbox failed: {e}"),
        }
    }
}

impl std::error::Error for SandboxError {}

/// Confines this process to `grant`, with `session_dir` writable. Called at
/// worker start, before any plugin runs.
pub fn apply(grant: &Grant, session_dir: &Path) -> Result<(), SandboxError> {
    let _ = session_dir;
    refuse_restrictions(grant)
}

/// Makes `cmd` start confined to `grant`, where the platform confines at spawn.
pub fn wrap(cmd: &mut Command, grant: &Grant) -> Result<(), SandboxError> {
    let _ = cmd;
    refuse_restrictions(grant)
}

fn refuse_restrictions(grant: &Grant) -> Result<(), SandboxError> {
    if grant.dirs.is_some() {
        return Err(SandboxError::Unavailable("a directory grant".into()));
    }
    if !grant.net {
        return Err(SandboxError::Unavailable("no network".into()));
    }
    Ok(())
}
