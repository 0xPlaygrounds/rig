//! The `RIG_HOME` layout. Every file the launcher and the agent write lives
//! under this one root.

use std::fs::{self, File, TryLockError};
use std::path::{Path, PathBuf};

use super::Result;

/// The root directory, `RIG_HOME` or `~/.rig`.
pub struct Home {
    root: PathBuf,
}

impl Home {
    /// The root from `RIG_HOME`, else `~/.rig`, made absolute so the agent
    /// and its children agree on it from any directory.
    pub fn from_env() -> Self {
        let root = std::env::var_os("RIG_HOME")
            .filter(|root| !root.is_empty())
            .map(PathBuf::from)
            .or_else(|| std::env::home_dir().map(|home| home.join(".rig")))
            .unwrap_or_else(|| PathBuf::from(".rig"));
        let root = std::path::absolute(&root).unwrap_or(root);
        Self { root }
    }

    /// The root itself.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The plugin list and build settings.
    pub fn config(&self) -> PathBuf {
        self.root.join("rig.toml")
    }

    /// The generated agent project.
    pub fn project(&self) -> PathBuf {
        self.root.join("project")
    }

    /// Cargo's target directory for the agent project.
    pub fn target(&self) -> PathBuf {
        self.root.join("target")
    }

    /// A file in `bin/`: `staged`, `good`, `built`, `lock`, or a
    /// launcher's own `trial-<session>`.
    pub fn bin(&self, name: &str) -> PathBuf {
        self.root.join("bin").join(name)
    }

    /// A session's directory. The agent keeps its state, effect log and
    /// text log there, and writes `ready` once it started.
    pub fn session(&self, session: &str) -> PathBuf {
        self.root.join("sessions").join(session)
    }

    /// Waits for, then holds, the lock on generating, building and staging
    /// the agent, which every launcher on this root shares. It is released
    /// when the returned file is dropped.
    pub fn lock(&self) -> Result<File> {
        fs::create_dir_all(self.root.join("bin"))?;
        let file = File::options()
            .create(true)
            .write(true)
            .truncate(false)
            .open(self.bin("lock"))?;
        match file.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                eprintln!(
                    "Waiting for another rig on {} to finish building…",
                    self.root.display()
                );
                file.lock()?;
            }
            Err(TryLockError::Error(failure)) => return Err(failure.into()),
        }
        Ok(file)
    }
}
