//! The `RIG_HOME` layout. Every file the launcher and the agent write lives
//! under this one root.

use std::fs::{self, File, TryLockError};
use std::io::ErrorKind;
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
        self.root.join("plugins.toml")
    }

    /// The generated agent project.
    pub fn project(&self) -> PathBuf {
        self.root.join("project")
    }

    /// Cargo's target directory for the agent project.
    pub fn target(&self) -> PathBuf {
        self.root.join("target")
    }

    /// A file in `bin/`: `staged`, `good`, `tried`, `lock`, or a
    /// launcher's own `staged-<session>` and `trial-<session>`.
    pub fn bin(&self, name: &str) -> PathBuf {
        self.root.join("bin").join(name)
    }

    /// The build staged for the launcher of `session` alone.
    pub fn staged_for(&self, session: &str) -> PathBuf {
        self.bin(&format!("staged-{session}"))
    }

    /// The build the launcher of `session` runs until it is ready.
    pub fn trial_for(&self, session: &str) -> PathBuf {
        self.bin(&format!("trial-{session}"))
    }

    /// A session's directory. The agent keeps its state, effect log and
    /// text log there, and writes `ready` once it started.
    pub fn session(&self, session: &str) -> PathBuf {
        self.root.join("sessions").join(session)
    }

    /// The file naming the session to resume in `directory`: the last one
    /// run there that did not quit cleanly. The name is a hash of the path,
    /// so a session comes back only where it ran.
    pub fn resume_marker(&self, directory: &Path) -> PathBuf {
        // FNV-1a: stable across builds and toolchains, unlike std's hasher.
        let key = directory
            .as_os_str()
            .as_encoded_bytes()
            .iter()
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
                (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
            });
        self.root.join("resume").join(format!("{key:016x}"))
    }

    /// Takes the lock that marks the launcher of `session` as running,
    /// unless another launcher holds it. It is released when the returned
    /// file is dropped or the launcher dies.
    pub fn hold_session(&self, session: &str) -> Result<Option<File>> {
        let file = session_lock(&self.session(session))?;
        match file.try_lock() {
            Ok(()) => Ok(Some(file)),
            Err(TryLockError::WouldBlock) => Ok(None),
            Err(TryLockError::Error(failure)) => Err(failure.into()),
        }
    }

    /// Removes the staged and trial builds of launchers that are gone,
    /// such as one killed with its terminal. Call it holding [`Home::lock`].
    pub fn sweep(&self) -> Result<()> {
        let Ok(entries) = fs::read_dir(self.root.join("bin")) else {
            return Ok(());
        };
        for entry in entries {
            let entry = entry?;
            let name = entry.file_name();
            let Some(session) = name.to_str().and_then(|name| {
                name.strip_prefix("trial-")
                    .or_else(|| name.strip_prefix("staged-"))
            }) else {
                continue;
            };
            let lock = File::options()
                .write(true)
                .open(self.session(session).join("launcher.lock"));
            let gone = match lock {
                Ok(lock) => lock.try_lock().is_ok(),
                Err(failure) => failure.kind() == ErrorKind::NotFound,
            };
            if gone {
                fs::remove_file(entry.path())?;
            }
        }
        Ok(())
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

fn session_lock(session: &Path) -> std::io::Result<File> {
    fs::create_dir_all(session)?;
    File::options()
        .create(true)
        .write(true)
        .truncate(false)
        .open(session.join("launcher.lock"))
}
