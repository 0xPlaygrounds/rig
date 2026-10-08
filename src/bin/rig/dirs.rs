//! Where rig keeps its config, cache and data: all under `RIG_HOME` when it
//! is set, otherwise in the platform's usual places.

use std::path::{Path, PathBuf};

use crate::Failure;

/// The three directory roots.
pub(crate) struct Dirs {
    /// `plugins.toml`.
    pub(crate) config: PathBuf,
    /// The agent project and its `target` directory.
    pub(crate) cache: PathBuf,
    /// Binaries, sessions, logs and the files shared with the agent.
    pub(crate) data: PathBuf,
}

impl Dirs {
    /// `$RIG_HOME/{config,cache,data}` when `RIG_HOME` is set; otherwise
    /// the XDG directories. Always
    /// absolute: cargo and the agent run in other directories.
    pub(crate) fn resolve() -> Result<Self, Failure> {
        let dirs = Self::relative()?;
        Ok(Self {
            config: absolute(&dirs.config)?,
            cache: absolute(&dirs.cache)?,
            data: absolute(&dirs.data)?,
        })
    }

    fn relative() -> Result<Self, Failure> {
        if let Some(home) = variable("RIG_HOME") {
            return Ok(Self::under(&home));
        }
        let home = variable("HOME");
        let base = |xdg: &str, fallback: &str| {
            variable(xdg)
                .or_else(|| home.as_ref().map(|home| home.join(fallback)))
                .map(|base| base.join("rig"))
                .ok_or_else(|| Failure::config(format!("set RIG_HOME, {xdg} or HOME")))
        };
        Ok(Self {
            config: base("XDG_CONFIG_HOME", ".config")?,
            cache: base("XDG_CACHE_HOME", ".cache")?,
            data: base("XDG_DATA_HOME", ".local/share")?,
        })
    }

    fn under(root: &Path) -> Self {
        Self {
            config: root.join("config"),
            cache: root.join("cache"),
            data: root.join("data"),
        }
    }

    /// The generated agent project.
    pub(crate) fn project(&self) -> PathBuf {
        self.cache.join("project")
    }

    /// The last binary that started.
    pub(crate) fn current_bin(&self) -> PathBuf {
        self.data.join("bin").join(executable("current"))
    }

    /// The modification time of the build staged as the candidate.
    pub(crate) fn staged_stamp(&self) -> PathBuf {
        self.data.join("bin").join("staged")
    }

    /// The modification time of the build that crashed during startup, so
    /// it is not staged again.
    pub(crate) fn rejected_stamp(&self) -> PathBuf {
        self.data.join("bin").join("rejected")
    }

    /// A newer binary that has not started yet.
    pub(crate) fn candidate_bin(&self) -> PathBuf {
        self.data.join("bin").join(executable("candidate"))
    }

    /// Created by the agent once it is up. One per launcher process, so
    /// two launchers sharing a home do not read each other's.
    pub(crate) fn ready_file(&self) -> PathBuf {
        self.data
            .join("run")
            .join(format!("ready-{}", std::process::id()))
    }

    /// Where the agent's stdout and stderr go.
    pub(crate) fn stdio_log(&self) -> PathBuf {
        self.data.join("logs").join("agent-stdio.log")
    }
}

/// `path` made absolute against the working directory.
pub(crate) fn absolute(path: &Path) -> Result<PathBuf, Failure> {
    std::path::absolute(path)
        .map_err(|error| Failure::io(format!("cannot resolve {}", path.display()), error))
}

/// `name` with the platform's executable suffix.
pub(crate) fn executable(name: &str) -> String {
    format!("{name}{}", std::env::consts::EXE_SUFFIX)
}

/// The non-empty value of the environment variable `name`, as a path.
pub(crate) fn variable(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// A log over this size is moved to `<name>.old` before it is reopened.
const LOG_LIMIT: u64 = 8 * 1024 * 1024;

/// Moves `log` to `<log>.old` when it is over [`LOG_LIMIT`], so logs that
/// are appended to on every start stay bounded.
pub(crate) fn rotate(log: &Path) {
    if log.metadata().is_ok_and(|meta| meta.len() > LOG_LIMIT) {
        let _ = std::fs::rename(log, log.with_extension("log.old"));
    }
}
