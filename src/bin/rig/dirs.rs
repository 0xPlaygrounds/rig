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
    /// the XDG directories (`%APPDATA%\rig\...` on Windows).
    pub(crate) fn resolve() -> Result<Self, Failure> {
        if let Some(home) = variable("RIG_HOME") {
            return Ok(Self::under(&home));
        }
        if cfg!(windows) {
            let app_data =
                variable("APPDATA").ok_or_else(|| Failure::config("set RIG_HOME or APPDATA"))?;
            return Ok(Self::under(&app_data.join("rig")));
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

    /// A newer binary that has not started yet.
    pub(crate) fn candidate_bin(&self) -> PathBuf {
        self.data.join("bin").join(executable("candidate"))
    }

    /// Created by the agent once it is up.
    pub(crate) fn ready_file(&self) -> PathBuf {
        self.data.join("run").join("ready")
    }

    /// Where the agent's stdout and stderr go.
    pub(crate) fn stdio_log(&self) -> PathBuf {
        self.data.join("logs").join("agent-stdio.log")
    }
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
