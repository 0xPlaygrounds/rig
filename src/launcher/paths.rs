//! Where the launcher keeps its files: one root under `RIG_HOME`, or the
//! platform's config, cache and data directories.

use std::{
    env::{self, consts::EXE_SUFFIX},
    path::PathBuf,
};

use super::{Error, Result};

/// The launcher's directories.
pub struct Paths {
    /// Holds `plugins.toml`.
    pub config: PathBuf,
    /// The generated agent project and its `target/`.
    pub project: PathBuf,
    /// The binaries `current` and `good`.
    pub bin: PathBuf,
    /// One directory per session, and `latest`.
    pub sessions: PathBuf,
}

impl Paths {
    /// `$RIG_HOME/{config,cache/agent,bin,sessions}` when `RIG_HOME` is
    /// set, else the XDG directories (`%APPDATA%` and `%LOCALAPPDATA%` on
    /// Windows).
    pub fn from_env() -> Result<Self> {
        if let Some(home) = env::var_os("RIG_HOME").filter(|home| !home.is_empty()) {
            let home = PathBuf::from(home);
            return Ok(Self {
                config: home.join("config"),
                project: home.join("cache").join("agent"),
                bin: home.join("bin"),
                sessions: home.join("sessions"),
            });
        }
        let data = base("XDG_DATA_HOME", ".local/share", "LOCALAPPDATA")?.join("rig");
        Ok(Self {
            config: base("XDG_CONFIG_HOME", ".config", "APPDATA")?.join("rig"),
            project: base("XDG_CACHE_HOME", ".cache", "LOCALAPPDATA")?
                .join("rig")
                .join("agent"),
            bin: data.join("bin"),
            sessions: data.join("sessions"),
        })
    }

    /// The plugin list.
    pub fn plugins(&self) -> PathBuf {
        self.config.join("plugins.toml")
    }

    /// The binary cargo builds.
    pub fn built(&self) -> PathBuf {
        self.project
            .join("target")
            .join("debug")
            .join(format!("rig-code-agent{EXE_SUFFIX}"))
    }

    /// The newest binary that built.
    pub fn current(&self) -> PathBuf {
        self.bin.join(format!("current{EXE_SUFFIX}"))
    }

    /// The newest binary that got through startup.
    pub fn good(&self) -> PathBuf {
        self.bin.join(format!("good{EXE_SUFFIX}"))
    }
}

/// `$xdg`, else `%windows%` on Windows, else `~/unix`.
fn base(xdg: &str, unix: &str, windows: &str) -> Result<PathBuf> {
    if let Some(dir) = env::var_os(xdg).filter(|dir| !dir.is_empty()) {
        return Ok(PathBuf::from(dir));
    }
    if cfg!(windows)
        && let Some(dir) = env::var_os(windows)
    {
        return Ok(PathBuf::from(dir));
    }
    env::home_dir()
        .map(|home| home.join(unix))
        .ok_or_else(|| Error("cannot find the home directory; set RIG_HOME".into()))
}
