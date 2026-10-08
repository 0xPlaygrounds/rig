use std::path::PathBuf;

/// Where rig keeps its files: everything under `RIG_HOME` when set,
/// otherwise the XDG directories.
pub struct Home {
    /// Holds `plugins.toml`.
    pub config: PathBuf,
    /// Holds the generated agent project and its `target/`.
    pub cache: PathBuf,
    /// Holds `bin/` and `sessions/`.
    pub data: PathBuf,
}

impl Home {
    /// The directories the environment names.
    pub fn from_env() -> Result<Self, String> {
        if let Some(root) = var("RIG_HOME") {
            let root = PathBuf::from(root);
            return Ok(Self {
                config: root.join("config"),
                cache: root.join("cache"),
                data: root.join("data"),
            });
        }
        let home = var("HOME").map(PathBuf::from);
        let dir = |xdg: &str, fallback: &str| {
            var(xdg)
                .map(PathBuf::from)
                .or_else(|| home.as_ref().map(|home| home.join(fallback)))
                .map(|dir| dir.join("rig"))
                .ok_or_else(|| format!("set RIG_HOME, {xdg} or HOME"))
        };
        Ok(Self {
            config: dir("XDG_CONFIG_HOME", ".config")?,
            cache: dir("XDG_CACHE_HOME", ".cache")?,
            data: dir("XDG_DATA_HOME", ".local/share")?,
        })
    }

    /// The plugin list.
    pub fn plugins(&self) -> PathBuf {
        self.config.join("plugins.toml")
    }

    /// The generated agent project.
    pub fn project(&self) -> PathBuf {
        self.cache.join("agent")
    }

    /// The last binary that reached ready.
    pub fn current(&self) -> PathBuf {
        self.data.join("bin").join("current")
    }

    /// A built binary not yet started.
    pub fn next(&self) -> PathBuf {
        self.data.join("bin").join("next")
    }

    /// The directory of session `id`.
    pub fn session(&self, id: &str) -> PathBuf {
        self.data.join("sessions").join(id)
    }
}

fn var(name: &str) -> Option<std::ffi::OsString> {
    std::env::var_os(name).filter(|value| !value.is_empty())
}
