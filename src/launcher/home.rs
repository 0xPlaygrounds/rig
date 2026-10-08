//! The `RIG_HOME` layout. Every file the launcher and the agent write lives
//! under this one root.

use std::path::{Path, PathBuf};

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

    /// A file in `bin/`: `staged`, `trial`, `good` or `ready`.
    pub fn bin(&self, name: &str) -> PathBuf {
        self.root.join("bin").join(name)
    }

    /// The text log of a session.
    pub fn session_log(&self, session: &str) -> PathBuf {
        self.root.join("sessions").join(session).join("agent.log")
    }
}
