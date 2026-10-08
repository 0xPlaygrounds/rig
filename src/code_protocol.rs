//! The protocol between the `rig` launcher, this crate's binary, and the
//! rig-code agent it builds and runs: the `RIG_HOME` layout, session ids,
//! the environment the launcher passes to the agent, and the exit code with
//! which the agent asks for a restart. The launcher and `rig-code` both use
//! this module, so the two sides cannot drift apart. It uses only std, and
//! it changes with rig-code.

use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

/// The exit code with which the agent asks its launcher to restart it on
/// the staged build.
pub const RELOAD_EXIT_CODE: u8 = 75;

/// The environment variables of the protocol.
pub mod env {
    /// The root of every rig directory, `~/.rig` when unset. The launcher
    /// passes it on made absolute; see [`Home::from_env`](super::Home::from_env).
    pub const HOME: &str = "RIG_HOME";
    /// The [`SessionId`](super::SessionId) the agent runs. `rig build`,
    /// run by the agent's `/reload`, stages for that session's launcher.
    pub const SESSION: &str = "RIG_SESSION";
    /// The launcher executable, which `/reload` runs as `rig build`.
    pub const LAUNCHER: &str = "RIG_LAUNCHER";
    /// A line the agent shows at startup, such as a rollback.
    pub const NOTICE: &str = "RIG_NOTICE";
    /// What the launcher tells the agent it runs, and only that agent: a
    /// command the agent runs, such as a nested agent while working on
    /// rig-code itself, must not act as this agent.
    pub const AGENT_ONLY: [&str; 3] = [SESSION, LAUNCHER, NOTICE];
}

/// A session's id: digits and dashes, so it is safe as a path segment. The
/// launcher makes one per new session; the agent makes its own when it
/// runs without the launcher.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SessionId(String);

impl SessionId {
    /// A new id: the Unix time in seconds and this process's id.
    pub fn generate() -> Self {
        let seconds = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs())
            .unwrap_or_default();
        Self(format!("{seconds}-{}", std::process::id()))
    }

    /// The id in [`env::SESSION`], or `None` when it is unset or empty.
    pub fn from_env() -> Result<Option<Self>, InvalidSessionId> {
        let id = std::env::var_os(env::SESSION).filter(|id| !id.is_empty());
        id.map(|id| id.to_string_lossy().parse()).transpose()
    }

    /// The id as text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for SessionId {
    type Err = InvalidSessionId;

    fn from_str(id: &str) -> Result<Self, Self::Err> {
        let valid = id.starts_with(|c: char| c.is_ascii_digit())
            && id.chars().all(|c| c.is_ascii_digit() || c == '-');
        if valid {
            Ok(Self(id.to_owned()))
        } else {
            Err(InvalidSessionId(id.to_owned()))
        }
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Text that is not a [`SessionId`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvalidSessionId(String);

impl fmt::Display for InvalidSessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "`{}` is not a session id: it must be digits and dashes",
            self.0
        )
    }
}

impl std::error::Error for InvalidSessionId {}

/// The `RIG_HOME` layout. Every file the launcher and the agent write lives
/// under this one root.
#[derive(Clone, Debug)]
pub struct Home {
    root: PathBuf,
}

impl Home {
    /// The root from [`env::HOME`], else `~/.rig`, made absolute so the
    /// launcher, the agent and their children agree on it from any
    /// directory.
    pub fn from_env() -> Self {
        let root = std::env::var_os(env::HOME)
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

    /// `plugins.toml`, the plugin list.
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

    /// The directory of the builds and the build lock.
    pub fn bin(&self) -> PathBuf {
        self.root.join("bin")
    }

    /// The last build that started up.
    pub fn good(&self) -> PathBuf {
        self.bin().join("good")
    }

    /// The build `rig build` staged for whichever launcher starts next.
    pub fn staged(&self) -> PathBuf {
        self.bin().join("staged")
    }

    /// The stamp of the last build that stopped before it was ready.
    pub fn rejected(&self) -> PathBuf {
        self.bin().join("rejected")
    }

    /// The lock every launcher on this root takes to generate, build and
    /// stage the agent.
    pub fn build_lock(&self) -> PathBuf {
        self.bin().join("lock")
    }

    /// The build staged for the launcher of `session` alone.
    pub fn staged_for(&self, session: &SessionId) -> PathBuf {
        self.bin().join(format!("staged-{session}"))
    }

    /// The build the launcher of `session` runs until it is ready.
    pub fn trial_for(&self, session: &SessionId) -> PathBuf {
        self.bin().join(format!("trial-{session}"))
    }

    /// The session whose launcher owns a file of [`Home::bin`] named
    /// `name`: a `staged-<session>` or `trial-<session>`.
    pub fn owner_of(name: &str) -> Option<SessionId> {
        name.strip_prefix("trial-")
            .or_else(|| name.strip_prefix("staged-"))?
            .parse()
            .ok()
    }

    /// The directory of `session`.
    pub fn session(&self, session: &SessionId) -> SessionDir {
        SessionDir(self.root.join("sessions").join(session.as_str()))
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
}

/// A session's directory: the agent's saved state, effect log and text
/// log, the file that tells the launcher the agent started, and the lock
/// of the launcher running it.
#[derive(Clone, Debug)]
pub struct SessionDir(PathBuf);

impl SessionDir {
    /// The directory itself.
    pub fn path(&self) -> &Path {
        &self.0
    }

    /// The saved agents.
    pub fn state(&self) -> PathBuf {
        self.0.join("state.json")
    }

    /// The effect log, one effect record per line.
    pub fn effects(&self) -> PathBuf {
        self.0.join("effects.jsonl")
    }

    /// The agent's text log.
    pub fn log(&self) -> PathBuf {
        self.0.join("agent.log")
    }

    /// Written by the agent once it started; the launcher then keeps the
    /// build.
    pub fn ready(&self) -> PathBuf {
        self.0.join("ready")
    }

    /// Held by the launcher running the session while it lives.
    pub fn launcher_lock(&self) -> PathBuf {
        self.0.join("launcher.lock")
    }
}
