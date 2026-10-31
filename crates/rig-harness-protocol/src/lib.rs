//! The protocol between the `rig` launcher (the `rig` crate's binary) and the
//! rig-harness agent it builds and runs: the `RIG_HOME` layout, session ids,
//! the environment the launcher passes to the agent, and the exit codes that
//! say how the agent ended. The launcher and `rig-harness` both use
//! this crate, so the two sides cannot drift apart. It uses only std, and
//! it changes with rig-harness.

use std::ffi::OsString;
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// The exit code with which the agent asks its launcher to restart it on
/// the staged build.
pub const RELOAD_EXIT_CODE: u8 = 75;

/// The exit code of a run refused before it did any work, such as for a
/// command line or a `--tools` name it does not take.
pub const REFUSED_EXIT_CODE: u8 = 2;

/// The exit code of a run a signal ended, as a shell reports Ctrl+C.
pub const SIGNAL_EXIT_CODE: u8 = 130;

/// The version of rig: of the workspace, the launcher and the harness they
/// build together. It stamps generated projects, so a new launcher
/// regenerates and rebuilds the agent; a plugin reads it to say which rig
/// wrote something.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The one Bevy version of the agent and of every plugin: a Bevy plugin
/// only works with the exact Bevy version of its app.
pub const BEVY_VERSION: &str = "0.20.0";

/// The environment variables of the protocol.
pub mod env {
    /// The root of every rig directory, `~/.rig` when unset. The launcher
    /// passes it on made absolute; see [`Home::from_env`](super::Home::from_env).
    pub const HOME: &str = "RIG_HOME";
    /// The [`SessionId`](super::SessionId) the agent runs. `rig build`,
    /// run by the agent's `/reload`, stages for that session's launcher.
    pub const SESSION: &str = "RIG_SESSION";
    /// The launcher executable, which `/reload` runs as `rig build`. Unlike
    /// [`AGENT_ONLY`], the agent's commands keep it, so the model runs
    /// `"$RIG_LAUNCHER" plugin check --build`; an agent counts as started by
    /// a launcher only with [`SESSION`] set too.
    pub const LAUNCHER: &str = "RIG_LAUNCHER";
    /// A line the agent shows at startup, such as a rollback.
    pub const NOTICE: &str = "RIG_NOTICE";
    /// Why the build before this start failed, so an older build runs:
    /// the reason and the first compiler errors. The whole output is in
    /// [`Home::build_log`](super::Home::build_log). The agent puts it in
    /// its conversation, for the model to read.
    pub const BUILD_FAILURE: &str = "RIG_BUILD_FAILURE";
    /// The launcher's [`RunDir`](super::RunDir), where the agent says it
    /// started.
    pub const RUN: &str = "RIG_RUN";
    /// Why the launcher started the agent: a
    /// [`StartReason`](super::StartReason), as text.
    pub const START: &str = "RIG_START";
    /// What the launcher tells the agent it runs, and only that agent: a
    /// command the agent runs, such as a nested agent while working on
    /// rig-harness itself, must not act as this agent. Without [`SESSION`]
    /// a nested agent ignores the [`LAUNCHER`] it inherits.
    pub const AGENT_ONLY: &[&str] = &[SESSION, NOTICE, BUILD_FAILURE, RUN, START];
    /// On every process the agent starts for its tools: the path of the
    /// launcher's [`RunDir::groups`](super::RunDir::groups), which the
    /// launcher finds on a listed group's leader before it kills the
    /// group. The name `rig_tools::process::LEDGER_VARIABLE` sets.
    pub const PROCESS_LEDGER: &str = "RIG_PROCESS_LEDGER";
    /// Set by the launcher to a file's path: the agent builds its app,
    /// writes what its plugins add there ([`AgentDescription`](super::AgentDescription))
    /// and exits without running a frame.
    pub const DESCRIBE: &str = "RIG_DESCRIBE";
    /// The start of a secret's variable, such as `RIG_SECRET_CRYPT`
    /// ([`secret_variable`](super::secret_variable)): the launcher sets it
    /// on the agent's process alone, and every process rig starts drops it
    /// ([`secret_variables`](super::secret_variables)).
    pub const SECRET_PREFIX: &str = "RIG_SECRET_";
    /// A rig checkout the agent is built from instead of crates.io.
    pub const SOURCE: &str = "RIG_SOURCE";
}

/// The launcher that started this agent, if one did: it sets both
/// [`env::LAUNCHER`] and [`env::SESSION`]. A nested agent, run by this
/// agent's shell, inherits only the launcher's path, which the model uses.
pub fn launcher() -> Option<OsString> {
    std::env::var_os(env::SESSION).filter(|session| !session.is_empty())?;
    std::env::var_os(env::LAUNCHER).filter(|launcher| !launcher.is_empty())
}

/// The variable that carries the secret `name`: [`env::SECRET_PREFIX`]
/// and the name in upper case, `-` as `_` (`crypt` is `RIG_SECRET_CRYPT`).
pub fn secret_variable(name: &str) -> String {
    format!(
        "{}{}",
        env::SECRET_PREFIX,
        name.to_ascii_uppercase().replace('-', "_")
    )
}

/// Every secret variable ([`env::SECRET_PREFIX`]) in this process's
/// environment: what a process this one starts must not inherit.
pub fn secret_variables() -> Vec<OsString> {
    std::env::vars_os()
        .map(|(name, _)| name)
        .filter(|name| {
            name.to_str()
                .is_some_and(|name| name.starts_with(env::SECRET_PREFIX))
        })
        .collect()
}

/// At most `count` lines of a build's output from the first one that
/// starts with `error`, as cargo writes a compiler error; none when no
/// line does.
pub fn first_errors<'a>(lines: impl IntoIterator<Item = &'a str>, count: usize) -> Vec<&'a str> {
    lines
        .into_iter()
        .skip_while(|line| !line.starts_with("error"))
        .take(count)
        .collect()
}

/// A session's id: digits and dashes, so it is safe as a path segment. The
/// launcher makes one per new session; the agent makes its own when it
/// runs without the launcher.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SessionId(String);

impl SessionId {
    /// A new id: the Unix time in seconds and this process's id, then,
    /// after the process's first, how many it made before, so no two of
    /// one process are the same.
    pub fn generate() -> Self {
        static MADE: AtomicU64 = AtomicU64::new(0);
        let seconds = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs())
            .unwrap_or_default();
        let id = format!("{seconds}-{}", std::process::id());
        match MADE.fetch_add(1, Ordering::Relaxed) {
            0 => Self(id),
            made => Self(format!("{id}-{made}")),
        }
    }

    /// The id in [`env::SESSION`], or `None` when it is unset or empty.
    pub fn from_env() -> Result<Option<Self>, String> {
        let id = std::env::var_os(env::SESSION).filter(|id| !id.is_empty());
        id.map(|id| id.to_string_lossy().parse()).transpose()
    }

    /// The id as text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for SessionId {
    type Err = String;

    fn from_str(id: &str) -> Result<Self, Self::Err> {
        let valid = id.starts_with(|c: char| c.is_ascii_digit())
            && id.chars().all(|c| c.is_ascii_digit() || c == '-');
        if valid {
            Ok(Self(id.to_owned()))
        } else {
            Err(format!(
                "`{id}` is not a session id: it must be digits and dashes"
            ))
        }
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Why the agent process started, which the launcher passes in
/// [`env::START`]. A switch to another session (`/new`, `/resume`) starts
/// it as `New` or `Resumed`; a crash is no reason, since the launcher
/// never restarts a crashed agent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StartReason {
    /// The session's first start: a new session, or `/new`.
    New,
    /// A session that ran before: `rig --resume`, or `/resume` switching
    /// to it.
    Resumed,
    /// The same session again after `/reload` or a plugin's reload exit.
    Reloaded,
    /// The same session again on the previous build, after the launcher
    /// rolled back a new build that stopped during startup.
    Restarted,
}

impl StartReason {
    /// Its text in [`env::START`].
    pub fn as_str(self) -> &'static str {
        match self {
            Self::New => "new",
            Self::Resumed => "resumed",
            Self::Reloaded => "reloaded",
            Self::Restarted => "restarted",
        }
    }

    /// The reason in [`env::START`], or `None` when it is unset or not one.
    pub fn from_env() -> Option<Self> {
        std::env::var(env::START).ok()?.parse().ok()
    }
}

impl FromStr for StartReason {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        [Self::New, Self::Resumed, Self::Reloaded, Self::Restarted]
            .into_iter()
            .find(|reason| reason.as_str() == text)
            .ok_or_else(|| format!("`{text}` is not a start reason"))
    }
}

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
        Self::at(root)
    }

    /// The layout under `root`, made absolute: a home of its own, such as
    /// a test's.
    pub fn at(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let root = std::path::absolute(&root).unwrap_or(root);
        Self { root }
    }

    /// The root itself.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// `plugins.toml`, the plugin list.
    pub fn plugins_toml(&self) -> PathBuf {
        self.root.join("plugins.toml")
    }

    /// `plugins/`, where `rig plugin new` makes plugin crates, outside any
    /// workspace: `plugins/<name>` is listed in `plugins.toml` as
    /// `path = "plugins/<name>"`.
    pub fn plugins(&self) -> PathBuf {
        self.root.join("plugins")
    }

    /// `settings.toml`, the user's settings for plugins: one table per
    /// plugin, which the agent hands to the plugin that reads it.
    pub fn settings(&self) -> PathBuf {
        self.root.join("settings.toml")
    }

    /// `defaults.json`, the model and reasoning setting a new session
    /// starts with: the last ones chosen.
    pub fn defaults(&self) -> PathBuf {
        self.root.join("defaults.json")
    }

    /// `auth/`, where `/login` keeps each subscription credential, as
    /// `<provider>.json`.
    pub fn auth(&self) -> PathBuf {
        self.root.join("auth")
    }

    /// The generated agent project.
    pub fn project(&self) -> PathBuf {
        self.root.join("project")
    }

    /// The whole output of the last build, the launcher's and cargo's,
    /// rewritten by every build: the one before each start and `rig build`,
    /// which `/reload` runs. Its last line says how the build ended.
    pub fn build_log(&self) -> PathBuf {
        self.root.join("build.log")
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

    /// The [`AgentDescription`] report of the build whose stamp is `stamp` (its
    /// modification time in nanoseconds and its length, as `"{mtime} {len}"`),
    /// one per build, which every copy of the build shares.
    pub fn described(&self, stamp: &str) -> PathBuf {
        self.bin()
            .join("described")
            .join(stamp.trim().replace(' ', "-"))
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

    /// `data/`, where plugins keep files across sessions and projects.
    pub fn data(&self) -> PathBuf {
        self.root.join("data")
    }

    /// `projects/`, where plugins keep files per project, one directory
    /// for each project the agent worked in.
    pub fn projects(&self) -> PathBuf {
        self.root.join("projects")
    }

    /// The directory holding every running launcher's [`RunDir`].
    pub fn runs(&self) -> PathBuf {
        self.root.join("run")
    }

    /// The run directory of the launcher with the process id `launcher`.
    pub fn run(&self, launcher: u32) -> RunDir {
        RunDir(self.runs().join(launcher.to_string()))
    }

    /// The directory holding every session's directory.
    pub fn sessions(&self) -> PathBuf {
        self.root.join("sessions")
    }

    /// The directory of `session`.
    pub fn session(&self, session: &SessionId) -> SessionDir {
        SessionDir(self.sessions().join(session.as_str()))
    }

    /// A new session, its directory created: a [`SessionId::generate`]d
    /// id, or, when a session of this home has that one, the first free
    /// one after it, such as `1700000000-42-1`.
    ///
    /// # Errors
    ///
    /// When the directory cannot be created.
    pub fn new_session(&self) -> std::io::Result<(SessionId, SessionDir)> {
        std::fs::create_dir_all(self.sessions())?;
        let base = SessionId::generate();
        let mut id = base.clone();
        let mut taken = 0u64;
        loop {
            let dir = self.session(&id);
            match std::fs::create_dir(dir.path()) {
                Ok(()) => return Ok((id, dir)),
                Err(failure) if failure.kind() == std::io::ErrorKind::AlreadyExists => {
                    taken += 1;
                    id = SessionId(format!("{base}-{taken}"));
                }
                Err(failure) => return Err(failure),
            }
        }
    }

    /// Every saved session of this home (not one being deleted), in no
    /// order, with its directory: what every listing and [`Self::find`]
    /// read.
    pub fn saved_sessions(&self) -> Vec<(SessionId, SessionDir)> {
        let entries = std::fs::read_dir(self.sessions()).into_iter().flatten();
        let ids = entries.filter_map(|entry| entry.ok()?.file_name().to_str()?.parse().ok());
        ids.map(|id| {
            let dir = self.session(&id);
            (id, dir)
        })
        .filter(|(_, dir)| dir.is_saved())
        .collect()
    }

    /// The saved session `name_or_id` names: the one with that id when it
    /// parses as one and is saved, else the only saved session with that
    /// [`SessionDir::name`].
    ///
    /// # Errors
    ///
    /// When no saved session has that id or name, or several have that
    /// name: the error lists their ids.
    pub fn find(&self, name_or_id: &str) -> Result<SessionId, String> {
        let wanted = name_or_id.trim();
        if let Ok(id) = wanted.parse::<SessionId>()
            && self.session(&id).is_saved()
        {
            return Ok(id);
        }
        let mut named: Vec<SessionId> = self
            .saved_sessions()
            .into_iter()
            .filter(|(_, dir)| dir.name().as_deref() == Some(wanted))
            .map(|(id, _)| id)
            .collect();
        named.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        match named.as_slice() {
            [id] => Ok(id.clone()),
            [] => Err(format!("`{wanted}` names no saved session")),
            several => Err(format!(
                "`{wanted}` names {} saved sessions: {}; resume one by its id",
                several.len(),
                several
                    .iter()
                    .map(SessionId::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        }
    }
}

/// A launcher's own directory, outside every session, which lives while
/// the launcher runs: the file that tells it the agent started, and the
/// log of a sessionless run that failed, which outlives the launcher.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunDir(PathBuf);

impl RunDir {
    /// The directory the launcher named in [`env::RUN`], or `None` when
    /// it is unset or empty.
    pub fn from_env() -> Option<Self> {
        let path = std::env::var_os(env::RUN).filter(|path| !path.is_empty())?;
        Some(Self(PathBuf::from(path)))
    }

    /// The run directory at `path`, such as one a supervisor other than
    /// the launcher makes for an agent it starts and names in [`env::RUN`].
    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self(path.into())
    }

    /// The directory itself.
    pub fn path(&self) -> &Path {
        &self.0
    }

    /// Written by the agent once its build started, or exited on purpose
    /// before that; the launcher then keeps the build.
    pub fn ready(&self) -> PathBuf {
        self.0.join("ready")
    }

    /// The text log of a sessionless run that failed, moved here from its
    /// session's directory, which is removed.
    pub fn log(&self) -> PathBuf {
        self.0.join("agent.log")
    }

    /// Why the agent refused its run, as it said it on stderr: the
    /// launcher says it again once it took the terminal back from a
    /// full-screen view that a restart kept.
    pub fn refusal(&self) -> PathBuf {
        self.0.join("refusal")
    }

    /// Written by the agent once its front takes the terminal's screen,
    /// as the terminal view does: only then does the launcher put the
    /// terminal back after the agent. Doing so after a front that writes
    /// lines, such as `--json` typed at a terminal, would overwrite them.
    pub fn screen(&self) -> PathBuf {
        self.0.join("screen")
    }

    /// Held by the launcher while it lives.
    pub fn lock(&self) -> PathBuf {
        self.0.join("lock")
    }

    /// The process groups the agent owns, one id per line: each from its
    /// start until it is empty or the agent killed it. The launcher kills
    /// those left once the agent exits for anything but a reload, such as
    /// after `SIGKILL`; after a reload the next agent takes them over
    /// ([`Self::listed_groups`]).
    pub fn groups(&self) -> PathBuf {
        self.0.join("groups")
    }

    /// The group ids listed in [`Self::groups`], none when there is no
    /// list. An id whose group emptied may name another group since: an
    /// agent that takes them over after a reload owns each only while it
    /// has members ([`Self::live_groups`] checks more).
    ///
    /// # Errors
    ///
    /// When the list cannot be read.
    pub fn listed_groups(&self) -> std::io::Result<Vec<i32>> {
        let listed = match std::fs::read_to_string(self.groups()) {
            Ok(listed) => listed,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error),
        };
        // A group id of 0 or 1 would name the caller's own group or every
        // process.
        Ok(listed
            .lines()
            .filter_map(|id| id.trim().parse::<i32>().ok())
            .filter(|&id| id > 1)
            .collect())
    }

    /// The groups listed in [`Self::groups`] that still have a process
    /// carrying [`env::PROCESS_LEDGER`] naming the list, its leader or one
    /// the leader left behind as it exited: so an id the system gave
    /// another process since is not among them.
    ///
    /// # Errors
    ///
    /// When the list cannot be read.
    pub fn live_groups(&self) -> std::io::Result<Vec<i32>> {
        let ledger = self.groups();
        Ok(self
            .listed_groups()?
            .into_iter()
            .filter(|&id| members(id).into_iter().any(|pid| marked(pid, &ledger)))
            .collect())
    }

    /// Ends the process groups the agent listed in [`Self::groups`] and
    /// left running ([`Self::live_groups`]), as after `SIGKILL` or when it
    /// exits with a job still running in the background, and removes the
    /// list: asks them to stop (`SIGTERM`), so an agent among them, such as
    /// a session of `rig acp`, ends its own; waits at most
    /// [`STOP_GRACE`](Self::STOP_GRACE) for them to empty; then kills what
    /// is left (`SIGKILL`). `kill(1)` sends the signals, so this keeps to
    /// std.
    ///
    /// # Errors
    ///
    /// When the list cannot be read or removed.
    pub fn kill_left_groups(&self) -> std::io::Result<()> {
        let signal = |name: &str, groups: &[i32]| {
            if !groups.is_empty() {
                child("kill")
                    .args(["-s", name, "--"])
                    .args(groups.iter().map(|id| format!("-{id}")))
                    .stderr(std::process::Stdio::null())
                    .status()
                    .ok();
            }
        };
        let groups = self.live_groups()?;
        signal("TERM", &groups);
        let deadline = std::time::Instant::now() + Self::STOP_GRACE;
        let mut left = groups;
        while !left.is_empty() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(20));
            left = self.live_groups()?;
        }
        signal("KILL", &left);
        match std::fs::remove_file(self.groups()) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(error),
            _ => Ok(()),
        }
    }

    /// How long [`Self::kill_left_groups`] waits for the groups it asked to
    /// stop: longer than an agent among them takes to end its own (it gives
    /// them 2 s).
    pub const STOP_GRACE: std::time::Duration = std::time::Duration::from_secs(5);
}

/// The processes of the group `group`, its leader first: Linux reads each
/// process's group in `/proc`, macOS through `ps`; elsewhere only the
/// leader.
fn members(group: i32) -> Vec<i32> {
    let mut members = vec![group];
    let of_group = |pid: i32, pgid: i32| (pgid == group && pid != group).then_some(pid);
    if cfg!(target_os = "linux") {
        let entries = std::fs::read_dir("/proc").into_iter().flatten().flatten();
        members.extend(entries.filter_map(|entry| {
            let pid = entry.file_name().to_str()?.parse().ok()?;
            let stat = std::fs::read_to_string(entry.path().join("stat")).ok()?;
            // After the command's closing parenthesis: state, parent, group.
            let (_, fields) = stat.rsplit_once(')')?;
            of_group(pid, fields.split_whitespace().nth(2)?.parse().ok()?)
        }));
    } else if cfg!(target_os = "macos") {
        let listed = child("ps").args(["-A", "-o", "pid=,pgid="]).output();
        let listed = listed.map(|listed| String::from_utf8_lossy(&listed.stdout).into_owned());
        members.extend(listed.unwrap_or_default().lines().filter_map(|line| {
            let mut ids = line.split_whitespace().map(str::parse::<i32>);
            of_group(ids.next()?.ok()?, ids.next()?.ok()?)
        }));
    }
    members
}

/// Whether the process `id` carries [`env::PROCESS_LEDGER`] set to
/// `ledger`: it is a process the agent started, or one of its children.
/// Linux reads its environment in `/proc`, macOS through `ps -E`;
/// elsewhere no process is.
fn marked(id: i32, ledger: &Path) -> bool {
    let mark = format!("{}={}", env::PROCESS_LEDGER, ledger.display());
    if cfg!(target_os = "linux") {
        let environ = std::fs::read(format!("/proc/{id}/environ"));
        environ.is_ok_and(|environ| {
            environ
                .split(|&byte| byte == 0)
                .any(|entry| entry == mark.as_bytes())
        })
    } else if cfg!(target_os = "macos") {
        let shown = child("ps")
            .args(["-E", "-ww", "-o", "command=", "-p", &id.to_string()])
            .stderr(std::process::Stdio::null())
            .output();
        // `ps -E` appends each variable after a space.
        let entry = format!(" {mark}");
        shown.is_ok_and(|shown| {
            let shown = String::from_utf8_lossy(&shown.stdout);
            shown
                .match_indices(&entry)
                .filter_map(|(at, _)| shown.get(at + entry.len()..))
                .any(|rest| rest.is_empty() || rest.starts_with([' ', '\n']))
        })
    } else {
        false
    }
}

/// `program`, without the plugins' secrets in its environment.
fn child(program: &str) -> std::process::Command {
    let mut command = std::process::Command::new(program);
    for secret in secret_variables() {
        command.env_remove(secret);
    }
    command
}

/// A session's directory. The agent keeps the session there through its
/// store, which a plugin may replace (by default in the folders `logs`,
/// `blobs` and `files`), so only the plain files beside the store are
/// named here: the markers of a session an agent started in
/// ([`Self::started`]), saved ([`Self::is_saved`]) and being deleted
/// ([`Self::is_deleting`]), the text log, the lock of the launcher running
/// it, its working directory, its name and the session to switch to.
#[derive(Clone, Debug)]
pub struct SessionDir(PathBuf);

impl SessionDir {
    /// The session kept in the directory `path`, such as one a command
    /// names, rather than one of a [`Home`]'s.
    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self(path.into())
    }

    /// The directory itself.
    pub fn path(&self) -> &Path {
        &self.0
    }

    /// The session's id: the directory's name, when it is one.
    pub fn id(&self) -> Option<SessionId> {
        self.0.file_name()?.to_str()?.parse().ok()
    }

    /// Whether the session holds a conversation, so it can be resumed:
    /// the session has the marker [`Self::mark_saved`] writes (and
    /// [`Self::unmark_saved`] takes away) and is not being deleted.
    pub fn is_saved(&self) -> bool {
        self.saved().exists() && !self.is_deleting()
    }

    /// Whether the session is being deleted: it has the marker
    /// [`Self::mark_deleting`] writes. Listings leave it out, and the
    /// agent's next start finishes deleting it.
    pub fn is_deleting(&self) -> bool {
        self.deleting().exists()
    }

    /// Marks the session as being deleted, before anything of it is
    /// removed, so a delete a crash cuts short is finished later and never
    /// leaves a half-removed session to resume.
    ///
    /// # Errors
    ///
    /// When the marker cannot be written.
    pub fn mark_deleting(&self) -> std::io::Result<()> {
        std::fs::write(self.deleting(), b"")
    }

    /// The plain, empty marker of a session being deleted.
    fn deleting(&self) -> PathBuf {
        self.0.join("deleting")
    }

    /// Marks the session saved, as the harness does once the session's log
    /// wrote and an agent's conversation has a user or assistant message.
    ///
    /// # Errors
    ///
    /// When the marker cannot be written.
    pub fn mark_saved(&self) -> std::io::Result<()> {
        std::fs::write(self.saved(), b"")
    }

    /// Takes away the marker [`Self::mark_saved`] writes, as the harness
    /// does once no agent's conversation has a user or assistant message;
    /// a session without it stays as it is.
    ///
    /// # Errors
    ///
    /// When the marker is there and cannot be removed.
    pub fn unmark_saved(&self) -> std::io::Result<()> {
        match std::fs::remove_file(self.saved()) {
            Err(failure) if failure.kind() == std::io::ErrorKind::NotFound => Ok(()),
            removed => removed,
        }
    }

    /// The plain, empty marker of a saved session.
    fn saved(&self) -> PathBuf {
        self.0.join("saved")
    }

    /// The plain, empty marker of a session an agent started in: its
    /// first app ran a whole frame.
    pub fn started(&self) -> PathBuf {
        self.0.join("started")
    }

    /// The agent's text log, unless a plugin keeps it elsewhere or not at
    /// all.
    pub fn log(&self) -> PathBuf {
        self.0.join("agent.log")
    }

    /// Held by the launcher running the session while it lives.
    pub fn launcher_lock(&self) -> PathBuf {
        self.0.join("launcher.lock")
    }

    /// Takes [`Self::launcher_lock`], creating the directory, unless
    /// another launcher holds it: the lock that marks the session as run,
    /// held until the returned file is dropped or the process dies. A
    /// launcher holds it while it runs the session, and a program that
    /// changes a saved session holds it while it does, so no launcher
    /// starts the session meanwhile. `None` when another holds it.
    ///
    /// # Errors
    ///
    /// When the directory or the lock file cannot be made, or locking
    /// fails otherwise.
    pub fn hold(&self) -> std::io::Result<Option<std::fs::File>> {
        std::fs::create_dir_all(&self.0)?;
        let file = lock_file(&self.launcher_lock(), true)?;
        match file.try_lock() {
            Ok(()) => Ok(Some(file)),
            Err(std::fs::TryLockError::WouldBlock) => Ok(None),
            Err(std::fs::TryLockError::Error(failure)) => Err(failure),
        }
    }

    /// Whether a launcher runs the session: one holds
    /// [`Self::launcher_lock`]. A command that rewrites or removes a saved
    /// session leaves such a one alone.
    pub fn runs_elsewhere(&self) -> bool {
        !is_free(&self.launcher_lock())
    }

    /// The working directory the session runs in, as plain text: written
    /// when the session starts, and where the launcher starts the agent
    /// when it resumes the session from elsewhere.
    pub fn directory(&self) -> PathBuf {
        self.0.join("directory")
    }

    /// The name the user gave the session, kept as plain text in the
    /// file `name` beside [`Self::directory`], so the launcher finds the
    /// session by it ([`Home::find`]) without opening its store. Like the
    /// directory, it is readable on disk when the store seals what it
    /// keeps.
    pub fn name(&self) -> Option<String> {
        let text = std::fs::read_to_string(self.name_file()).ok()?;
        let name = text.trim();
        (!name.is_empty()).then(|| name.to_owned())
    }

    /// Names the session, or removes its name with `None` or a blank name.
    /// The name is written beside the file and renamed over it, so a
    /// reader sees the old name or the new one, never a part.
    ///
    /// # Errors
    ///
    /// When the file cannot be written or removed.
    pub fn set_name(&self, name: Option<&str>) -> std::io::Result<()> {
        let file = self.name_file();
        match name.map(str::trim).filter(|name| !name.is_empty()) {
            Some(name) => {
                let part = self.0.join("name.part");
                std::fs::write(&part, name)?;
                std::fs::rename(part, file)
            }
            None => match std::fs::remove_file(file) {
                Err(failure) if failure.kind() == std::io::ErrorKind::NotFound => Ok(()),
                removed => removed,
            },
        }
    }

    /// Writes `directory` as the one the session runs in
    /// ([`Self::directory`]), beside the file and renamed over it, so a
    /// reader sees the old one or the new one, never a part.
    ///
    /// # Errors
    ///
    /// When the file cannot be written.
    pub fn set_working_directory(&self, directory: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.0)?;
        let part = self.0.join("directory.part");
        std::fs::write(&part, directory.to_string_lossy().as_bytes())?;
        std::fs::rename(part, self.directory())
    }

    /// The plain-text file of [`Self::name`].
    fn name_file(&self) -> PathBuf {
        self.0.join("name")
    }

    /// Written by the agent before it exits with [`RELOAD_EXIT_CODE`] to
    /// run another session: the [`SessionId`] to resume, or nothing for a
    /// new session. The launcher reads and removes it.
    pub fn switch(&self) -> PathBuf {
        self.0.join("switch")
    }

    /// The session's working directory, when [`Self::directory`] names one
    /// that still exists.
    pub fn working_directory(&self) -> Option<PathBuf> {
        let text = std::fs::read_to_string(self.directory()).ok()?;
        let directory = PathBuf::from(text.trim_end_matches('\n'));
        directory.is_dir().then_some(directory)
    }
}

/// The agent's arguments, in the canonical form the launcher passes on
/// ([`Invocation::to_args`]): the model, each plugin option as
/// `--name=VALUE`, then the print prompt after `--print --`, or a plugin's
/// command and its arguments. The launcher reads the user's command line
/// against the agent's [`AgentDescription`] report first
/// ([`AgentDescription::invocation`]), so `rig -p "…"` and the agent binary run
/// alone agree on them.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Invocation {
    /// `Some` for print mode: answer this one prompt, then exit, with the
    /// answer on stdout and text piped in on stdin after the prompt (which
    /// may be empty when stdin is piped). `None` runs the terminal view, or
    /// no view at all without one.
    pub print: Option<String>,
    /// A catalog model (`vendor/model`) for the session's first agent.
    pub model: Option<String>,
    /// The options of the agent's plugins, in the order given: each
    /// `--name` with its value, if any. The agent refuses one no plugin of
    /// its build takes.
    pub flags: Vec<(String, Option<String>)>,
    /// A plugin's command, `rig <command> …`, which chooses the front that
    /// runs.
    pub command: Option<String>,
    /// The words after the command, as given.
    pub arguments: Vec<String>,
}

/// The arguments [`Invocation::parse`] takes, for usage texts.
pub const INVOCATION_USAGE: &str =
    "  -p, --print [prompt…]  Answer one prompt and exit: the answer goes to stdout.
                         Text piped in on stdin follows the prompt. With
                         --json, the run as JSON lines; --json alone reads
                         requests on stdin (`rig help-agents`).
  -m, --model <model>    The catalog model (vendor/model) to use.
  --<option> [<value>]   An option of one of the agent's plugins, also written
                         --<option>=<value>.
";

impl Invocation {
    /// Reads the agent's arguments in their canonical form, without the
    /// program name: a plugin option's value comes after `=`. Without
    /// `--print`, the first other word is a command, and every word after
    /// it is the command's.
    pub fn parse<S: AsRef<str>>(args: &[S]) -> Result<Self, String> {
        match read_args(args, |flag, _| plugin_flag(flag))? {
            (invocation, None) => Ok(invocation),
            (_, Some(_)) => Err("`--resume` is the launcher's".to_owned()),
        }
    }

    /// The agent's own arguments, from the process's.
    pub fn from_env() -> Result<Self, String> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        Self::parse(&args)
    }

    /// Whether nobody sits at a terminal view, and the run ends by itself.
    pub fn is_headless(&self) -> bool {
        self.print.is_some()
    }

    /// The arguments that [`Self::parse`] reads back as `self`.
    pub fn to_args(&self) -> Vec<OsString> {
        let mut args: Vec<OsString> = Vec::new();
        if let Some(model) = &self.model {
            args.extend(["--model".into(), model.into()]);
        }
        for (name, value) in &self.flags {
            args.push(match value {
                Some(value) => format!("--{name}={value}").into(),
                None => format!("--{name}").into(),
            });
        }
        if let Some(prompt) = &self.print {
            args.push("--print".into());
            if !prompt.is_empty() {
                args.extend(["--".into(), prompt.into()]);
            }
        } else if let Some(command) = &self.command {
            args.push(command.into());
            args.extend(self.arguments.iter().map(OsString::from));
        }
        args
    }
}

/// Reads a command line left to right, as [`Invocation::parse`] and
/// [`AgentDescription`] do, into the invocation and the session `-r` names; they
/// differ in how a plugin option is read: `option` reads one, `--…`,
/// given the words after it, from which it may take its value.
fn read_args<'a, S: AsRef<str>>(
    args: &'a [S],
    mut option: impl FnMut(
        &'a str,
        &mut dyn Iterator<Item = &'a str>,
    ) -> Result<(String, Option<String>), String>,
) -> Result<(Invocation, Option<String>), String> {
    let mut print = false;
    let mut model = None;
    let mut resume = None;
    let mut flags = Vec::new();
    let mut words: Vec<&str> = Vec::new();
    let mut args = args.iter().map(AsRef::as_ref);
    while let Some(arg) = args.next() {
        match arg {
            "-p" | "--print" => print = true,
            "-m" | "--model" => {
                let name = args.next().ok_or("--model needs a vendor/model")?;
                model = Some(name.to_owned());
            }
            "--" => words.extend(args.by_ref()),
            flag if flag.starts_with("--model=") => {
                model = flag.strip_prefix("--model=").map(str::to_owned);
            }
            "-r" | "--resume" => {
                let session = args.next().ok_or("--resume needs a session id or name")?;
                resume = Some(session.to_owned());
            }
            flag if flag.starts_with("--resume=") => {
                resume = flag.strip_prefix("--resume=").map(str::to_owned);
            }
            flag if flag.starts_with("--") => flags.push(option(flag, &mut args)?),
            flag if flag.starts_with('-') && flag.len() > 1 => {
                return Err(format!("unknown option `{flag}`"));
            }
            // A command takes every word after it as it is.
            word if !print => {
                words.push(word);
                words.extend(args.by_ref());
            }
            word => words.push(word),
        }
    }
    let mut words = words.into_iter().map(str::to_owned);
    let (print, command) = if print {
        (Some(words.by_ref().collect::<Vec<_>>().join(" ")), None)
    } else {
        (None, words.next())
    };
    let invocation = Invocation {
        print,
        model,
        flags,
        command,
        arguments: words.collect(),
    };
    Ok((invocation, resume))
}

/// A plugin's option, `--name` or `--name=value`: the name is lowercase
/// ASCII letters, digits and dashes, starting with a letter.
fn plugin_flag(flag: &str) -> Result<(String, Option<String>), String> {
    let option = flag.strip_prefix("--").unwrap_or(flag);
    let (name, value) = match option.split_once('=') {
        Some((name, value)) => (name, Some(value.to_owned())),
        None => (option, None),
    };
    if !is_name(name) {
        return Err(format!("unknown option `{flag}`"));
    }
    Ok((name.to_owned(), value))
}

/// Opens the lock file at `path` for writing, creating it when `create`.
/// Every lock between launchers and agents is such a file, held with
/// [`std::fs::File::try_lock`] or [`std::fs::File::lock`].
///
/// # Errors
///
/// When the file cannot be opened (or is missing and not `create`).
pub fn lock_file(path: &Path, create: bool) -> std::io::Result<std::fs::File> {
    std::fs::File::options()
        .create(create)
        .write(true)
        .truncate(false)
        .open(path)
}

/// Whether no live process holds the lock file at `path`: it is missing, or
/// it can be locked now. A failure that hides the answer counts as held.
pub fn is_free(path: &Path) -> bool {
    match lock_file(path, false) {
        Ok(lock) => lock.try_lock().is_ok(),
        Err(failure) => failure.kind() == std::io::ErrorKind::NotFound,
    }
}

/// Whether `name` can name a plugin's option or command: lowercase ASCII
/// letters, digits and dashes, starting with a letter.
pub fn is_name(name: &str) -> bool {
    name.starts_with(|c: char| c.is_ascii_lowercase())
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

/// The launcher's own first words, which no plugin's command may take.
pub const RESERVED_COMMANDS: &[&str] = &["build", "plugin", "help", "help-agents"];

/// The plugins whose absence takes away the way back from inside the
/// agent, by type path, with what goes: `rig plugin disable`, `rig plugin
/// list` and `/plugins` warn on them. `rig plugin enable` from a shell
/// always brings one back.
pub const ESSENTIAL_PLUGINS: &[(&str, &str)] = &[
    (
        "rig_tui::TuiPlugin",
        "the terminal view, leaving only -p and piped prompts",
    ),
    (
        "rig_basics::reload::plugins::PluginsPlugin",
        "/plugins itself",
    ),
    (
        "rig_basics::reload::ReloadPlugin",
        "/reload, so a change applies only when rig starts again",
    ),
];

/// What goes when the plugin `type_path` is left out, if it is one of the
/// [`ESSENTIAL_PLUGINS`].
pub fn essential(type_path: &str) -> Option<&'static str> {
    ESSENTIAL_PLUGINS
        .iter()
        .find(|(essential, _)| *essential == type_path)
        .map(|(_, what)| *what)
}

/// The first line of a [`AgentDescription`] report.
const DESCRIBED_HEADER: &str = "rig-describe 1";

/// What a built agent's plugins add to the command line, as the agent
/// writes it when the launcher sets [`env::DESCRIBE`]: their commands,
/// their options and the secrets they ask for, and the plugins whose
/// agent-loop systems clash, for `rig plugin check --build`. The launcher
/// keeps one per build ([`Home::described`]) and reads the command line
/// against it.
///
/// The text form is a header line, then one line each, its fields separated by
/// tabs (shown here as `⇥`):
///
/// ```text
/// rig-describe 1
/// command ⇥ stats ⇥ - ⇥ headless ⇥ sessionless ⇥ Count the saved sessions
/// flag ⇥ remote ⇥ HOST:PATH ⇥ Work in a directory on another machine over SSH
/// secret ⇥ crypt ⇥ Passphrase for this machine's sessions
/// conflict ⇥ `a::APlugin` and `b::BPlugin`: a::count and b::reset, on b::Tally
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AgentDescription {
    /// The plugins' commands.
    pub commands: Vec<DescribedCommand>,
    /// The plugins' options, the fronts' included.
    pub flags: Vec<DescribedFlag>,
    /// The secrets the plugins ask for.
    pub secrets: Vec<DescribedSecret>,
    /// Each pair of systems of one agent-loop stage that use the same data
    /// with no order between them, one writing it, naming their plugins.
    pub conflicts: Vec<String>,
}

/// A plugin's command, `rig <name> <usage>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DescribedCommand {
    /// Its name.
    pub name: String,
    /// What its arguments are, such as `<tool> [path]`; may be empty.
    pub usage: String,
    /// Whether it runs on an interactive terminal; otherwise it runs
    /// headless, never restarted, its output passed through.
    pub terminal: bool,
    /// Whether its runs keep no session.
    pub sessionless: bool,
    /// What it does.
    pub help: String,
}

/// A plugin's option, `--name` or `--name VALUE`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DescribedFlag {
    /// Its name.
    pub name: String,
    /// What its value is, such as `ADDR`, when it takes one.
    pub value: Option<String>,
    /// What it does.
    pub help: String,
}

/// A secret a plugin asks for, which reaches the agent as
/// [`secret_variable`] of its name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DescribedSecret {
    /// Its name.
    pub name: String,
    /// What to ask for.
    pub help: String,
}

impl AgentDescription {
    /// Reads a report. Without the header it is empty: a build from before
    /// commands, which has none. A record of a kind it does not know is
    /// skipped, and fields after those it knows are ignored, so a later
    /// version can add both.
    pub fn parse(text: &str) -> Self {
        let mut lines = text.lines();
        let mut described = Self::default();
        if lines.next() != Some(DESCRIBED_HEADER) {
            return described;
        }
        let given = |field: &str| (field != "-").then(|| field.to_owned());
        for line in lines {
            match line.split('\t').collect::<Vec<_>>().as_slice() {
                ["command", name, usage, run, session, help, ..] => {
                    described.commands.push(DescribedCommand {
                        name: (*name).to_owned(),
                        usage: given(usage).unwrap_or_default(),
                        terminal: *run == "terminal",
                        sessionless: *session == "sessionless",
                        help: (*help).to_owned(),
                    });
                }
                ["flag", name, value, help, ..] => described.flags.push(DescribedFlag {
                    name: (*name).to_owned(),
                    value: given(value),
                    help: (*help).to_owned(),
                }),
                ["secret", name, help, ..] => described.secrets.push(DescribedSecret {
                    name: (*name).to_owned(),
                    help: (*help).to_owned(),
                }),
                ["conflict", conflict, ..] => described.conflicts.push((*conflict).to_owned()),
                // A record of a later version.
                _ => {}
            }
        }
        described
    }

    /// The report as text, which [`Self::parse`] reads back. Tabs and line
    /// breaks in a field become spaces.
    pub fn write(&self) -> String {
        let clean = |field: &str| field.replace(['\t', '\n', '\r'], " ");
        let or_dash = |field: &str| match field {
            "" => "-".to_owned(),
            field => clean(field),
        };
        let mut lines = vec![DESCRIBED_HEADER.to_owned()];
        for command in &self.commands {
            lines.push(format!(
                "command\t{}\t{}\t{}\t{}\t{}",
                clean(&command.name),
                or_dash(&command.usage),
                if command.terminal {
                    "terminal"
                } else {
                    "headless"
                },
                if command.sessionless {
                    "sessionless"
                } else {
                    "session"
                },
                clean(&command.help),
            ));
        }
        for flag in &self.flags {
            lines.push(format!(
                "flag\t{}\t{}\t{}",
                clean(&flag.name),
                or_dash(flag.value.as_deref().unwrap_or_default()),
                clean(&flag.help),
            ));
        }
        for secret in &self.secrets {
            lines.push(format!(
                "secret\t{}\t{}",
                clean(&secret.name),
                clean(&secret.help)
            ));
        }
        for conflict in &self.conflicts {
            lines.push(format!("conflict\t{}", clean(conflict)));
        }
        lines.push(String::new());
        lines.join("\n")
    }

    /// The command `name`, if a plugin adds it.
    pub fn command(&self, name: &str) -> Option<&DescribedCommand> {
        self.commands.iter().find(|command| command.name == name)
    }

    /// Reads the user's command line, without the launcher's own first
    /// word, against this agent, into the canonical [`Invocation`]. `-p`,
    /// `-m` and `-r` ([`Self::resume`]) are the launcher's; a plugin option
    /// takes a value as `--name=VALUE`, or as the next word when it takes
    /// one (`--name VALUE`); without `-p`, the first other word is a
    /// command, and every later word is its argument. An option or command
    /// no plugin adds, and a launcher's word as a command, are refused.
    pub fn invocation<S: AsRef<str>>(&self, args: &[S]) -> Result<Invocation, String> {
        let invocation = read_args(args, |flag, rest| {
            let (name, value) = plugin_flag(flag).map_err(|_| self.unknown_option(flag))?;
            let Some(known) = self.flags.iter().find(|known| known.name == name) else {
                return Err(self.unknown_option(flag));
            };
            match (&known.value, value) {
                (Some(_), Some(value)) => Ok((name, Some(value))),
                (Some(what), None) => match rest.next() {
                    Some(value) => Ok((name, Some(value.to_owned()))),
                    None => Err(format!("--{name} needs a value: --{name} {what}")),
                },
                (None, Some(_)) => Err(format!("--{name} takes no value")),
                (None, None) => Ok((name, None)),
            }
        })?
        .0;
        if let Some(command) = &invocation.command {
            if RESERVED_COMMANDS.contains(&command.as_str()) {
                return Err(format!("`{command}` is the launcher's; put it first"));
            }
            if self.command(command).is_none() {
                return Err(format!(
                    "unknown command `{command}`; commands: {}",
                    self.command_names().join(", ")
                ));
            }
        }
        Ok(invocation)
    }

    /// The session `-r <id|name>` (or `--resume`) names on the command
    /// line `args`, read as [`Self::invocation`] reads it: wherever an
    /// option goes, so not after `--` or among a command's words. This
    /// build's options that take a value take the next word as theirs; an
    /// option it does not know takes none. `None` without `-r`, or when
    /// the command line cannot be read.
    pub fn resume<S: AsRef<str>>(&self, args: &[S]) -> Option<String> {
        let read = read_args(args, |flag, rest| {
            let (name, value) = plugin_flag(flag)?;
            let spaced = value.is_none()
                && self
                    .flags
                    .iter()
                    .any(|known| known.name == name && known.value.is_some());
            Ok((
                name,
                if spaced {
                    rest.next().map(str::to_owned)
                } else {
                    value
                },
            ))
        });
        read.ok().and_then(|(_, resume)| resume)
    }

    /// What `rig help` lists of the build: the plugins' commands, their
    /// options, and the secrets they ask for, each with the variable that
    /// gives it (never its value).
    pub fn usage(&self) -> String {
        let mut text = String::from("\nPlugin commands (of the last build):\n");
        if self.commands.is_empty() {
            text.push_str("  (none)\n");
        }
        for command in &self.commands {
            let usage = format!("rig {} {}", command.name, command.usage);
            text.push_str(&help_entry(usage.trim_end(), &command.help));
        }
        text.push_str("\nPlugin options (of the last build):\n");
        if self.flags.is_empty() {
            text.push_str("  (none)\n");
        }
        for flag in &self.flags {
            let option = match &flag.value {
                Some(value) => format!("--{} {value}", flag.name),
                None => format!("--{}", flag.name),
            };
            text.push_str(&help_entry(&option, &flag.help));
        }
        if !self.secrets.is_empty() {
            text.push_str("\nPlugin secrets (of the last build), asked for when not set:\n");
        }
        for secret in &self.secrets {
            let variable = secret_variable(&secret.name);
            text.push_str(&help_entry(&variable, &secret.help));
        }
        text
    }

    /// The launcher's commands, then the plugins'.
    pub fn command_names(&self) -> Vec<&str> {
        RESERVED_COMMANDS
            .iter()
            .copied()
            .chain(self.commands.iter().map(|command| command.name.as_str()))
            .collect()
    }

    /// Why `flag` is refused, with the options there are.
    fn unknown_option(&self, flag: &str) -> String {
        let options: Vec<String> = ["-p", "-m"]
            .into_iter()
            .map(str::to_owned)
            .chain(self.flags.iter().map(|flag| match &flag.value {
                Some(value) => format!("--{} {value}", flag.name),
                None => format!("--{}", flag.name),
            }))
            .collect();
        format!("unknown option {flag}; options: {}", options.join(", "))
    }
}

/// A line of `rig help`: `name`, then `help` from the 26th column, as the
/// launcher's own lines are, or on the next line there when `name` leaves
/// no gap before it.
fn help_entry(name: &str, help: &str) -> String {
    const COLUMN: usize = 25;
    let name = format!("  {name}");
    if name.chars().count() + 2 <= COLUMN {
        format!("{name:<COLUMN$}{help}\n")
    } else {
        format!("{name}\n{:COLUMN$}{help}\n", "")
    }
}

#[cfg(test)]
mod tests;
