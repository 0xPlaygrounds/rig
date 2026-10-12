//! The protocol between the `rig` launcher, this crate's binary, and the
//! rig-harness agent it builds and runs: the `RIG_HOME` layout, session ids,
//! the environment the launcher passes to the agent, and the exit code with
//! which the agent asks for a restart. The launcher and `rig-harness` both use
//! this module, so the two sides cannot drift apart. It uses only std, and
//! it changes with rig-harness.

use std::ffi::OsString;
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

/// The exit code with which the agent asks its launcher to restart it on
/// the staged build.
pub const RELOAD_EXIT_CODE: u8 = 75;

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
    /// What the launcher tells the agent it runs, and only that agent: a
    /// command the agent runs, such as a nested agent while working on
    /// rig-harness itself, must not act as this agent. Without [`SESSION`]
    /// a nested agent ignores the [`LAUNCHER`] it inherits.
    pub const AGENT_ONLY: [&str; 4] = [SESSION, NOTICE, BUILD_FAILURE, RUN];
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

/// The extension of an agent log, and of the effect log.
const AGENT_LOG: &str = ".jsonl";

/// The effect log's file stem, which no agent id takes.
const EFFECTS: &str = "effects";

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
    pub fn config(&self) -> PathBuf {
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

    /// `auth/<provider>.json`, the subscription credential `/login` keeps
    /// for `provider`, such as `chatgpt`.
    pub fn auth(&self, provider: &str) -> PathBuf {
        self.root.join("auth").join(format!("{provider}.json"))
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

    /// The options the plugins of the last build that started take, one per
    /// line as `--name[=VALUE]  what it does`, for the launcher's usage
    /// text.
    pub fn flags(&self) -> PathBuf {
        self.bin().join("flags")
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

    /// The prompts typed into the terminal view, one JSON string per line,
    /// shared by every session on this root.
    pub fn history(&self) -> PathBuf {
        self.root.join("history.jsonl")
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

    /// Held by the launcher while it lives.
    pub fn lock(&self) -> PathBuf {
        self.0.join("lock")
    }
}

/// A session's directory: one append-only log per agent, the listing cache,
/// the stored images, the effect log and text log, and the lock of the
/// launcher running it. Each top-level `*.jsonl` file but the effect log is
/// an agent's log; a plugin's files go in a directory of its own.
#[derive(Clone, Debug)]
pub struct SessionDir(PathBuf);

impl SessionDir {
    /// The directory itself.
    pub fn path(&self) -> &Path {
        &self.0
    }

    /// The session's id: the directory's name, when it is one.
    pub fn id(&self) -> Option<SessionId> {
        self.0.file_name()?.to_str()?.parse().ok()
    }

    /// Whether any agent of the session wrote its log, so it can be
    /// resumed.
    pub fn is_saved(&self) -> bool {
        std::fs::read_dir(&self.0).is_ok_and(|entries| {
            entries.filter_map(Result::ok).any(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .and_then(|name| name.strip_suffix(AGENT_LOG))
                    .is_some_and(|stem| !stem.is_empty() && stem != EFFECTS)
            })
        })
    }

    /// The agent's text log.
    pub fn log(&self) -> PathBuf {
        self.0.join("agent.log")
    }

    /// Held by the launcher running the session while it lives.
    pub fn launcher_lock(&self) -> PathBuf {
        self.0.join("launcher.lock")
    }

    /// The working directory the session runs in, as plain text: written
    /// when the session starts, and where the launcher starts the agent
    /// when it resumes the session from elsewhere.
    pub fn directory(&self) -> PathBuf {
        self.0.join("directory")
    }

    /// What `/resume` lists about the session, as JSON the agent rewrites
    /// at the end of each turn: its name, title, cost and when it was last
    /// updated. The agent logs hold the same facts.
    pub fn meta(&self) -> PathBuf {
        self.0.join("meta.json")
    }

    /// The session this one was forked from and where the copy stopped,
    /// as JSON the agent writes when it forks a session; absent for a
    /// session started afresh.
    pub fn parent(&self) -> PathBuf {
        self.0.join("parent.json")
    }

    /// Written by the agent before it exits with [`RELOAD_EXIT_CODE`] to
    /// run another session: the [`SessionId`] to resume, or nothing for a
    /// new session. The launcher reads and removes it.
    pub fn switch(&self) -> PathBuf {
        self.0.join("switch")
    }

    /// The images pasted into the session's input, which messages attach
    /// by path.
    pub fn images(&self) -> PathBuf {
        self.0.join("images")
    }

    /// The session's working directory, when [`Self::directory`] names one
    /// that still exists.
    pub fn working_directory(&self) -> Option<PathBuf> {
        let text = std::fs::read_to_string(self.directory()).ok()?;
        let directory = PathBuf::from(text.trim_end_matches('\n'));
        directory.is_dir().then_some(directory)
    }
}

/// The agent's arguments. The launcher takes the same arguments after its
/// own session options and passes them on unchanged
/// ([`Invocation::to_args`]), so `rig -p "…"` and the agent binary run
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
    /// `--name` with the text after `=`, if any. The agent refuses one no
    /// plugin of its build takes.
    pub flags: Vec<(String, Option<String>)>,
}

/// The arguments [`Invocation::parse`] takes, for usage texts.
pub const INVOCATION_USAGE: &str =
    "  -p, --print [prompt…]  Answer one prompt and exit: the answer goes to stdout.
                         Text piped in on stdin follows the prompt.
  -m, --model <model>    The catalog model (vendor/model) to use.
  --<option>[=<value>]   An option of one of the agent's plugins.
";

impl Invocation {
    /// Reads the agent's arguments, without the program name.
    pub fn parse<S: AsRef<str>>(args: &[S]) -> Result<Self, String> {
        let mut print = false;
        let mut model = None;
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
                flag if flag.starts_with("--") => flags.push(plugin_flag(flag)?),
                flag if flag.starts_with('-') && flag.len() > 1 => {
                    return Err(format!("unknown option `{flag}`"));
                }
                word => words.push(word),
            }
        }
        if !print && !words.is_empty() {
            return Err("a prompt needs --print (-p)".to_owned());
        }
        let print = print.then(|| words.join(" "));
        Ok(Self {
            print,
            model,
            flags,
        })
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
        }
        args
    }
}

/// A plugin's option, `--name` or `--name=value`: the name is lowercase
/// ASCII letters, digits and dashes, starting with a letter.
fn plugin_flag(flag: &str) -> Result<(String, Option<String>), String> {
    let option = flag.strip_prefix("--").unwrap_or(flag);
    let (name, value) = match option.split_once('=') {
        Some((name, value)) => (name, Some(value.to_owned())),
        None => (option, None),
    };
    let valid = name.starts_with(|c: char| c.is_ascii_lowercase())
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
    if !valid {
        return Err(format!("unknown option `{flag}`"));
    }
    Ok((name.to_owned(), value))
}

#[cfg(test)]
mod tests;
