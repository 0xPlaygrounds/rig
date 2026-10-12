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
    /// Set by the launcher to a file's path: the agent builds its app,
    /// writes what its plugins add there ([`Described`](super::Described))
    /// and exits without running a frame.
    pub const DESCRIBE: &str = "RIG_DESCRIBE";
    /// The start of a secret's variable, such as `RIG_SECRET_CRYPT`
    /// ([`secret_variable`](super::secret_variable)): the launcher sets it
    /// on the agent's process alone, and every process rig starts drops it
    /// ([`secret_variables`](super::secret_variables)).
    pub const SECRET_PREFIX: &str = "RIG_SECRET_";
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

    /// The [`Described`] report of the build whose stamp is `stamp` (its
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

    /// Why the agent refused its run, as it said it on stderr: the
    /// launcher says it again once it took the terminal back from a
    /// full-screen view that a restart kept.
    pub fn refusal(&self) -> PathBuf {
        self.0.join("refusal")
    }

    /// Held by the launcher while it lives.
    pub fn lock(&self) -> PathBuf {
        self.0.join("lock")
    }
}

/// A session's directory. The agent keeps the session's files there
/// through its store, which a plugin may replace, so only the plain files
/// kept beside the store are named here ([`Self::PLAIN_FILES`]): the
/// markers of a session an agent started in ([`Self::started`]) and
/// logged to ([`Self::is_saved`]), the text log, the lock of the launcher
/// running it, its working directory and the session to switch to.
#[derive(Clone, Debug)]
pub struct SessionDir(PathBuf);

impl SessionDir {
    /// The top-level names in a session's directory that are plain files
    /// beside its store, which hold no conversation: the files named
    /// below, and the directories of output cut short (`spill`) and pasted
    /// images (`images`) the agent keeps there by default. A store over
    /// the directory leaves them out, and a copy of a session into another
    /// store copies them as they are, if at all.
    pub const PLAIN_FILES: [&str; 8] = [
        "launcher.lock",
        "directory",
        "switch",
        "started",
        "saved",
        "agent.log",
        "spill",
        "images",
    ];

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

    /// Whether an agent of the session wrote its log, so it can be
    /// resumed: the session has the marker [`Self::mark_saved`] writes.
    pub fn is_saved(&self) -> bool {
        self.saved().exists()
    }

    /// Marks the session saved, as the agent's store does when it first
    /// writes an agent's log there.
    ///
    /// # Errors
    ///
    /// When the marker cannot be written.
    pub fn mark_saved(&self) -> std::io::Result<()> {
        std::fs::write(self.saved(), b"")
    }

    /// The plain, empty marker of a saved session.
    fn saved(&self) -> PathBuf {
        self.0.join("saved")
    }

    /// The plain, empty marker of a session an agent started in: its
    /// first app ran a whole frame. A fork does not copy it.
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

    /// Whether a launcher runs the session: one holds
    /// [`Self::launcher_lock`]. A command that rewrites or removes a saved
    /// session leaves such a one alone.
    pub fn runs_elsewhere(&self) -> bool {
        std::fs::File::options()
            .write(true)
            .open(self.launcher_lock())
            .is_ok_and(|lock| matches!(lock.try_lock(), Err(std::fs::TryLockError::WouldBlock)))
    }

    /// The working directory the session runs in, as plain text: written
    /// when the session starts, and where the launcher starts the agent
    /// when it resumes the session from elsewhere.
    pub fn directory(&self) -> PathBuf {
        self.0.join("directory")
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
/// against the agent's [`Described`] report first
/// ([`Described::invocation`]), so `rig -p "…"` and the agent binary run
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
                         Text piped in on stdin follows the prompt.
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
        read_args(args, |flag, _| plugin_flag(flag))
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
/// [`Described::invocation`] both do; they differ in how a plugin option
/// is read: `option` reads one, `--…`, given the words after it, from
/// which it may take its value.
fn read_args<'a, S: AsRef<str>>(
    args: &'a [S],
    mut option: impl FnMut(
        &'a str,
        &mut dyn Iterator<Item = &'a str>,
    ) -> Result<(String, Option<String>), String>,
) -> Result<Invocation, String> {
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
    Ok(Invocation {
        print,
        model,
        flags,
        command,
        arguments: words.collect(),
    })
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

/// Whether `name` can name a plugin's option or command: lowercase ASCII
/// letters, digits and dashes, starting with a letter.
pub fn is_name(name: &str) -> bool {
    name.starts_with(|c: char| c.is_ascii_lowercase())
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

/// The launcher's own first words, which no plugin's command may take.
pub const RESERVED_COMMANDS: [&str; 3] = ["build", "plugin", "help"];

/// The first line of a [`Described`] report.
const DESCRIBED_HEADER: &str = "rig-describe 1";

/// What a built agent's plugins add to the command line, as the agent
/// writes it when the launcher sets [`env::DESCRIBE`]: their commands,
/// their options and the secrets they ask for. The launcher keeps one per
/// build ([`Home::described`]) and reads the command line against it.
///
/// The text form is a header line, then one line each, its fields separated by tabs (shown here as `⇥`):
///
/// ```text
/// rig-describe 1
/// command ⇥ stats ⇥ - ⇥ headless ⇥ sessionless ⇥ Count the saved sessions
/// flag ⇥ remote ⇥ HOST:PATH ⇥ Work in a directory on another machine over SSH
/// secret ⇥ crypt ⇥ Passphrase for this machine's sessions
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Described {
    /// The plugins' commands.
    pub commands: Vec<CommandInfo>,
    /// The plugins' options, the fronts' included.
    pub flags: Vec<FlagInfo>,
    /// The secrets the plugins ask for.
    pub secrets: Vec<SecretInfo>,
}

/// A plugin's command, `rig <name> <usage>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandInfo {
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
pub struct FlagInfo {
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
pub struct SecretInfo {
    /// Its name.
    pub name: String,
    /// What to ask for.
    pub help: String,
}

impl Described {
    /// Reads a report. Without the header it is empty: a build from before
    /// commands, which has none.
    pub fn parse(text: &str) -> Self {
        let mut lines = text.lines();
        let mut described = Self::default();
        if lines.next() != Some(DESCRIBED_HEADER) {
            return described;
        }
        let given = |field: &str| (field != "-").then(|| field.to_owned());
        for line in lines {
            match line.split('\t').collect::<Vec<_>>().as_slice() {
                ["command", name, usage, run, session, help] => {
                    described.commands.push(CommandInfo {
                        name: (*name).to_owned(),
                        usage: given(usage).unwrap_or_default(),
                        terminal: *run == "terminal",
                        sessionless: *session == "sessionless",
                        help: (*help).to_owned(),
                    });
                }
                ["flag", name, value, help] => described.flags.push(FlagInfo {
                    name: (*name).to_owned(),
                    value: given(value),
                    help: (*help).to_owned(),
                }),
                ["secret", name, help] => described.secrets.push(SecretInfo {
                    name: (*name).to_owned(),
                    help: (*help).to_owned(),
                }),
                // A line of a later version.
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
        lines.push(String::new());
        lines.join("\n")
    }

    /// The command `name`, if a plugin adds it.
    pub fn command(&self, name: &str) -> Option<&CommandInfo> {
        self.commands.iter().find(|command| command.name == name)
    }

    /// Reads the user's command line, without the launcher's own first
    /// word and `--resume`, against this agent, into the canonical
    /// [`Invocation`]. `-p` and `-m` are the launcher's; a plugin option
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
        })?;
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

    /// What `rig help` lists of the build: the plugins' commands, their
    /// options, and the secrets they ask for, each with the variable that
    /// gives it (never its value).
    pub fn usage(&self) -> String {
        let mut text = String::from("\nPlugin commands (of the last build):\n");
        if self.commands.is_empty() {
            text.push_str("  (none)\n");
        }
        for command in &self.commands {
            let usage = format!("{} {}", command.name, command.usage);
            text.push_str(&format!("  rig {:<19}{}\n", usage.trim_end(), command.help));
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
            text.push_str(&format!("  {option:<23}{}\n", flag.help));
        }
        if !self.secrets.is_empty() {
            text.push_str("\nPlugin secrets (of the last build), asked for when not set:\n");
        }
        for secret in &self.secrets {
            let variable = secret_variable(&secret.name);
            text.push_str(&format!("  {variable:<23}{}\n", secret.help));
        }
        text
    }

    /// The launcher's commands, then the plugins'.
    pub fn command_names(&self) -> Vec<&str> {
        RESERVED_COMMANDS
            .into_iter()
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

#[cfg(test)]
mod tests;
