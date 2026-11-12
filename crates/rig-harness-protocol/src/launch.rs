//! What a launcher is told when it starts, and the one place that builds
//! every process it starts: its agent ([`Launch::agent`]), the describing
//! build ([`Launch::describing`]), cargo, `stty` and the installed
//! launcher's `rig build` ([`Launch::build_child`]), and the launcher that
//! replaces it ([`Launch::reexec`]). Each command first drops every
//! variable of the protocol and every secret, then sets only what that
//! process gets, so no state reaches a process by being inherited.
//!
//! [`Launch::from_process`] is the launcher's only read of its directory,
//! terminal, file and environment. A re-exec's command reads back as the
//! same value ([`Launch::from_parts`]), so the new launcher carries on with
//! the old one's state ([`Carry`]) and typed secrets.
//!
//! ```no_run
//! use rig_harness_protocol::launch::{Launch, Purpose};
//!
//! let launch = Launch::from_process()?;
//! let status = launch.build_child(Purpose::Stty).arg("sane").status()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::io::IsTerminal;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::{Home, RunDir, SessionId, Stamp, StartReason, env, warnings};

/// A path made absolute by [`std::path::absolute`], so a process started
/// in another directory reads the same file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbsPath(PathBuf);

impl AbsPath {
    /// `path` made absolute against this process's directory.
    pub fn new(path: impl AsRef<Path>) -> std::io::Result<Self> {
        std::path::absolute(path).map(Self)
    }

    /// `path` made absolute against `base`, the directory it was given
    /// in; a relative path without one is an error.
    fn within(base: Option<&AbsPath>, path: &Path) -> std::io::Result<Self> {
        match base {
            Some(base) => Self::new(base.join(path)),
            None if path.is_absolute() => Self::new(path),
            None => Err(std::io::Error::other(format!(
                "`{}` is relative and the directory it is relative to is unknown",
                path.display()
            ))),
        }
    }
}

impl Deref for AbsPath {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for AbsPath {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<OsStr> for AbsPath {
    fn as_ref(&self) -> &OsStr {
        self.0.as_os_str()
    }
}

/// A launch value that does not read: the variable or part, and why.
#[derive(Debug)]
pub struct LaunchError {
    what: &'static str,
    why: String,
}

impl LaunchError {
    fn new(what: &'static str, why: impl fmt::Display) -> Self {
        Self {
            what,
            why: why.to_string(),
        }
    }
}

impl fmt::Display for LaunchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.what, self.why)
    }
}

impl std::error::Error for LaunchError {}

/// The state a launcher hands on: to its agent, and across a re-exec to
/// the launcher that replaces it. Each field is one variable of [`env`](mod@crate::env).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Carry {
    /// The session the agent runs ([`env::SESSION`]). `rig build` stages
    /// for it; `rig` itself resumes only what its command line names.
    pub session: Option<SessionId>,
    /// The session the agent switched away from, which a refused first
    /// start goes back to ([`env::SWITCHED_FROM`]). Launchers only.
    pub left: Option<SessionId>,
    /// Why the agent starts ([`env::START`]). A re-exec'd launcher's first
    /// start goes on with it instead of its claim's.
    pub reason: Option<StartReason>,
    /// Whether the run warned of a 0.46 home already ([`env::WARNED`]).
    /// Launchers only.
    pub warned: bool,
    /// What the agent warns of at startup ([`env::WARNING`]).
    pub warnings: Vec<String>,
    /// A line the agent shows at startup ([`env::NOTICE`]).
    pub notice: Option<String>,
    /// Why the build before this start failed ([`env::BUILD_FAILURE`]).
    pub build_failure: Option<String>,
}

impl Carry {
    /// The state `vars` hold; an unset or empty variable is none. A value
    /// that does not read is an error naming its variable.
    fn read(vars: &BTreeMap<&OsStr, &OsStr>) -> Result<Self, LaunchError> {
        let text = |name: &'static str| {
            vars.get(OsStr::new(name))
                .filter(|value| !value.is_empty())
                .map(|value| {
                    value
                        .to_str()
                        .map(str::to_owned)
                        .ok_or_else(|| LaunchError::new(name, "it is not UTF-8"))
                })
                .transpose()
        };
        let session = |name: &'static str| -> Result<Option<SessionId>, LaunchError> {
            text(name)?
                .map(|id| id.parse().map_err(|why| LaunchError::new(name, why)))
                .transpose()
        };
        let reason = text(env::START)?
            .map(|reason| {
                reason
                    .parse()
                    .map_err(|why| LaunchError::new(env::START, why))
            })
            .transpose()?;
        let warned = match text(env::WARNED)?.as_deref() {
            None => false,
            Some("1") => true,
            Some(other) => {
                return Err(LaunchError::new(env::WARNED, format!("`{other}` is not 1")));
            }
        };
        Ok(Self {
            session: session(env::SESSION)?,
            left: session(env::SWITCHED_FROM)?,
            reason,
            warned,
            warnings: text(env::WARNING)?.map_or_else(Vec::new, |text| warnings(&text)),
            notice: text(env::NOTICE)?,
            build_failure: text(env::BUILD_FAILURE)?,
        })
    }

    /// Sets each of its variables on `command`, or removes it when none.
    fn write(&self, command: &mut Command) {
        let Self {
            session,
            left,
            reason,
            warned,
            warnings,
            notice,
            build_failure,
        } = self;
        let set = |command: &mut Command, name: &str, value: Option<&str>| {
            match value {
                Some(value) => command.env(name, value),
                None => command.env_remove(name),
            };
        };
        set(
            command,
            env::SESSION,
            session.as_ref().map(SessionId::as_str),
        );
        set(
            command,
            env::SWITCHED_FROM,
            left.as_ref().map(SessionId::as_str),
        );
        set(command, env::START, reason.map(StartReason::as_str));
        set(command, env::WARNED, warned.then_some("1"));
        let warning = (!warnings.is_empty()).then(|| warning_text(warnings));
        set(command, env::WARNING, warning.as_deref());
        set(command, env::NOTICE, notice.as_deref());
        set(command, env::BUILD_FAILURE, build_failure.as_deref());
    }
}

/// The text of [`env::WARNING`] for `warnings`: a TOML array of strings,
/// so a line break in a warning stays in that warning. Read by
/// [`warnings`](crate::warnings).
fn warning_text(warnings: &[String]) -> String {
    warnings
        .iter()
        .map(String::as_str)
        .collect::<toml_edit::Array>()
        .to_string()
}

/// Every variable a [`Launch`] command sets or drops, besides the secrets.
const VARIABLES: [&str; 7] = [
    env::HOME,
    env::LAUNCHER,
    env::SOURCE,
    env::DESCRIBE,
    env::DESCRIBE_SETTINGS,
    env::SWITCHED_FROM,
    env::WARNED,
];

/// One start of an agent ([`Launch::agent`]).
#[derive(Clone, Copy, Debug)]
pub struct Agent<'a> {
    /// The agent build to run.
    pub program: &'a Path,
    /// Its command line, after the program.
    pub args: &'a [OsString],
    /// The directory it runs in; the caller's when `None`.
    pub dir: Option<&'a Path>,
    /// The run directory the agent says it started in ([`env::RUN`]).
    pub run: &'a RunDir,
    /// What `rig --version` prints after `rig` ([`env::VERSION`]).
    pub version: &'a str,
    /// The [`Stamp`] of the launcher's file when it started, for an agent
    /// this launcher restarts on a reload; `None` for one that has no
    /// launcher, such as an ACP session's, which gets no [`env::LAUNCHER`].
    pub launcher: Option<Stamp>,
}

/// A process a launcher starts that is not an agent nor a launcher. None
/// gets a secret or a variable of the protocol, but as said here.
#[derive(Clone, Copy, Debug)]
pub enum Purpose<'a> {
    /// cargo in the agent project ([`Home::project`]), building it or
    /// reading its dependency tree.
    Cargo,
    /// `cargo install` of a new launcher, in `dir`, whose toolchain rustup
    /// picks: a checkout's own, or the install root's.
    Install(&'a Path),
    /// The launcher at this launcher's path, as `cargo install` just put
    /// it there, building the agent: it gets the home, the source and the
    /// session, so a `/update`'s build stages for that session.
    Rebuild,
    /// `stty`, on the launcher's terminal.
    Stty,
}

/// What a launcher process was told when it started, and what it hands
/// on. Build it with [`Self::from_process`]; every process the launcher
/// starts is built from it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Launch {
    /// The root of every rig directory ([`env::HOME`]), `~/.rig` when unset.
    pub home: AbsPath,
    /// The directory the launcher runs in; `None` when it is gone.
    pub here: Option<AbsPath>,
    /// The launcher's own file.
    pub launcher: AbsPath,
    /// Whether its stdin is a terminal.
    pub terminal: bool,
    /// A rig checkout to build the agent from ([`env::SOURCE`]).
    pub source: Option<AbsPath>,
    /// Its command line, after the program.
    pub args: Vec<OsString>,
    /// The secrets the launcher holds, by variable
    /// ([`secret_variable`](crate::secret_variable)): those it was given
    /// and those typed at its prompt. Only its agent and the launcher
    /// replacing it get them.
    pub secrets: BTreeMap<OsString, OsString>,
    /// What it hands on.
    pub carry: Carry,
}

impl Launch {
    /// This process's launch: its directory, terminal, file, environment
    /// and arguments, read once.
    pub fn from_process() -> Result<Self, LaunchError> {
        let here = std::env::current_dir().ok();
        let launcher = std::env::current_exe().map_err(|why| LaunchError::new("its file", why))?;
        let vars: Vec<(OsString, OsString)> = std::env::vars_os().collect();
        let args: Vec<OsString> = std::env::args_os().skip(1).collect();
        Self::from_parts(
            &launcher,
            std::io::stdin().is_terminal(),
            vars.iter()
                .map(|(name, value)| (name.as_os_str(), Some(value.as_os_str()))),
            here.as_deref(),
            args.iter().map(OsString::as_os_str),
        )
    }

    /// The launch of a process at `launcher`, with `terminal`, given
    /// `envs` (a removed variable as `None`, as
    /// [`Command::get_envs`] lists it), in `here`, with `args`. Relative
    /// paths are made absolute against `here`. Without [`env::HOME`] the
    /// home is `~/.rig`.
    pub fn from_parts<'a>(
        launcher: &Path,
        terminal: bool,
        envs: impl IntoIterator<Item = (&'a OsStr, Option<&'a OsStr>)>,
        here: Option<&Path>,
        args: impl IntoIterator<Item = &'a OsStr>,
    ) -> Result<Self, LaunchError> {
        let vars: BTreeMap<&OsStr, &OsStr> = envs
            .into_iter()
            .filter_map(|(name, value)| Some((name, value?)))
            .collect();
        let here = here
            .map(AbsPath::new)
            .transpose()
            .map_err(|why| LaunchError::new("its directory", why))?;
        let path = |name: &'static str| {
            vars.get(OsStr::new(name))
                .filter(|value| !value.is_empty())
                .map(|value| AbsPath::within(here.as_ref(), Path::new(value)))
                .transpose()
                .map_err(|why| LaunchError::new(name, why))
        };
        let home = match path(env::HOME)? {
            Some(home) => home,
            None => {
                let home = std::env::home_dir().ok_or_else(|| {
                    LaunchError::new(env::HOME, "unset, and there is no home directory")
                })?;
                AbsPath::new(home.join(".rig")).map_err(|why| LaunchError::new(env::HOME, why))?
            }
        };
        let secrets = vars
            .iter()
            .filter(|(name, _)| {
                name.to_str()
                    .is_some_and(|name| name.starts_with(env::SECRET_PREFIX))
            })
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect();
        let source = path(env::SOURCE)?;
        let launcher = AbsPath::within(here.as_ref(), launcher)
            .map_err(|why| LaunchError::new("its file", why))?;
        Ok(Self {
            home,
            launcher,
            here,
            terminal,
            source,
            args: args.into_iter().map(OsStr::to_owned).collect(),
            secrets,
            carry: Carry::read(&vars)?,
        })
    }

    /// The `RIG_HOME` layout under [`Self::home`].
    pub fn home(&self) -> Home {
        Home::at(self.home.to_path_buf())
    }

    /// `program`, with no variable of the protocol and no secret.
    fn bare(&self, program: impl AsRef<OsStr>) -> Command {
        let mut command = Command::new(program);
        for name in VARIABLES.iter().chain(env::AGENT_ONLY) {
            command.env_remove(name);
        }
        for secret in self.secrets.keys() {
            command.env_remove(secret);
        }
        command
    }

    /// Sets the home and the source, made absolute, on `command`.
    fn paths(&self, command: &mut Command) {
        command.env(env::HOME, &self.home);
        if let Some(source) = &self.source {
            command.env(env::SOURCE, source);
        }
    }

    /// The agent `agent` says, told [`Self::carry`] but what only a
    /// launcher reads, and given the secrets.
    pub fn agent(&self, agent: &Agent<'_>) -> Command {
        let mut command = self.bare(agent.program);
        command.args(agent.args);
        self.paths(&mut command);
        Carry {
            left: None,
            warned: false,
            ..self.carry.clone()
        }
        .write(&mut command);
        command
            .env(env::RUN, agent.run.path())
            .env(env::VERSION, agent.version)
            .envs(&self.secrets);
        if let Some(stamp) = agent.launcher {
            command
                .env(env::LAUNCHER, &self.launcher)
                .env(env::LAUNCHER_STAMP, stamp.text());
        }
        if let Some(dir) = agent.dir {
            command.current_dir(dir);
        }
        command
    }

    /// The agent build `binary` asked to describe itself into `report`
    /// ([`env::DESCRIBE`]), in the scratch home `home`, which is its
    /// directory, with stdin and stdout closed. With `settings`
    /// ([`env::DESCRIBE_SETTINGS`]) it reads that settings file.
    pub fn describing(
        &self,
        binary: &Path,
        home: &Path,
        report: &Path,
        settings: Option<&Path>,
    ) -> Command {
        let mut command = self.bare(binary);
        command
            .env(env::DESCRIBE, report)
            .env(env::HOME, home)
            .current_dir(home)
            .stdin(Stdio::null())
            .stdout(Stdio::null());
        if let Some(settings) = settings {
            command.env(env::DESCRIBE_SETTINGS, settings);
        }
        command
    }

    /// The process `purpose` names.
    pub fn build_child(&self, purpose: Purpose<'_>) -> Command {
        match purpose {
            Purpose::Cargo => {
                let mut command = self.bare("cargo");
                command.current_dir(self.home().project());
                command
            }
            Purpose::Install(dir) => {
                let mut command = self.bare("cargo");
                command.current_dir(dir);
                command
            }
            Purpose::Rebuild => {
                let mut command = self.bare(&self.launcher);
                command.arg("build");
                self.paths(&mut command);
                if let Some(session) = &self.carry.session {
                    command.env(env::SESSION, session.as_str());
                }
                command
            }
            Purpose::Stty => self.bare("stty"),
        }
    }

    /// The launcher at [`Self::launcher`] in this one's place, with
    /// [`Self::args`], in [`Self::here`], carrying [`Self::carry`] and the
    /// secrets: [`Self::from_parts`] reads its command back as this value.
    pub fn reexec(&self) -> Command {
        let mut command = self.bare(&self.launcher);
        command.args(&self.args).envs(&self.secrets);
        self.paths(&mut command);
        self.carry.write(&mut command);
        if let Some(here) = &self.here {
            command.current_dir(here);
        }
        command
    }
}

#[cfg(test)]
mod tests;
