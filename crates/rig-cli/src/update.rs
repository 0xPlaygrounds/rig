//! `rig update`: reinstalls this launcher with `cargo install`, from where
//! it was installed, into the directory it runs from, then builds the agent
//! with the new launcher. It says the exact command before it runs it, and
//! a failed install leaves this launcher as it was: cargo moves the new
//! binary in only once it is built. cargo runs in the checkout, or for
//! crates.io in the install root, not in the directory rig was started in,
//! whose `rust-toolchain.toml` may pin a toolchain too old for rig
//! ([`install_command`]).
//!
//! A launcher whose own file was replaced while its agent ran, by `rig
//! update` or by hand, starts the new file on the agent's next reload, so
//! the session goes on with a new launcher and agent ([`Reexec`]). A
//! switch to a session the new agent refuses goes back to the session it
//! left, as it does without a new launcher ([`LEFT`]). `/update` in the
//! agent runs `rig update`, then reloads.

use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use rig_harness_protocol::{Home, SessionId, UPDATE_INSTALLED, UPDATE_SAYS, env, shell_word};

use super::Result;
use super::build::child;
use super::project::RigSource;

/// Set by a launcher on the new launcher it starts in its place
/// ([`Reexec`]), and only on it: the session its agent switched away from
/// to the one the new launcher resumes.
pub const LEFT: &str = "RIG_SWITCHED_FROM";

/// The session [`LEFT`] names, which this launcher's first start goes back
/// to when its agent refuses the session; `None` when it is unset. A value
/// that is not a session id is an error naming it.
pub fn left() -> Result<Option<SessionId>> {
    parse_left(std::env::var_os(LEFT).as_deref())
}

/// The session a [`LEFT`] value names.
fn parse_left(value: Option<&OsStr>) -> Result<Option<SessionId>> {
    let Some(left) = value else {
        return Ok(None);
    };
    let left = left.to_string_lossy();
    Ok(Some(left.parse().map_err(|why| format!("{LEFT}: {why}"))?))
}

/// The crate `cargo install` installs the launcher from.
const PACKAGE: &str = "rig-cli";

/// Runs `rig update`. Refuses, in a line that starts as each of its own
/// does, when this launcher is not in a `bin` directory, as `cargo install`
/// puts it, or the rig source cannot be told.
pub fn update() -> ExitCode {
    let install: Result<(Command, PathBuf)> = (std::env::current_exe().map_err(Into::into))
        .and_then(|launcher| {
            // The checkout without the `..` of its compile-time path, as shown.
            let source = match RigSource::detect()? {
                RigSource::Local(checkout) => RigSource::Local(fs::canonicalize(checkout)?),
                RigSource::Registry => RigSource::Registry,
            };
            let install = install_command(&source, &install_root(&launcher)?);
            Ok((install, launcher))
        });
    match install {
        Ok((install, launcher)) => install_and_build(install, &launcher),
        Err(failure) => {
            eprintln!("{UPDATE_SAYS}{failure}");
            ExitCode::FAILURE
        }
    }
}

/// The directory `cargo install --root` takes for the launcher at
/// `launcher`: the one above its `bin` directory, made absolute, since
/// cargo starts elsewhere and macOS may give a launcher path relative to
/// the directory rig was started in.
fn install_root(launcher: &Path) -> Result<PathBuf> {
    let absolute = std::path::absolute(launcher)?;
    absolute
        .parent()
        .filter(|bin| bin.file_name().is_some_and(|name| name == "bin"))
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| {
            format!(
                "it replaces a rig that `cargo install` put in a `bin` directory; this one \
                 is {}, so it is left as it is. Install rig with `cargo install` to update \
                 it this way, or rebuild it the way you built it",
                launcher.display()
            )
            .into()
        })
}

/// The `cargo install` that replaces the launcher in `root` from `source`:
/// crates.io's latest `rig-cli`, or the checkout as it is now, whose new
/// commits are the user's to fetch. rustup picks the toolchain of the
/// directory cargo starts in, so it starts in the checkout, whose own
/// `rust-toolchain.toml` builds rig, or for crates.io in `root`, which the
/// launcher's own `bin` directory is in, not in the directory rig was
/// started in.
pub fn install_command(source: &RigSource, root: &Path) -> Command {
    let mut command = child("cargo");
    command.args(["install", "--force"]);
    match source {
        RigSource::Registry => {
            command.arg(PACKAGE).current_dir(root);
        }
        RigSource::Local(checkout) => {
            let path = checkout.join("crates").join(PACKAGE);
            command.arg("--locked").arg("--path").arg(path);
            command.current_dir(checkout);
        }
    }
    command.arg("--root").arg(root);
    command
}

/// Runs `install`, then the new launcher's `rig build`; their output goes
/// to stderr, where the agent's `/update` reads it, and neither gets a
/// secret ([`child`]). Says each step first.
/// `ExitCode::FAILURE` when either fails.
fn install_and_build(mut install: Command, launcher: &Path) -> ExitCode {
    let shown = shell_words(&install);
    eprintln!("{UPDATE_SAYS}running {shown}");
    match install.status() {
        Ok(status) if status.success() => {}
        Ok(status) => {
            eprintln!(
                "{UPDATE_SAYS}`{shown}` failed ({status}); this rig is unchanged. Fix what \
                 cargo says above and run `rig update` again."
            );
            return ExitCode::FAILURE;
        }
        Err(failure) => {
            eprintln!(
                "{UPDATE_SAYS}could not run cargo: {failure}; this rig is unchanged. Put \
                 cargo on PATH and run `rig update` again."
            );
            return ExitCode::FAILURE;
        }
    }
    eprintln!(
        "{UPDATE_INSTALLED} Building the agent with the new rig (`rig build`); after a \
         change this takes 2-3 minutes."
    );
    match child(launcher).arg("build").status() {
        Ok(status) if status.success() => {
            eprintln!("{UPDATE_SAYS}done.");
            ExitCode::SUCCESS
        }
        Ok(status) => {
            eprintln!(
                "{UPDATE_SAYS}rig is updated, but building the agent failed ({status}); \
                 `rig build` tries again, and the next start or /reload runs the new rig."
            );
            ExitCode::FAILURE
        }
        Err(failure) => {
            eprintln!(
                "{UPDATE_SAYS}rig is updated, but could not run `rig build`: {failure}; the \
                 next start or /reload runs the new rig."
            );
            ExitCode::FAILURE
        }
    }
}

/// `command` as POSIX shell words ([`shell_word`]), after a `cd` to the
/// directory it runs in, so a user who copies it runs the same thing.
fn shell_words(command: &Command) -> String {
    let words = std::iter::once(command.get_program())
        .chain(command.get_args())
        .map(|word| shell_word(&word.to_string_lossy()))
        .collect::<Vec<_>>()
        .join(" ");
    match command.get_current_dir() {
        Some(directory) => format!("cd {} && {words}", shell_word(&directory.to_string_lossy())),
        None => words,
    }
}

/// The launcher to start in place of this one once its session is let go:
/// the new file at the same path, with the same arguments and home.
#[derive(Debug, PartialEq)]
pub struct Reexec {
    launcher: PathBuf,
    args: Vec<OsString>,
    directory: Option<PathBuf>,
    left: Option<SessionId>,
    home: PathBuf,
}

impl Reexec {
    /// The launcher at `launcher` with `args`, the command line's own,
    /// which never name a session: resuming `session`, or starting a new
    /// one when it is `None`, in `directory` when given, on `home`. `left`
    /// is the session the agent switched away from to `session` ([`LEFT`]).
    pub fn new(
        home: &Home,
        launcher: &Path,
        session: Option<&str>,
        args: Vec<OsString>,
        directory: Option<PathBuf>,
        left: Option<SessionId>,
    ) -> Self {
        let resume = session.map(|session| ["--resume".into(), session.into()]);
        Self {
            launcher: launcher.to_path_buf(),
            args: resume.into_iter().flatten().chain(args).collect(),
            directory,
            left,
            home: home.root().to_path_buf(),
        }
    }

    /// The new launcher's command, told the session left, or not told
    /// one the environment holds from an earlier re-exec. It may start in
    /// another directory, so it gets the home and `RIG_SOURCE` as absolute
    /// paths.
    fn command(&self) -> Result<Command> {
        let mut command = Command::new(&self.launcher);
        command.args(&self.args).env(env::HOME, &self.home);
        if let Some(source) = RigSource::from_env()? {
            command.env(env::SOURCE, source);
        }
        match &self.left {
            Some(left) => command.env(LEFT, left.as_str()),
            None => command.env_remove(LEFT),
        };
        if let Some(directory) = &self.directory {
            command.current_dir(directory);
        }
        Ok(command)
    }

    /// Runs the new launcher in this process's place, or on other systems
    /// as a child whose exit code it returns. Returns only on failure on
    /// unix.
    pub fn run(self) -> Result<ExitCode> {
        eprintln!("rig was updated; restarting on the new launcher.");
        let mut command = self.command()?;
        #[cfg(unix)]
        {
            let failure = std::os::unix::process::CommandExt::exec(&mut command);
            Err(format!("could not start the new rig: {failure}").into())
        }
        #[cfg(not(unix))]
        {
            let status = command.status()?;
            let code = status.code().and_then(|code| u8::try_from(code).ok());
            Ok(code.map_or(ExitCode::FAILURE, ExitCode::from))
        }
    }
}

#[cfg(all(test, unix))]
mod tests;
