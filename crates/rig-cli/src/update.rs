//! `rig update`: reinstalls this launcher with `cargo install`, from where
//! it was installed, into the directory it runs from, then builds the agent
//! with the new launcher. It says the exact command before it runs it, and
//! a failed install leaves this launcher as it was: cargo moves the new
//! binary in only once it is built.
//!
//! A launcher whose own file was replaced while its agent ran, by `rig
//! update` or by hand, starts the new file on the agent's next reload, so
//! the session goes on with a new launcher and agent ([`Stamp`],
//! [`Reexec`]). `/update` in the agent runs `rig update`, then reloads.

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::SystemTime;

use rig_harness_protocol::{UPDATE_INSTALLED, UPDATE_SAYS};

use super::Result;
use super::project::RigSource;

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
            let install = install_command(&source, install_root(&launcher)?);
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
/// `launcher`: the one above its `bin` directory.
fn install_root(launcher: &Path) -> Result<&Path> {
    launcher
        .parent()
        .filter(|bin| bin.file_name().is_some_and(|name| name == "bin"))
        .and_then(Path::parent)
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
/// commits are the user's to fetch.
pub fn install_command(source: &RigSource, root: &Path) -> Command {
    let mut command = Command::new("cargo");
    command.args(["install", "--force"]);
    match source {
        RigSource::Registry => {
            command.arg(PACKAGE);
        }
        RigSource::Local(checkout) => {
            let path = checkout.join("crates").join(PACKAGE);
            command.arg("--locked").arg("--path").arg(path);
        }
    }
    command.arg("--root").arg(root);
    command
}

/// Runs `install`, then the new launcher's `rig build`; their output goes
/// to stderr, where the agent's `/update` reads it. Says each step first.
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
    match Command::new(launcher).arg("build").status() {
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

/// `command` as POSIX shell words, single-quoting those that need it.
fn shell_words(command: &Command) -> String {
    std::iter::once(command.get_program())
        .chain(command.get_args())
        .map(|word| {
            let word = word.to_string_lossy();
            let plain = !word.is_empty()
                && word
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || "-_./=:@+".contains(c));
            if plain {
                word.into_owned()
            } else {
                format!("'{}'", word.replace('\'', r"'\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Which file a path names, to tell when it was replaced: a file renamed
/// over it, as `cargo install` does, has another inode and modification
/// time. A file that cannot be read has none.
#[derive(Debug, PartialEq, Eq)]
pub struct Stamp(Option<(SystemTime, u64)>);

impl Stamp {
    /// Whether the file is another than at `now`. A file gone, or never
    /// read, was not replaced: only a new file at the path is a new launcher.
    pub fn replaced(&self, now: &Stamp) -> bool {
        matches!((&self.0, &now.0), (Some(then), Some(now)) if then != now)
    }

    /// The stamp of the file at `path` now.
    pub fn of(path: &Path) -> Self {
        Self(fs::metadata(path).ok().and_then(|metadata| {
            #[cfg(unix)]
            let identity = std::os::unix::fs::MetadataExt::ino(&metadata);
            #[cfg(not(unix))]
            let identity = metadata.len();
            Some((metadata.modified().ok()?, identity))
        }))
    }
}

/// The launcher to start in place of this one once its session is let go:
/// the new file at the same path, with the same arguments.
#[derive(Debug, PartialEq)]
pub struct Reexec {
    launcher: PathBuf,
    args: Vec<OsString>,
    directory: Option<PathBuf>,
}

impl Reexec {
    /// The launcher at `launcher` with `args`, the command line's own,
    /// which never name a session: resuming `session`, or starting a new
    /// one when it is `None`, in `directory` when given.
    pub fn new(
        launcher: &Path,
        session: Option<&str>,
        args: Vec<OsString>,
        directory: Option<PathBuf>,
    ) -> Self {
        let resume = session.map(|session| ["--resume".into(), session.into()]);
        Self {
            launcher: launcher.to_path_buf(),
            args: resume.into_iter().flatten().chain(args).collect(),
            directory,
        }
    }

    /// Runs the new launcher in this process's place, or on other systems
    /// as a child whose exit code it returns. Returns only on failure on
    /// unix.
    pub fn run(self) -> Result<ExitCode> {
        eprintln!("rig was updated; restarting on the new launcher.");
        let mut command = Command::new(&self.launcher);
        command.args(&self.args);
        if let Some(directory) = &self.directory {
            command.current_dir(directory);
        }
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
