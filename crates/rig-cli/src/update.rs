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
//! the session goes on with a new launcher and agent ([`reexec`]), as the
//! reload would on this one: the new launcher gets this one's state and
//! secrets ([`Launch::reexec`]). `/update` in the agent runs `rig update`,
//! then reloads.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use rig_harness_protocol::launch::{Launch, Purpose};
use rig_harness_protocol::{UPDATE_SAYS, shell_word};

use super::Result;
use super::project::RigSource;

/// The crate `cargo install` installs the launcher from.
const PACKAGE: &str = "rig-cli";

/// Runs `rig update`. Refuses, in a line that starts as each of its own
/// does, when this launcher is not in a `bin` directory, as `cargo install`
/// puts it, or the rig source cannot be told.
pub fn update(launch: &Launch) -> ExitCode {
    let install = (|| -> Result<Command> {
        // The checkout without the `..` of its compile-time path, as shown.
        let source = match RigSource::detect(launch.source.as_deref())? {
            RigSource::Local(checkout) => RigSource::Local(fs::canonicalize(checkout)?),
            RigSource::Registry => RigSource::Registry,
        };
        Ok(install_command(
            launch,
            &source,
            &install_root(&launch.launcher)?,
        ))
    })();
    match install {
        Ok(install) => install_and_build(launch, install),
        Err(failure) => {
            eprintln!("{UPDATE_SAYS}{failure}");
            ExitCode::FAILURE
        }
    }
}

/// The directory `cargo install --root` takes for the launcher at
/// `launcher`: the one above its `bin` directory.
fn install_root(launcher: &Path) -> Result<PathBuf> {
    launcher
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
/// commits are the user's to fetch. Both build with the lock rig was
/// tested with: the checkout's, or the one published in the package.
/// rustup picks the toolchain of the directory cargo starts in, so it
/// starts in the checkout, whose own `rust-toolchain.toml` builds rig, or
/// for crates.io in `root`, which the launcher's own `bin` directory is
/// in, not in the directory rig was started in.
pub fn install_command(launch: &Launch, source: &RigSource, root: &Path) -> Command {
    let mut command = match source {
        RigSource::Registry => {
            let mut command = launch.build_child(Purpose::Install(root));
            command.args(["install", "--force", "--locked", PACKAGE]);
            command
        }
        RigSource::Local(checkout) => {
            let mut command = launch.build_child(Purpose::Install(checkout));
            let path = checkout.join("crates").join(PACKAGE);
            command
                .args(["install", "--force", "--locked", "--path"])
                .arg(path);
            command
        }
    };
    command.arg("--root").arg(root);
    command
}

/// Runs `install`, then the new launcher's `rig build`
/// ([`Purpose::Rebuild`]); their output goes to stderr, where the agent's
/// `/update` reads it, and neither gets a secret. Says each step first.
/// `ExitCode::FAILURE` when either fails.
fn install_and_build(launch: &Launch, mut install: Command) -> ExitCode {
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
        "{UPDATE_SAYS}installed. Building the agent with the new rig (`rig build`); after a \
         change this takes 2-3 minutes."
    );
    match launch.build_child(Purpose::Rebuild).status() {
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

/// Starts the launcher at `next`'s path in this one's place, carrying on
/// as `next` says ([`Launch::reexec`]), or on other systems as a child
/// whose exit code it returns. Returns only on failure on unix.
pub fn reexec(next: &Launch) -> Result<ExitCode> {
    eprintln!("rig was updated; restarting on the new launcher.");
    let mut command = next.reexec();
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

#[cfg(all(test, unix))]
mod tests;
