//! The command-line verbs (`rig start`, `send`, `wait`, ..., and `rig -p`)
//! run in the agent binary's `cli` role: the launcher finds a build,
//! building one only when there is none, and replaces itself with it under
//! `RIG_ROLE=cli`. A build `rig build` staged is promoted first, so the
//! verbs and the server they start use it. No cargo runs when a build
//! exists, and nothing is said on a stderr that is not chatty
//! ([`chatty`](rig_harness_protocol::chatty)): stdout and stderr stay the
//! verb's.

use std::fs;
use std::io::IsTerminal;
use std::path::PathBuf;
use std::process::{Command, ExitCode};

use rig_harness_protocol::{Home, env, guide};

use super::Result;
use super::build::{self, Staging};
use super::locks;

/// Whether `args` run a verb of the `cli` role: its first word names one,
/// they hold `-p`/`--print`, or there are none and stdin is not a terminal
/// (a prompt piped to a bare `rig`).
pub fn is_cli(args: &[String]) -> bool {
    match args.first() {
        Some(first) => {
            guide::VERBS
                .iter()
                .any(|verb| verb.name == first && verb.name != "-p")
                || args.iter().any(|arg| arg == "-p" || arg == "--print")
        }
        None => !std::io::stdin().is_terminal(),
    }
}

/// Runs `args` in the `cli` role of the current build.
pub fn run(home: &Home, args: &[String]) -> Result<ExitCode> {
    let binary = match current_build(home) {
        Ok(binary) => binary,
        Err(failure) => {
            eprintln!(
                "rig: no agent build could be made: {failure}\nnext: RIG_VERBOSE=1 rig build"
            );
            return Ok(ExitCode::from(1));
        }
    };
    let mut command = Command::new(&binary);
    command
        .args(args)
        .env(env::HOME, home.root())
        .env(env::ROLE, "cli")
        .env(env::LAUNCHER, std::env::current_exe()?);
    for name in env::AGENT_ONLY.iter().filter(|name| **name != env::ROLE) {
        command.env_remove(name);
    }
    exec(command)
}

/// The build the verbs run: the staged one, promoted to the good one, else
/// the good one, else a new build.
fn current_build(home: &Home) -> Result<PathBuf> {
    let good = home.good();
    if good.exists() && !home.staged().exists() {
        return Ok(good);
    }
    let _lock = locks::hold_build_lock(home)?;
    if home.staged().exists() {
        // `rig build` checked it describes itself, so it starts.
        fs::rename(home.staged(), &good)?;
    }
    if !good.exists() {
        build::compile(home, &home.staged(), Staging::Always)?;
        fs::rename(home.staged(), &good)?;
    }
    Ok(good)
}

#[cfg(unix)]
fn exec(mut command: Command) -> Result<ExitCode> {
    use std::os::unix::process::CommandExt;
    // Only returns on failure.
    Err(command.exec().into())
}

#[cfg(not(unix))]
fn exec(mut command: Command) -> Result<ExitCode> {
    let status = command.status()?;
    Ok(ExitCode::from(
        status
            .code()
            .and_then(|code| u8::try_from(code).ok())
            .unwrap_or(1),
    ))
}
