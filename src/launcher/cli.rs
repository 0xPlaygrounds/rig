//! The command-line verbs (`rig start`, `send`, `wait`, ..., and `rig -p`)
//! run in the agent binary's `cli` role, and `rig` on a terminal in its
//! `view` role: the launcher finds a build,
//! building one only when there is none, and replaces itself with it under
//! `RIG_ROLE=cli`. A build `rig build` staged is promoted first, so the
//! verbs and the server they start use it. No cargo runs when a build
//! exists, and nothing is said on a stderr that is not chatty
//! ([`chatty`](rig_harness_protocol::chatty)): stdout and stderr stay the
//! verb's.

use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use rig_harness_protocol::{Home, env, guide};

use super::Result;
use super::build::{self, Staging};
use super::locks;
use super::project::Profile;

/// Whether `args` run a verb of the `cli` role: its first word names one,
/// they hold `-p`/`--print`, or there are none and stdin is not a terminal
/// (a prompt piped to a bare `rig`).
pub fn is_cli(args: &[String]) -> bool {
    match args.first() {
        Some(first) => {
            guide::VERBS
                .iter()
                .any(|verb| verb.name == first && !matches!(verb.name, "-p" | "help-agents"))
                || args.iter().any(|arg| arg == "-p" || arg == "--print")
        }
        None => !std::io::stdin().is_terminal(),
    }
}

/// What refuses `rig -r SESSION` when there is no terminal for the view
/// (`terminal` false), with the commands that work headless; `None` when
/// it may run.
pub fn resume_refusal(session: &str, terminal: bool) -> Option<String> {
    let _ = (session, terminal);
    None
}

/// Runs `args` in the `cli` role of the current build.
pub fn run(home: &Home, args: &[String]) -> Result<ExitCode> {
    run_role(home, "cli", args)
}

/// `rig` on a terminal: rebuilds what changed, as every interactive start
/// did, then runs the terminal view, which starts a session in this
/// directory and drives it. A build that fails leaves the last one
/// running, and says so.
pub fn view(home: &Home) -> Result<ExitCode> {
    if let Err(failure) = build::build(home) {
        if !home.good().exists() && !home.staged().exists() {
            return Err(failure);
        }
        eprintln!("rig: the build failed, so the previous one runs: {failure}");
    }
    run_role(home, "view", &[])
}

/// Replaces this process with the current build in `role`, given `args`.
fn run_role(home: &Home, role: &str, args: &[String]) -> Result<ExitCode> {
    let binary = match current_build(home).and_then(|binary| install(home, &binary)) {
        Ok((_, installed)) => installed,
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
        .env(env::ROLE, role)
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

/// The id of `binary`, from its stamp: its modification time and size in
/// hex, which tell builds apart as [`build::stamp`] does.
pub fn build_id(binary: &Path) -> Result<String> {
    let stamp = build::stamp(binary)?;
    let mut parts = stamp.split_whitespace().map(str::parse::<u128>);
    match (parts.next(), parts.next()) {
        (Some(Ok(modified)), Some(Ok(size))) => Ok(format!("{modified:x}-{size:x}")),
        _ => Err(format!("{} has no build stamp", binary.display()).into()),
    }
}

/// Copies `binary` to its immutable place, [`Home::build`], unless it is
/// there already (never replacing it), and returns its id and that path. Every process of the
/// build runs from there, so `rigd` and its workers name the build they
/// run by their directory, and a later `rig build` replaces nothing they
/// run.
pub fn install(home: &Home, binary: &Path) -> Result<(String, PathBuf)> {
    let id = build_id(binary)?;
    let installed = home.build(&id);
    if installed.exists() {
        return Ok((id, installed));
    }
    let directory = installed.parent().ok_or("a build has no directory")?;
    fs::create_dir_all(directory)?;
    // A name of this install's own: other processes, and other threads of
    // this one, may install the same build at the same time.
    static INSTALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let temporary = directory.join(format!(
        ".{}-{}.tmp",
        std::process::id(),
        INSTALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    fs::copy(binary, &temporary)?;
    // A hard link never replaces: the first install wins, and a build a
    // process already runs from is never swapped under it.
    let linked = fs::hard_link(&temporary, &installed);
    fs::remove_file(&temporary).ok();
    match linked {
        Err(failure) if failure.kind() != std::io::ErrorKind::AlreadyExists => Err(failure.into()),
        _ => Ok((id, installed)),
    }
}

/// `rig build --print-build-id [--plugins FILE]`: builds like `rig build`
/// (or the plugin list `FILE` as a profile of its own) and checks the
/// build describes itself, then installs it and prints its id, for a
/// session's `/reload` (`rig/session/reload`) or `rig start --plugins` to
/// name. It leaves the build new sessions run as it is.
pub fn print_build_id(home: &Home, plugins: Option<&Path>) -> Result<ExitCode> {
    let profile = match plugins {
        Some(list) => Profile::of(home, list)?,
        None => Profile::default_of(home),
    };
    let _lock = locks::hold_build_lock(home)?;
    build::compile_profile(home, &profile, &home.staged(), Staging::Never)?;
    let (id, _) = install(home, &build::artifact(home))?;
    println!("{id}");
    Ok(ExitCode::SUCCESS)
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

#[cfg(test)]
mod tests;
