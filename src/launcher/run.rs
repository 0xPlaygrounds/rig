//! `rig`: build the agent when needed, then run it. Exit code 75 restarts
//! it on the staged build. A new build runs as this launcher's own trial
//! and becomes the good build once it signals ready; one that exits before
//! that is rolled back to the last good build. Several launchers can share
//! one `RIG_HOME`: building and staging take the root's lock, and a
//! `/reload` build, its trial and the ready file are per session.
//!
//! A session belongs to the directory it runs in. Until it quits cleanly
//! (exit code 0), `resume/<hash of the directory>` names it, so after a
//! crash, a kill or a closed terminal the next `rig` there resumes it from
//! its last autosave, unless another launcher still runs it.

use std::fs::{self, File};
use std::io::{ErrorKind, IsTerminal};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, ExitStatus};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::home::Home;
use super::{RELOAD_EXIT_CODE, Result, build};

/// How often the launcher looks for the ready file while the agent runs.
const POLL: Duration = Duration::from_millis(100);

/// Runs the agent until it exits with anything but the reload code, and
/// returns its exit code.
pub fn run(home: &Home) -> Result<ExitCode> {
    let (claimed, mut notice) = {
        let _lock = home.lock()?;
        // Before claiming, so a resumed session's leftover builds from its
        // dead launcher are removed too.
        home.sweep()?;
        let claimed = claim(home)?;
        let built = rebuild(home, &claimed.id)?;
        let resumed = claimed
            .resumed
            .then(|| "Resumed this directory's session from its last autosave.".to_owned());
        let notice: Vec<String> = resumed.into_iter().chain(built).collect();
        (claimed, (!notice.is_empty()).then(|| notice.join("\n")))
    };
    let session = claimed.id.as_str();
    let launcher = std::env::current_exe()?;
    let trial = home.trial_for(session);
    let ready = home.session(session).join("ready");
    loop {
        let (binary, is_trial) = pick(home, session, &trial)?;
        remove_if_present(&ready)?;
        let mut command = Command::new(&binary);
        command
            .env("RIG_HOME", home.root())
            .env("RIG_SESSION", session)
            .env("RIG_LAUNCHER", &launcher)
            .env("RIG_READY_FILE", &ready)
            .env_remove("RIG_NOTICE");
        if let Some(notice) = notice.take() {
            command.env("RIG_NOTICE", notice);
        }
        let mut child = command.spawn()?;
        // A trial becomes the good build as soon as it is ready, so a
        // launcher killed later (a closed terminal) keeps it.
        let mut started = false;
        let mut promoted: Result<()> = Ok(());
        let status = loop {
            let status = child.try_wait()?;
            if !started && ready.exists() {
                started = true;
                if is_trial {
                    promoted = fs::rename(&trial, home.bin("good")).map_err(Into::into);
                }
            }
            match status {
                Some(status) => break status,
                None => std::thread::sleep(POLL),
            }
        };
        promoted?;
        let rejected = is_trial && !started;
        if rejected {
            fs::remove_file(&trial)?;
        }
        if status.code() == Some(RELOAD_EXIT_CODE) {
            continue;
        }
        let log = home.session(session).join("agent.log");
        if !rejected && status.success() {
            claimed.forget()?;
            return Ok(exit_code(status));
        }
        if !rejected {
            restore_terminal();
            eprintln!(
                "The agent stopped ({status}). Run `rig` here again to resume the session \
                 from its last autosave. Log: {}",
                log.display()
            );
            return Ok(exit_code(status));
        }
        let fallback = home.bin("good").exists();
        let message = if fallback {
            format!(
                "The new build stopped during startup ({status}); rolled back to the previous \
                 build. `rig build` tries it again. Log: {}",
                log.display()
            )
        } else {
            format!(
                "The new build stopped during startup ({status}) and there is no previous \
                 build to roll back to; the next `rig` tries it again. Log: {}",
                log.display()
            )
        };
        restore_terminal();
        eprintln!("{message}");
        if !fallback {
            return Ok(exit_code(status));
        }
        notice = Some(message);
    }
}

/// The session this launcher runs, held locked while it lives.
struct Claimed {
    id: String,
    /// Whether it resumes a session that did not quit cleanly.
    resumed: bool,
    /// The working directory's resume marker, when there is a working
    /// directory.
    marker: Option<PathBuf>,
    _lock: File,
}

impl Claimed {
    /// After a clean quit, the next `rig` in this directory starts a new
    /// session: the marker goes, when it still names this one.
    fn forget(&self) -> Result<()> {
        match &self.marker {
            Some(marker)
                if fs::read_to_string(marker).is_ok_and(|named| named.trim() == self.id) =>
            {
                remove_if_present(marker)
            }
            _ => Ok(()),
        }
    }
}

/// The session to run: the one the working directory's resume marker names,
/// when it has saved state and no other launcher runs it, or else a new
/// one. The marker then names the new one, unless another launcher runs the
/// session it names. Call it holding [`Home::lock`].
fn claim(home: &Home) -> Result<Claimed> {
    let marker = std::env::current_dir()
        .ok()
        .map(|directory| home.resume_marker(&directory));
    let previous = marker
        .as_ref()
        .and_then(|marker| fs::read_to_string(marker).ok())
        .map(|id| id.trim().to_owned())
        .filter(|id| is_session_id(id))
        .filter(|id| home.session(id).join("state.json").is_file());
    let mut taken = false;
    if let Some(id) = previous {
        match home.hold_session(&id)? {
            Some(lock) => {
                return Ok(Claimed {
                    id,
                    resumed: true,
                    marker,
                    _lock: lock,
                });
            }
            None => taken = true,
        }
    }
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or_default();
    // The agent makes the same id when it runs without the launcher.
    let id = format!("{seconds}-{}", std::process::id());
    let lock = home
        .hold_session(&id)?
        .ok_or_else(|| format!("session {id} is already running"))?;
    if let Some(marker) = marker.as_ref().filter(|_| !taken) {
        if let Some(parent) = marker.parent() {
            fs::create_dir_all(parent)?;
        }
        let written = marker.with_extension("tmp");
        fs::write(&written, &id)?;
        fs::rename(&written, marker)?;
    }
    Ok(Claimed {
        id,
        resumed: false,
        marker,
        _lock: lock,
    })
}

/// Whether `id` has the form the launcher gives session ids, so a marker
/// cannot name a path elsewhere.
fn is_session_id(id: &str) -> bool {
    !id.is_empty() && id.chars().all(|c| c.is_ascii_digit() || c == '-')
}

/// Puts the terminal back after an agent that could not restore it: leaves
/// the alternate screen and, on a terminal, raw mode.
fn restore_terminal() {
    eprint!("\x1b[?1049l\x1b[?25h");
    if std::io::stdin().is_terminal() {
        Command::new("stty").arg("sane").status().ok();
    }
}

fn remove_if_present(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Err(failure) if failure.kind() != ErrorKind::NotFound => Err(failure.into()),
        _ => Ok(()),
    }
}

/// Builds before the first start when there is no binary yet, when the
/// generated project changed, and always from a local checkout, staging for
/// this launcher. With no good build yet, the build is staged even if it
/// was rejected before, so each start retries it. A failed build with an
/// older binary at hand becomes the notice that binary starts with.
fn rebuild(home: &Home, session: &str) -> Result<Option<String>> {
    let first_run = !["staged", "good"]
        .into_iter()
        .any(|name| home.bin(name).exists());
    let built = build::prepare(home).and_then(|project| {
        if first_run || project.changed || project.local {
            eprintln!(
                "Building the rig agent{}…",
                if first_run { " (first run)" } else { "" }
            );
            build::compile(home, &project, &home.staged_for(session), first_run)?;
        }
        Ok(())
    });
    match built {
        Ok(()) => Ok(None),
        Err(failure) if first_run => Err(failure),
        Err(failure) => {
            eprintln!("error: {failure}; starting the previous build");
            Ok(Some(format!(
                "Building the agent failed, so the previous build is running: {failure}"
            )))
        }
    }
}

/// The binary to start: a build staged for this launcher or for any, claimed
/// as its `trial`, or else the last good one. The flag says it is a trial.
fn pick(home: &Home, session: &str, trial: &Path) -> Result<(PathBuf, bool)> {
    for staged in [home.staged_for(session), home.bin("staged")] {
        // The rename claims the staged build for this launcher alone.
        match fs::rename(staged, trial) {
            Ok(()) => return Ok((trial.to_path_buf(), true)),
            Err(failure) if failure.kind() != ErrorKind::NotFound => return Err(failure.into()),
            Err(_) => {}
        }
    }
    let good = home.bin("good");
    if good.exists() {
        return Ok((good, false));
    }
    Err("no agent build exists yet; run `rig build`".into())
}

/// The agent's exit code; 1 when a signal ended it.
fn exit_code(status: ExitStatus) -> ExitCode {
    ExitCode::from(
        status
            .code()
            .and_then(|code| u8::try_from(code).ok())
            .unwrap_or(1),
    )
}
