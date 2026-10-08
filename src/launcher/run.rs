//! `rig`: build the agent when needed, then run it. Exit code 75 restarts
//! it on the staged build. A new build runs as this launcher's own trial
//! and becomes the good build once it signals ready; one that exits before
//! that is rolled back to the last good build. Several launchers can share
//! one `RIG_HOME`: building and staging take the root's lock, and trials
//! and ready files are per session.

use std::fs;
use std::io::ErrorKind;
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
    let mut notice = {
        let _lock = home.lock()?;
        rebuild(home)?
    };
    let launcher = std::env::current_exe()?;
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or_default();
    // The agent makes the same id when it runs without the launcher.
    let session = format!("{seconds}-{}", std::process::id());
    let trial = home.bin(&format!("trial-{session}"));
    let ready = home.session(&session).join("ready");
    fs::create_dir_all(home.session(&session))?;
    loop {
        let (binary, is_trial) = pick(home, &trial)?;
        remove_if_present(&ready)?;
        let mut command = Command::new(&binary);
        command
            .env("RIG_HOME", home.root())
            .env("RIG_SESSION", &session)
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
        if status.code() == Some(RELOAD_EXIT_CODE) {
            continue;
        }
        if !is_trial || started {
            return Ok(exit_code(status));
        }
        fs::remove_file(&trial)?;
        let log = home.session(&session).join("agent.log");
        let fallback = home.bin("good").exists();
        let message = if fallback {
            format!(
                "The new build crashed during startup ({status}); rolled back to the previous \
                 build. Log: {}",
                log.display()
            )
        } else {
            format!(
                "The new build crashed during startup ({status}) and there is no previous build \
                 to roll back to. Log: {}",
                log.display()
            )
        };
        // Leave the alternate screen the crashed build may have left behind.
        eprintln!("\x1b[?1049l{message}");
        if !fallback {
            return Ok(exit_code(status));
        }
        notice = Some(message);
    }
}

fn remove_if_present(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Err(failure) if failure.kind() != ErrorKind::NotFound => Err(failure.into()),
        _ => Ok(()),
    }
}

/// Builds before the first start when there is no binary yet, when the
/// generated project changed, and always from a local checkout. A failed
/// build with an older binary at hand becomes the notice that binary
/// starts with.
fn rebuild(home: &Home) -> Result<Option<String>> {
    let first_run = !["staged", "good"]
        .into_iter()
        .any(|name| home.bin(name).exists());
    let built = build::prepare(home).and_then(|project| {
        if first_run || project.changed || project.local {
            eprintln!(
                "Building the rig agent{}…",
                if first_run { " (first run)" } else { "" }
            );
            build::compile(home, &project)?;
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

/// The binary to start: a newly staged build, claimed as this launcher's
/// `trial`, or else the last good one. The flag says it is a trial.
fn pick(home: &Home, trial: &Path) -> Result<(PathBuf, bool)> {
    // The rename claims the staged build for this launcher alone.
    match fs::rename(home.bin("staged"), trial) {
        Ok(()) => return Ok((trial.to_path_buf(), true)),
        Err(failure) if failure.kind() != ErrorKind::NotFound => return Err(failure.into()),
        Err(_) => {}
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
