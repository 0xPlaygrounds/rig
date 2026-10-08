//! `rig`: build the agent when needed, then run it. Exit code 75 restarts
//! it on the staged build. A new build that exits before it signals ready
//! is rolled back to the last build that did.

use std::fs;
use std::io::ErrorKind;
use std::path::PathBuf;
use std::process::{Command, ExitCode, ExitStatus};
use std::time::{SystemTime, UNIX_EPOCH};

use super::home::Home;
use super::{RELOAD_EXIT_CODE, Result, build};

/// Runs the agent until it exits with anything but the reload code, and
/// returns its exit code.
pub fn run(home: &Home) -> Result<ExitCode> {
    let mut notice = rebuild(home)?;
    let launcher = std::env::current_exe()?;
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or_default();
    let session = format!("{seconds}-{}", std::process::id());
    let ready = home.bin("ready");
    loop {
        let (binary, trial) = pick(home)?;
        if let Err(failure) = fs::remove_file(&ready)
            && failure.kind() != ErrorKind::NotFound
        {
            return Err(failure.into());
        }
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
        let status = command.status()?;
        let reached_ready = ready.exists();
        if trial && reached_ready {
            fs::rename(home.bin("trial"), home.bin("good"))?;
        }
        if status.code() == Some(RELOAD_EXIT_CODE) {
            continue;
        }
        if !trial || reached_ready {
            return Ok(exit_code(status));
        }
        fs::remove_file(home.bin("trial"))?;
        let log = home.session_log(&session);
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

/// Builds before the first start when there is no binary yet, when the
/// generated project changed, and always from a local checkout. A failed
/// build with an older binary at hand becomes the notice that binary
/// starts with.
fn rebuild(home: &Home) -> Result<Option<String>> {
    let first_run = !["staged", "trial", "good"]
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

/// The binary to start: a newly staged build (moved to `trial`), a trial
/// that never finished, or the last good one. The flag says it is a trial.
fn pick(home: &Home) -> Result<(PathBuf, bool)> {
    let trial = home.bin("trial");
    let staged = home.bin("staged");
    if staged.exists() {
        fs::rename(&staged, &trial)?;
        return Ok((trial, true));
    }
    if trial.exists() {
        return Ok((trial, true));
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
