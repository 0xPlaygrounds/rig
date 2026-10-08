//! The run loop: start the agent, restart it on a new build when it exits
//! with [`RELOAD`], promote a new build once it is ready, and roll back
//! when a new build dies before that.

use std::{
    path::Path,
    process::{Command, ExitCode, ExitStatus},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use crate::{RELOAD, build, home::Home};

/// How often the running agent is checked for exit and readiness.
const POLL: Duration = Duration::from_millis(100);

/// `rig`: build when needed, then run the agent until it quits.
pub fn run(home: &Home) -> ExitCode {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    let session = format!("{secs}-{}", std::process::id());
    let session_dir = home.session(&session);
    let launcher = match std::env::current_exe() {
        Ok(path) => path,
        Err(error) => {
            eprintln!("error: cannot find the rig executable: {error}");
            return ExitCode::FAILURE;
        }
    };
    let mut notice = None;
    if build::needed(home) {
        eprintln!(
            "Building the rig-code agent in {}",
            home.project().display()
        );
        if let Err(failure) = build::build(home) {
            let code = failure.report();
            if !home.current().is_file() {
                return code;
            }
            notice = Some(
                "Building the agent for this launcher failed; running the previous build. \
                 Run `rig build` to see the errors."
                    .to_owned(),
            );
        }
    }
    loop {
        let (binary, is_next) = if home.next().is_file() {
            (home.next(), true)
        } else {
            (home.current(), false)
        };
        let ready = session_dir.join("ready");
        // A stale ready file would promote a binary that never started.
        let _ = std::fs::remove_file(&ready);
        let mut command = Command::new(&binary);
        command
            .env("RIG_LAUNCHER", &launcher)
            .env("RIG_SESSION", &session)
            .env("RIG_DATA_DIR", &home.data)
            .env_remove("RIG_NOTICE");
        if let Some(text) = notice.take() {
            command.env("RIG_NOTICE", text);
        }
        let mut promoted = !is_next;
        let status = match command.spawn() {
            Ok(mut child) => loop {
                match child.try_wait() {
                    Ok(Some(status)) => break Ok(status),
                    Ok(None) => {}
                    Err(error) => break Err(error.to_string()),
                }
                if !promoted && ready.exists() {
                    match std::fs::rename(&binary, home.current()) {
                        Ok(()) => promoted = true,
                        Err(error) => break Err(format!("cannot promote the new build: {error}")),
                    }
                }
                std::thread::sleep(POLL);
            },
            Err(error) => Err(format!("cannot start {}: {error}", binary.display())),
        };
        let status = match status {
            Ok(status) => status,
            Err(error) => {
                eprintln!("error: {error}");
                return ExitCode::FAILURE;
            }
        };
        if status.code() == Some(RELOAD) {
            continue;
        }
        if status.success() {
            return ExitCode::SUCCESS;
        }
        if !promoted {
            // A new build that died before it was ready: drop it.
            let _ = std::fs::remove_file(&binary);
            if home.current().is_file() {
                notice = Some(format!(
                    "The new build crashed during startup ({}). Rolled back to the previous build.",
                    describe(status)
                ));
                continue;
            }
        }
        return crashed(status, &session_dir);
    }
}

/// Report an agent that ended with a failure, and exit the same way.
fn crashed(status: ExitStatus, session_dir: &Path) -> ExitCode {
    eprintln!(
        "rig-code {}. The log is {}",
        describe(status),
        session_dir.join("rig-code.log").display()
    );
    status
        .code()
        .and_then(|code| u8::try_from(code).ok())
        .filter(|code| *code != 0)
        .map_or(ExitCode::FAILURE, ExitCode::from)
}

fn describe(status: ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("exited with code {code}"),
        None => "was killed by a signal".to_owned(),
    }
}
