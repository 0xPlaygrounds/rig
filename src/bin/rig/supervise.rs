//! Running the agent: start the newest binary, promote it once it reports
//! ready, restart it on the reload exit code, and roll back to the last
//! working binary when a new one crashes during startup.

use std::fs::OpenOptions;
use std::path::Path;
use std::process::{Command, ExitStatus, Stdio};
use std::time::Duration;

use crate::Failure;
use crate::dirs::Dirs;

/// The agent asks to be restarted with the newest build (`EX_TEMPFAIL`).
const RELOAD: i32 = 75;
/// How often the running agent is checked.
const POLL: Duration = Duration::from_millis(50);
/// Lines of the agent's output shown when it cannot start at all.
const TAIL_LINES: usize = 20;

/// Runs the agent until it quits, and returns the exit code for `rig`.
/// `notice` is shown by the first agent started, through `RIG_NOTICE`.
pub(crate) fn supervise(dirs: &Dirs, mut notice: Option<String>) -> Result<u8, Failure> {
    let launcher = std::env::current_exe()
        .map_err(|error| Failure::io("cannot find the rig executable", error))?;
    let log = dirs.stdio_log();
    crate::dirs::rotate(&log);
    // The agent logs to agent.log; only what bypasses its logger, such as
    // an early panic, lands in agent-stdio.log.
    let logs = dirs.data.join("logs");
    let details = format!("{} (agent.log, agent-stdio.log)", logs.display());
    loop {
        let candidate = dirs.candidate_bin();
        let on_candidate = candidate.exists();
        let exe = if on_candidate {
            candidate
        } else {
            dirs.current_bin()
        };
        let ready = dirs.ready_file();
        remove(&ready)?;
        let mut command = Command::new(&exe);
        command
            .env("RIG_LAUNCHER", &launcher)
            .env("RIG_DATA_DIR", &dirs.data)
            .env("RIG_READY_FILE", &ready)
            .stdin(Stdio::inherit());
        match notice.take() {
            Some(text) => command.env("RIG_NOTICE", text),
            None => command.env_remove("RIG_NOTICE"),
        };
        let (status, was_ready) = run(command, &log, &ready, on_candidate.then_some(dirs))?;
        match status.code() {
            Some(0) => return Ok(0),
            Some(RELOAD) => continue,
            _ => {}
        }
        let what = match status.code() {
            Some(code) => format!("exit {code}"),
            None => status.to_string(),
        };
        if was_ready {
            eprintln!(
                "rig: the agent stopped ({what}); run rig again to resume the session. Logs: \
                 {details}"
            );
            return Ok(status
                .code()
                .and_then(|code| u8::try_from(code).ok())
                .unwrap_or(1));
        }
        if on_candidate && dirs.current_bin().exists() {
            remove(&dirs.candidate_bin())?;
            // The same build is not staged again until the source changes.
            if let Err(error) = std::fs::rename(dirs.staged_stamp(), dirs.rejected_stamp()) {
                eprintln!("rig: cannot record the rejected build: {error}");
            }
            let message = format!(
                "the new build crashed during startup ({what}); rolled back to the previous \
                 build. Logs: {details}"
            );
            eprintln!("rig: {message}");
            notice = Some(message);
            continue;
        }
        eprintln!("rig: the agent could not start ({what}). Logs: {details}");
        for file in [logs.join("agent.log"), log] {
            eprintln!("--- the end of {}:\n{}", file.display(), tail(&file));
        }
        return Ok(1);
    }
}

/// Runs `command` with its stdout and stderr appended to `log`, waits for
/// it, and says whether it reported ready first. When `promote` is given,
/// the candidate binary becomes the current one once it is ready.
fn run(
    mut command: Command,
    log: &Path,
    ready: &Path,
    promote: Option<&Dirs>,
) -> Result<(ExitStatus, bool), Failure> {
    let failed = |error| Failure::io(format!("cannot open {}", log.display()), error);
    if let Some(parent) = log.parent() {
        std::fs::create_dir_all(parent).map_err(failed)?;
    }
    let output = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log)
        .map_err(failed)?;
    let mut child = command
        .stdout(output.try_clone().map_err(failed)?)
        .stderr(output)
        .spawn()
        .map_err(|error| Failure::io("cannot start the agent", error))?;
    let mut was_ready = false;
    loop {
        let exited = child
            .try_wait()
            .map_err(|error| Failure::io("cannot wait for the agent", error))?;
        if !was_ready && ready.exists() {
            was_ready = true;
            if let Some(dirs) = promote {
                std::fs::rename(dirs.candidate_bin(), dirs.current_bin())
                    .map_err(|error| Failure::io("cannot promote the new build", error))?;
            }
        }
        if let Some(status) = exited {
            return Ok((status, was_ready));
        }
        std::thread::sleep(POLL);
    }
}

/// Deletes `path` if it exists.
fn remove(path: &Path) -> Result<(), Failure> {
    match std::fs::remove_file(path) {
        Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(Failure::io(
            format!("cannot remove {}", path.display()),
            error,
        )),
        _ => Ok(()),
    }
}

/// The last lines of `log`.
fn tail(log: &Path) -> String {
    let text = std::fs::read_to_string(log).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    lines
        .get(lines.len().saturating_sub(TAIL_LINES)..)
        .unwrap_or_default()
        .join("\n")
}
