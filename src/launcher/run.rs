//! `rig`: build the agent, then run it until it exits for good.
//!
//! The agent writes `<session>/ready` after its first frame. A binary that
//! got that far and then exited cleanly or for a reload is promoted to
//! `bin/good`. Exit code 75 asks for the binary `/reload` just built; a
//! binary that crashes before the marker is replaced by `bin/good`, with a
//! notice saying so.

use std::{
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, ExitCode, ExitStatus},
    time::{SystemTime, UNIX_EPOCH},
};

use super::{Error, Result, build, paths::Paths};

/// The exit code with which the agent asks for the freshly built binary.
const RELOAD_EXIT_CODE: i32 = 75;

/// Which binary is running.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Binary {
    Current,
    Good,
}

/// Build the agent and run it on a new session, or on the latest one with
/// `resume`.
pub fn run(paths: &Paths, jobs: Option<u32>, resume: bool) -> Result<ExitCode> {
    let session = session(paths, resume)?;
    let mut notice = None;
    let mut running = Binary::Current;
    eprintln!("rig: building the agent in {}", paths.project.display());
    match build::build(paths, jobs) {
        Ok(()) => install(paths)?,
        Err(error) if paths.good().is_file() => {
            eprintln!("rig: {error}; starting the last working binary");
            running = Binary::Good;
            notice = Some(format!(
                "The agent failed to build ({error}), so the last working binary is running. \
                 Run `rig build` in a terminal to see the errors."
            ));
        }
        Err(error) => return Err(error),
    }
    supervise(paths, jobs, &session, running, notice).inspect_err(|_| leave_alternate_screen())
}

/// Run the agent, starting the next binary after each reload, until it exits
/// for good.
fn supervise(
    paths: &Paths,
    jobs: Option<u32>,
    session: &Path,
    mut running: Binary,
    mut notice: Option<String>,
) -> Result<ExitCode> {
    let ready = session.join("ready");
    let launcher = env::current_exe()?;
    loop {
        let binary = match running {
            Binary::Current => paths.current(),
            Binary::Good => paths.good(),
        };
        let _ = fs::remove_file(&ready);
        let mut agent = Command::new(&binary);
        agent
            .env("RIG_CODE_SESSION_DIR", session)
            .env("RIG_CODE_LAUNCHER", &launcher);
        match notice.take() {
            Some(text) => agent.env("RIG_CODE_NOTICE", text),
            None => agent.env_remove("RIG_CODE_NOTICE"),
        };
        if let Some(jobs) = jobs {
            agent.env("CARGO_BUILD_JOBS", jobs.to_string());
        }
        let status = agent
            .status()
            .map_err(|error| Error(format!("cannot start {}: {error}", binary.display())))?;
        let started = ready.is_file();
        let clean = matches!(status.code(), Some(0 | RELOAD_EXIT_CODE));
        if started && clean && running == Binary::Current {
            promote(paths)?;
        }
        match status.code() {
            Some(0) => return Ok(ExitCode::SUCCESS),
            Some(RELOAD_EXIT_CODE) => {
                install(paths)?;
                running = Binary::Current;
            }
            _ if !started && running == Binary::Current && paths.good().is_file() => {
                running = Binary::Good;
                notice = Some(format!(
                    "The new build crashed during startup ({status}). Rolled back to the last \
                     working binary."
                ));
            }
            _ => {
                leave_alternate_screen();
                let log = session.join("agent.log");
                if started {
                    eprintln!(
                        "rig: the agent stopped ({status}). The session was saved after the \
                         last turn; `rig --resume` continues it. Log: {}",
                        log.display()
                    );
                } else {
                    eprintln!(
                        "rig: the agent crashed during startup ({status}). Log: {}",
                        log.display()
                    );
                }
                return Ok(exit_code(status));
            }
        }
    }
}

/// The session directory: a new one, or with `resume` the one named in
/// `sessions/latest`, which then names this one.
fn session(paths: &Paths, resume: bool) -> Result<PathBuf> {
    let latest = paths.sessions.join("latest");
    let id = if resume {
        fs::read_to_string(&latest)
            .map(|id| id.trim().to_owned())
            .ok()
            .filter(|id| !id.is_empty())
            .ok_or_else(|| Error("there is no session to resume".into()))?
    } else {
        session_id()
    };
    let dir = paths.sessions.join(&id);
    fs::create_dir_all(&dir)?;
    fs::write(latest, &id)?;
    Ok(dir)
}

/// `YYYYMMDD-HHMMSS-<pid>`, in UTC.
fn session_id() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or_default();
    let (days, time) = (seconds / 86_400, seconds % 86_400);
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let shifted = days + 719_468;
    let era = shifted / 146_097;
    let day_of_era = shifted % 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + u64::from(month <= 2);
    format!(
        "{year:04}{month:02}{day:02}-{:02}{:02}{:02}-{}",
        time / 3600,
        time / 60 % 60,
        time % 60,
        std::process::id()
    )
}

/// Copy the binary cargo built to `bin/current`. It goes through a new file
/// and a rename, because `current` may be a hard link to `good`.
fn install(paths: &Paths) -> Result<()> {
    fs::create_dir_all(&paths.bin)?;
    replace(&paths.built(), &paths.current())
}

/// Make `bin/current` the last working binary.
fn promote(paths: &Paths) -> Result<()> {
    let (current, good) = (paths.current(), paths.good());
    let staged = good.with_extension("new");
    let _ = fs::remove_file(&staged);
    if fs::hard_link(&current, &staged).is_ok() {
        fs::rename(&staged, &good)?;
        return Ok(());
    }
    replace(&current, &good)
}

fn replace(from: &Path, to: &Path) -> Result<()> {
    let staged = to.with_extension("new");
    fs::copy(from, &staged)
        .map_err(|error| Error(format!("cannot copy {}: {error}", from.display())))?;
    fs::rename(&staged, to)?;
    Ok(())
}

/// A reload keeps the terminal on the alternate screen for the next binary;
/// leave it before printing.
fn leave_alternate_screen() {
    let mut stdout = std::io::stdout();
    let _ = stdout.write_all(b"\x1b[?1049l");
    let _ = stdout.flush();
}

fn exit_code(status: ExitStatus) -> ExitCode {
    status
        .code()
        .and_then(|code| u8::try_from(code).ok())
        .filter(|code| *code != 0)
        .map_or(ExitCode::FAILURE, ExitCode::from)
}
