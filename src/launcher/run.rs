//! `rig`: build the agent before every start, then run it.
//! [`RELOAD_EXIT_CODE`] restarts it on the staged build. A new build runs as this launcher's own trial
//! and becomes the good build once it signals ready; one that exits before
//! that is rolled back to the last good build. Several launchers can share
//! one `RIG_HOME`: building and staging take the root's lock, and a
//! `/reload` build, its trial and the ready file are per session.
//!
//! `rig` starts a new session, and `rig --resume <id>` an earlier one, in
//! the directory it ran in. The agent's `/new` and `/resume` leave a
//! [`SessionDir::switch`] file and exit with the reload code; the launcher
//! then runs that session instead, in its own directory. A session left
//! without a message, by a clean quit or a switch, leaves no directory.
//!
//! The agent's arguments ([`Invocation`]) pass through unchanged. A
//! headless run (`--print`) is not restarted on the reload code.

use std::fs::{self, File};
use std::io::{ErrorKind, IsTerminal};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, ExitStatus};
use std::time::Duration;

use rig::harness_protocol::{Home, Invocation, RELOAD_EXIT_CODE, SessionDir, SessionId, env};

use super::build::{self, BuildFailure, Staging};
use super::{Result, home};

/// How often the launcher looks for the ready file while the agent runs.
const POLL: Duration = Duration::from_millis(100);

/// Which session `rig` runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Start {
    /// The named session, in the directory it ran in.
    Resume(SessionId),
    /// A new session.
    New,
}

/// Runs the agent until it exits with anything but the reload code, and
/// returns its exit code.
pub fn run(home: &Home, start: Start, invocation: &Invocation) -> Result<ExitCode> {
    let here = std::env::current_dir().ok();
    let headless = invocation.is_headless();
    let (mut claimed, mut notice, mut build_failure) = {
        let _lock = home::lock(home)?;
        // Before claiming, so a resumed session's leftover builds from its
        // dead launcher are removed too.
        home::sweep(home)?;
        let (claimed, claim_notice) = claim(home, start, here.as_deref())?;
        let failure = rebuild(home, &claimed.id)?;
        let built = failure.as_ref().map(|failure| {
            format!("Building the agent failed, so the previous build is running: {failure}")
        });
        let notice: Vec<String> = claim_notice.into_iter().chain(built).collect();
        (
            claimed,
            (!notice.is_empty()).then(|| notice.join("\n")),
            failure.map(|failure| failure.details()),
        )
    };
    let launcher = std::env::current_exe()?;
    loop {
        let session = claimed.id.clone();
        let trial = home.trial_for(&session);
        let directory = home.session(&session);
        let ready = directory.ready();
        let log = directory.log();
        let binary = pick(home, &session, &trial)?;
        absent(fs::remove_file(&ready))?;
        let mut command = Command::new(binary.path());
        command
            .args(invocation.to_args())
            .env(env::HOME, home.root())
            .env(env::SESSION, session.as_str())
            .env(env::LAUNCHER, &launcher)
            .env_remove(env::NOTICE)
            .env_remove(env::BUILD_FAILURE);
        if let Some(working) = &claimed.directory {
            command.current_dir(working);
        }
        if let Some(notice) = notice.take() {
            command.env(env::NOTICE, notice);
        }
        // Only the first start follows the failed build.
        if let Some(failure) = build_failure.take() {
            command.env(env::BUILD_FAILURE, failure);
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
                if let Binary::Trial(trial) = &binary {
                    promoted = fs::rename(trial, home.good()).map_err(Into::into);
                }
            }
            match status {
                Some(status) => break status,
                None => std::thread::sleep(POLL),
            }
        };
        promoted?;
        let rejected = matches!(binary, Binary::Trial(_)) && !started;
        if rejected {
            build::reject(home, &trial)?;
            fs::remove_file(&trial)?;
        }
        let reload = status.code() == Some(i32::from(RELOAD_EXIT_CODE));
        if reload && !headless {
            if let Some(target) = take_switch(&directory)? {
                (claimed, notice) = switch(home, claimed, target)?;
            }
            continue;
        }
        if !rejected && (status.success() || headless) {
            if status.success() {
                claimed.discard_if_unsaved(home);
            }
            return Ok(exit_code(status));
        }
        if !headless {
            restore_terminal();
        }
        if !rejected {
            eprintln!(
                "The agent stopped ({status}). `rig --resume {session}` resumes the session \
                 where it stopped. Log: {}",
                log.display()
            );
            return Ok(exit_code(status));
        }
        if !home.good().exists() {
            eprintln!(
                "The new build stopped during startup ({status}) and there is no previous \
                 build to roll back to; the next `rig` tries it again. Log: {}",
                log.display()
            );
            return Ok(exit_code(status));
        }
        let message = format!(
            "The new build stopped during startup ({status}); rolled back to the previous \
             build. `rig build` tries it again. Log: {}",
            log.display()
        );
        eprintln!("{message}");
        notice = Some(message);
    }
}

/// The session the agent asked to run next in `directory`'s switch file,
/// which is removed: a session id, or a new session when the file is empty.
fn take_switch(directory: &SessionDir) -> Result<Option<Start>> {
    let path = directory.switch();
    let Some(text) = absent(fs::read_to_string(&path))? else {
        return Ok(None);
    };
    fs::remove_file(&path)?;
    let text = text.trim();
    if text.is_empty() {
        return Ok(Some(Start::New));
    }
    Ok(Some(text.parse().map(Start::Resume)?))
}

/// Leaves the `current` session for `target`, which the agent asked for.
/// A build `/reload` staged for it goes with the switch. When the target
/// cannot be run, the current session carries on, told why.
fn switch(home: &Home, current: Claimed, target: Start) -> Result<(Claimed, Option<String>)> {
    let _lock = home::lock(home)?;
    let here = current.directory.clone();
    match claim(home, target, here.as_deref()) {
        Ok((next, notice)) => {
            current.discard_if_unsaved(home);
            absent(fs::rename(
                home.staged_for(&current.id),
                home.staged_for(&next.id),
            ))?;
            Ok((next, notice))
        }
        Err(failure) => Ok((
            current,
            Some(format!("Could not switch sessions: {failure}.")),
        )),
    }
}

/// The session this launcher runs, held locked while it lives.
struct Claimed {
    id: SessionId,
    /// The directory the session runs in, when known and still there.
    directory: Option<PathBuf>,
    _lock: File,
}

impl Claimed {
    /// Removes the session's directory when no agent logged a message in
    /// it, such as a `/new` session left at once: a session with nothing
    /// to resume leaves nothing behind. Call it once the agent exited
    /// cleanly or switched away. Best effort: a directory left behind has
    /// no agent log, so `/resume` does not list it.
    fn discard_if_unsaved(&self, home: &Home) {
        let session = home.session(&self.id);
        if !session.is_saved() {
            fs::remove_dir_all(session.path()).ok();
        }
    }
}

/// The session to run for `start` from the working directory `here`, held
/// locked, and the line the agent shows about it. Call it holding
/// [`home::lock`].
fn claim(home: &Home, start: Start, here: Option<&Path>) -> Result<(Claimed, Option<String>)> {
    let (id, lock, notice) = match start {
        Start::Resume(id) => {
            if !home.session(&id).is_saved() {
                return Err(format!("session {id} has no saved conversation").into());
            }
            let lock = home::hold_session(home, &id)?
                .ok_or_else(|| format!("session {id} is open in another rig"))?;
            (id, lock, Some("Resumed the session.".to_owned()))
        }
        Start::New => {
            let id = SessionId::generate();
            let lock = home::hold_session(home, &id)?
                .ok_or_else(|| format!("session {id} is already running"))?;
            (id, lock, None)
        }
    };
    let session = home.session(&id);
    let directory = match session.working_directory() {
        Some(directory) => Some(directory),
        None => {
            if let Some(here) = here {
                write_atomic(&session.directory(), &here.to_string_lossy())?;
            }
            here.map(Path::to_path_buf)
        }
    };
    Ok((
        Claimed {
            id,
            directory,
            _lock: lock,
        },
        notice,
    ))
}

/// Writes `text` to `path` through a temporary file, creating its
/// directory.
fn write_atomic(path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let written = path.with_extension("tmp");
    fs::write(&written, text)?;
    fs::rename(&written, path)?;
    Ok(())
}

/// Puts the terminal back after an agent that could not restore it: leaves
/// the alternate screen and, on a terminal, raw mode.
fn restore_terminal() {
    eprint!("\x1b[?1049l\x1b[?25h");
    if std::io::stdin().is_terminal() {
        Command::new("stty").arg("sane").status().ok();
    }
}

/// `result`, with a missing file as `None`.
fn absent<T>(result: std::io::Result<T>) -> std::io::Result<Option<T>> {
    match result {
        Err(failure) if failure.kind() == ErrorKind::NotFound => Ok(None),
        result => result.map(Some),
    }
}

/// Builds before every start, staging for this launcher; cargo does no
/// work when nothing changed. With no good build yet, the build is staged
/// even if it was rejected before, so each start retries it. A failed build
/// with an older binary at hand is returned, for that binary to start with.
fn rebuild(home: &Home, session: &SessionId) -> Result<Option<BuildFailure>> {
    let first_run = !home.staged().exists() && !home.good().exists();
    eprintln!(
        "Building the rig agent{}…",
        if first_run { " (first run)" } else { "" }
    );
    let staging = if first_run {
        Staging::Always
    } else {
        Staging::OnlyNew
    };
    match build::compile(home, &home.staged_for(session), staging) {
        Ok(()) => Ok(None),
        Err(failure) if first_run => Err(failure.into()),
        Err(failure) => {
            eprintln!("error: {failure}; starting the previous build");
            Ok(Some(failure))
        }
    }
}

/// The binary a launcher starts.
enum Binary {
    /// A staged build, claimed as this launcher's trial until it is ready.
    Trial(PathBuf),
    /// The last build that started up.
    Good(PathBuf),
}

impl Binary {
    fn path(&self) -> &Path {
        match self {
            Self::Trial(path) | Self::Good(path) => path,
        }
    }
}

/// The binary to start: a build staged for this launcher or for any, claimed
/// as its `trial`, or else the last good one.
fn pick(home: &Home, session: &SessionId, trial: &Path) -> Result<Binary> {
    for staged in [home.staged_for(session), home.staged()] {
        // The rename claims the staged build for this launcher alone.
        if absent(fs::rename(staged, trial))?.is_some() {
            return Ok(Binary::Trial(trial.to_path_buf()));
        }
    }
    let good = home.good();
    if good.exists() {
        return Ok(Binary::Good(good));
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
