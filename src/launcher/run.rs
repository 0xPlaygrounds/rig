//! `rig`: build the agent before every start, then run it.
//! [`RELOAD_EXIT_CODE`] restarts it on the staged build. A new build runs as this launcher's own trial
//! and becomes the good build once it signals ready; one that exits before
//! that is rolled back to the last good build. Several launchers can share
//! one `RIG_HOME`: building and staging take the root's lock, and a
//! `/reload` build, its trial and the ready file are per session.
//!
//! A session belongs to the directory it runs in. Until it quits cleanly
//! (exit code 0), `resume/<hash of the directory>` names it, so after a
//! crash, a kill or a closed terminal the next `rig` there resumes it where
//! it stopped, unless another launcher still runs it. The agent's `/new`
//! and `/resume` leave a
//! [`SessionDir::switch`] file and exit with the reload code; the launcher
//! then runs that session instead, in its own directory. A session left
//! without a message, by a clean quit or a switch, leaves no directory.
//!
//! The agent's arguments ([`Invocation`]) pass through unchanged. A
//! headless run (`--print`) never becomes the session its directory
//! resumes, and is not restarted on the reload code.

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
    /// The working directory's session that did not quit cleanly, else a
    /// new one.
    Default,
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
        let (claimed, claim_notice) = claim(home, start, here.as_deref(), !headless)?;
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
        remove_if_present(&ready)?;
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
                (claimed, notice) = switch(home, claimed, target, !headless)?;
            }
            continue;
        }
        if !rejected && (status.success() || headless) {
            if status.success() {
                claimed.forget()?;
                claimed.discard_if_unsaved(home);
            }
            return Ok(exit_code(status));
        }
        if !headless {
            restore_terminal();
        }
        if !rejected {
            eprintln!(
                "The agent stopped ({status}). Run `rig` here again to resume the session \
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
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(failure) if failure.kind() == ErrorKind::NotFound => return Ok(None),
        Err(failure) => return Err(failure.into()),
    };
    fs::remove_file(&path)?;
    let text = text.trim();
    if text.is_empty() {
        return Ok(Some(Start::New));
    }
    Ok(Some(text.parse().map(Start::Resume)?))
}

/// Leaves the `current` session for `target`, which the agent asked for.
/// The current session quit cleanly, so its directory no longer resumes
/// it, and a build `/reload` staged for it goes with the switch. When the
/// target cannot be run, the current session carries on, told why.
fn switch(
    home: &Home,
    current: Claimed,
    target: Start,
    mark: bool,
) -> Result<(Claimed, Option<String>)> {
    let _lock = home::lock(home)?;
    let here = current.directory.clone();
    match claim(home, target, here.as_deref(), mark) {
        Ok((next, notice)) => {
            current.forget()?;
            current.discard_if_unsaved(home);
            match fs::rename(home.staged_for(&current.id), home.staged_for(&next.id)) {
                Err(failure) if failure.kind() != ErrorKind::NotFound => {
                    return Err(failure.into());
                }
                _ => {}
            }
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
    /// That directory's resume marker.
    marker: Option<PathBuf>,
    _lock: File,
}

impl Claimed {
    /// After a clean quit, the next `rig` in this directory starts a new
    /// session: the marker goes, when it still names this one.
    fn forget(&self) -> Result<()> {
        match &self.marker {
            Some(marker) if names(marker, &self.id) => remove_if_present(marker),
            _ => Ok(()),
        }
    }
}

impl Claimed {
    /// Removes the session's directory when no agent logged a message in
    /// it, such as a `/new` session left at once: a session with nothing
    /// to resume leaves nothing behind. Call it once the agent exited
    /// cleanly or switched away. Best effort: a directory left behind has
    /// no agent log, so neither `/resume` nor the launcher resumes it.
    fn discard_if_unsaved(&self, home: &Home) {
        let session = home.session(&self.id);
        if !session.is_saved() {
            fs::remove_dir_all(session.path()).ok();
        }
    }
}

/// Whether `marker` names `session`.
fn names(marker: &Path, session: &SessionId) -> bool {
    fs::read_to_string(marker).is_ok_and(|named| named.trim() == session.as_str())
}

/// The session a marker file names, when it has an agent log.
fn named_session(home: &Home, marker: Option<&Path>) -> Option<SessionId> {
    fs::read_to_string(marker?)
        .ok()
        // A marker that is not a session id cannot name a path elsewhere.
        .and_then(|id| id.trim().parse::<SessionId>().ok())
        .filter(|id| home.session(id).is_saved())
}

/// The session to run for `start` from the working directory `here`, held
/// locked, and the line the agent shows about it. Its directory's marker
/// then names it, unless the resume marker names a session another launcher
/// runs. A headless run passes `mark` false: its directory's resume marker
/// is left alone. Call it holding [`home::lock`].
fn claim(
    home: &Home,
    start: Start,
    here: Option<&Path>,
    mark: bool,
) -> Result<(Claimed, Option<String>)> {
    let resume_marker = here.map(|here| home.resume_marker(here));
    let (id, lock, notice) = match start {
        Start::Default => match named_session(home, resume_marker.as_deref()) {
            Some(id) => match home::hold_session(home, &id)? {
                Some(lock) => (
                    id,
                    lock,
                    Some("Resumed this directory's session where it stopped.".to_owned()),
                ),
                None => fresh(home)?,
            },
            None => fresh(home)?,
        },
        Start::Resume(id) => {
            if !home.session(&id).is_saved() {
                return Err(format!("session {id} has no saved conversation").into());
            }
            let lock = home::hold_session(home, &id)?
                .ok_or_else(|| format!("session {id} is open in another rig"))?;
            (id, lock, Some("Resumed the session.".to_owned()))
        }
        Start::New => fresh(home)?,
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
    let marker = directory
        .as_deref()
        .filter(|_| mark)
        .map(|directory| home.resume_marker(directory));
    if let Some(marker) = &marker
        && !held_elsewhere(home, marker, &id)?
    {
        write_atomic(marker, id.as_str())?;
    }
    Ok((
        Claimed {
            id,
            directory,
            marker,
            _lock: lock,
        },
        notice,
    ))
}

/// A new session, held locked.
fn fresh(home: &Home) -> Result<(SessionId, File, Option<String>)> {
    let id = SessionId::generate();
    let lock =
        home::hold_session(home, &id)?.ok_or_else(|| format!("session {id} is already running"))?;
    Ok((id, lock, None))
}

/// Whether `marker` names a session other than `own` that another launcher
/// runs, which the marker must keep naming.
fn held_elsewhere(home: &Home, marker: &Path, own: &SessionId) -> Result<bool> {
    match named_session(home, Some(marker)) {
        Some(id) if id != *own => Ok(home::hold_session(home, &id)?.is_none()),
        _ => Ok(false),
    }
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

fn remove_if_present(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Err(failure) if failure.kind() != ErrorKind::NotFound => Err(failure.into()),
        _ => Ok(()),
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
        match fs::rename(staged, trial) {
            Ok(()) => return Ok(Binary::Trial(trial.to_path_buf())),
            Err(failure) if failure.kind() != ErrorKind::NotFound => return Err(failure.into()),
            Err(_) => {}
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
