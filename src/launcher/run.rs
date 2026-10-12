//! `rig`: build the agent before every start, then run it.
//! [`RELOAD_EXIT_CODE`] restarts it on the staged build. A new build runs as this launcher's own trial
//! and becomes the good build once it signals ready, in the launcher's
//! [`RunDir`]; one that exits before that is rolled back to the last good
//! build. Several launchers can share one `RIG_HOME`: building and staging
//! take the root's lock, a `/reload` build and its trial are per session,
//! and the ready file is per launcher.
//!
//! `rig` starts a new session, and `rig --resume <id>` an earlier one, in
//! the directory it ran in. The agent's `/new` and `/resume` leave a
//! [`SessionDir::switch`] file and exit with the reload code; the launcher
//! then runs that session instead, in its own directory. A session left
//! without a message, by a clean quit or a switch, leaves no directory.
//!
//! The command line is read once the agent is built, against what its
//! plugins add ([`Described::invocation`]), and the agent gets it in its
//! canonical form ([`Invocation`]). A headless run (`--print`, stdin piped
//! in, which the agent answers as `--print` does, or a plugin's command
//! that does not take the terminal) is not restarted on the reload code,
//! nor run again on the previous build when its new build is rolled back;
//! its exit code and output pass through. Before each start with stdin on
//! a terminal, whatever the front, the launcher asks for each secret the
//! build's plugins want that the environment does not hold
//! ([`ask_secret`]); only the agent's process gets the answers.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{ErrorKind, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, ExitStatus, Stdio};
use std::time::Duration;

use rig::harness_protocol::{
    Described, Home, Invocation, RELOAD_EXIT_CODE, RunDir, SecretInfo, SessionDir, SessionId, env,
    secret_variable,
};

use super::build::{self, BuildFailure, Staging, child};
use super::{Result, home};

/// How often the launcher looks for the ready file while the agent runs.
const POLL: Duration = Duration::from_millis(100);

/// The exit code of a run the user stopped with Ctrl+C at a prompt.
const INTERRUPTED: u8 = 130;

/// Which session `rig` runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Start {
    /// The named session, in the directory it ran in.
    Resume(SessionId),
    /// A new session.
    New,
}

/// Builds the agent, reads the command line `args` against it, runs it
/// until it exits with anything but the reload code, and returns its exit
/// code: 2 for a command line the build does not take.
pub fn run(home: &Home, start: Start, args: &[String]) -> Result<ExitCode> {
    let here = std::env::current_dir().ok();
    let terminal = std::io::stdin().is_terminal();
    // Before the build, only whether `-p` is there: the type-ahead guard.
    let print = args.iter().any(|arg| arg == "-p" || arg == "--print");
    let resumed = matches!(start, Start::Resume(_));
    let (run, claimed, notice, build_failure, report) = {
        let _typing = (terminal && !print).then(TypeAhead::hold).flatten();
        let _lock = home::lock(home)?;
        // Before claiming, so a resumed session's leftover builds from its
        // dead launcher are removed too.
        home::sweep(home)?;
        let run = home::hold_run(home)?;
        let (claimed, claim_notice) = claim(home, start, here.as_deref())?;
        let failure = rebuild(home, &claimed.id)?;
        let built = failure.as_ref().map(|failure| {
            format!("Building the agent failed, so the previous build is running: {failure}")
        });
        let notice: Vec<String> = claim_notice.into_iter().chain(built).collect();
        let report = report(home, &claimed.id);
        (
            run,
            claimed,
            (!notice.is_empty()).then(|| notice.join("\n")),
            failure.map(|failure| failure.details()),
            report,
        )
    };
    let invocation = report
        .invocation(args)
        .and_then(|invocation| match &invocation.command {
            Some(command) if resumed || invocation.print.is_some() => Err(format!(
                "the command `{command}` takes neither --resume nor --print"
            )),
            _ => Ok(invocation),
        });
    let invocation = match invocation {
        Ok(invocation) => invocation,
        Err(why) => {
            claimed.discard_if_unsaved(home);
            eprintln!("rig: {why}");
            return Ok(ExitCode::from(2));
        }
    };
    let command = invocation
        .command
        .as_deref()
        .and_then(|command| report.command(command));
    let headless =
        invocation.is_headless() || !terminal || command.is_some_and(|command| !command.terminal);
    launch(
        home,
        run.dir(),
        &invocation,
        headless,
        claimed,
        notice,
        build_failure,
    )
}

/// What the plugins of the build the launcher of `session` starts next add
/// to the command line; nothing when there is no build or it cannot say.
fn report(home: &Home, session: &SessionId) -> Described {
    let Some(binary) = next_build(home, session) else {
        return Described::default();
    };
    build::described(home, &binary).unwrap_or_else(|failure| {
        eprintln!("warning: the agent build does not say what its plugins add: {failure}");
        Described::default()
    })
}

/// The build the launcher of `session` starts next, as [`pick`] picks it,
/// without claiming it.
fn next_build(home: &Home, session: &SessionId) -> Option<PathBuf> {
    [home.staged_for(session), home.staged(), home.good()]
        .into_iter()
        .find(|binary| binary.exists())
}

/// Starts the agent of `claimed` with the build [`pick`] finds, again on
/// the reload code, and returns its exit code. `notice` and
/// `build_failure` go to the first start.
fn launch(
    home: &Home,
    run: &RunDir,
    invocation: &Invocation,
    headless: bool,
    mut claimed: Claimed,
    mut notice: Option<String>,
    mut build_failure: Option<String>,
) -> Result<ExitCode> {
    let launcher = std::env::current_exe()?;
    // The secrets asked for, for every restart: asked whenever someone is
    // at the terminal, whatever the front, so a command typed there can be
    // asked too; a piped run passes only what its environment has.
    let ask = std::io::stdin().is_terminal();
    let mut answers: BTreeMap<String, String> = BTreeMap::new();
    // Whether the agent's full-screen view kept its screen across a
    // restart, so what the agent said before it took the terminal is gone
    // once the launcher takes the terminal back.
    let mut kept_screen = false;
    loop {
        let session = claimed.id.clone();
        let trial = home.trial_for(&session);
        let directory = home.session(&session);
        let ready = run.ready();
        if ask {
            for secret in report(home, &session).secrets {
                let variable = secret_variable(&secret.name);
                if std::env::var_os(&variable).is_some() || answers.contains_key(&variable) {
                    continue;
                }
                let Some(answer) = ask_secret(&secret)? else {
                    claimed.discard_if_unsaved(home);
                    return Ok(ExitCode::from(INTERRUPTED));
                };
                answers.insert(variable, answer);
            }
        }
        let binary = pick(home, &session, &trial)?;
        absent(fs::remove_file(&ready))?;
        absent(fs::remove_file(run.refusal()))?;
        let mut command = Command::new(binary.path());
        command
            .args(invocation.to_args())
            .env(env::HOME, home.root())
            .env(env::SESSION, session.as_str())
            .env(env::LAUNCHER, &launcher)
            .env(env::RUN, run.path())
            .env_remove(env::NOTICE)
            .env_remove(env::BUILD_FAILURE)
            .envs(&answers);
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
        // A sessionless run that failed moved its log here, out of the
        // session directory it removed.
        let log = Some(run.log())
            .filter(|log| log.exists())
            .unwrap_or_else(|| directory.log());
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
            kept_screen = true;
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
            let refusal = fs::read_to_string(run.refusal()).ok();
            // A refused session would be refused again.
            let resumable = directory.is_saved() && refusal.is_none();
            let lost = refusal.filter(|_| kept_screen);
            let said = stopped(status, &session, resumable, lost.as_deref(), &log);
            eprintln!("{said}");
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
        // A one-shot run is not run twice: its output is already out.
        if headless {
            return Ok(exit_code(status));
        }
        kept_screen = false;
        notice = Some(message);
    }
}

/// What the launcher says when the agent of `session` stopped with a
/// failure: the `refusal` the agent said where it is no longer seen, where
/// its log is and, when the session can be resumed, how.
fn stopped(
    status: ExitStatus,
    session: &SessionId,
    resumable: bool,
    refusal: Option<&str>,
    log: &Path,
) -> String {
    let refusal = refusal.unwrap_or_default();
    let resume = if resumable {
        format!(" `rig --resume {session}` resumes the session where it stopped.")
    } else {
        String::new()
    };
    format!(
        "{refusal}The agent stopped ({status}).{resume} Log: {}",
        log.display()
    )
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
            let (id, _) = home.new_session()?;
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
        child("stty").arg("sane").status().ok();
    }
}

/// The terminal's line mode, put aside while the launcher waits for and
/// runs a build before an interactive start, and given back when dropped,
/// before the agent starts. Meanwhile the terminal neither echoes what is
/// typed over the build's output nor turns a typed Enter (CR) into a line
/// feed, which the agent, in raw mode, would read as Ctrl+J.
struct TypeAhead {
    /// The mode `stty -g` printed.
    saved: String,
}

impl TypeAhead {
    /// Puts the mode aside, or `None` when `stty` cannot.
    fn hold() -> Option<Self> {
        Self::set(&["-icrnl", "-echo"])
    }

    /// Puts the mode aside and sets `mode` (`stty` arguments), or `None`
    /// when `stty` cannot.
    fn set(mode: &[&str]) -> Option<Self> {
        let output = child("stty")
            .arg("-g")
            .stdin(Stdio::inherit())
            .stderr(Stdio::null())
            .output()
            .ok()
            .filter(|output| output.status.success())?;
        let saved = String::from_utf8(output.stdout).ok()?.trim().to_owned();
        let set = child("stty").args(mode).status();
        set.is_ok_and(|status| status.success())
            .then_some(Self { saved })
    }
}

impl Drop for TypeAhead {
    fn drop(&mut self) {
        child("stty").arg(&self.saved).status().ok();
    }
}

/// Asks on the terminal for `secret`, without echo, and returns the
/// answer; `None` when the user pressed Ctrl+C. The terminal reads byte by
/// byte with its signals off, so Ctrl+C arrives as a byte and the mode is
/// given back, echo included, whatever the answer.
fn ask_secret(secret: &SecretInfo) -> Result<Option<String>> {
    let mut stderr = std::io::stderr();
    write!(stderr, "{} ({}): ", secret.help, secret.name)?;
    stderr.flush()?;
    let mode = TypeAhead::set(&["-echo", "-icanon", "-isig"]);
    let answer = read_secret(&mut std::io::stdin().lock());
    drop(mode);
    writeln!(stderr)?;
    Ok(answer?.map(|bytes| String::from_utf8_lossy(&bytes).into_owned()))
}

/// Reads a line typed without the terminal's line editing: Enter (or the
/// input's end) ends it, Backspace erases the last character, and Ctrl+C
/// gives `None`.
fn read_secret(input: &mut impl Read) -> std::io::Result<Option<Vec<u8>>> {
    let mut answer = Vec::new();
    let mut byte = [0_u8];
    loop {
        if input.read(&mut byte)? == 0 {
            return Ok(Some(answer));
        }
        match byte {
            [b'\r' | b'\n'] => return Ok(Some(answer)),
            [0x03] => return Ok(None),
            [0x7f | 0x08] => {
                // A whole UTF-8 character: its continuation bytes, then its
                // first.
                while answer.pop().is_some_and(|last| last & 0xc0 == 0x80) {}
            }
            [typed] => answer.push(typed),
        }
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

#[cfg(test)]
mod tests;
