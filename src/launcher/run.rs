//! Running the agent. [`RELOAD_EXIT_CODE`] restarts it on the staged build;
//! a new build is this launcher's trial until it signals ready in the
//! launcher's [`RunDir`]. Several launchers can share one `RIG_HOME`:
//! building and staging take the root's lock, a `/reload` build and its
//! trial are per session, and the ready file is per launcher.
//!
//! `rig` starts a new session, and `rig --resume <id or name>` an earlier one,
//! in the directory it ran in. The agent's `/new` and `/resume` leave a
//! [`SessionDir::switch`] file and exit with the reload code; the launcher then
//! runs that session instead, in its own directory. A session left without a
//! message, by a clean quit or a switch, leaves no directory. Each start tells
//! the agent why it started ([`StartReason`]).
//!
//! The command line is read once the agent is built, against what its
//! plugins add ([`AgentDescription::invocation`]), and the agent gets it in its
//! canonical form ([`Invocation`]). A plugin's command runs in a new
//! session, or with `--resume` in that one, unless its front keeps no
//! session, which the agent refuses. A headless run (`--print`, stdin piped
//! in, which the agent answers as `--print` does, or a plugin's command
//! that does not take the terminal) passes its exit code and output
//! through. One that answers a prompt is restarted on the reload code, as
//! when its model asked for a reload, without the prompt it already
//! answered, and run again on the previous build when that reload's new
//! build is rolled back; the restarted agent carries on with what the
//! reload left, such as its report to the model. A plugin's command is not
//! restarted, nor a run that asked to switch sessions. Before each start with stdin on
//! a terminal, whatever the front, the launcher asks for each secret the
//! build's plugins want that the environment does not hold
//! ([`ask_secret`]); only the agent's process gets the answers.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{ErrorKind, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, ExitStatus, Stdio};
use std::time::Duration;

use rig_harness_protocol::{
    AgentDescription, DescribedSecret, Home, Invocation, REFUSED_EXIT_CODE, RELOAD_EXIT_CODE,
    RunDir, SIGNAL_EXIT_CODE, SessionDir, SessionId, StartReason, env, secret_variable,
};

use super::build::{self, BuildFailure, BuildOutput, Staging, child};
use super::{Result, locks};

mod stop;

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

/// Builds the agent, reads the command line `args` against it, runs it
/// until it exits with anything but the reload code, and returns its exit
/// code: 2 for a command line the build does not take, or a `--resume`
/// that names no saved session.
pub fn run(home: &Home, args: &[String]) -> Result<ExitCode> {
    let here = std::env::current_dir().ok();
    let terminal = std::io::stdin().is_terminal();
    // Before the build, only whether `-p` is there, by a plain scan that a
    // prompt that is just `-p` matches too: the type-ahead guard, and the
    // build's output, which a headless run keeps off its stderr.
    let print = args.iter().any(|arg| arg == "-p" || arg == "--print");
    let output = if print || !terminal {
        BuildOutput::LogOnly
    } else {
        BuildOutput::Live
    };
    // Before the build, `-r` as the last build reads it; the new build may
    // read it otherwise (below).
    let asked = last_report(home).unwrap_or_default().resume(args);
    let start = match start_of(home, asked.as_deref()) {
        Ok(start) => start,
        Err(why) => return Ok(refused(&why)),
    };
    let (run, mut claimed, mut claim_notice, build_notice, build_failure, report) = {
        let _typing = (terminal && !print).then(TypeAhead::hold).flatten();
        let _lock = locks::hold_build_lock(home, output)?;
        // Before claiming, so a resumed session's leftover builds from its
        // dead launcher are removed too.
        locks::sweep(home)?;
        let run = locks::hold_run(home)?;
        let (claimed, claim_notice) = claim(home, start, here.as_deref())?;
        let failure = rebuild(home, &claimed.id, output)?;
        let built = failure.as_ref().map(|failure| {
            format!("Building the agent failed, so the previous build is running: {failure}")
        });
        let report = report(home, &claimed.id);
        (
            run,
            claimed,
            claim_notice,
            built,
            failure.map(|failure| failure.details()),
            report,
        )
    };
    let resume = report.resume(args);
    if resume != asked {
        // Only the build says which of its options take a value, and so
        // whether a word after one was `-r`.
        let start = match start_of(home, resume.as_deref()) {
            Ok(start) => start,
            Err(why) => {
                claimed.discard_if_unsaved(home);
                return Ok(refused(&why));
            }
        };
        let _lock = locks::hold_build_lock(home, output)?;
        let (next, notice) = claim(home, start, here.as_deref())?;
        absent(fs::rename(
            home.staged_for(&claimed.id),
            home.staged_for(&next.id),
        ))?;
        claimed.discard_if_unsaved(home);
        (claimed, claim_notice) = (next, notice);
    }
    let notice: Vec<String> = claim_notice.into_iter().chain(build_notice).collect();
    let notice = (!notice.is_empty()).then(|| notice.join("\n"));
    let invocation = match report.invocation(args) {
        Ok(invocation) => invocation,
        Err(why) => {
            claimed.discard_if_unsaved(home);
            eprintln!("rig: {why}");
            return Ok(ExitCode::from(REFUSED_EXIT_CODE));
        }
    };
    let command = invocation
        .command
        .as_deref()
        .and_then(|command| report.command(command));
    let headless =
        invocation.is_headless() || !terminal || command.is_some_and(|command| !command.terminal);
    let ended = launch(
        home,
        run.dir(),
        &invocation,
        headless,
        claimed,
        notice,
        build_failure,
    )?;
    let code = ended.code;
    let close = move || {
        if ended.restore {
            restore_terminal();
        }
        if let Some(said) = ended.said {
            eprintln!("{said}");
        }
    };
    // After a signal, a terminal or pipe nobody reads must not hold the
    // launcher: its closing output gets a moment, then it ends anyway.
    match stop::heard() {
        None => close(),
        Some(_) => stop::bounded(close),
    }
    // The run's own cleanup, before the signal ends the launcher.
    drop(run);
    stop::end_as_heard();
    Ok(code)
}

/// The session `--resume` names, by id or name ([`Home::find`]), or a new
/// one without it.
fn start_of(home: &Home, resume: Option<&str>) -> std::result::Result<Start, String> {
    resume.map_or(Ok(Start::New), |session| {
        home.find(session).map(Start::Resume)
    })
}

/// Says why the command line is refused, and returns its exit code.
fn refused(why: &str) -> ExitCode {
    eprintln!("rig: {why}\nRun `rig help` for usage.");
    ExitCode::from(REFUSED_EXIT_CODE)
}

/// What the plugins of the build the launcher of `session` starts next add
/// to the command line; nothing when there is no build or it cannot say.
fn report(home: &Home, session: &SessionId) -> AgentDescription {
    let Some(binary) = next_build(home, session) else {
        return AgentDescription::default();
    };
    build::described(home, &binary).unwrap_or_else(|failure| {
        eprintln!("warning: the agent build does not say what its plugins add: {failure}");
        AgentDescription::default()
    })
}

/// The report of the build a new session starts with, as kept from its
/// last build; `None` when it has none. It never builds.
pub fn last_report(home: &Home) -> Option<AgentDescription> {
    [home.staged(), home.good()]
        .into_iter()
        .find(|binary| binary.exists())
        .and_then(|binary| build::good_report(home, &binary))
}

/// The build the launcher of `session` starts next, as [`pick`] picks it,
/// without claiming it.
fn next_build(home: &Home, session: &SessionId) -> Option<PathBuf> {
    [home.staged_for(session), home.staged(), home.good()]
        .into_iter()
        .find(|binary| binary.exists())
}

/// Starts the agent of `claimed` with the build [`pick`] finds, again on
/// the reload code, and returns how it ended. `notice` and
/// `build_failure` go to the first start.
fn launch(
    home: &Home,
    run: &RunDir,
    invocation: &Invocation,
    headless: bool,
    mut claimed: Claimed,
    mut notice: Option<String>,
    mut build_failure: Option<String>,
) -> Result<Ended> {
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
    // Whether an agent's front took the terminal's screen since the
    // launcher last put it back: a view keeps it across a restart, so a
    // later start that never takes it may still need it put back.
    let mut screen = false;
    // The session the agent switched away from to start this one, which a
    // refused start goes back to.
    let mut switched_from: Option<SessionId> = None;
    let mut reason = claimed.reason;
    // A headless run that answers a prompt, restarted on a reload: without
    // the prompt, which it answered already.
    let answers_prompt = headless && invocation.command.is_none();
    let restarted = Invocation {
        print: invocation.print.as_ref().map(|_| String::new()),
        ..invocation.clone()
    };
    let mut invocation: &Invocation = invocation;
    loop {
        let left = switched_from.take();
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
                    return Ok(Ended::quiet(ExitCode::from(SIGNAL_EXIT_CODE)));
                };
                answers.insert(variable, answer);
            }
        }
        let binary = pick(home, &session, &trial)?;
        absent(fs::remove_file(&ready))?;
        absent(fs::remove_file(run.refusal()))?;
        absent(fs::remove_file(run.screen()))?;
        let mut command = Command::new(binary.path());
        command
            .args(invocation.to_args())
            .env(env::HOME, home.root())
            .env(env::SESSION, session.as_str())
            .env(env::LAUNCHER, &launcher)
            .env(env::RUN, run.path())
            .env(env::START, reason.as_str())
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
        let mut watch = stop::Watch::start();
        let mut child = command.spawn()?;
        // A trial becomes the good build as soon as it is ready, so a
        // launcher killed later (a closed terminal) keeps it.
        let mut started = false;
        let mut promoted: Result<()> = Ok(());
        let status = loop {
            watch.pass_on(&mut child);
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
        drop(watch);
        let reload = status.code() == Some(i32::from(RELOAD_EXIT_CODE)) && stop::heard().is_none();
        let restart = !headless || (answers_prompt && !directory.switch().exists());
        screen |= run.screen().exists();
        // A reload that restarts leaves the jobs running, listed for the
        // next build to take over. Best effort: a list that cannot be read
        // must not keep the terminal from being put back below.
        if !(reload && restart) {
            run.kill_left_groups().ok();
        }
        promoted?;
        let rejected = matches!(binary, Binary::Trial(_)) && !started;
        if rejected {
            build::reject(home, &trial)?;
            fs::remove_file(&trial)?;
        }
        if reload && restart {
            if headless {
                invocation = &restarted;
            }
            if let Some(target) = take_switch(&directory)? {
                (claimed, notice) = switch(home, claimed, target)?;
            }
            reason = if claimed.id == session {
                StartReason::Reloaded
            } else {
                claimed.reason
            };
            switched_from = (claimed.id != session).then_some(session);
            kept_screen = !headless;
            continue;
        }
        if !rejected
            && !headless
            && stop::heard().is_none()
            && let Some(left) = left
            && let Some(refusal) = absent(fs::read_to_string(run.refusal()))?
        {
            let back = if home.session(&left).is_saved() {
                Start::Resume(left)
            } else {
                Start::New
            };
            let (previous, _) = switch(home, claimed, back)?;
            claimed = previous;
            reason = claimed.reason;
            if claimed.id != session {
                // Told in the session it went back to, so nothing is kept.
                absent(fs::remove_file(run.log()))?;
                let why = refusal.trim().trim_start_matches("rig: ");
                notice = Some(format!("Could not switch sessions: {why}"));
                continue;
            }
        }
        // Only a session rolled back for someone at the terminal, or after
        // a reload of a run that answers a prompt, runs again.
        let again = (!headless || (answers_prompt && reason == StartReason::Reloaded))
            && stop::heard().is_none();
        let ended = !rejected || !again || !home.good().exists();
        let log = if ended {
            claimed.leave(home, run, status.success())
        } else {
            directory.log()
        };
        let refusal = fs::read_to_string(run.refusal()).ok();
        if !rejected && headless {
            return Ok(Ended::quiet(exit_code(status)));
        }
        let resume = started
            .then(|| resume_command(home, &session, refusal.is_some()))
            .flatten();
        if !rejected && status.success() {
            let said = resume.map(|resume| format!("To resume this session: {resume}"));
            // The agent left the terminal as it found it; the line needs
            // it out of the full screen in case it did not.
            let restore = said.is_some() && screen;
            return Ok(Ended::new(status, said).restoring(restore));
        }
        let restore = std::mem::take(&mut screen);
        if !rejected {
            let log = Some(log.as_path()).filter(|log| log.exists());
            let refusal = refusal.as_deref();
            // A refusal said while a view had the screen was not seen.
            let hidden = kept_screen || restore;
            let said = stopped(status, resume.as_deref(), refusal, hidden, log);
            return Ok(Ended::new(status, said).restoring(restore));
        }
        if !home.good().exists() {
            let said = format!(
                "The new build stopped during startup ({status}) and there is no previous \
                 build to roll back to; the next `rig` tries it again. Log: {}",
                log.display()
            );
            return Ok(Ended::new(status, Some(said)).restoring(restore));
        }
        let message = format!(
            "The new build stopped during startup ({status}); rolled back to the previous \
             build. `rig build` tries it again. Log: {}",
            log.display()
        );
        // A one-shot run is not run twice, its output already out, unless
        // its agent reloaded: the restart tells the agent of the rollback.
        if !again {
            return Ok(Ended::new(status, Some(message)).restoring(restore));
        }
        if restore {
            restore_terminal();
        }
        eprintln!("{message}");
        kept_screen = false;
        notice = Some(message);
        reason = StartReason::Restarted;
    }
}

/// How the launcher's runs of a session ended: the exit code, what the
/// launcher says last, and whether it puts the terminal back first, after
/// a front that took its screen.
#[derive(Debug, PartialEq)]
struct Ended {
    code: ExitCode,
    said: Option<String>,
    restore: bool,
}

impl Ended {
    /// The end of an agent that exited with `status`, followed by `said`.
    fn new(status: ExitStatus, said: Option<String>) -> Self {
        Self {
            code: exit_code(status),
            said,
            restore: false,
        }
    }

    /// An end with `code` and nothing said.
    fn quiet(code: ExitCode) -> Self {
        Self {
            code,
            said: None,
            restore: false,
        }
    }

    /// The same end, putting the terminal back first when `restore`.
    fn restoring(self, restore: bool) -> Self {
        Self { restore, ..self }
    }
}

/// What the launcher says when the agent exited with `status`, not 0. A
/// run a signal ended, which the agent was told of, gets the `resume`
/// command as a quit does. A refused run said why itself: its `refusal`
/// comes again only when a view had the screen (`kept_screen`), which hid
/// it, such as a restarted view, or one that took the screen before a
/// `--tools` name was refused. Any other end is a failure: the agent
/// stopped, where its log is when it kept one and, when the session can be
/// resumed, the `resume` command.
fn stopped(
    status: ExitStatus,
    resume: Option<&str>,
    refusal: Option<&str>,
    kept_screen: bool,
    log: Option<&Path>,
) -> Option<String> {
    let code = status.code().and_then(|code| u8::try_from(code).ok());
    match (code, refusal) {
        (Some(SIGNAL_EXIT_CODE), _) => {
            resume.map(|resume| format!("To resume this session: {resume}"))
        }
        (Some(REFUSED_EXIT_CODE), Some(refusal)) => {
            kept_screen.then(|| refusal.trim_end().to_owned())
        }
        _ => {
            let resume = resume.map_or_else(String::new, |resume| {
                format!(" `{resume}` resumes the session where it stopped.")
            });
            let log = log.map_or_else(String::new, |log| format!(" Log: {}", log.display()));
            Some(format!("The agent stopped ({status}).{resume}{log}"))
        }
    }
}

/// The command that resumes `session`, when it is saved and was not
/// `refused` (a refused session would be refused again). It names the
/// session by its name, single-quoted for the shell, when [`Home::find`]
/// finds this session by it; by its id when the name is not unique, could
/// be read as an id or an option, or holds a control or format character.
/// The quoting is POSIX shell's, as the launcher runs on Linux and macOS.
fn resume_command(home: &Home, session: &SessionId, refused: bool) -> Option<String> {
    let directory = home.session(session);
    if refused || !directory.is_saved() {
        return None;
    }
    let name = directory.name().filter(|name| {
        !name.starts_with('-')
            && !name.chars().any(|c| c.is_control() || is_format(c))
            && name.parse::<SessionId>().is_err()
            && home.find(name).as_ref() == Ok(session)
    });
    Some(match name {
        Some(name) => format!("rig --resume '{}'", name.replace('\'', r"'\''")),
        None => format!("rig --resume {session}"),
    })
}

/// Whether `c` is an invisible Unicode format character, line or paragraph
/// separator, or variation selector, such as a bidi override, which shows a
/// name otherwise than it is typed.
fn is_format(c: char) -> bool {
    matches!(
        c,
        '\u{ad}'
            | '\u{61c}'
            | '\u{180b}'..='\u{180f}'
            | '\u{200b}'..='\u{200f}'
            | '\u{2028}'..='\u{202e}'
            | '\u{2060}'..='\u{206f}'
            | '\u{fe00}'..='\u{fe0f}'
            | '\u{feff}'
            | '\u{fff9}'..='\u{fffb}'
            | '\u{e0001}'
            | '\u{e0020}'..='\u{e007f}'
            | '\u{e0100}'..='\u{e01ef}'
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
    let _lock = locks::hold_build_lock(home, BuildOutput::Live)?;
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
    /// Why its first start starts it: `New` or `Resumed`.
    reason: StartReason,
    /// The directory the session runs in, when known and still there.
    directory: Option<PathBuf>,
    _lock: File,
}

impl Claimed {
    /// Removes the session's directory when no agent logged a message in
    /// it, such as a `/new` session left at once: a session with nothing
    /// to resume leaves nothing behind. Call it once the agent exited for
    /// good or switched away. Best effort: a directory left behind is not
    /// marked saved, so `/resume` does not list it.
    fn discard_if_unsaved(&self, home: &Home) {
        let session = home.session(&self.id);
        if !session.is_saved() {
            fs::remove_dir_all(session.path()).ok();
        }
    }

    /// Ends this launcher's runs of the session for good: discards it when
    /// unsaved ([`Self::discard_if_unsaved`]), first moving the text log of
    /// a run that failed into `run`, which keeps it, and returns where the
    /// text log is. A sessionless run that failed, and a refused run, moved
    /// it there themselves.
    fn leave(&self, home: &Home, run: &RunDir, succeeded: bool) -> PathBuf {
        let session = home.session(&self.id);
        if session.is_saved() {
            return if run.refusal().exists() {
                run.log()
            } else {
                session.log()
            };
        }
        if !succeeded {
            fs::rename(session.log(), run.log()).ok();
        }
        self.discard_if_unsaved(home);
        run.log()
    }
}

/// The session to run for `start` from the working directory `here`, held
/// locked, and the line the agent shows about it. Call it holding
/// [`locks::hold_build_lock`].
fn claim(home: &Home, start: Start, here: Option<&Path>) -> Result<(Claimed, Option<String>)> {
    let (id, lock, notice, reason) = match start {
        Start::Resume(id) => {
            if !home.session(&id).is_saved() {
                return Err(format!("session {id} has no saved conversation").into());
            }
            let lock = home
                .session(&id)
                .hold()?
                .ok_or_else(|| format!("session {id} is open in another rig"))?;
            (
                id,
                lock,
                Some("Resumed the session.".to_owned()),
                StartReason::Resumed,
            )
        }
        Start::New => {
            let (id, _) = home.new_session()?;
            let lock = home
                .session(&id)
                .hold()?
                .ok_or_else(|| format!("session {id} is already running"))?;
            (id, lock, None, StartReason::New)
        }
    };
    let session = home.session(&id);
    let directory = match session.working_directory() {
        Some(directory) => Some(directory),
        None => {
            if let Some(here) = here {
                session.set_working_directory(here)?;
            }
            here.map(Path::to_path_buf)
        }
    };
    Ok((
        Claimed {
            id,
            reason,
            directory,
            _lock: lock,
        },
        notice,
    ))
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
fn ask_secret(secret: &DescribedSecret) -> Result<Option<String>> {
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

/// Builds before every start, staging for this launcher, its output as
/// `output` says; cargo does no work when nothing changed. With no good
/// build yet, the build is staged even if it was rejected before, so each
/// start retries it. A failed build with an older binary at hand is
/// returned, for that binary to start with.
fn rebuild(home: &Home, session: &SessionId, output: BuildOutput) -> Result<Option<BuildFailure>> {
    let first_run = !home.staged().exists() && !home.good().exists();
    if output == BuildOutput::Live {
        eprintln!(
            "Building the rig agent{}…",
            if first_run { " (first run)" } else { "" }
        );
    }
    let staging = if first_run {
        Staging::Always
    } else {
        Staging::OnlyNew
    };
    match build::compile(home, &home.staged_for(session), staging, output) {
        Ok(()) => Ok(None),
        Err(failure) if first_run => Err(failure.told(output).into()),
        Err(failure) => {
            let told = failure.told(output);
            eprintln!("error: building the agent failed; starting the previous build: {told}");
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
