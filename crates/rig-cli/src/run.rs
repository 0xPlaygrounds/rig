//! Running the agent. [`RELOAD_EXIT_CODE`] restarts it on the staged build;
//! a new build is this launcher's trial until it signals ready in the
//! launcher's [`RunDir`]. Several launchers can share one `RIG_HOME`:
//! building, staging and selecting a build take the root's lock, a
//! `/reload` build is staged per session, and each launcher holds the
//! builds it selected ([`BuildRef`]) and its ready file in its run
//! directory. It starts the build it read the command line against, and
//! rolls back to the one its session ran before.
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
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use rig_harness_protocol::config::SettingsFile;
use rig_harness_protocol::{
    AgentDescription, DescribedSecret, Home, Invocation, REFUSED_EXIT_CODE, RELOAD_EXIT_CODE,
    RunDir, SIGNAL_EXIT_CODE, SessionDir, SessionId, Stamp, StartReason, env, secret_variable,
    warning_text,
};

use super::build::{self, BuildFailure, BuildOutput, FailedAt, Staging, child};
use super::project::RigSource;
use super::update::{self, Reexec};
use super::{LONG_VERSION, Result, locks};

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
    // Before the build, the command line as the last build reads it; the
    // new build may read it otherwise (below). Its `-p` decides the
    // type-ahead guard, the build's output, which a headless run keeps off
    // its stderr, and whether a build that fails refuses the run. A line it
    // cannot read, such as one with an option it does not report, is read
    // as the agent reads its own, which takes any option, so the previous
    // build never answers a `-p` the new build failed on.
    let last = last_report(home).unwrap_or_default();
    let print = (last.invocation(args))
        .or_else(|_| Invocation::parse(args))
        .is_ok_and(|invocation| invocation.print.is_some());
    let output = if print || !terminal {
        BuildOutput::LogOnly
    } else {
        BuildOutput::Live
    };
    let asked = last.resume(args);
    let start = match start_of(home, asked.as_deref()) {
        Ok(start) => start,
        Err(why) => return Ok(refused(&why)),
    };
    // The session a replaced launcher left for this run's, which a
    // refused start goes back to, as its own switch would.
    let left = update::left()?;
    let (run, mut claimed, mut claim_notice, build_notice, build_failure, running) = {
        let _typing = (terminal && !print).then(TypeAhead::hold).flatten();
        let _lock = locks::hold_build_lock(home, output)?;
        // Before claiming, so a resumed session's leftover builds from its
        // dead launcher are removed too.
        locks::sweep(home)?;
        let run = locks::hold_run(home)?;
        let (claimed, claim_notice) = claim(home, start, here.as_deref())?;
        let failure = rebuild(home, &claimed.id, output, print)?;
        let built = failure.as_ref().map(|failure| {
            format!("Building the agent failed, so the previous build is running: {failure}")
        });
        let running = select(home, &claimed.id, run.dir(), None)?;
        (
            run,
            claimed,
            claim_notice,
            built,
            failure.map(|failure| failure.details()),
            running,
        )
    };
    let report = &running.current.description;
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
        claimed.discard_if_unsaved(home);
        (claimed, claim_notice) = (next, notice);
    }
    claimed.left = left;
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
    let mut ended = launch(
        home,
        &std::env::current_exe()?,
        run.dir(),
        &invocation,
        headless,
        claimed,
        running,
        notice,
        build_failure,
    )?;
    let code = ended.code;
    let reexec = ended.reexec.take();
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
    match reexec {
        Some(reexec) => reexec.run(),
        None => Ok(code),
    }
}

/// A warning for each file rig 0.46 kept in `home` that rig no longer
/// reads, when its `settings.toml` has no table for what the file held:
/// `plugins.toml` without `[plugins]`, `defaults.json` without `[model]`.
/// The files are never changed. A `settings.toml` rig cannot read refuses
/// the start before this.
fn legacy_warnings(home: &Home) -> Vec<String> {
    let settings = home.settings();
    let Ok(file) = SettingsFile::read(&settings) else {
        return Vec::new();
    };
    [
        ("plugins.toml", "plugin list", "plugins"),
        ("defaults.json", "model and effort", "model"),
    ]
    .into_iter()
    .filter(|(old, _, table)| home.root().join(old).exists() && !file.document.contains_key(table))
    .map(|(old, what, table)| {
        format!(
            "rig no longer reads {}. Move its {what} to [{table}] in {} (see MIGRATING.md). \
             The file was not changed.",
            home.root().join(old).display(),
            settings.display(),
        )
    })
    .collect()
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

/// The report of the build a new session starts with, as kept from its
/// last build; `None` when it has none. It never builds.
pub fn last_report(home: &Home) -> Option<AgentDescription> {
    [home.staged(), home.good()]
        .into_iter()
        .find(|binary| binary.exists())
        .and_then(|binary| build::good_report(home, &binary))
}

/// Starts the agent of `claimed` with the build `running` holds, again on
/// the reload code with the build [`select`] then holds, and returns how
/// it ended. `notice` and `build_failure` go to the first start.
fn launch(
    home: &Home,
    launcher: &Path,
    run: &RunDir,
    invocation: &Invocation,
    headless: bool,
    mut claimed: Claimed,
    mut running: Running,
    mut notice: Option<String>,
    mut build_failure: Option<String>,
) -> Result<Ended> {
    let installed = Stamp::of(launcher);
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
    let mut reason = claimed.reason;
    // A headless run that answers a prompt, restarted on a reload: without
    // the prompt, which it answered already.
    let answers_prompt = headless && invocation.command.is_none();
    let restarted = Invocation {
        print: invocation.print.as_ref().map(|_| String::new()),
        ..invocation.clone()
    };
    let mut invocation: &Invocation = invocation;
    let mut legacy = legacy_warnings(home);
    // The agent runs in its session's directory, and its `/reload` runs
    // `rig build` there.
    let source = RigSource::from_env()?;
    loop {
        let left = claimed.left.take();
        let session = claimed.id.clone();
        let directory = home.session(&session);
        let ready = run.ready();
        if ask {
            for secret in &running.current.description.secrets {
                let variable = secret_variable(&secret.name);
                if std::env::var_os(&variable).is_some() || answers.contains_key(&variable) {
                    continue;
                }
                let Some(answer) = ask_secret(secret)? else {
                    claimed.discard_if_unsaved(home);
                    return Ok(Ended::quiet(ExitCode::from(SIGNAL_EXIT_CODE)));
                };
                answers.insert(variable, answer);
            }
        }
        absent(fs::remove_file(&ready))?;
        absent(fs::remove_file(run.refusal()))?;
        absent(fs::remove_file(run.screen()))?;
        let mut command = Command::new(&running.current.executable);
        command
            .args(invocation.to_args())
            .env(env::HOME, home.root())
            .env(env::SESSION, session.as_str())
            .env(env::LAUNCHER, launcher)
            .env(env::VERSION, LONG_VERSION)
            .env(env::LAUNCHER_STAMP, installed.text())
            .env(env::RUN, run.path())
            .env(env::START, reason.as_str())
            .env_remove(env::NOTICE)
            .env_remove(env::WARNING)
            .env_remove(env::BUILD_FAILURE)
            .env_remove(update::LEFT)
            .envs(&answers);
        if let Some(source) = &source {
            command.env(env::SOURCE, source);
        }
        if let Some(working) = &claimed.directory {
            command.current_dir(working);
        }
        if let Some(notice) = notice.take() {
            command.env(env::NOTICE, notice);
        }
        // Only the first start of a claim runs elsewhere than it was, and
        // only the run's starts until one is ready warn of a 0.46 home.
        let moved = claimed.moved.take();
        let warnings: Vec<&str> = (moved.iter().chain(&legacy)).map(String::as_str).collect();
        if !warnings.is_empty() {
            command.env(env::WARNING, warning_text(warnings));
        }
        // Only the first start follows the failed build.
        if let Some(failure) = build_failure.take() {
            command.env(env::BUILD_FAILURE, failure);
        }
        let mut watch = stop::Watch::start();
        let mut child = command.spawn()?;
        // A trial becomes the good build as soon as it is ready and the
        // build lock is free, so a launcher killed later (a closed
        // terminal) keeps it. Once the agent exited, the launcher polls for
        // the lock until a signal asks it to end; an unpromoted trial is
        // staged again by the next start's build.
        let mut started = false;
        let mut promoting = false;
        let mut promoted: Result<()> = Ok(());
        let mut exited = None;
        // When the launcher began to wait for the lock, and whether it said so.
        let mut waiting: Option<(std::time::Instant, bool)> = None;
        let status = loop {
            if exited.is_none() {
                watch.pass_on(&mut child);
                exited = child.try_wait()?;
            }
            if !started && ready.exists() {
                started = true;
                legacy.clear();
                promoting = running.current.is_trial();
            }
            if promoting {
                match promote(home, &mut running.current) {
                    Ok(done) => promoting = !done,
                    Err(failure) => (promoting, promoted) = (false, Err(failure)),
                }
            }
            match exited {
                Some(status) if !promoting || stop::heard().is_some() => break status,
                Some(_) => {
                    let (since, said) =
                        waiting.get_or_insert_with(|| (std::time::Instant::now(), false));
                    if !*said && since.elapsed() >= locks::QUIET_WAIT {
                        *said = true;
                        locks::say_waiting(home);
                    }
                }
                None => {}
            }
            std::thread::sleep(POLL);
        };
        drop(watch);
        let reload = status.code() == Some(i32::from(RELOAD_EXIT_CODE)) && stop::heard().is_none();
        let restart = !headless || (answers_prompt && !directory.switch().exists());
        // A view's reload after the launcher's file was replaced restarts on
        // the new launcher, never the new agent under this one. A headless
        // agent does not reload once its launcher was replaced, as told by
        // the stamp passed on, since only this launcher restarts it without
        // its prompt.
        let replaced = reload && restart && !headless && installed.replaced(&Stamp::of(launcher));
        screen |= run.screen().exists();
        // A reload that restarts leaves the jobs running, listed for the
        // next build to take over. Best effort: a list that cannot be read
        // must not keep the terminal from being put back below.
        if !(reload && restart) || replaced {
            run.kill_left_groups().ok();
        }
        promoted?;
        let rejected = running.current.is_trial() && !started && stop::heard().is_none();
        if rejected {
            build::reject(home, &running.current.executable)?;
        }
        if replaced {
            // The session the agent switched to, claimed as a switch claims
            // it: one that cannot run leaves this one on, told why.
            let mut said = None;
            if let Some(target) = take_switch(&directory)? {
                (claimed, said) = switch(home, claimed, target)?;
            }
            let resume = home
                .session(&claimed.id)
                .is_saved()
                .then(|| claimed.id.clone());
            let here = claimed.directory.clone();
            claimed.discard_if_unsaved(home);
            let left = (claimed.id != session).then_some(session);
            let resume = resume.as_ref().map(SessionId::as_str);
            let reexec = Reexec::new(home, launcher, resume, invocation.to_args(), here, left);
            return Ok(Ended {
                said,
                reexec: Some(reexec),
                ..Ended::quiet(ExitCode::SUCCESS).restoring(screen)
            });
        }
        if reload && restart {
            if headless {
                invocation = &restarted;
            }
            if let Some(target) = take_switch(&directory)? {
                (claimed, notice) = switch(home, claimed, target)?;
            }
            let Running { current, previous } = running;
            let current = if rejected { previous } else { Some(current) };
            running = {
                let output = if headless {
                    BuildOutput::LogOnly
                } else {
                    BuildOutput::Live
                };
                let _lock = locks::hold_build_lock(home, output)?;
                select(home, &claimed.id, run, current)?
            };
            reason = if claimed.id == session {
                StartReason::Reloaded
            } else {
                claimed.reason
            };
            claimed.left = (claimed.id != session).then_some(session);
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
        let ended = !rejected || !again || running.previous.is_none();
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
        let Some(previous) = running.previous.take() else {
            let said = format!(
                "The new build stopped during startup ({status}) and there is no previous \
                 build to roll back to; the next `rig` tries it again. Log: {}",
                log.display()
            );
            return Ok(Ended::new(status, Some(said)).restoring(restore));
        };
        running.current = previous;
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
    /// The new launcher to start once the session is let go.
    reexec: Option<Reexec>,
}

impl Ended {
    /// The end of an agent that exited with `status`, followed by `said`.
    fn new(status: ExitStatus, said: Option<String>) -> Self {
        Self {
            code: exit_code(status),
            said,
            restore: false,
            reexec: None,
        }
    }

    /// An end with `code` and nothing said.
    fn quiet(code: ExitCode) -> Self {
        Self {
            code,
            said: None,
            restore: false,
            reexec: None,
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
    /// Why it runs elsewhere than the directory it saved, which is gone.
    moved: Option<String>,
    /// The session the agent switched away from to start this one, which
    /// its first start, when refused, goes back to.
    left: Option<SessionId>,
    _lock: File,
}

impl Claimed {
    /// Removes the session's directory when it is not saved and its agent
    /// logs hold nothing ([`SessionDir::holds_logs`]), such as a `/new`
    /// session left at once: a session with nothing to resume leaves
    /// nothing behind. Call it once the agent exited for good or switched
    /// away. Best effort: a directory left behind is not marked saved, so
    /// `/resume` does not list it.
    fn discard_if_unsaved(&self, home: &Home) {
        let session = home.session(&self.id);
        if !session.is_saved() && !session.holds_logs() {
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
    let mut moved = None;
    let directory = match session.working_directory() {
        Some(directory) => Some(directory),
        None => {
            if let Some(here) = here {
                moved = session.saved_directory().map(|gone| {
                    format!(
                        "The session's directory {} is gone, so it runs in {}.",
                        gone.display(),
                        here.display()
                    )
                });
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
            moved,
            left: None,
            _lock: lock,
        },
        notice,
    ))
}

/// Puts the terminal back after an agent that could not restore it: leaves
/// the alternate screen, mouse reports and, on a terminal, raw mode.
fn restore_terminal() {
    eprint!("\x1b[?1049l\x1b[?25h\x1b[?1006l\x1b[?1002l");
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
/// start retries it. A build that failed with an older binary at hand is
/// returned, for that binary to start with, unless the run is `print`'s
/// (`-p`), whose caller asked for this build's answer. Settings that name no
/// build ([`FailedAt::Settings`]) refuse the start too, since that binary
/// was built from other settings.
fn rebuild(
    home: &Home,
    session: &SessionId,
    output: BuildOutput,
    print: bool,
) -> Result<Option<BuildFailure>> {
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
    let staged = home.staged_for(session);
    match build::compile(home, &staged, staging, output) {
        Ok(()) => {
            if build::passed_over(home, &staged) {
                eprintln!(
                    "The new build stopped during startup before, so the previous build is \
                     running; `rig build` tries it again."
                );
            }
            Ok(None)
        }
        Err(failure) if failure.at == FailedAt::Settings => Err(failure.reason.into()),
        Err(failure) if first_run || print => Err(failure.told(output).into()),
        Err(failure) => {
            let told = failure.told(output);
            eprintln!("error: building the agent failed; starting the previous build: {told}");
            Ok(Some(failure))
        }
    }
}

/// The builds a launcher holds for its session: the one it starts, and
/// the one it rolls back to when that one is a trial that stops during
/// startup.
struct Running {
    current: BuildRef,
    previous: Option<BuildRef>,
}

/// A build a launcher holds from the moment it selects it until it lets it
/// go: a file of its own in its run directory ([`locks::builds`]), which
/// no other launcher claims or replaces, so the build whose description
/// the launcher read its command line and secrets against is the one it
/// starts, and starts again. Dropping it removes the file; an agent
/// started from it keeps running.
struct BuildRef {
    executable: PathBuf,
    /// What its plugins add to the command line.
    description: AgentDescription,
    role: Role,
}

/// What a [`BuildRef`]'s start decides about the build.
enum Role {
    /// A staged build on trial until it is ready. It then becomes the good
    /// build, if the good build is still the one whose stamp is `good`
    /// (`None` for no good build) as when the trial was selected: another
    /// launcher's build that became good meanwhile stays good.
    Trial { good: Option<String> },
    /// A build that started up before.
    Proven,
}

impl BuildRef {
    /// Holds the build at `executable`, which this launcher alone names,
    /// reading its description ([`build::described`]); a build that cannot
    /// say adds nothing.
    fn new(home: &Home, executable: PathBuf, role: Role) -> Self {
        let description = build::described(home, &executable).unwrap_or_else(|failure| {
            eprintln!("warning: the agent build does not say what its plugins add: {failure}");
            AgentDescription::default()
        });
        Self {
            executable,
            description,
            role,
        }
    }

    fn is_trial(&self) -> bool {
        matches!(self.role, Role::Trial { .. })
    }
}

impl Drop for BuildRef {
    fn drop(&mut self) {
        fs::remove_file(&self.executable).ok();
    }
}

/// The builds the launcher of `session` holds in `run` for its next start:
/// a build staged for it or for any launcher, claimed as a trial, rolling
/// back to `current` or else the good build; without one, `current`, else
/// the good build. Call it holding [`locks::hold_build_lock`], so the
/// claims and the good build it reads are the ones it holds.
fn select(
    home: &Home,
    session: &SessionId,
    run: &RunDir,
    current: Option<BuildRef>,
) -> Result<Running> {
    let good = build::stamp(&home.good()).ok();
    let held_before = match current {
        Some(current) => Some(current),
        None => hold_good(home, run)?,
    };
    for staged in [home.staged_for(session), home.staged()] {
        let held = held(run)?;
        // The rename claims the staged build for this launcher alone.
        if absent(fs::rename(staged, &held))?.is_some() {
            let current = BuildRef::new(home, held, Role::Trial { good });
            return Ok(Running {
                current,
                previous: held_before,
            });
        }
    }
    Ok(Running {
        current: held_before.ok_or("no agent build exists yet; run `rig build`")?,
        previous: None,
    })
}

/// The good build, held in `run`; `None` when there is none.
fn hold_good(home: &Home, run: &RunDir) -> Result<Option<BuildRef>> {
    let held = held(run)?;
    Ok(absent(build::hold(&home.good(), &held))?.map(|()| BuildRef::new(home, held, Role::Proven)))
}

/// A path in `run` for a build to hold, which no other build of this
/// launcher has.
fn held(run: &RunDir) -> Result<PathBuf> {
    static HELD: AtomicU32 = AtomicU32::new(0);
    let builds = locks::builds(run);
    fs::create_dir_all(&builds)?;
    Ok(builds.join(HELD.fetch_add(1, Ordering::Relaxed).to_string()))
}

/// Makes `build`, a trial that is ready, the good build as its
/// [`Role::Trial`] says, when the build lock is free. Returns whether it
/// was done; once done, the build is [`Role::Proven`].
fn promote(home: &Home, build: &mut BuildRef) -> Result<bool> {
    let Some(_lock) = locks::try_build_lock(home)? else {
        return Ok(false);
    };
    let Role::Trial { good } = std::mem::replace(&mut build.role, Role::Proven) else {
        return Ok(true);
    };
    if build::stamp(&home.good()).ok() == good {
        let temporary = home.bin().join("good.tmp");
        absent(fs::remove_file(&temporary))?;
        build::hold(&build.executable, &temporary)?;
        fs::rename(&temporary, home.good())?;
    }
    Ok(true)
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
