//! Running the agent. An agent that ends asking to reload ([`AgentEnd`])
//! restarts on the staged build; a new build is this launcher's trial until
//! it signals ready in the launcher's [`RunDir`]. Several launchers can
//! share one `RIG_HOME`: building, staging and selecting a build take the
//! root's lock, a `/reload` build is staged per session, and each launcher
//! holds the builds it selected ([`BuildRef`]) and its ready file in its
//! run directory. It starts the build it read the command line against, and
//! rolls back to the one its session ran before.
//!
//! `rig` starts a new session, and `rig --resume <id or name>` an earlier
//! one, in the directory it ran in. The agent's `/new` and `/resume` end it
//! with [`AgentEnd::Switch`]; the launcher then runs that session instead,
//! in its own directory. A session left without a message, by a clean quit
//! or a switch, leaves no directory. Each start tells the agent why it
//! started ([`StartReason`]).
//!
//! The command line is read once the agent is built, against what its
//! plugins add ([`AgentDescription::invocation`]), and the agent gets it in
//! its canonical form ([`Invocation`]). A plugin's command runs in a new
//! session, or with `--resume` in that one, unless its front keeps no
//! session, which the agent refuses. A headless run (`--print`, stdin piped
//! in, which the agent answers as `--print` does, or a plugin's command
//! that does not take the terminal) passes its exit code and output
//! through, but a reload or switch it does not follow ends it with 0. One
//! that answers a prompt is restarted when it reloads, as when its model
//! asked for a reload, without the prompt it already answered, and run
//! again on the previous build when that reload's new build is rolled back;
//! the restarted agent carries on with what the reload left, such as its
//! report to the model. A plugin's command is not restarted, nor a run that
//! asked to switch sessions. Before each start with stdin on a terminal,
//! whatever the front, the launcher asks for each secret the build's
//! plugins want that the environment does not hold ([`ask_secret`]); only
//! the agent's process, and a launcher that replaces this one, get the
//! answers.

use std::ffi::OsString;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use rig_harness_protocol::config::SettingsFile;
use rig_harness_protocol::end::AgentEnd;
use rig_harness_protocol::fs;
use rig_harness_protocol::launch::{AbsPath, Agent, Carry, Launch, Purpose};
use rig_harness_protocol::{
    AgentDescription, DescribedSecret, Home, Invocation, REFUSED_EXIT_CODE, RunDir,
    SIGNAL_EXIT_CODE, SessionId, Stamp, StartReason, secret_variable,
};

use super::build::{self, BuildFailure, BuildOutput, FailedAt, Staging};
use super::update;
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
/// until it ends without asking to reload, and returns its exit code: 2
/// for a command line the build does not take, or a `--resume` that names
/// no saved session.
pub fn run(mut launch: Launch, args: &[String]) -> Result<ExitCode> {
    let home = &launch.home();
    let terminal = launch.terminal;
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
    // What the launcher this one replaced handed on, for the first start.
    let carried = std::mem::take(&mut launch.carry);
    let (run, mut claimed, mut claim_notice, build_notice, build_failure, running) = {
        let _typing = (terminal && !print)
            .then(|| TypeAhead::hold(&launch))
            .flatten();
        let _lock = locks::hold_build_lock(home, output)?;
        // Before claiming, so a resumed session's leftover builds from its
        // dead launcher are removed too.
        locks::sweep(home)?;
        let run = locks::hold_run(home)?;
        let (claimed, claim_notice) = claim(home, start, launch.here.as_deref())?;
        let failure = rebuild(&launch, &claimed.id, output, print)?;
        let built = failure.as_ref().map(|failure| {
            format!("Building the agent failed, so the previous build is running: {failure}")
        });
        let running = select(&launch, &claimed.id, run.dir(), None)?;
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
        let (next, notice) = claim(home, start, launch.here.as_deref())?;
        claimed.discard_if_unsaved(home);
        (claimed, claim_notice) = (next, notice);
    }
    let claim_notice = carry_on(&mut launch, carried, &mut claimed, claim_notice);
    let notice: Vec<String> = claim_notice.into_iter().chain(build_notice).collect();
    launch.carry.notice = (!notice.is_empty()).then(|| notice.join("\n"));
    launch.carry.build_failure = build_failure;
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
    let mut ended = run_agent(
        &mut launch,
        run.dir(),
        &invocation,
        headless,
        claimed,
        running,
    )?;
    let code = ended.code;
    let reexec = ended.reexec.take();
    let close = move || {
        if ended.restore {
            restore_terminal(sane(&launch).as_mut());
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
        Some(next) => update::reexec(&next),
        None => Ok(code),
    }
}

/// Applies what the launcher this one replaced handed on, `carried`, to
/// the first start of `claimed`, and returns that start's notice: the
/// session it left and, after a re-exec, why it starts and what it is told
/// in place of the claim's `notice`. A run that warned of a 0.46 home
/// already does not again.
fn carry_on(
    launch: &mut Launch,
    carried: Carry,
    claimed: &mut Claimed,
    notice: Option<String>,
) -> Option<String> {
    claimed.left = carried.left;
    launch.carry.warned = carried.warned;
    match carried.reason {
        Some(reason) => {
            claimed.reason = reason;
            carried.notice
        }
        None => notice,
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
/// last build; `None` when it has none. It never builds. A report that
/// cannot be read is warned of on stderr and is `None`, so a start still
/// builds and `rig help` still answers.
pub fn last_report(home: &Home) -> Option<AgentDescription> {
    let read = || {
        for binary in [home.staged(), home.good()] {
            if fs::optional(fs::metadata(&binary))?.is_some() {
                return build::good_report(home, &binary);
            }
        }
        Ok(None)
    };
    read().unwrap_or_else(|why: std::io::Error| {
        eprintln!("warning: the last build's report cannot be read: {why}");
        None
    })
}

/// Starts the agent of `claimed` with the build `running` holds, again on
/// a reload or switch with the build [`select`] then holds, and returns how
/// it ended. The notice and build failure of `launch`'s carry go to the
/// first start. Each time the agent ends, [`next`] decides the step from
/// what it left; this function carries the step out. An error after a
/// front took the terminal's screen puts the terminal back ([`Screen`]).
fn run_agent(
    launch: &mut Launch,
    run: &RunDir,
    invocation: &Invocation,
    headless: bool,
    mut claimed: Claimed,
    mut running: Running,
) -> Result<Ended> {
    let home = &launch.home();
    let installed = Stamp::of(&launch.launcher)?;
    // The secrets asked for are kept with the launch, for every restart and
    // a re-exec: asked whenever someone is at the terminal, whatever the
    // front, so a command typed there can be asked too; a piped run passes
    // only what its environment has.
    let ask = launch.terminal;
    let mut screen = Screen::new(launch);
    // Whether a restarted view kept its screen ([`Runs::kept_screen`]).
    let mut kept_screen = false;
    let mut reason = claimed.reason;
    // A headless run that answers a prompt, restarted on a reload: without
    // the prompt, which it answered already.
    let answers_prompt = headless && invocation.command.is_none();
    let restarted = Invocation {
        print: invocation.print.as_ref().map(|_| String::new()),
        ..invocation.clone()
    };
    let mut invocation: &Invocation = invocation;
    let legacy = legacy_warnings(home);
    loop {
        let session = claimed.id.clone();
        let directory = home.session(&session);
        if ask {
            for secret in &running.current.description.secrets {
                let variable = OsString::from(secret_variable(&secret.name));
                if launch.secrets.contains_key(&variable) {
                    continue;
                }
                let Some(answer) = ask_secret(launch, secret)? else {
                    claimed.discard_if_unsaved(home);
                    let code = ExitCode::from(SIGNAL_EXIT_CODE);
                    return Ok(Ended::quiet(code).restoring(screen.hand_over()));
                };
                launch.secrets.insert(variable, answer.into());
            }
        }
        run.clear_messages()?;
        // Only the first start of a claim runs elsewhere than it was, and
        // only the run's starts until one is ready warn of a 0.46 home.
        let moved = claimed.moved.take();
        let unwarned = (!launch.carry.warned)
            .then_some(&legacy)
            .into_iter()
            .flatten();
        launch.carry.warnings = moved.into_iter().chain(unwarned.cloned()).collect();
        launch.carry.session = Some(session.clone());
        launch.carry.reason = Some(reason);
        let args = invocation.to_args();
        let mut command = launch.agent(&Agent {
            program: &running.current.executable,
            args: &args,
            dir: claimed.directory.as_deref(),
            run,
            version: LONG_VERSION,
            launcher: Some(installed),
        });
        // Only the first start follows the notice and the failed build.
        launch.carry.notice = None;
        launch.carry.build_failure = None;
        let mut watch = stop::Watch::start();
        let mut child = command
            .spawn()
            .map_err(fs::named("start", &running.current.executable))?;
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
            if !started && run.heard_ready() {
                started = true;
                launch.carry.warned = true;
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
        // Once the watch is dropped, a signal no longer counts as heard: it
        // ends the launcher. So what was heard is final here.
        drop(watch);
        let left = Left {
            end: AgentEnd::read(run, status),
            ready: started,
            screen: screen.took(run.took_screen()),
            heard: stop::heard().is_some(),
            replaced: Stamp::of(&launch.launcher).is_ok_and(|now| installed.replaced(&now)),
            saved: directory.is_saved(),
        };
        let runs = Runs {
            headless,
            answers_prompt,
            reason,
            kept_screen,
            back: claimed.left.take(),
            trial: running.current.is_trial(),
            previous: running.previous.is_some(),
        };
        let step = next(&runs, &left);
        // A reload that restarts under this launcher leaves the jobs
        // running, listed for the next build to take over. Best effort: a
        // list that cannot be read must not keep the terminal from being
        // put back.
        if !matches!(step, Step::Restart { .. }) {
            run.kill_left_groups().ok();
        }
        promoted?;
        let rejected = runs.rejected(&left);
        if rejected {
            build::reject(home, &running.current.executable)?;
        }
        let ending = match step {
            Step::Reexec { to } => {
                // The session the agent switched to, claimed as a switch
                // claims it: one that cannot run leaves this one on, told
                // why. The new launcher goes on as this one would: resuming
                // it by its command line, which names no session else, in
                // its directory.
                if let Some(target) = to {
                    (claimed, launch.carry.notice) = switch(home, claimed, target)?;
                }
                let resume = home
                    .session(&claimed.id)
                    .is_saved()
                    .then(|| ["--resume".into(), OsString::from(claimed.id.as_str())]);
                if let Some(here) = &claimed.directory {
                    launch.here = Some(AbsPath::new(here)?);
                }
                claimed.discard_if_unsaved(home);
                launch.args = resume.into_iter().flatten().chain(args).collect();
                launch.carry = Carry {
                    reason: Some(if claimed.id == session {
                        StartReason::Reloaded
                    } else {
                        claimed.reason
                    }),
                    left: (claimed.id != session).then_some(session),
                    warned: launch.carry.warned,
                    notice: launch.carry.notice.take(),
                    ..Carry::default()
                };
                return Ok(Ended {
                    reexec: Some(launch.clone()),
                    ..Ended::quiet(ExitCode::SUCCESS).restoring(screen.hand_over())
                });
            }
            Step::Restart { to } => {
                if headless {
                    invocation = &restarted;
                }
                if let Some(target) = to {
                    (claimed, launch.carry.notice) = switch(home, claimed, target)?;
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
                    select(launch, &claimed.id, run, current)?
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
            Step::GoBack { to, why, otherwise } => {
                let back = if home.session(&to).is_saved() {
                    Start::Resume(to)
                } else {
                    Start::New
                };
                (claimed, _) = switch(home, claimed, back)?;
                if claimed.id != session {
                    // Told in the session it went back to, so nothing is kept.
                    fs::optional(fs::remove_file(run.log()))?;
                    launch.carry.notice = Some(format!("Could not switch sessions: {why}"));
                    reason = claimed.reason;
                    continue;
                }
                otherwise
            }
            Step::Stop(stop) => stop,
        };
        let (said, restore) = match ending {
            Stop::End { said, restore } => (said, restore),
            Stop::RollBack => {
                // `next` rolls back only to a previous build.
                if let Some(previous) = running.previous.take() {
                    running.current = previous;
                }
                let message = rolled_back(status, &directory.log());
                screen.restore();
                eprintln!("{message}");
                kept_screen = false;
                launch.carry.notice = Some(message);
                reason = StartReason::Restarted;
                continue;
            }
        };
        let log = claimed.leave(home, run, &left.end);
        let refused = matches!(left.end, AgentEnd::Refused(_));
        let resume = (left.ready)
            .then(|| resume_command(home, &session, refused))
            .flatten();
        let said = match said {
            Said::Nothing => None,
            Said::Resume => resume.map(|resume| format!("To resume this session: {resume}")),
            Said::Stopped { hidden } => {
                let log = Some(log.as_path()).filter(|log| log.exists());
                stopped(&left.end, status, resume.as_deref(), hidden, log)
            }
            Said::NoPrevious => Some(format!(
                "The new build stopped during startup ({status}) and there is no previous \
                 build to roll back to; the next `rig` tries it again. Log: {}",
                log.display()
            )),
            Said::RolledBack => Some(rolled_back(status, &log)),
        };
        screen.hand_over();
        return Ok(Ended::new(&left.end, said).restoring(restore));
    }
}

/// What the launcher says when it rolls a new build that stopped during
/// startup with `status` back to the previous one; `log` is the run's.
fn rolled_back(status: ExitStatus, log: &Path) -> String {
    format!(
        "The new build stopped during startup ({status}); rolled back to the previous \
         build. `rig build` tries it again. Log: {}",
        log.display()
    )
}

/// What the launcher knows of its runs of the session when an agent ends,
/// for [`next`].
#[derive(Clone, Debug, PartialEq)]
struct Runs {
    /// Whether the run passes its exit code and output through ([`run`]).
    headless: bool,
    /// Whether it is a headless run that answers a prompt.
    answers_prompt: bool,
    /// Why the agent that ended started.
    reason: StartReason,
    /// Whether the agent is a restarted view's, which kept the screen, so
    /// what the agent said before it took the screen was not seen.
    kept_screen: bool,
    /// The session the agent switched away from to start this one, on its
    /// first start.
    back: Option<SessionId>,
    /// Whether the agent ran a trial build.
    trial: bool,
    /// Whether there is a previous build to roll back to.
    previous: bool,
}

impl Runs {
    /// Whether the launcher rejects the agent's build: a trial that ended
    /// before it said it started, but for a signal the launcher heard,
    /// which did not give it its chance.
    fn rejected(&self, left: &Left) -> bool {
        self.trial && !left.ready && !left.heard
    }
}

/// What an agent left when it ended, for [`next`].
#[derive(Clone, Debug, PartialEq)]
struct Left {
    end: AgentEnd,
    /// Whether it said it started.
    ready: bool,
    /// Whether a front took the terminal's screen since the launcher last
    /// put it back ([`Screen`]).
    screen: bool,
    /// Whether the launcher heard a signal while the agent ran.
    heard: bool,
    /// Whether the launcher's file was replaced since it started, as
    /// `rig update` does.
    replaced: bool,
    /// Whether its session is saved.
    saved: bool,
}

/// The launcher's step once an agent ended ([`next`]).
#[derive(Clone, Debug, PartialEq)]
enum Step {
    /// Hand the session to the launcher's new file ([`update::reexec`]),
    /// first switching to `to` when the agent asked to.
    Reexec { to: Option<Start> },
    /// Start the agent again, on the session `to` names when the agent
    /// switched; on the previous build when the trial was rejected.
    Restart { to: Option<Start> },
    /// Go back to `to`, the session the agent switched away from, as its
    /// start was refused for `why`; else `otherwise`, when it cannot run.
    GoBack {
        to: SessionId,
        why: String,
        otherwise: Stop,
    },
    /// Follow neither a reload nor a switch.
    Stop(Stop),
}

/// A [`Step`] that does not follow a reload or switch.
#[derive(Clone, Debug, PartialEq)]
enum Stop {
    /// Start the session again on the previous build, after the trial
    /// stopped during startup.
    RollBack,
    /// End the launcher's runs, saying `said`, after putting the terminal
    /// back when `restore`.
    End { said: Said, restore: bool },
}

/// What the launcher says last ([`Ended::said`]).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Said {
    Nothing,
    /// How to resume the session, when it can be.
    Resume,
    /// That the agent stopped ([`stopped`]), the `hidden` way.
    Stopped {
        hidden: bool,
    },
    /// That the trial stopped during startup with nothing to roll back to.
    NoPrevious,
    /// That the trial stopped during startup and was rolled back.
    RolledBack,
}

/// The step after an agent of `runs` ended as `left` says. A reload or
/// switch restarts the agent of a run with someone at the terminal, and a
/// headless run's reload when it answers a prompt, on a new launcher once
/// the launcher's file was replaced; a signal the launcher heard follows
/// neither. A refused first start after a switch goes back to the session
/// it left.
fn next(runs: &Runs, left: &Left) -> Step {
    let to = match &left.end {
        AgentEnd::Switch(target) => Some(target.clone().map_or(Start::New, Start::Resume)),
        _ => None,
    };
    let reload = left.end == AgentEnd::Reload || to.is_some();
    let restart = !runs.headless || (runs.answers_prompt && left.end == AgentEnd::Reload);
    if reload && restart && !left.heard {
        // A headless agent does not reload onto a new launcher, as told by
        // the stamp passed on, since only this one restarts it without its
        // prompt.
        return if left.replaced && !runs.headless {
            Step::Reexec { to }
        } else {
            Step::Restart { to }
        };
    }
    let otherwise = unfollowed(runs, left);
    match (&left.end, &runs.back) {
        (AgentEnd::Refused(why), Some(back))
            if !runs.headless && !left.heard && !runs.rejected(left) =>
        {
            Step::GoBack {
                to: back.clone(),
                why: why.clone(),
                otherwise,
            }
        }
        _ => Step::Stop(otherwise),
    }
}

/// How the runs end when the agent's end is not followed ([`next`]). A
/// rejected trial runs again on the previous build only for someone at the
/// terminal, or after a reload of a run that answers a prompt: a one-shot
/// run's output is already out.
fn unfollowed(runs: &Runs, left: &Left) -> Stop {
    let end = |said, restore| Stop::End { said, restore };
    if !runs.rejected(left) {
        if runs.headless {
            return end(Said::Nothing, false);
        }
        if left.end == AgentEnd::Quit {
            // The agent left the terminal as it found it; the line needs
            // it out of the full screen in case it did not.
            let resume = left.ready && left.saved;
            let said = if resume { Said::Resume } else { Said::Nothing };
            return end(said, resume && left.screen);
        }
        // A refusal said while a view had the screen was not seen.
        let hidden = runs.kept_screen || left.screen;
        return end(Said::Stopped { hidden }, left.screen);
    }
    if !runs.previous {
        return end(Said::NoPrevious, left.screen);
    }
    if !runs.headless || (runs.answers_prompt && runs.reason == StartReason::Reloaded) {
        Stop::RollBack
    } else {
        end(Said::RolledBack, left.screen)
    }
}

/// Whether an agent's front took the terminal's screen since the launcher
/// last put it back. A view keeps it across a restart, so a later start
/// that never takes it may still need it put back. Dropped while taken,
/// as when an error ends the launcher's runs, it puts the terminal back.
struct Screen {
    taken: bool,
    /// `stty sane` on the launcher's terminal, when it has one.
    sane: Option<Command>,
}

impl Screen {
    fn new(launch: &Launch) -> Self {
        Self {
            taken: false,
            sane: sane(launch),
        }
    }

    /// Notes whether the last agent `took` the screen, and returns
    /// whether it is taken.
    fn took(&mut self, took: bool) -> bool {
        self.taken |= took;
        self.taken
    }

    /// Puts the terminal back when the screen is taken.
    fn restore(&mut self) {
        if std::mem::take(&mut self.taken) {
            restore_terminal(self.sane.as_mut());
        }
    }

    /// Leaves the restore to the caller, and returns whether the screen
    /// was taken.
    fn hand_over(&mut self) -> bool {
        std::mem::take(&mut self.taken)
    }
}

impl Drop for Screen {
    fn drop(&mut self) {
        if !self.taken {
            return;
        }
        let sane = self.sane.take();
        let restore = move || {
            let mut sane = sane;
            restore_terminal(sane.as_mut());
        };
        // After a signal, a terminal nobody reads must not hold the launcher.
        match stop::heard() {
            None => restore(),
            Some(_) => stop::bounded(restore),
        }
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
    /// The launch of the new launcher to start once the session is let go
    /// ([`update::reexec`]).
    reexec: Option<Launch>,
}

impl Ended {
    /// The end of an agent that ended as `end`, followed by `said`.
    fn new(end: &AgentEnd, said: Option<String>) -> Self {
        Self {
            code: ExitCode::from(end.launcher_code()),
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

/// What the launcher says when the agent ended as `end`, not a quit, with
/// `status`. A run a signal ended, which the agent was told of, gets the
/// `resume` command as a quit does. A refused run said why itself: it
/// comes again only when a view had the screen (`kept_screen`), which hid
/// it, such as a restarted view, or one that took the screen before a
/// `--tools` name was refused. Any other end is a failure: the agent
/// stopped, where its log is when it kept one and, when the session can be
/// resumed, the `resume` command.
fn stopped(
    end: &AgentEnd,
    status: ExitStatus,
    resume: Option<&str>,
    kept_screen: bool,
    log: Option<&Path>,
) -> Option<String> {
    match end {
        AgentEnd::Signal => resume.map(|resume| format!("To resume this session: {resume}")),
        AgentEnd::Refused(why) => kept_screen.then(|| format!("rig: {why}")),
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
    // A name that cannot be read is left out: the id resumes the session.
    let name = directory.name().ok().flatten().filter(|name| {
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

/// Leaves the `current` session for `target`, which the agent asked for:
/// the session named, else a new one. A build `/reload` staged for it goes
/// with the switch. When the target cannot be run, the current session
/// carries on, told why.
fn switch(home: &Home, current: Claimed, target: Start) -> Result<(Claimed, Option<String>)> {
    let _lock = locks::hold_build_lock(home, BuildOutput::Live)?;
    let here = current.directory.clone();
    match claim(home, target, here.as_deref()) {
        Ok((next, notice)) => {
            current.discard_if_unsaved(home);
            fs::optional(fs::rename(
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
    fn leave(&self, home: &Home, run: &RunDir, end: &AgentEnd) -> PathBuf {
        let session = home.session(&self.id);
        if session.is_saved() {
            return if matches!(end, AgentEnd::Refused(_)) {
                run.log()
            } else {
                session.log()
            };
        }
        if *end != AgentEnd::Quit {
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
    let directory = match session.working_directory()? {
        Some(directory) => Some(directory),
        None => {
            if let Some(here) = here {
                moved = session.saved_directory()?.map(|gone| {
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
/// the alternate screen, mouse reports and, with `sane` ([`sane`]), raw
/// mode.
fn restore_terminal(sane: Option<&mut Command>) {
    eprint!("\x1b[?1049l\x1b[?25h\x1b[?1006l\x1b[?1002l");
    if let Some(sane) = sane {
        sane.status().ok();
    }
}

/// `stty sane` on the launcher's terminal; `None` without one.
fn sane(launch: &Launch) -> Option<Command> {
    launch.terminal.then(|| {
        let mut stty = launch.build_child(Purpose::Stty);
        stty.arg("sane");
        stty
    })
}

/// The terminal's line mode, put aside while the launcher waits for and
/// runs a build before an interactive start, and given back when dropped,
/// before the agent starts. Meanwhile the terminal neither echoes what is
/// typed over the build's output nor turns a typed Enter (CR) into a line
/// feed, which the agent, in raw mode, would read as Ctrl+J.
struct TypeAhead {
    /// `stty` with the mode `stty -g` printed.
    restore: Command,
}

impl TypeAhead {
    /// Puts the mode aside, or `None` when `stty` cannot.
    fn hold(launch: &Launch) -> Option<Self> {
        Self::set(launch, &["-icrnl", "-echo"])
    }

    /// Puts the mode aside and sets `mode` (`stty` arguments), or `None`
    /// when `stty` cannot.
    fn set(launch: &Launch, mode: &[&str]) -> Option<Self> {
        let output = launch
            .build_child(Purpose::Stty)
            .arg("-g")
            .stdin(Stdio::inherit())
            .stderr(Stdio::null())
            .output()
            .ok()
            .filter(|output| output.status.success())?;
        let saved = String::from_utf8(output.stdout).ok()?;
        let mut restore = launch.build_child(Purpose::Stty);
        restore.arg(saved.trim());
        let set = launch.build_child(Purpose::Stty).args(mode).status();
        set.is_ok_and(|status| status.success())
            .then_some(Self { restore })
    }
}

impl Drop for TypeAhead {
    fn drop(&mut self) {
        self.restore.status().ok();
    }
}

/// Asks on the terminal for `secret`, without echo, and returns the
/// answer; `None` when the user pressed Ctrl+C. The terminal reads byte by
/// byte with its signals off, so Ctrl+C arrives as a byte and the mode is
/// given back, echo included, whatever the answer.
fn ask_secret(launch: &Launch, secret: &DescribedSecret) -> Result<Option<String>> {
    let mut stderr = std::io::stderr();
    write!(stderr, "{} ({}): ", secret.help, secret.name)?;
    stderr.flush()?;
    let mode = TypeAhead::set(launch, &["-echo", "-icanon", "-isig"]);
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

/// Builds before every start, staging for this launcher, its output as
/// `output` says; cargo does no work when nothing changed. With no good
/// build yet, the build is staged even if it was rejected before, so each
/// start retries it. A build that failed with an older binary at hand is
/// returned, for that binary to start with, unless the run is `print`'s
/// (`-p`), whose caller asked for this build's answer. Settings that name no
/// build ([`FailedAt::Settings`]) refuse the start too, since that binary
/// was built from other settings.
fn rebuild(
    launch: &Launch,
    session: &SessionId,
    output: BuildOutput,
    print: bool,
) -> Result<Option<BuildFailure>> {
    let home = &launch.home();
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
    match build::compile(launch, &staged, staging, output) {
        Ok(()) => {
            match build::passed_over(home, &staged) {
                Ok(true) => eprintln!(
                    "The new build stopped during startup before, so the previous build is \
                     running; `rig build` tries it again."
                ),
                Ok(false) => {}
                Err(why) => eprintln!("warning: which build runs is not known: {why}"),
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
    fn new(launch: &Launch, executable: PathBuf, role: Role) -> Self {
        let description = build::described(launch, &executable).unwrap_or_else(|failure| {
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
/// the good build. A staged path that is no file, or cannot be looked at,
/// is warned of and passed over. Call it holding [`locks::hold_build_lock`],
/// so the claims and the good build it reads are the ones it holds.
fn select(
    launch: &Launch,
    session: &SessionId,
    run: &RunDir,
    current: Option<BuildRef>,
) -> Result<Running> {
    let home = &launch.home();
    let good = build::stamp_if_any(&home.good())?;
    let held_before = match current {
        Some(current) => Some(current),
        None => hold_good(launch, run)?,
    };
    for staged in [home.staged_for(session), home.staged()] {
        match fs::optional(fs::metadata(&staged)) {
            Ok(Some(metadata)) if metadata.is_file() => {}
            Ok(None) => continue,
            Ok(Some(_)) => {
                eprintln!(
                    "warning: {} is not a build, so it is not started",
                    staged.display()
                );
                continue;
            }
            Err(why) => {
                eprintln!("warning: a staged build is not started: {why}");
                continue;
            }
        }
        let held = held(run)?;
        // The rename claims the staged build for this launcher alone.
        if fs::optional(fs::rename(staged, &held))?.is_some() {
            let current = BuildRef::new(launch, held, Role::Trial { good });
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
fn hold_good(launch: &Launch, run: &RunDir) -> Result<Option<BuildRef>> {
    let held = held(run)?;
    let good = launch.home().good();
    Ok(
        fs::optional(build::hold(&good, &held))?
            .map(|()| BuildRef::new(launch, held, Role::Proven)),
    )
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
    if build::stamp_if_any(&home.good())? == good {
        let temporary = home.bin().join("good.tmp");
        fs::optional(fs::remove_file(&temporary))?;
        build::hold(&build.executable, &temporary)?;
        fs::rename(&temporary, home.good())?;
    }
    Ok(true)
}

#[cfg(test)]
mod tests;
