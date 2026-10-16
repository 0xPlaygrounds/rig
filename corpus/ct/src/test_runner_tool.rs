//! CT-04: a structured test runner tool, `run_tests {filter?, only_failed?,
//! timeout_secs?}`.
//!
//! The command comes from the project's `[tests]` table (`.rig/settings.toml`
//! over the user's `settings.toml`: `command = "cargo test"`, `kind =
//! "cargo"`), else from what the project looks like (`Cargo.toml`,
//! `go.mod`, `pyproject.toml`, `package.json`). `run_tests` is an open tool:
//! each call starts the command through the project's workspace and the
//! shared launcher ([`Processes`]) on a thread of its own, whose output
//! lines come back through a [`Running::stream`] on the call's job entity.
//! Each line is read by the kind's [`Parser`] and kept in a tail of 80, and
//! becomes the call's [`ToolProgress`] (shown by every front) and its
//! [`IfStopped`]: the counts and failures so far, which the kernel answers
//! with when the call is interrupted or its turn stopped.
//!
//! The result gives the counts, the duration and each failure's name,
//! `file:line`, message and first 30 lines; passing tests are never named.
//! Output no parser can read falls back to the exit code and the last 80
//! lines, and says so. The last finished run's failures are [`LastRun`],
//! saved with the session, which `only_failed: true` reruns exactly.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::RecvTimeoutError;
use std::thread;
use std::time::Instant;

use bevy_tasks::{IoTaskPool, TaskPool};
use futures::channel::mpsc::{UnboundedSender, unbounded};
use rig_harness::prelude::*;
use rig_harness::rig_core::tool::ToolErrorKind;
use rig_harness::rig_tools::process::{Group, KillOnDrop, Launcher, Process, Spawn};
use rig_harness::rig_tools::workspace::{Stat, Workspace};
use schemars::JsonSchema;
use serde::Deserialize;

mod parse;
pub use parse::{FAILURE_LINES, Failure, Filter, Kind, Parser};

#[cfg(test)]
mod tests;

/// The tool's name.
pub const RUN_TESTS: &str = "run_tests";
/// Who decides a `run_tests` call.
const NAME: &str = "test-runner";
/// The last lines of output a fallback result keeps.
pub const TAIL_LINES: usize = 80;
/// Seconds a run may take unless the call says otherwise.
const DEFAULT_TIMEOUT: u64 = 600;
/// How often a run's thread checks its deadline and whether its call is
/// still waiting for it.
const POLL: Duration = Duration::from_millis(100);

const DESCRIPTION: &str = "Run the project's test suite and get a short report: how many \
    tests passed, failed and were skipped, how long it took, and for each failure its name, \
    file:line, message and first 30 lines of output. Passing tests are not listed. `filter` \
    runs the tests the runner's own filter matches; `only_failed: true` reruns exactly the \
    tests that failed in the last run of this session. `timeout_secs` defaults to 600. Run \
    tests with this, not through `shell`. After fixing failures, rerun them with \
    `only_failed: true`, then run the whole suite once.";

/// The `run_tests` tool.
#[derive(Default)]
pub struct TestRunnerPlugin;

impl Plugin for TestRunnerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Processes>()
            .init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .init_resource::<LastRun>()
            .keeps_saved::<LastRun>()
            .add_project_setting::<TestsConfig>("tests")
            .add_open_tool(RUN_TESTS, DESCRIPTION, ToolOptions::default(), start)
            .add_observer(take_event)
            .add_observer(end_job);
    }
}

/// The `[tests]` table, which a project's `.rig/settings.toml` may set.
#[derive(Resource, Deserialize, Default, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct TestsConfig {
    /// The command line, run with `sh -c`, the filter's arguments after it;
    /// `None` picks one from what the project looks like.
    pub command: Option<String>,
    /// How its output is read; `None` guesses from the command.
    pub kind: Option<Kind>,
}

/// The last finished run of this session, saved with it.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Resource, Saved, Default)]
pub struct LastRun {
    /// The full names of the tests that failed, as their runner takes them
    /// back as filters.
    pub failed: Vec<String>,
    /// Its counts and duration, as the result said them.
    pub summary: String,
}

/// The arguments of `run_tests`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RunTestsArgs {
    /// Run only the tests this matches, as the test runner's own filter.
    #[serde(default)]
    pub filter: Option<String>,
    /// Rerun exactly the tests that failed in the last run of this session.
    #[serde(default)]
    pub only_failed: bool,
    /// Seconds before the run is stopped (default 600).
    #[serde(default)]
    pub timeout_secs: Option<u64>,
}

/// What a run's thread reports, in order.
pub enum RunEvent {
    /// The command started: how its output is read, its line, and its
    /// process group.
    Started(Kind, String, Group),
    /// A line of its output, stdout and stderr in order.
    Line(String),
    /// It ended.
    Ended(Ending),
}

/// How a run ended.
pub enum Ending {
    /// The command exited, with its code (`None` for a signal).
    Exited(Option<i32>),
    /// It ran past its timeout, after this many seconds, and was killed.
    TimedOut(u64),
    /// It could not run.
    Failed(String),
}

/// A `run_tests` call's run: its stream of [`RunEvent`]s and what it read.
#[derive(Component)]
#[relationship(relationship_target = Job)]
pub struct JobOf(pub Entity);

/// The run of a `run_tests` call, despawned with the call.
#[derive(Component)]
#[relationship_target(relationship = JobOf, linked_spawn)]
pub struct Job(Entity);

/// Kills a run's process group when its job goes, at once: when its call
/// settled, or the app exits.
#[derive(Component)]
struct Kill {
    _group: KillOnDrop,
}

/// What a run read so far, on its job.
#[derive(Component)]
pub struct TestRun {
    command: String,
    parser: Parser,
    tail: VecDeque<String>,
    started: Instant,
}

/// What a run's thread needs.
struct Plan {
    workspace: Arc<dyn Workspace>,
    dir: PathBuf,
    launcher: Launcher,
    config: TestsConfig,
    filter: Filter,
    timeout: u64,
}

/// Starts a `run_tests` call's run, or answers it at once when it asks to
/// rerun failures and none are recorded.
fn start(
    called: On<ToolCalled<RunTestsArgs>>,
    (processes, workspace, root): (Res<Processes>, Res<ProjectWorkspace>, Res<ProjectRoot>),
    (config, last): (Res<TestsConfig>, Res<LastRun>),
    dirs: Query<&WorkDir>,
    mut calls: Query<&mut Decision, ToolCalls>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let call = called.call;
    let args = &called.args;
    let filter = if args.only_failed {
        if last.failed.is_empty() {
            if let Ok(mut decision) = calls.get_mut(call) {
                let text = "No failures are recorded from the last run in this session, so \
                    nothing was run. Call `run_tests` without `only_failed` to run the suite.";
                decision.decide(
                    Decider::plugin(NAME),
                    ToolResult::success(text.to_owned().into()),
                );
            }
            return;
        }
        Filter::Exactly(last.failed.clone())
    } else {
        args.filter
            .clone()
            .filter(|pattern| !pattern.trim().is_empty())
            .map_or(Filter::All, Filter::Matching)
    };
    let dir = dirs
        .get(called.agent)
        .map_or_else(|_| root.0.clone(), |dir| dir.0.clone());
    let plan = Plan {
        workspace: workspace.0.clone(),
        dir,
        launcher: processes.launcher(),
        config: config.clone(),
        filter,
        timeout: args.timeout_secs.unwrap_or(DEFAULT_TIMEOUT).max(1),
    };
    let (events, received) = unbounded();
    let spawned = thread::Builder::new()
        .name("rig-run-tests".to_owned())
        .spawn(move || run(plan, &events));
    if let Err(error) = spawned {
        if let Ok(mut decision) = calls.get_mut(call) {
            let failed = ToolExecutionError::new(
                ToolErrorKind::Other,
                format!("could not start the run: {error}"),
            );
            decision.decide(Decider::plugin(NAME), ToolResult::failed(failed));
        }
        return;
    }
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    let run = TestRun {
        command: String::new(),
        parser: Parser::default(),
        tail: VecDeque::new(),
        started: Instant::now(),
    };
    commands.entity(call).insert(ToolProgress::default());
    commands.spawn((
        Name::new("run_tests run"),
        JobOf(call),
        run,
        Running::stream(pool, &wake, received),
    ));
}

/// The run's thread: picks the command, runs it until it ends, its
/// timeout passes or its call stops waiting, and reports each line.
fn run(plan: Plan, events: &UnboundedSender<RunEvent>) {
    let send = |event| events.unbounded_send(event).is_ok();
    let picked = match plan.config.command {
        Some(command) => Some((plan.config.kind.unwrap_or(Kind::guess(&command)), command)),
        None => detect(&*plan.workspace, &plan.dir).and_then(|kind| {
            let command = kind.default_command()?.to_owned();
            Some((plan.config.kind.unwrap_or(kind), command))
        }),
    };
    let Some((kind, command)) = picked else {
        let why = "no test command: set `command` in the `[tests]` table of the project's \
            `.rig/settings.toml`, or add a Cargo.toml, go.mod, pyproject.toml or package.json";
        send(RunEvent::Ended(Ending::Failed(why.to_owned())));
        return;
    };
    // `sh -c '<command> "$@"' sh <filter args…>` passes the filter
    // arguments through unquoted by any shell.
    let mut args: Vec<OsString> =
        vec!["-c".into(), format!("{command} \"$@\"").into(), "sh".into()];
    args.extend(
        kind.filter_args(&plan.filter)
            .into_iter()
            .map(OsString::from),
    );
    let how = Spawn {
        stdin: None,
        merge_stderr: true,
    };
    let spawned = plan
        .workspace
        .spawn(&plan.launcher, Some(&plan.dir), "sh".as_ref(), &args, how);
    let process = match spawned {
        Ok(process) => process,
        Err(error) => {
            let why = format!("could not start `{command}`: {error}");
            send(RunEvent::Ended(Ending::Failed(why)));
            return;
        }
    };
    if send(RunEvent::Started(kind, command, process.group())) {
        let deadline = Instant::now().checked_add(Duration::from_secs(plan.timeout));
        if let Some(ending) = follow(process, events, deadline, plan.timeout) {
            send(RunEvent::Ended(ending));
        }
    }
}

/// Reports `process`'s lines until it ends; `None` once nobody listens.
/// Dropping the process kills its group, whatever it left running.
fn follow(
    mut process: Process,
    events: &UnboundedSender<RunEvent>,
    deadline: Option<Instant>,
    timeout: u64,
) -> Option<Ending> {
    let past = || deadline.is_some_and(|deadline| Instant::now() >= deadline);
    loop {
        if past() {
            return Some(Ending::TimedOut(timeout));
        }
        match process.lines().recv_timeout(POLL) {
            Ok(line) => {
                let line = line.text().trim_end_matches(['\n', '\r']).to_owned();
                events.unbounded_send(RunEvent::Line(line)).ok()?;
            }
            Err(RecvTimeoutError::Timeout) if events.is_closed() => return None,
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    loop {
        match process.wait_for(POLL) {
            Ok(Some(status)) => return Some(Ending::Exited(status.code())),
            Ok(None) if events.is_closed() => return None,
            Ok(None) if past() => return Some(Ending::TimedOut(timeout)),
            Ok(None) => {}
            Err(error) => return Some(Ending::Failed(error.to_string())),
        }
    }
}

/// The kind of project in `dir`, from the files that mark it.
fn detect(workspace: &dyn Workspace, dir: &Path) -> Option<Kind> {
    let has = |name: &str| matches!(workspace.stat(&dir.join(name)), Ok(Stat::File { .. }));
    [
        (&["Cargo.toml"][..], Kind::Cargo),
        (&["go.mod"], Kind::Go),
        (
            &[
                "pyproject.toml",
                "pytest.ini",
                "setup.cfg",
                "tox.ini",
                "conftest.py",
            ],
            Kind::Pytest,
        ),
        (&["package.json"], Kind::Npm),
    ]
    .into_iter()
    .find(|(files, _)| files.iter().any(|file| has(file)))
    .map(|(_, kind)| kind)
}

/// Takes each event of a run: a line goes through the parser into the
/// call's progress and what it answers if stopped; the end answers it.
fn take_event(
    event: On<Insert<Done<RunEvent>>>,
    mut jobs: Query<(&Done<RunEvent>, &JobOf, &mut TestRun)>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut last: ResMut<LastRun>,
    mut commands: Commands,
) {
    let job = event.entity;
    let Ok((Done(event), &JobOf(call), mut run)) = jobs.get_mut(job) else {
        return;
    };
    match event {
        RunEvent::Started(kind, command, group) => {
            run.parser = Parser::new(*kind);
            run.command.clone_from(command);
            commands.entity(job).try_insert(Kill {
                _group: KillOnDrop(group.clone()),
            });
        }
        RunEvent::Line(line) => {
            run.parser.feed(line);
            run.tail.push_back(line.clone());
            if run.tail.len() > TAIL_LINES {
                run.tail.pop_front();
            }
            let stopped = ToolExecutionError::new(ToolErrorKind::Cancelled, "cancelled")
                .with_model_feedback(run.report("Cancelled before the run finished"));
            commands.entity(call).try_insert((
                ToolProgress(line.clone()),
                IfStopped(ToolResult::failed(stopped).into()),
            ));
        }
        RunEvent::Ended(ending) => {
            let (result, finished) = run.finish(ending);
            if let Some(finished) = finished {
                *last = finished;
            }
            if let Ok(mut decision) = calls.get_mut(call) {
                decision.decide(Decider::plugin(NAME), result);
            }
        }
    }
}

impl TestRun {
    /// The counts and duration so far.
    fn counts(&self) -> String {
        let [passed, failed, skipped] = self.parser.counts();
        let took = self.started.elapsed().as_secs_f64();
        format!("{passed} passed, {failed} failed, {skipped} skipped in {took:.1} s")
    }

    /// `headline`, then what finished: the counts and failures, or the
    /// tail when the output could not be read.
    fn report(&self, headline: &str) -> String {
        if !self.parser.parsed() {
            return format!("{headline}; no test results could be read. {}", self.tail());
        }
        format!(
            "{headline}. What finished: {}.\n{}",
            self.counts(),
            self.failures()
        )
    }

    /// The tail of the output, with how many lines it is.
    fn tail(&self) -> String {
        let lines: Vec<&str> = self.tail.iter().map(String::as_str).collect();
        format!(
            "Last {} lines of output:\n{}",
            lines.len(),
            lines.join("\n")
        )
    }

    /// Each failure: its name, place, message and output.
    fn failures(&self) -> String {
        let mut text = String::new();
        for failure in self.parser.failures() {
            text.push_str(&format!("\nFAILED {}", failure.name));
            if let Some(place) = &failure.location {
                text.push_str(&format!(" ({place})"));
            }
            if let Some(message) = &failure.message {
                text.push_str(&format!("\n  {message}"));
            }
            for line in &failure.output {
                text.push_str(&format!("\n    | {line}"));
            }
            if failure.cut > 0 {
                text.push_str(&format!("\n    | [{} more lines]", failure.cut));
            }
            text.push('\n');
        }
        text
    }

    /// The call's result once the run ended, and the session's new
    /// [`LastRun`] when it finished and could be read.
    fn finish(&self, ending: &Ending) -> (ToolResult, Option<LastRun>) {
        let code = match ending {
            Ending::Failed(why) => {
                let failed = ToolExecutionError::new(ToolErrorKind::Other, why);
                return (ToolResult::failed(failed), None);
            }
            Ending::TimedOut(secs) => {
                let headline = format!("Timed out after {secs} s and stopped");
                let timed_out = ToolExecutionError::new(ToolErrorKind::Timeout, "timed out")
                    .with_model_feedback(self.report(&headline));
                return (ToolResult::failed(timed_out), None);
            }
            Ending::Exited(code) => code.map_or("a signal".to_owned(), |code| code.to_string()),
        };
        if !self.parser.parsed() {
            let text = format!(
                "Fell back: the output of `{}` could not be read as test results, so here \
                 are its exit code and last lines.\nExit code: {code}\n{}",
                self.command,
                self.tail()
            );
            return (ToolResult::success(text.into()), None);
        }
        let counts = self.counts();
        let mut text = format!("`{}`: {counts} (exit code {code}).\n", self.command);
        let failures = self.parser.failures();
        if failures.is_empty() && code != "0" {
            text.push_str(&format!(
                "\nNo test failed, but the run did not succeed. {}\n",
                self.tail()
            ));
        }
        text.push_str(&self.failures());
        if !failures.is_empty() {
            text.push_str("\nRerun just these with `run_tests {\"only_failed\": true}`.");
        }
        let finished = LastRun {
            failed: failures
                .iter()
                .map(|failure| failure.name.clone())
                .collect(),
            summary: counts,
        };
        (ToolResult::success(text.into()), Some(finished))
    }
}

/// Ends a call's run once the call settled, such as when it was
/// interrupted, which kills its command's process group.
fn end_job(settled: On<Add<Settled>>, calls: Query<&Job>, mut commands: Commands) {
    if let Ok(&Job(job)) = calls.get(settled.entity) {
        commands.entity(job).try_despawn();
    }
}
