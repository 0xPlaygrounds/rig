//! The `helper` tool: the model hands tasks to other rig agents, each a
//! one-shot `rig -p --json` child process, and gets back each one's final
//! answer, or why it failed. `tasks` run side by side ([`AT_ONCE`] at a
//! time), a `chain` runs in order with `{previous}` standing for the answer
//! before, and `task` is one.
//!
//! A child works in its parent's directory, or one inside it, on its
//! parent's model, with its parent's tools or fewer (never `helper` unless
//! named), of those registered as the parent started: a fresh rig of the
//! same build has those, and refuses a `--tools` name it has not. A child
//! built from a plugin list changed since the parent started may lack one:
//! its refusal names it, and the model can name the tools it needs. It runs [`HelperCommand`] through [`Processes`], with an empty
//! stdin. Past its timeout or dead, it fails its task; stopping the turn
//! stops every child, so none outlives the call: on unix, while the run
//! goes on, each gets the time a launcher takes to end ([`STOP_WINDOW`])
//! before it is killed, so its own helpers are stopped in turn; as the run
//! exits, the usual grace.
//!
//! What a child spends is the calling agent's, as purpose `helper`: each
//! `usage` line it writes is recorded as a [`Spent`] as it arrives, and
//! its own helpers' are among them. A child that is not on this protocol
//! version, that ended before its `exit` line, or whose `exit` line says
//! its spend is not all known, counts in [`UnknownSpend`], and its result
//! says so. With a [`budget`](crate::budget), a child is
//! started with what is left of it.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use rig_harness::harness_protocol::{self, STOP_WINDOW};
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::{Usage, UsageTotals};
use rig_harness::rig_core::tool::{ToolExecutionError, ToolOutput, ToolResult};
use rig_tools::process::{Ended, Launcher, OutputLine, Scope, Spawn};
use rig_tools::workspace::{Local, Workspace};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::budget::{Budget, HandedDown, hand_down};
use crate::print::VERSION;
use crate::telemetry::usage::{SessionUsage, Spent, UnknownSpend, UsagePlugin};

/// The tool's name.
pub const HELPER: &str = "helper";

/// How many helpers run at once; a call takes at most twice as many.
pub const AT_ONCE: usize = 4;

/// The program a helper runs, before `-p --json …`: the launcher that
/// started this agent, else `rig` on the `PATH`.
#[derive(Resource, Clone, Debug, PartialEq, Eq)]
pub struct HelperCommand {
    /// The program.
    pub program: OsString,
    /// The arguments that come first.
    pub args: Vec<OsString>,
}

impl Default for HelperCommand {
    fn default() -> Self {
        Self {
            program: harness_protocol::launcher().unwrap_or_else(|| "rig".into()),
            args: Vec::new(),
        }
    }
}

/// Adds the `helper` tool, and [`UsagePlugin`] when it is missing.
#[derive(Default)]
pub struct HelpersPlugin;

impl Plugin for HelpersPlugin {
    fn build(&self, app: &mut App) {
        let description = format!(
            "Hand tasks to helpers: other rig agents that start fresh, without this \
             conversation, so a task says all it needs. `tasks` run side by side ({AT_ONCE} \
             at a time, {} at most), `chain` in order (`{{previous}}` in a task is the answer \
             before it), `task` alone. The call returns once every helper finished, and you \
             wait until then: give work that should run alongside, such as a slow test, to a \
             helper of the same call. Each works in `dir` (default: yours) with your tools \
             or the `tools` named; helpers in one directory see each other's edits, so give \
             them separate files or directories (such as git worktrees). Helpers pay for \
             independent parts and for slow work beside yours, not for edits you already \
             know. You get back each one's final answer, or why it failed, its session and \
             what it cost, or that some of that is unknown.",
            2 * AT_ONCE
        );
        if !app.is_plugin_added::<UsagePlugin>() {
            app.add_plugins(UsagePlugin);
        }
        app.init_resource::<HelperCommand>()
            .init_resource::<Processes>()
            .init_resource::<ProjectRoot>()
            .init_resource::<Reported>()
            .add_open_tool(HELPER, &description, ToolOptions::default(), start)
            .add_systems(AgentPass, record.in_set(Stage::Review))
            .add_systems(Update, record.before(AgentLoop))
            .add_systems(
                Last,
                record_at_exit
                    .after(rig_ecs::StopTurnsOnExit)
                    .before(rig_ecs::SaveSystems)
                    .run_if(on_message::<AppExit>),
            );
    }

    /// Notes the tools every plugin registered by now, which a child has
    /// too, unlike one registered while the app runs, such as an MCP
    /// server's.
    fn cleanup(&self, app: &mut App) {
        let world = app.world_mut();
        let mut tools = world.query::<&ToolDef>();
        let names = tools.iter(world).map(|tool| tool.0.name.to_string());
        let lendable = Lendable(names.collect());
        world.insert_resource(lendable);
    }
}

/// What children reported and is not yet recorded, sent from their
/// threads, and how many have not settled what they spent: still running,
/// or ended with some of it unknown, which is not recorded yet.
#[derive(Resource)]
struct Reported {
    sender: async_channel::Sender<Heard>,
    receiver: async_channel::Receiver<Heard>,
    unsettled: Arc<AtomicU64>,
}

/// A child's report: a call's spend, or that it ended with all of its
/// spend known, or some of it unknown.
enum Heard {
    Spent(Spent),
    Known,
    Unknown,
}

impl Default for Reported {
    fn default() -> Self {
        let (sender, receiver) = async_channel::unbounded();
        Self {
            sender,
            receiver,
            unsettled: Arc::default(),
        }
    }
}

/// Records what children reported: in the pass a call's result arrives,
/// which comes after everything its children reported, so the spend is
/// in before the next model call; and before the agent loop, for what came
/// while no call ended, such as from a stopped call's children.
fn record(reported: Res<Reported>, mut unknown: ResMut<UnknownSpend>, mut commands: Commands) {
    let unknowns = drain(&reported, &mut commands);
    if unknowns > 0 {
        unknown.0 += unknowns;
    }
}

/// On the exit, before the session is saved and the `exit` line written:
/// records what came, such as from the turns the exit stopped, then counts
/// each child that has not settled what it spent, which it would report
/// after this run is gone, as spend unknown: each child once.
fn record_at_exit(
    reported: Res<Reported>,
    mut unknown: ResMut<UnknownSpend>,
    mut commands: Commands,
) {
    let unknowns = drain(&reported, &mut commands);
    unknown.0 += unknowns + reported.unsettled.swap(0, Ordering::SeqCst);
}

/// Records each report that came: a spend, or how a child ended, which
/// settles it; so at the exit a child whose end was not recorded yet,
/// with the spend it reported before it, is counted as unknown. How many
/// children ended with some of their spend unknown.
fn drain(reported: &Reported, commands: &mut Commands) -> u64 {
    let mut unknowns = 0;
    while let Ok(heard) = reported.receiver.try_recv() {
        match heard {
            Heard::Spent(spent) => commands.trigger(spent),
            Heard::Known => settle(&reported.unsettled),
            Heard::Unknown => {
                unknowns += 1;
                settle(&reported.unsettled);
            }
        }
    }
    unknowns
}

/// One child fewer that has not settled what it spent.
fn settle(unsettled: &AtomicU64) {
    let fewer = |count: u64| count.checked_sub(1);
    unsettled
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, fewer)
        .ok();
}

/// The tools a child has: those registered as the app started.
#[derive(Resource, Default)]
struct Lendable(Vec<String>);

/// A task for one helper.
#[derive(Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
struct Job {
    /// What the helper does, with everything it needs to know.
    task: String,
    /// Its directory, relative to yours; yours by default.
    #[serde(default)]
    dir: Option<String>,
    /// Its tools, of yours; yours but `helper` by default.
    #[serde(default)]
    tools: Option<Vec<String>>,
}

/// `task` (with `dir` and `tools`), `tasks` or `chain`.
#[derive(Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
struct Args {
    /// One task.
    #[serde(default)]
    task: Option<String>,
    /// `task`'s directory.
    #[serde(default)]
    dir: Option<String>,
    /// `task`'s tools.
    #[serde(default)]
    tools: Option<Vec<String>>,
    /// Tasks side by side.
    #[serde(default)]
    tasks: Option<Vec<Job>>,
    /// Tasks in order, stopping at the first that fails.
    #[serde(default)]
    chain: Option<Vec<Job>>,
    /// Seconds each helper may take: 600 by default, 3600 at most.
    #[serde(default)]
    timeout: Option<u64>,
}

/// A child's task, and its arguments and directory.
type Planned = (String, (Vec<OsString>, PathBuf));

/// What the caller lends its helpers: its model, and its access to the
/// tools a helper may have.
struct Lent {
    model: Option<String>,
    access: ToolAccess,
    lendable: Vec<String>,
    base: PathBuf,
}

impl Lent {
    /// The arguments and directory of `job`'s child, or why it cannot run.
    fn child(&self, job: &Job) -> Result<(Vec<OsString>, PathBuf), String> {
        let lendable = self.lendable.iter().map(String::as_str);
        let tools: Vec<String> = match &job.tools {
            None => lendable
                .filter(|tool| *tool != HELPER && self.access.allows(tool))
                .map(str::to_owned)
                .collect(),
            Some(asked) => {
                let granted = self.access.grant(asked, lendable);
                let granted = granted.map_err(|refused| refused.to_string())?;
                // A child runs with what it may use in this run.
                let names = granted.names().unwrap_or_default().iter();
                names.filter(|tool| granted.allows(tool)).cloned().collect()
            }
        };
        let dir = Local.join(&self.base, Path::new(job.dir.as_deref().unwrap_or(".")));
        let dir = dir
            .canonicalize()
            .map_err(|error| format!("no directory {}: {error}", dir.display()))?;
        if !dir.starts_with(&self.base) {
            return Err(format!("{} is outside your directory", dir.display()));
        }
        let mut args: Vec<OsString> = vec!["-p".into(), "--json".into()];
        args.extend(
            self.model
                .as_ref()
                .map(|model| format!("--model={model}").into()),
        );
        args.push(format!("--tools={}", tools.join(",")).into());
        args.extend(["--".into(), job.task.clone().into()]);
        Ok((args, dir))
    }
}

/// Starts a `helper` call, whose children run off the main thread.
fn start(
    called: On<ToolCalled<Args>>,
    (agents, turns): (
        Query<(Option<&ModelChoice>, Option<&ToolAccess>, Option<&WorkDir>)>,
        Query<&CallOf>,
    ),
    lendable: Option<Res<Lendable>>,
    (root, processes, command): (Res<ProjectRoot>, Res<Processes>, Res<HelperCommand>),
    (budget, handed, spent): (
        Option<Res<Budget>>,
        Option<Res<HandedDown>>,
        Res<SessionUsage>,
    ),
    (wake, reported): (Res<Wake>, Res<Reported>),
    mut commands: Commands,
) {
    let Ok((model, access, dir)) = agents.get(called.agent) else {
        return;
    };
    let lent = Lent {
        model: model.map(|model| model.0.clone()),
        access: access.cloned().unwrap_or_default(),
        lendable: lendable
            .map(|lendable| lendable.0.clone())
            .unwrap_or_default(),
        base: dir.map_or(&root.0, |dir| &dir.0).clone(),
    };
    let args = called.args.clone();
    let (chain, jobs) = match (args.task, args.tasks, args.chain) {
        (Some(task), None, None) => {
            let job = Job {
                task,
                dir: args.dir,
                tools: args.tools,
            };
            (false, vec![job])
        }
        (None, Some(jobs), None) => (false, jobs),
        (None, None, Some(jobs)) => (true, jobs),
        _ => (false, Vec::new()),
    };
    let planned: Result<Vec<Planned>, String> = match jobs.len() {
        0 => Err("give one of `task`, `tasks` or `chain`".to_owned()),
        many if many > 2 * AT_ONCE => {
            Err(format!("give at most {} tasks, not {many}", 2 * AT_ONCE))
        }
        _ => jobs
            .iter()
            .map(|job| Ok((job.task.clone(), lent.child(job)?)))
            .collect(),
    };
    let mut launcher = processes.launcher();
    for (name, left) in hand_down(budget.as_deref(), handed.as_deref(), &spent) {
        launcher = launcher.with_env(name, left);
    }
    let runner = Runner {
        command: command.clone(),
        launcher,
        timeout: Duration::from_secs(args.timeout.unwrap_or(600).clamp(1, 3600)),
        reports: Reports {
            agent: called.agent,
            turn: turns.get(called.call).ok().map(|&CallOf(turn)| turn),
            sender: reported.sender.clone(),
            unsettled: reported.unsettled.clone(),
            wake: wake.clone(),
        },
    };
    // Dropped with the call, as a stop drops it: every child dies.
    let work = async move {
        let planned = planned.map_err(ToolExecutionError::refused)?;
        let ran = rig_tools::blocking(move || Ok(runner.run(chain, planned))).await?;
        ran.map(|text| ToolResult::success(ToolOutput::text(text)))
            .map_err(ToolExecutionError::other)
    };
    let work = async move { work.await.unwrap_or_else(ToolResult::failed) };
    let pool = bevy_tasks::IoTaskPool::get_or_init(bevy_tasks::TaskPool::default);
    commands
        .entity(called.call)
        .insert(Running::answering(pool, &wake, work));
}

/// Runs a call's children, on a thread of its own.
struct Runner {
    command: HelperCommand,
    /// The call's launcher: what [`Processes`] gives, with what is left of
    /// the budget.
    launcher: Launcher,
    timeout: Duration,
    reports: Reports,
}

/// Where a call's children's spend goes: the calling agent's, in its turn.
#[derive(Clone)]
struct Reports {
    agent: Entity,
    turn: Option<Entity>,
    sender: async_channel::Sender<Heard>,
    unsettled: Arc<AtomicU64>,
    wake: Wake,
}

impl Reports {
    /// Records `usage`, answered by `model`, as soon as the app can.
    fn send(&self, model: Option<String>, usage: Usage) {
        let spent = Spent {
            agent: Some(self.agent),
            turn: self.turn,
            model,
            purpose: Some(HELPER.to_owned()),
            usage,
        };
        self.send_heard(Heard::Spent(spent));
    }

    /// Records that a child ended with all its spend `known`, or not.
    fn ended(&self, known: bool) {
        self.send_heard(if known { Heard::Known } else { Heard::Unknown });
    }

    fn send_heard(&self, heard: Heard) {
        if self.sender.try_send(heard).is_ok() {
            self.wake.wake();
        }
    }
}

/// How a child ended: its answer or why it has none, and a note of its
/// session and cost.
type Report = (Result<String, String>, String);

impl Runner {
    /// The call's result text, or why it failed.
    fn run(&self, chain: bool, jobs: Vec<Planned>) -> Result<String, String> {
        if chain {
            let (mut previous, mut notes) = (String::new(), String::new());
            for (step, (task, (mut args, dir))) in jobs.into_iter().enumerate() {
                if let Some(last) = args.last_mut() {
                    *last = task.replace("{previous}", &previous).into();
                }
                let (answer, note) = self.child(&args, &dir);
                notes.push_str(&note);
                previous =
                    answer.map_err(|why| format!("step {} failed: {why}{notes}", step + 1))?;
            }
            return Ok(previous + &notes);
        }
        let mut reports: Vec<Report> = Vec::new();
        // The call's, so a stop reaches the children of every thread.
        let call = Scope::current().unwrap_or_default();
        for jobs in jobs.chunks(AT_ONCE) {
            std::thread::scope(|scope| {
                let started: Vec<_> = jobs
                    .iter()
                    .map(|(_, (args, dir))| {
                        let call = &call;
                        scope.spawn(move || call.enter(|| self.child(args, dir)))
                    })
                    .collect();
                for thread in started {
                    let crashed = || (Err("its thread crashed".to_owned()), String::new());
                    reports.push(thread.join().unwrap_or_else(|_| crashed()));
                }
            });
        }
        if let [(answer, note)] = reports.as_slice() {
            return answer
                .clone()
                .map(|text| text + note)
                .map_err(|why| why + note);
        }
        let answered = reports.iter().filter(|(answer, _)| answer.is_ok()).count();
        let mut text = format!("{answered} of {} helpers answered.", reports.len());
        for (index, ((task, _), (answer, note))) in jobs.iter().zip(&reports).enumerate() {
            let task: String = task
                .lines()
                .next()
                .unwrap_or_default()
                .chars()
                .take(60)
                .collect();
            let (how, said) = match answer {
                Ok(said) => ("answered", said),
                Err(why) => ("failed", why),
            };
            text.push_str(&format!(
                "\n\n## {}. {task}: {how}\n\n{said}{note}",
                index + 1
            ));
        }
        if answered == 0 { Err(text) } else { Ok(text) }
    }

    /// Runs one child to its end.
    fn child(&self, args: &[OsString], dir: &Path) -> Report {
        let args = self.command.args.iter().chain(args);
        let launcher = self.launcher.clone().in_dir(dir);
        let how = Spawn {
            grace: Some(STOP_WINDOW),
            ..Spawn::default()
        };
        // Unsettled from before it can spend until what it spent is known.
        self.reports.unsettled.fetch_add(1, Ordering::SeqCst);
        let mut process = match launcher.spawn(&self.command.program, args, how) {
            Ok(process) => process,
            Err(error) => {
                // A program that was not found or may not run, or a start
                // the stopped call refused, never ran; one whose process was
                // made before another error may have spent.
                let ran = !matches!(
                    error.kind(),
                    std::io::ErrorKind::NotFound
                        | std::io::ErrorKind::PermissionDenied
                        | std::io::ErrorKind::Interrupted
                );
                self.reports.ended(!ran);
                let why = format!("it could not start: {error}");
                let note = if ran { "\n[helper spend unknown]" } else { "" };
                return (Err(why), note.to_owned());
            }
        };
        let mut read = Read::new(self.reports.clone());
        let ended = process.run(Some(Instant::now() + self.timeout), |line| {
            read.line(line);
            true
        });
        let note = read.note();
        // It stays unsettled until how it ended is recorded, after the
        // spend it reported.
        self.reports.ended(read.known());
        let answer = match ended {
            Ok(Ended::Exited(status)) => read.answer(status.code()),
            Ok(Ended::TimedOut) => Err(format!("it timed out after {}s", self.timeout.as_secs())),
            Ok(_) => Err("stopped".to_owned()),
            Err(error) => Err(error.to_string()),
        };
        (answer, note)
    }
}

/// What a child wrote: its lines' version, its `exit` line's code and
/// answer, its last error, its last lines on stderr, its session, and the
/// spend it reported, each `usage` line recorded as it is read.
struct Read {
    version: Option<u64>,
    exit: Option<(u64, Option<String>)>,
    /// Whether its `exit` line says its spend is all known.
    complete: bool,
    /// Whether a `usage` line of it did not read.
    unread: bool,
    error: Option<String>,
    stderr: Vec<String>,
    session: Option<String>,
    spent: UsageTotals,
    reports: Reports,
}

impl Read {
    fn new(reports: Reports) -> Self {
        Self {
            version: None,
            exit: None,
            complete: false,
            unread: false,
            error: None,
            stderr: Vec::new(),
            session: None,
            spent: UsageTotals::default(),
            reports,
        }
    }

    fn line(&mut self, line: OutputLine) {
        let text = match &line {
            OutputLine::Out(_) => line.text(),
            OutputLine::Err(_) => {
                self.stderr.push(line.text().trim_end().to_owned());
                let cut = self.stderr.len().saturating_sub(5);
                self.stderr.drain(..cut);
                return;
            }
        };
        let Ok(event) = serde_json::from_str::<Value>(&text) else {
            return;
        };
        let field = |key: &str| event.get(key).and_then(Value::as_str).map(str::to_owned);
        self.version = self.version.or(event.get("v").and_then(Value::as_u64));
        match field("type").as_deref() {
            Some("ready") => self.session = field("session"),
            Some("usage") if self.version == Some(VERSION) => {
                let usage = event.get("usage").cloned().unwrap_or_default();
                match serde_json::from_value::<Usage>(usage) {
                    Ok(usage) => {
                        self.spent.record(&usage);
                        self.reports.send(field("model"), usage);
                    }
                    Err(_) => self.unread = true,
                }
            }
            Some("exit") => {
                let code = event.get("code").and_then(Value::as_u64).unwrap_or(1);
                self.exit = Some((code, field("answer")));
                self.complete = event.get("usage_complete") == Some(&Value::Bool(true));
                self.error = field("error").or(self.error.take());
            }
            Some("notice") if field("level").as_deref() == Some("error") => {
                self.error = field("text");
            }
            Some("turn_end") if field("error").is_some() => self.error = field("error"),
            _ => {}
        }
    }

    /// The answer of a child that exited with `code`.
    fn answer(self, code: Option<i32>) -> Result<String, String> {
        match self.exit {
            Some((0, answer)) => Ok(answer.unwrap_or_else(|| "(no answer)".to_owned())),
            Some((code, _)) => Err(self
                .error
                .unwrap_or_else(|| format!("it failed with exit code {code}"))),
            None => {
                let code = code.map_or("a signal".to_owned(), |code| format!("exit code {code}"));
                let said = format!(
                    "it stopped with {code} before it answered\n{}",
                    self.stderr.join("\n")
                );
                Err(said.trim_end().to_owned())
            }
        }
    }

    /// Whether all it spent is known: once it wrote its `exit` line on
    /// this protocol version, saying so, and every `usage` line read.
    fn known(&self) -> bool {
        self.version == Some(VERSION) && self.exit.is_some() && self.complete && !self.unread
    }

    /// Its session and what it spent, for the end of its text.
    fn note(&self) -> String {
        let session = self
            .session
            .as_ref()
            .map(|session| format!("session {session}"));
        let spent = match (self.known(), self.spent.calls) {
            (true, 0) => "no model call".to_owned(),
            (true, _) => self.spent.cost_or_tokens(),
            (false, 0) => "spend unknown".to_owned(),
            (false, _) => format!("at least {}, the rest unknown", self.spent.cost_or_tokens()),
        };
        let said: Vec<String> = session.into_iter().chain([spent]).collect();
        format!("\n[helper {}]", said.join(", "))
    }
}

#[cfg(test)]
mod tests;
