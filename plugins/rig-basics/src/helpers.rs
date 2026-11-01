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
//! kills every child, so none outlives the call.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use rig_harness::harness_protocol;
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::UsageTotals;
use rig_harness::rig_core::tool::{ToolExecutionError, ToolOutput, ToolResult};
use rig_tools::process::{Ended, OutputLine, Scope, Spawn};
use rig_tools::workspace::{Local, Workspace};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

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

/// Adds the `helper` tool.
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
             what it cost.",
            2 * AT_ONCE
        );
        app.init_resource::<HelperCommand>()
            .init_resource::<Processes>()
            .init_resource::<ProjectRoot>()
            .add_open_tool(HELPER, &description, ToolOptions::default(), start);
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
    agents: Query<(Option<&ModelChoice>, Option<&ToolAccess>, Option<&WorkDir>)>,
    lendable: Option<Res<Lendable>>,
    (root, processes, command): (Res<ProjectRoot>, Res<Processes>, Res<HelperCommand>),
    wake: Res<Wake>,
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
    let runner = Runner {
        command: command.clone(),
        processes: processes.clone(),
        timeout: Duration::from_secs(args.timeout.unwrap_or(600).clamp(1, 3600)),
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
    processes: Processes,
    timeout: Duration,
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
        let launcher = self.processes.launcher().in_dir(dir);
        let mut process = match launcher.spawn(&self.command.program, args, Spawn::default()) {
            Ok(process) => process,
            Err(error) => return (Err(format!("it could not start: {error}")), String::new()),
        };
        let mut read = Read::default();
        let ended = process.run(Some(Instant::now() + self.timeout), |line| {
            read.line(line);
            true
        });
        let note = read.note();
        let answer = match ended {
            Ok(Ended::Exited(status)) => read.answer(status.code()),
            Ok(Ended::TimedOut) => Err(format!("it timed out after {}s", self.timeout.as_secs())),
            Ok(_) => Err("stopped".to_owned()),
            Err(error) => Err(error.to_string()),
        };
        (answer, note)
    }
}

/// What a child wrote: its `exit` line's code and answer, its last error,
/// its last lines on stderr, its session and its usage.
#[derive(Default)]
struct Read {
    exit: Option<(u64, Option<String>)>,
    error: Option<String>,
    stderr: Vec<String>,
    session: Option<String>,
    usage: Option<UsageTotals>,
}

impl Read {
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
        match field("type").as_deref() {
            Some("ready") => self.session = field("session"),
            Some("exit") => {
                let usage = event.get("usage").cloned().unwrap_or_default();
                self.usage = serde_json::from_value(usage).ok();
                let code = event.get("code").and_then(Value::as_u64).unwrap_or(1);
                self.exit = Some((code, field("answer")));
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

    /// Its session and cost, for the end of its text.
    fn note(&self) -> String {
        let session = self
            .session
            .as_ref()
            .map(|session| format!("session {session}"));
        let cost = self.usage.map(|usage| usage.cost_or_tokens());
        let said: Vec<String> = session.into_iter().chain(cost).collect();
        match said.is_empty() {
            true => String::new(),
            false => format!("\n[helper {}]", said.join(", ")),
        }
    }
}

#[cfg(test)]
mod tests;
