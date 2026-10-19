//! The `helper` tool: the model hands tasks to other rig agents, each a
//! one-shot `rig -p --json` child process, and gets back each one's final
//! answer, or why it failed, as the call's result. One task runs alone,
//! `tasks` run side by side ([`MOST_TASKS`] at most, [`AT_ONCE`] at a
//! time), and a `chain` runs in order, `{previous}` in each task standing
//! for the answer before it.
//!
//! A child starts in a directory of its own (its parent's by default, and
//! inside it when the parent is confined to its [`WorkDir`]), on its
//! parent's model, with its parent's tools or fewer, never `helper`
//! unless asked for. It runs the program [`HelperCommand`] names, through
//! [`Processes`], so it inherits no secret and no session; stdin is empty,
//! so a child never waits on input. A child that runs past its timeout or
//! dies fails its task, and stopping the turn kills every child running.
//! No child outlives the call.
//!
//! ```
//! use rig_basics::helpers::HelpersPlugin;
//! use rig_harness::prelude::*;
//!
//! let mut app = App::new();
//! app.add_plugins(HelpersPlugin);
//! ```

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::time::{Duration, Instant};

use rig_harness::harness_protocol;
use rig_harness::prelude::*;
use rig_harness::rig_core::tool::{ToolExecutionError, ToolOutput, ToolResult};
use rig_tools::process::{Group, KillOnDrop, OutputLine, Spawn};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

/// The tool's name.
pub const HELPER: &str = "helper";

/// The most tasks one call runs side by side, or in a chain.
pub const MOST_TASKS: usize = 8;

/// How many helpers run at once.
pub const AT_ONCE: usize = 4;

/// How long a helper may take unless the call says; at most an hour.
pub const TIMEOUT: Duration = Duration::from_secs(600);

/// The program a helper runs, before its own arguments (`-p --json …`):
/// the launcher that started this agent, else `rig` on the `PATH`.
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
            "Hand tasks to helpers: other rig agents, each starting fresh with no view of \
             this conversation, so a task says everything it needs. `tasks` run side by \
             side (at most {MOST_TASKS}, {AT_ONCE} at a time), `chain` runs them in order \
             (`{{previous}}` in a task is the answer before it), and `task` is one. The call \
             returns once every helper has finished, and you wait until then: put work that \
             should run alongside, such as a slow test, in a helper of the same call. Each \
             helper works in `dir` (default: yours) with your tools or the `tools` named. \
             Helpers in one directory see each other's edits, so give side-by-side helpers \
             separate files, or separate directories (such as git worktrees). You get back \
             each helper's final answer, or why it failed, and its session."
        );
        app.init_resource::<HelperCommand>()
            .init_resource::<Processes>()
            .init_resource::<ProjectRoot>()
            .add_open_tool(HELPER, &description, ToolOptions::default(), start);
    }
}

/// A task for one helper.
#[derive(Deserialize, JsonSchema, Clone, Debug)]
#[serde(deny_unknown_fields)]
struct Job {
    /// What the helper does, with everything it needs to know.
    task: String,
    /// Its working directory, relative to yours; yours by default.
    #[serde(default)]
    dir: Option<String>,
    /// The tools it may use, of yours; yours without `helper` by default.
    #[serde(default)]
    tools: Option<Vec<String>>,
}

/// One of `task`, `tasks` or `chain`.
#[derive(Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
struct Args {
    /// One task for one helper.
    #[serde(default)]
    task: Option<String>,
    /// The working directory of `task`'s helper.
    #[serde(default)]
    dir: Option<String>,
    /// The tools of `task`'s helper.
    #[serde(default)]
    tools: Option<Vec<String>>,
    /// Tasks for helpers that work side by side.
    #[serde(default)]
    tasks: Option<Vec<Job>>,
    /// Tasks run one after another; `{previous}` in a task is the answer
    /// before it. It stops at the first that fails.
    #[serde(default)]
    chain: Option<Vec<Job>>,
    /// How many seconds each helper may take; 600 by default, 3600 at most.
    #[serde(default)]
    timeout: Option<u64>,
}

/// A helper ready to start: its arguments after the program's own, and
/// its directory.
#[derive(Clone, Debug)]
struct Child {
    args: Vec<OsString>,
    dir: PathBuf,
}

/// What a call runs.
enum Plan {
    One(Child),
    Side(Vec<(String, Child)>),
    /// The steps' tasks, each made into a child once the one before
    /// answered.
    Chain(Vec<(String, Child)>),
}

/// What the caller's agent lends its helpers.
struct Lent<'a> {
    model: Option<&'a str>,
    tools: Vec<String>,
    base: PathBuf,
    confined: bool,
}

impl Lent<'_> {
    /// The child for `job`, or why not.
    fn child(&self, job: &Job) -> Result<Child, String> {
        let tools = match &job.tools {
            None => self
                .tools
                .iter()
                .filter(|tool| *tool != HELPER)
                .cloned()
                .collect(),
            Some(asked) => {
                if let Some(tool) = asked.iter().find(|tool| !self.tools.contains(tool)) {
                    return Err(format!(
                        "`{tool}` is not one of your tools ({})",
                        self.tools.join(", ")
                    ));
                }
                asked.clone()
            }
        };
        let dir = match job.dir.as_deref() {
            None => self.base.clone(),
            Some(dir) => self.base.join(dir),
        };
        let dir = dir
            .canonicalize()
            .map_err(|error| format!("no directory {}: {error}", dir.display()))?;
        if self.confined && !dir.starts_with(&self.base) {
            return Err(format!("{} is outside your directory", dir.display()));
        }
        let mut args: Vec<OsString> = vec!["-p".into(), "--json".into()];
        if let Some(model) = self.model {
            args.push(format!("--model={model}").into());
        }
        args.push(format!("--tools={}", tools.join(",")).into());
        args.extend(["--".into(), job.task.clone().into()]);
        Ok(Child { args, dir })
    }

    /// What `args` runs, or why it cannot.
    fn plan(&self, args: Args) -> Result<Plan, String> {
        let modes = [
            args.task.is_some(),
            args.tasks.is_some(),
            args.chain.is_some(),
        ];
        if modes.iter().filter(|given| **given).count() != 1 {
            return Err("give one of `task`, `tasks` or `chain`".to_owned());
        }
        let jobs = |jobs: Vec<Job>| -> Result<Vec<(String, Child)>, String> {
            if jobs.is_empty() || jobs.len() > MOST_TASKS {
                return Err(format!("give 1 to {MOST_TASKS} tasks, not {}", jobs.len()));
            }
            jobs.iter()
                .map(|job| Ok((job.task.clone(), self.child(job)?)))
                .collect()
        };
        match (args.task, args.tasks, args.chain) {
            (Some(task), ..) => {
                let job = Job {
                    task,
                    dir: args.dir,
                    tools: args.tools,
                };
                Ok(Plan::One(self.child(&job)?))
            }
            (_, Some(tasks), _) => Ok(Plan::Side(jobs(tasks)?)),
            (.., Some(chain)) => Ok(Plan::Chain(jobs(chain)?)),
            _ => Err("give one of `task`, `tasks` or `chain`".to_owned()),
        }
    }
}

/// Starts a `helper` call: its children run off the main thread, and the
/// call ends with their answers.
fn start(
    called: On<ToolCalled<Args>>,
    agents: Query<(Option<&ModelChoice>, Option<&ToolAccess>, Option<&WorkDir>)>,
    tools: Query<&ToolDef>,
    (root, processes, command): (Res<ProjectRoot>, Res<Processes>, Res<HelperCommand>),
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Ok((model, access, dir)) = agents.get(called.agent) else {
        return;
    };
    let access = access.cloned().unwrap_or_default();
    let names = tools.iter().map(|def| def.0.name.as_str().to_owned());
    let lent = Lent {
        model: model.map(|model| model.0.as_str()),
        tools: names.filter(|name| access.allows(name)).collect(),
        base: dir.map_or(&root.0, |dir| &dir.0).clone(),
        confined: dir.is_some(),
    };
    let timeout = called.args.timeout.map_or(TIMEOUT, Duration::from_secs);
    let timeout = timeout.clamp(Duration::from_secs(1), Duration::from_secs(3600));
    let plan = lent.plan(called.args.clone());
    let runner = Runner {
        command: command.clone(),
        processes: processes.clone(),
        timeout,
        running: Arc::new(Children::default()),
    };
    let pool = bevy_tasks::IoTaskPool::get_or_init(bevy_tasks::TaskPool::default);
    let work = async move {
        let plan = match plan {
            Ok(plan) => plan,
            Err(why) => return ToolResult::failed(ToolExecutionError::refused(why)),
        };
        // Dropped with the call, as a stop does: every child running dies.
        let running = runner.running.clone();
        let answered = rig_tools::blocking(move || Ok(runner.run(plan))).await;
        drop(running);
        match answered {
            Ok(Ok(text)) => ToolResult::success(ToolOutput::text(text)),
            Ok(Err(why)) => ToolResult::failed(ToolExecutionError::other(why)),
            Err(error) => ToolResult::failed(error),
        }
    };
    commands
        .entity(called.call)
        .insert(Running::answering(pool, &wake, work));
}

/// The children of a call that are running, killed when it is dropped.
#[derive(Default)]
struct Children {
    groups: Mutex<Vec<(usize, Group)>>,
}

impl Drop for Children {
    fn drop(&mut self) {
        let groups = self
            .groups
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner);
        for (_, group) in groups.drain(..) {
            drop(KillOnDrop(group));
        }
    }
}

/// Runs a call's children, on a thread of its own.
struct Runner {
    command: HelperCommand,
    processes: Processes,
    timeout: Duration,
    running: Arc<Children>,
}

impl Runner {
    /// The call's result text, or why it failed.
    fn run(self, plan: Plan) -> Result<String, String> {
        let running = Arc::downgrade(&self.running);
        drop(self.running);
        let one = |slot: usize, child: &Child| -> Report {
            let to = (&running, slot);
            run_child(&self.command, &self.processes, child, self.timeout, to)
        };
        match plan {
            Plan::One(child) => {
                let report = one(0, &child);
                let note = report.note();
                report
                    .answer
                    .map(|text| text + &note)
                    .map_err(|why| why + &note)
            }
            Plan::Chain(steps) => {
                let (mut previous, mut notes) = (String::new(), String::new());
                for (step, (task, child)) in steps.iter().enumerate() {
                    let mut child = child.clone();
                    if let Some(last) = child.args.last_mut() {
                        *last = task.replace("{previous}", &previous).into();
                    }
                    let report = one(step, &child);
                    notes.push_str(&report.note());
                    previous = report.answer.map_err(|why| {
                        format!("step {} of {} failed: {why}{notes}", step + 1, steps.len())
                    })?;
                }
                Ok(previous + &notes)
            }
            Plan::Side(tasks) => {
                let next = Mutex::new(0usize);
                let reports: Vec<Mutex<Option<Report>>> =
                    tasks.iter().map(|_| Mutex::new(None)).collect();
                std::thread::scope(|scope| {
                    for _ in 0..AT_ONCE.min(tasks.len()) {
                        scope.spawn(|| {
                            loop {
                                let index = {
                                    let mut next =
                                        next.lock().unwrap_or_else(PoisonError::into_inner);
                                    *next += 1;
                                    *next - 1
                                };
                                let (Some((_, child)), Some(report)) =
                                    (tasks.get(index), reports.get(index))
                                else {
                                    break;
                                };
                                let done = one(index, child);
                                *report.lock().unwrap_or_else(PoisonError::into_inner) = Some(done);
                            }
                        });
                    }
                });
                let reports: Vec<Report> = reports
                    .into_iter()
                    .map(|report| {
                        let report = report.into_inner().unwrap_or_else(PoisonError::into_inner);
                        report.unwrap_or_else(|| Report::failed("not run"))
                    })
                    .collect();
                let answered = reports
                    .iter()
                    .filter(|report| report.answer.is_ok())
                    .count();
                let parts: Vec<String> = tasks
                    .iter()
                    .zip(&reports)
                    .enumerate()
                    .map(|(index, ((task, _), report))| {
                        let (head, note) = (first_line(task), report.note());
                        match &report.answer {
                            Ok(text) => {
                                format!("## {}. {head}: answered\n\n{text}{note}", index + 1)
                            }
                            Err(why) => format!("## {}. {head}: failed\n\n{why}{note}", index + 1),
                        }
                    })
                    .collect();
                let text = format!(
                    "{answered} of {} helpers answered.\n\n{}",
                    tasks.len(),
                    parts.join("\n\n")
                );
                if answered == 0 { Err(text) } else { Ok(text) }
            }
        }
    }
}

/// How one helper ended: its answer or why it has none, and its session
/// and cost when it said them.
struct Report {
    answer: Result<String, String>,
    session: Option<String>,
    cost: Option<f64>,
}

impl Report {
    fn failed(why: impl Into<String>) -> Self {
        Self {
            answer: Err(why.into()),
            session: None,
            cost: None,
        }
    }

    /// Where its session is and what it cost, for the end of its text.
    fn note(&self) -> String {
        let mut said: Vec<String> = Vec::new();
        said.extend(
            self.session
                .as_ref()
                .map(|session| format!("session {session}")),
        );
        said.extend(self.cost.map(|cost| format!("${cost:.4}")));
        match said.is_empty() {
            true => String::new(),
            false => format!("\n[helper {}]", said.join(", ")),
        }
    }
}

/// A task's first line, cut short.
fn first_line(task: &str) -> String {
    let line = task.lines().next().unwrap_or_default().trim();
    match line.char_indices().nth(60) {
        Some((at, _)) => format!("{}…", line.get(..at).unwrap_or(line)),
        None => line.to_owned(),
    }
}

/// Runs one child to its end.
fn run_child(
    command: &HelperCommand,
    processes: &Processes,
    child: &Child,
    timeout: Duration,
    (running, slot): (&Weak<Children>, usize),
) -> Report {
    let Some(registry) = running.upgrade() else {
        return Report::failed("stopped");
    };
    let args = command.args.iter().chain(&child.args);
    let launcher = processes.launcher().in_dir(&child.dir);
    let mut process = match launcher.spawn(&command.program, args, Spawn::default()) {
        Ok(process) => process,
        Err(error) => {
            let program = Path::new(&command.program).display();
            return Report::failed(format!("could not start {program}: {error}"));
        }
    };
    registry
        .groups
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push((slot, process.group()));
    // Only the call holds the registry, so a stop drops it.
    drop(registry);
    let mut read = Read::default();
    let ended = process.run(Some(Instant::now() + timeout), None, |line| {
        read.line(line);
        true
    });
    if let Some(registry) = running.upgrade() {
        let mut groups = registry
            .groups
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        groups.retain(|(at, _)| *at != slot);
    }
    let (session, cost) = (read.session.take(), read.cost);
    let answer = match ended {
        Ok(rig_tools::process::Ended::Exited(status)) => read.answer(status.code()),
        Ok(rig_tools::process::Ended::TimedOut) => {
            Err(format!("timed out after {}s", timeout.as_secs()))
        }
        Ok(_) => Err("stopped".to_owned()),
        Err(error) => Err(error.to_string()),
    };
    Report {
        answer,
        session,
        cost,
    }
}

/// What a child wrote.
#[derive(Default)]
struct Read {
    /// Its `exit` line: the code and the answer.
    exit: Option<(u64, Option<String>)>,
    /// The last error it said, from a notice, a turn's end or its exit.
    error: Option<String>,
    /// Its last lines on stderr.
    stderr: Vec<String>,
    /// Its session, from its `ready` line.
    session: Option<String>,
    /// What it cost, from its `exit` line.
    cost: Option<f64>,
}

impl Read {
    fn line(&mut self, line: OutputLine) {
        match line {
            OutputLine::Out(text) => {
                let Ok(event) = serde_json::from_str::<Value>(&text) else {
                    return;
                };
                let field = |key: &str| event.get(key).and_then(Value::as_str).map(str::to_owned);
                match event.get("type").and_then(Value::as_str) {
                    Some("ready") => self.session = field("session"),
                    Some("exit") => {
                        self.cost = event.pointer("/usage/cost").and_then(Value::as_f64);
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
            OutputLine::Err(text) => {
                self.stderr.push(text.trim_end().to_owned());
                if self.stderr.len() > 5 {
                    self.stderr.remove(0);
                }
            }
        }
    }

    /// The answer of a child that exited with `code`.
    fn answer(self, code: Option<i32>) -> Result<String, String> {
        match self.exit {
            Some((0, answer)) => {
                Ok(answer.unwrap_or_else(|| "(it ended without an answer)".to_owned()))
            }
            Some((code, _)) => Err(self
                .error
                .unwrap_or_else(|| format!("it failed with exit code {code}"))),
            None => {
                let code = code.map_or("a signal".to_owned(), |code| format!("exit code {code}"));
                let said = self.stderr.join("\n");
                Err(format!("it stopped with {code} before it answered\n{said}")
                    .trim_end()
                    .to_owned())
            }
        }
    }
}

#[cfg(test)]
mod tests;
