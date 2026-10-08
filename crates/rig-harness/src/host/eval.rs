//! `rig eval <spec.json>`: the tasks of a spec run across several models
//! in this one process, each trial an agent entity of its own with a
//! [`WorkDir`] holding a fresh copy of the task's directory. Trials share
//! the task pools, the model connections and the effect log (each under
//! its agent's id), and up to `parallel` of them run at once. A trial is
//! scored by its task's `check` command (the tests passing) and by what
//! its model calls cost at the catalog's prices.
//!
//! The spec, with paths relative to the spec file:
//!
//! ```json
//! {
//!   "models": ["anthropic/claude-sonnet-4-5", "openai/gpt-5"],
//!   "effort": "low",
//!   "runs": 1,
//!   "parallel": 4,
//!   "timeout_secs": 1200,
//!   "tasks": [{
//!     "name": "fix-parser",
//!     "directory": "fixtures/parser",
//!     "setup": "git init -q && git add -A && git commit -qm base",
//!     "prompt": "The parser drops trailing commas. Fix it.",
//!     "check": "cargo test -q",
//!     "check_timeout_secs": 600
//!   }]
//! }
//! ```
//!
//! Each trial works in `<session>/eval/<task>.<model>.<run>/work`, beside
//! its `setup.log` and `check.log`; the run ends with `report.json` there
//! and a summary on stdout (the report itself with `--json`). A call the
//! approval policy leaves to the user is refused: nobody watches an eval.

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_core::completion::{AssistantContent, Message, Reasoning};
use serde::{Deserialize, Serialize};

use super::headless::emit;
use super::process::{detach, kill_group};
use crate::core::agent::{
    ActiveTurn, Agent, AgentId, Conversation, Effort, Interrupt, ModelChoice, Notice, NoticeLevel,
    Submit, SystemPrompt, TurnFinished,
};
use crate::core::approval::Policy;
use crate::core::blocking::blocking;
use crate::core::calls::{Done, Running, Wake, poll_calls};
use crate::core::models;
use crate::core::save::SessionPaths;
use crate::core::turn::PollCalls;
use crate::core::usage::Spending;
use crate::core::workdir::WorkDir;

/// The most of a trial's answer and of a command's output kept in the
/// report.
const EXCERPT_CHARS: usize = 2000;
/// How long a task's `setup` may run.
const SETUP_TIMEOUT: Duration = Duration::from_secs(600);
/// How often a running command is looked at.
const POLL: Duration = Duration::from_millis(100);
/// The errors kept per trial.
const KEPT_ERRORS: usize = 5;

/// Runs an eval spec, then exits.
pub struct EvalPlugin {
    /// The spec file.
    pub spec: PathBuf,
    /// Whether stdout gets the report as JSON instead of a summary.
    pub json: bool,
}

impl Plugin for EvalPlugin {
    fn build(&self, app: &mut App) {
        let spec = Spec::load(&self.spec);
        app.insert_resource(EvalRun {
            spec_path: self.spec.clone(),
            spec,
            json: self.json,
            directory: PathBuf::new(),
            reported: false,
        })
        .add_systems(Startup, plan_trials)
        .add_systems(
            Update,
            (
                (poll_calls::<Prepared>, poll_calls::<Checked>).in_set(PollCalls),
                (collect_errors, start_trials, watch_trials, report)
                    .chain()
                    .after(PollCalls),
            ),
        )
        .add_observer(on_prepared)
        .add_observer(on_checked);
    }
}

/// An eval spec. See the [module docs](self).
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    /// The catalog models (`vendor/model`) every task runs on.
    pub models: Vec<String>,
    /// A reasoning setting by its `/effort` name, used on each model that
    /// takes it.
    #[serde(default)]
    pub effort: Option<String>,
    /// How many times each task runs on each model.
    #[serde(default = "Spec::default_runs")]
    pub runs: u32,
    /// How many trials run at once.
    #[serde(default = "Spec::default_parallel")]
    pub parallel: usize,
    /// How long a trial's agent may work before it is stopped and scored.
    #[serde(default = "Spec::default_timeout")]
    pub timeout_secs: u64,
    /// The tasks.
    pub tasks: Vec<TaskSpec>,
}

impl Spec {
    fn default_runs() -> u32 {
        1
    }

    fn default_parallel() -> usize {
        4
    }

    fn default_timeout() -> u64 {
        1200
    }

    /// Reads and checks the spec at `path`.
    fn load(path: &Path) -> Result<Self, String> {
        let text = fs::read_to_string(path)
            .map_err(|failure| format!("cannot read {}: {failure}", path.display()))?;
        let spec: Self = serde_json::from_str(&text)
            .map_err(|failure| format!("{} is not an eval spec: {failure}", path.display()))?;
        if spec.models.is_empty() || spec.tasks.is_empty() {
            return Err(format!("{} names no models or no tasks", path.display()));
        }
        let mut names: Vec<&str> = spec.tasks.iter().map(|task| task.name.as_str()).collect();
        names.sort_unstable();
        if names.windows(2).any(|pair| pair.first() == pair.last()) {
            return Err("two tasks share a name".to_owned());
        }
        Ok(spec)
    }
}

/// One task of a spec.
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct TaskSpec {
    /// Its name, unique in the spec.
    pub name: String,
    /// What the agent is asked.
    pub prompt: String,
    /// The directory each trial starts from a copy of; an empty one when
    /// absent.
    #[serde(default)]
    pub directory: Option<PathBuf>,
    /// A shell command run in the copy before the agent starts.
    #[serde(default)]
    pub setup: Option<String>,
    /// The shell command that scores the trial: it passed when this exits
    /// with 0. Without one, trials are not scored.
    #[serde(default)]
    pub check: Option<String>,
    /// How long `check` may run.
    #[serde(default = "TaskSpec::default_check_timeout")]
    pub check_timeout_secs: u64,
    /// This task's own agent timeout, in place of the spec's.
    #[serde(default)]
    pub timeout_secs: Option<u64>,
}

impl TaskSpec {
    fn default_check_timeout() -> u64 {
        600
    }
}

/// The eval run.
#[derive(Resource)]
struct EvalRun {
    spec_path: PathBuf,
    spec: Result<Spec, String>,
    json: bool,
    /// Where the trials work and the report goes.
    directory: PathBuf,
    reported: bool,
}

/// An eval trial: one task on one model, one run of it.
#[derive(Component, Clone, Debug)]
pub struct Trial {
    task: TaskSpec,
    model: String,
    effort: Option<Reasoning>,
    run: u32,
    /// The trial's own directory, which holds `work`.
    directory: PathBuf,
    /// How long its agent may work.
    timeout: Duration,
}

impl Trial {
    fn work(&self) -> PathBuf {
        self.directory.join("work")
    }
}

/// A trial not started yet.
#[derive(Component)]
struct Pending;

/// A trial whose agent works.
#[derive(Component)]
struct Working {
    agent: Entity,
    started: Instant,
    timed_out: bool,
}

/// A trial's errors: its setup's, and its agent's error notices.
#[derive(Component, Default)]
struct Errors(Vec<String>);

impl Errors {
    fn push(&mut self, error: String) {
        if self.0.len() < KEPT_ERRORS {
            self.0.push(error);
        }
    }
}

/// On the agent of a trial: the trial entity. Its subagents do not have
/// it; they inherit the trial's [`WorkDir`].
#[derive(Component, Clone, Copy, Debug)]
pub struct TrialAgent(pub Entity);

/// What preparing a trial's directory came to.
type Prepared = Result<(), String>;

/// How a check command ended.
#[derive(Clone, Debug, Default, Serialize)]
struct CommandEnd {
    /// The exit code; `None` when killed.
    code: Option<i32>,
    timed_out: bool,
    seconds: f64,
    /// The end of what it printed.
    output: String,
}

/// What a trial's check came to; `None` when the task has no check.
type Checked = Result<Option<CommandEnd>, String>;

/// A scored trial.
#[derive(Component, Clone, Debug, Serialize)]
struct Outcome {
    task: String,
    model: String,
    run: u32,
    /// The trial agent's id, which its effects are recorded under.
    agent: Option<String>,
    /// Whether the check passed; `None` without a check or when the trial
    /// never ran.
    passed: Option<bool>,
    check: Option<CommandEnd>,
    timed_out: bool,
    /// What the agent's model calls used and cost.
    spending: Spending,
    tool_calls: usize,
    /// How long the agent worked.
    seconds: f64,
    /// The start of the agent's last answer.
    answer: String,
    errors: Vec<String>,
    directory: PathBuf,
}

/// Spawns a trial for each task, model and run, or exits on a bad spec.
fn plan_trials(
    mut run: ResMut<EvalRun>,
    paths: Option<Res<SessionPaths>>,
    mut commands: Commands,
    mut exits: MessageWriter<AppExit>,
) {
    let spec = match &run.spec {
        Ok(spec) => spec.clone(),
        Err(failure) => {
            eprintln!("rig eval: {failure}");
            exits.write(AppExit::from_code(2));
            return;
        }
    };
    let Some(paths) = paths else {
        eprintln!("rig eval: no session directory to work in");
        exits.write(AppExit::from_code(1));
        return;
    };
    for model in &spec.models {
        if models::resolve(model).is_none() {
            eprintln!("rig eval: the catalog has no model `{model}`; use vendor/model");
            exits.write(AppExit::from_code(2));
            return;
        }
    }
    run.directory = paths.eval();
    let base = run
        .spec_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    let mut count = 0;
    for task in &spec.tasks {
        let mut task = task.clone();
        task.directory = task.directory.map(|directory| base.join(directory));
        for model in &spec.models {
            let effort = spec.effort.as_deref().and_then(|name| {
                let spec = models::resolve(model)?;
                models::effort_options(spec)
                    .into_iter()
                    .find(|option| option.0 == name)
                    .and_then(|option| option.1)
            });
            for number in 1..=spec.runs.max(1) {
                let trial = Trial {
                    directory: run.directory.join(format!(
                        "{}.{}.{number}",
                        slug(&task.name),
                        slug(model)
                    )),
                    task: task.clone(),
                    model: model.clone(),
                    effort,
                    run: number,
                    timeout: Duration::from_secs(task.timeout_secs.unwrap_or(spec.timeout_secs)),
                };
                commands.spawn((
                    Name::new(format!("trial {} · {model} · {number}", task.name)),
                    trial,
                    Pending,
                    Errors::default(),
                ));
                count += 1;
            }
        }
    }
    eprintln!(
        "rig eval: {} tasks × {} models × {} runs = {count} trials, {} at a time, in {}",
        spec.tasks.len(),
        spec.models.len(),
        spec.runs.max(1),
        spec.parallel.max(1),
        run.directory.display()
    );
}

/// Starts pending trials while fewer than `parallel` run.
fn start_trials(
    run: Res<EvalRun>,
    pending: Query<(Entity, &Trial), With<Pending>>,
    unfinished: Query<(), (With<Trial>, Without<Pending>, Without<Outcome>)>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Ok(spec) = &run.spec else {
        return;
    };
    let free = spec
        .parallel
        .max(1)
        .saturating_sub(unfinished.iter().count());
    // Spawn order: tasks, then models, then runs.
    let mut waiting: Vec<(Entity, &Trial)> = pending.iter().collect();
    waiting.sort_by_key(|(entity, _)| *entity);
    for (entity, trial) in waiting.into_iter().take(free) {
        let trial = trial.clone();
        let work = async move {
            blocking(move || Ok(prepare(&trial)))
                .await
                .unwrap_or_else(|failure| Err(failure.to_string()))
        };
        commands
            .entity(entity)
            .remove::<Pending>()
            .insert(Running::spawn(pool(), &wake, work));
    }
}

/// Copies the task's directory into the trial's and runs its setup.
fn prepare(trial: &Trial) -> Prepared {
    let work = trial.work();
    if trial.directory.exists() {
        fs::remove_dir_all(&trial.directory)
            .map_err(|failure| format!("cannot clear {}: {failure}", trial.directory.display()))?;
    }
    fs::create_dir_all(&work)
        .map_err(|failure| format!("cannot create {}: {failure}", work.display()))?;
    if let Some(from) = &trial.task.directory {
        copy_tree(from, &work)
            .map_err(|failure| format!("cannot copy {}: {failure}", from.display()))?;
    }
    if let Some(setup) = &trial.task.setup {
        let end = run_command(
            setup,
            &work,
            &trial.directory.join("setup.log"),
            SETUP_TIMEOUT,
        )?;
        if end.code != Some(0) {
            return Err(format!(
                "setup failed ({}): {}",
                describe(&end),
                end.output.trim()
            ));
        }
    }
    Ok(())
}

/// Starts the trial's agent on its prompt, or scores a failed setup.
fn on_prepared(
    done: On<Add<Done<Prepared>>>,
    mut trials: Query<(&Trial, &Done<Prepared>, &mut Errors)>,
    mut commands: Commands,
) {
    let entity = done.entity;
    let Ok((trial, Done(prepared), mut errors)) = trials.get_mut(entity) else {
        return;
    };
    if let Err(failure) = prepared {
        errors.push(failure.clone());
        let outcome = Outcome::unrun(trial, &errors);
        progress(&outcome);
        commands
            .entity(entity)
            .remove::<Done<Prepared>>()
            .insert(outcome);
        return;
    }
    let work = trial.work();
    let agent = commands
        .spawn((
            Name::new(format!("eval {} · {}", trial.task.name, trial.model)),
            Agent,
            TrialAgent(entity),
            WorkDir(work.clone()),
            SystemPrompt(format!(
                "{}\n\n<environment>\nWorking directory: {}\nPlatform: {} ({})\n\
                 Nobody watches this run: do the task without asking questions, and stop \
                 when it is done.\n</environment>",
                SystemPrompt::default().0,
                work.display(),
                std::env::consts::OS,
                std::env::consts::ARCH,
            )),
            // Trials work on copies: nothing to ask about.
            Policy::default(),
            Effort(trial.effort),
            ModelChoice(trial.model.clone()),
        ))
        .id();
    commands.trigger(Submit {
        entity: agent,
        text: trial.task.prompt.clone(),
    });
    commands
        .entity(entity)
        .remove::<Done<Prepared>>()
        .insert(Working {
            agent,
            started: Instant::now(),
            timed_out: false,
        });
}

/// Keeps each trial agent's error notices with its trial.
fn collect_errors(
    mut notices: MessageReader<Notice>,
    agents: Query<&TrialAgent>,
    mut trials: Query<&mut Errors>,
) {
    for notice in notices.read() {
        if notice.level != NoticeLevel::Error {
            continue;
        }
        let Some(TrialAgent(trial)) = notice.agent.and_then(|agent| agents.get(agent).ok()) else {
            continue;
        };
        if let Ok(mut errors) = trials.get_mut(*trial) {
            errors.push(notice.text.clone());
        }
    }
}

/// Stops agents past their timeout, and checks each trial whose agent's
/// turn ended, or whose agent is idle past its timeout.
fn watch_trials(
    mut finished: MessageReader<TurnFinished>,
    agents: Query<(&TrialAgent, Has<ActiveTurn>)>,
    mut trials: Query<(Entity, &Trial, &mut Working), Without<Running<Checked>>>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let mut ended: Vec<Entity> = finished
        .read()
        .filter_map(|turn| agents.get(turn.agent).ok())
        .map(|(&TrialAgent(trial), _)| trial)
        .collect();
    for (entity, trial, mut working) in &mut trials {
        if working.started.elapsed() <= trial.timeout {
            continue;
        }
        let busy = agents.get(working.agent).is_ok_and(|(_, busy)| busy);
        if !busy {
            ended.push(entity);
        } else if !working.timed_out {
            working.timed_out = true;
            commands.trigger(Interrupt {
                entity: working.agent,
            });
        }
    }
    ended.sort_unstable();
    ended.dedup();
    for entity in ended {
        let Ok((entity, trial, _)) = trials.get(entity) else {
            continue;
        };
        let trial = trial.clone();
        let work = async move {
            blocking(move || Ok(check(&trial)))
                .await
                .unwrap_or_else(|failure| Err(failure.to_string()))
        };
        commands
            .entity(entity)
            .insert(Running::spawn(pool(), &wake, work));
    }
}

/// Runs the task's check in the trial's directory.
fn check(trial: &Trial) -> Checked {
    let Some(check) = &trial.task.check else {
        return Ok(None);
    };
    run_command(
        check,
        &trial.work(),
        &trial.directory.join("check.log"),
        Duration::from_secs(trial.task.check_timeout_secs),
    )
    .map(Some)
}

/// Scores a checked trial.
fn on_checked(
    done: On<Add<Done<Checked>>>,
    trials: Query<(&Trial, &Working, &Done<Checked>, &Errors)>,
    agents: Query<(&AgentId, &Spending, &Conversation)>,
    mut commands: Commands,
) {
    let entity = done.entity;
    let Ok((trial, working, Done(checked), errors)) = trials.get(entity) else {
        return;
    };
    let mut errors = errors.0.clone();
    let check = match checked {
        Ok(check) => check.clone(),
        Err(failure) => {
            errors.push(format!("check could not run: {failure}"));
            None
        }
    };
    let agent = agents.get(working.agent).ok();
    let (tool_calls, answer) = agent.map_or((0, String::new()), |(_, _, conversation)| {
        summarize(conversation)
    });
    let outcome = Outcome {
        task: trial.task.name.clone(),
        model: trial.model.clone(),
        run: trial.run,
        agent: agent.map(|(id, ..)| id.0.clone()),
        passed: match (&trial.task.check, &check) {
            (None, _) => None,
            (Some(_), Some(end)) => Some(end.code == Some(0)),
            (Some(_), None) => Some(false),
        },
        check,
        timed_out: working.timed_out,
        spending: agent.map(|(_, spending, _)| *spending).unwrap_or_default(),
        tool_calls,
        seconds: working.started.elapsed().as_secs_f64(),
        answer,
        errors,
        directory: trial.work(),
    };
    progress(&outcome);
    commands
        .entity(entity)
        .remove::<(Working, Done<Checked>)>()
        .insert(outcome);
}

/// How many tool calls the conversation made, and the start of its last
/// answer.
fn summarize(conversation: &Conversation) -> (usize, String) {
    let mut calls = 0;
    let mut answer = String::new();
    for message in &conversation.0 {
        let Message::Assistant(reply) = message else {
            continue;
        };
        let mut text = String::new();
        for part in &reply.content {
            match part {
                AssistantContent::ToolCall(_) => calls += 1,
                AssistantContent::Text(part) => text.push_str(&part.text),
                _ => {}
            }
        }
        if !text.trim().is_empty() {
            answer = text;
        }
    }
    (calls, excerpt(&answer))
}

impl Outcome {
    /// A trial that never got to run its agent.
    fn unrun(trial: &Trial, errors: &Errors) -> Self {
        Self {
            task: trial.task.name.clone(),
            model: trial.model.clone(),
            run: trial.run,
            agent: None,
            passed: trial.task.check.as_ref().map(|_| false),
            check: None,
            timed_out: false,
            spending: Spending::default(),
            tool_calls: 0,
            seconds: 0.0,
            answer: String::new(),
            errors: errors.0.clone(),
            directory: trial.work(),
        }
    }

    fn verdict(&self) -> &'static str {
        match (self.passed, self.timed_out) {
            (Some(true), _) => "pass",
            (Some(false), true) => "fail (timed out)",
            (Some(false), false) => "fail",
            (None, _) if !self.errors.is_empty() && self.agent.is_none() => "error",
            (None, _) => "done",
        }
    }
}

/// One line on stderr per finished trial.
fn progress(outcome: &Outcome) {
    eprintln!(
        "rig eval: {} · {} · run {}: {}, {}, {:.0}s, {} tool calls",
        outcome.task,
        outcome.model,
        outcome.run,
        outcome.verdict(),
        cost(&outcome.spending),
        outcome.seconds,
        outcome.tool_calls,
    );
    if let Some(error) = outcome.errors.first() {
        eprintln!("  {error}");
    }
}

/// Once every trial is scored: writes the report and exits.
fn report(
    mut run: ResMut<EvalRun>,
    trials: Query<(Entity, Option<&Outcome>), With<Trial>>,
    mut exits: MessageWriter<AppExit>,
) {
    if run.reported || trials.is_empty() || trials.iter().any(|(_, outcome)| outcome.is_none()) {
        return;
    }
    run.reported = true;
    let mut outcomes: Vec<(Entity, &Outcome)> = trials
        .iter()
        .filter_map(|(entity, outcome)| Some((entity, outcome?)))
        .collect();
    outcomes.sort_by_key(|(entity, _)| *entity);
    let outcomes: Vec<&Outcome> = outcomes.into_iter().map(|(_, outcome)| outcome).collect();
    let models: Vec<ModelSummary> = match &run.spec {
        Ok(spec) => spec
            .models
            .iter()
            .map(|model| ModelSummary::of(model, &outcomes))
            .collect(),
        Err(_) => Vec::new(),
    };
    let report = Report {
        spec: run.spec_path.clone(),
        directory: run.directory.clone(),
        models,
        trials: outcomes,
    };
    let path = run.directory.join("report.json");
    let written = fs::create_dir_all(&run.directory)
        .and_then(|()| File::create(&path))
        .map_err(|failure| failure.to_string())
        .and_then(|file| {
            serde_json::to_writer_pretty(file, &report).map_err(|failure| failure.to_string())
        });
    if let Err(failure) = &written {
        eprintln!("rig eval: could not write {}: {failure}", path.display());
    }
    if run.json {
        emit(&serde_json::to_value(&report).unwrap_or_default());
    } else {
        print!("{}", report.table());
        if written.is_ok() {
            println!("Report: {}", path.display());
        }
    }
    exits.write(AppExit::Success);
}

/// `report.json`.
#[derive(Serialize)]
struct Report<'a> {
    spec: PathBuf,
    directory: PathBuf,
    models: Vec<ModelSummary>,
    trials: Vec<&'a Outcome>,
}

/// One model's trials summed.
#[derive(Serialize)]
struct ModelSummary {
    model: String,
    trials: usize,
    /// Trials whose check passed, of those with a check.
    passed: usize,
    checked: usize,
    timed_out: usize,
    /// USD, at the catalog's prices where the provider did not say.
    cost: f64,
    /// Whether some calls had no known price, so `cost` is a lower bound.
    unpriced: bool,
    mean_seconds: f64,
    tool_calls: usize,
}

impl ModelSummary {
    fn of(model: &str, outcomes: &[&Outcome]) -> Self {
        let mine: Vec<&&Outcome> = outcomes
            .iter()
            .filter(|outcome| outcome.model == model)
            .collect();
        let trials = mine.len();
        Self {
            model: model.to_owned(),
            trials,
            passed: mine
                .iter()
                .filter(|outcome| outcome.passed == Some(true))
                .count(),
            checked: mine
                .iter()
                .filter(|outcome| outcome.passed.is_some())
                .count(),
            timed_out: mine.iter().filter(|outcome| outcome.timed_out).count(),
            cost: mine.iter().map(|outcome| outcome.spending.cost).sum(),
            unpriced: mine.iter().any(|outcome| outcome.spending.unpriced > 0),
            mean_seconds: if trials == 0 {
                0.0
            } else {
                mine.iter().map(|outcome| outcome.seconds).sum::<f64>() / trials as f64
            },
            tool_calls: mine.iter().map(|outcome| outcome.tool_calls).sum(),
        }
    }
}

impl Report<'_> {
    /// The summary: a row per model, then a row per task with each
    /// model's verdicts.
    fn table(&self) -> String {
        let width = self
            .models
            .iter()
            .map(|model| model.model.chars().count())
            .max()
            .unwrap_or(5)
            .max(5);
        let mut out = format!(
            "{:<width$}  {:>8}  {:>10}  {:>8}  {:>6}\n",
            "model", "passed", "cost", "time", "tools"
        );
        for model in &self.models {
            let passed = if model.checked == 0 {
                "-".to_owned()
            } else {
                format!("{}/{}", model.passed, model.checked)
            };
            let cost = format!(
                "{}${:.4}",
                if model.unpriced { "≥" } else { "" },
                model.cost
            );
            out.push_str(&format!(
                "{:<width$}  {passed:>8}  {cost:>10}  {:>7.0}s  {:>6}\n",
                model.model, model.mean_seconds, model.tool_calls
            ));
        }
        out.push('\n');
        for trial in &self.trials {
            out.push_str(&format!(
                "{} · {} · run {}: {}\n",
                trial.task,
                trial.model,
                trial.run,
                trial.verdict()
            ));
        }
        out
    }
}

/// A cost for a progress line.
fn cost(spending: &Spending) -> String {
    spending
        .cost_label()
        .unwrap_or_else(|| "no cost known".to_owned())
}

/// Runs `command` with `sh -c` in `dir`, its output into `log`, killing it
/// after `timeout`.
fn run_command(
    command: &str,
    dir: &Path,
    log: &Path,
    timeout: Duration,
) -> Result<CommandEnd, String> {
    let output = File::create(log)
        .map_err(|failure| format!("cannot create {}: {failure}", log.display()))?;
    let errors = output
        .try_clone()
        .map_err(|failure| format!("cannot open {}: {failure}", log.display()))?;
    let mut shell = Command::new("sh");
    shell
        .arg("-c")
        .arg(command)
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(output)
        .stderr(errors);
    for name in rig::harness_protocol::env::AGENT_ONLY {
        shell.env_remove(name);
    }
    detach(&mut shell);
    let started = Instant::now();
    let mut child = shell
        .spawn()
        .map_err(|failure| format!("cannot run sh: {failure}"))?;
    let (code, timed_out) = loop {
        match child.try_wait() {
            Ok(Some(status)) => break (status.code(), false),
            Ok(None) if started.elapsed() >= timeout => {
                kill_group(&mut child);
                child.wait().ok();
                break (None, true);
            }
            Ok(None) => std::thread::sleep(POLL),
            Err(failure) => {
                kill_group(&mut child);
                return Err(format!("waiting for `{command}` failed: {failure}"));
            }
        }
    };
    let printed = fs::read(log).unwrap_or_default();
    Ok(CommandEnd {
        code,
        timed_out,
        seconds: started.elapsed().as_secs_f64(),
        output: tail(&String::from_utf8_lossy(&printed)),
    })
}

/// How a command ended, in words.
fn describe(end: &CommandEnd) -> String {
    match (end.code, end.timed_out) {
        (_, true) => format!("timed out after {:.0}s", end.seconds),
        (Some(code), false) => format!("exit code {code}"),
        (None, false) => "killed by a signal".to_owned(),
    }
}

/// Copies the directory `from` into `to`, links as links.
fn copy_tree(from: &Path, to: &Path) -> std::io::Result<()> {
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let target = to.join(entry.file_name());
        if kind.is_dir() {
            fs::create_dir_all(&target)?;
            copy_tree(&entry.path(), &target)?;
        } else if kind.is_symlink() {
            let link = fs::read_link(entry.path())?;
            #[cfg(unix)]
            std::os::unix::fs::symlink(link, &target)?;
            #[cfg(not(unix))]
            fs::copy(entry.path(), &target).map(drop)?;
        } else {
            fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

/// The first [`EXCERPT_CHARS`] of `text`.
fn excerpt(text: &str) -> String {
    match text.char_indices().nth(EXCERPT_CHARS) {
        Some((cut, _)) => format!("{}…", text.get(..cut).unwrap_or(text)),
        None => text.to_owned(),
    }
}

/// The last [`EXCERPT_CHARS`] of `text`.
fn tail(text: &str) -> String {
    let count = text.chars().count();
    if count <= EXCERPT_CHARS {
        return text.to_owned();
    }
    let skip = count - EXCERPT_CHARS;
    let start = text
        .char_indices()
        .nth(skip)
        .map_or(text.len(), |(index, _)| index);
    format!("…{}", text.get(start..).unwrap_or_default())
}

/// `name` as a file name: letters, digits, `-` and `_`.
fn slug(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

/// Preparing and checking trials wait on other threads: the IO pool.
fn pool() -> &'static IoTaskPool {
    IoTaskPool::get_or_init(TaskPool::default)
}
