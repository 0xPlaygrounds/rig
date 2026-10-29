//! Relay pipelines: named pipelines of stages, read from
//! `RIG_HOME/pipelines.toml` and `<project>/.rig/pipelines.toml` (a
//! project's pipeline replaces one of the same name), where each stage is
//! a fresh one-job helper agent of the user's agent, one after another.
//!
//! ```toml
//! [[feature.stage]]
//! name = "plan"
//! prompt = "Plan how to {{task}}."
//! model = "anthropic/claude-sonnet-4-5"
//! tools = ["read", "search"]
//!
//! [[feature.stage]]
//! name = "implement"
//! prompt = "Implement this plan: {{prev}}\n{{feedback}}"
//!
//! [[feature.stage]]
//! name = "test"
//! prompt = "Test this change: {{prev}}. Start your answer with PASS or FAIL."
//! tools = ["read", "shell"]
//! loop_to = "implement"
//! pass_when = "^PASS"
//! max_loops = 2
//! ```
//!
//! `/pipe feature add retry to the client` runs the stages in order. A
//! stage's prompt fills in `{{task}}`, `{{prev}}` (the latest answer of the
//! stage before it) and `{{feedback}}` (the answer that last sent the
//! pipeline back). A stage whose answer misses its `pass_when` sends the
//! pipeline back to its `loop_to`, up to `max_loops` times (1 when not
//! set), and then the pipeline fails. At the end the user's agent gets a
//! queued message: one line per stage run and the last stage's answer.
//! `/pipe status` says where it is; `/pipe stop` interrupts the running
//! stage and skips the rest. Interrupting the user's agent leaves the
//! pipeline running, and a pipeline running at `/reload` resumes at its
//! current stage, which starts over.

use std::collections::BTreeMap;
use std::path::Path;

use regex::Regex;
use rig_basics::subagents::{SpawnSubagent, SpawnedAgent, SubagentsPlugin};
use rig_harness::prelude::*;
use serde::Deserialize;

/// The file of pipelines, in `RIG_HOME` and in a project's `.rig`.
pub const FILE: &str = "pipelines.toml";

/// How often a stage sends the pipeline back when its `max_loops` is not
/// set.
const MAX_LOOPS: u32 = 1;

/// The most characters of an answer a stage's line quotes.
const QUOTE: usize = 80;

/// A stage of a pipeline, as `pipelines.toml` writes it.
#[derive(Reflect, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StageDef {
    /// Its name, which `loop_to` names.
    pub name: String,
    /// The first message of its agent: `{{task}}`, `{{prev}}` and
    /// `{{feedback}}` are filled in.
    pub prompt: String,
    /// A catalog model as `vendor/model`; the user's agent's when not set.
    pub model: Option<String>,
    /// The tools it may use, from the user's agent's; all of them when not
    /// set.
    pub tools: Option<Vec<String>>,
    /// The stage to go back to when its answer misses `pass_when`.
    pub loop_to: Option<String>,
    /// A regular expression its answer must match to pass.
    pub pass_when: Option<String>,
    /// How often it may send the pipeline back.
    pub max_loops: Option<u32>,
}

/// A pipeline of `pipelines.toml`: its stages, in order.
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct PipelineDef {
    stage: Vec<StageDef>,
}

/// The pipelines of `RIG_HOME` and of the project, by name; a project's
/// replaces one of `RIG_HOME` of the same name. Each is checked.
fn load(home: &Path, root: &Path) -> Result<BTreeMap<String, Vec<StageDef>>, String> {
    let mut pipelines = BTreeMap::new();
    for path in [home.join(FILE), root.join(".rig").join(FILE)] {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let file: BTreeMap<String, PipelineDef> =
            toml::from_str(&text).map_err(|why| format!("{}: {why}", path.display()))?;
        for (name, pipeline) in file {
            check(&pipeline.stage).map_err(|why| format!("{}: `{name}`: {why}", path.display()))?;
            pipelines.insert(name, pipeline.stage);
        }
    }
    Ok(pipelines)
}

/// Why `stages` cannot run, if they cannot.
fn check(stages: &[StageDef]) -> Result<(), String> {
    if stages.is_empty() {
        return Err("it has no stage".to_owned());
    }
    for (at, stage) in stages.iter().enumerate() {
        let name = &stage.name;
        if name.trim().is_empty() || stage.prompt.trim().is_empty() {
            return Err(format!(
                "stage {}: `name` and `prompt` must not be empty",
                at + 1
            ));
        }
        if stages.iter().filter(|other| other.name == *name).count() > 1 {
            return Err(format!("two stages are named `{name}`"));
        }
        if let Some(pattern) = &stage.pass_when {
            Regex::new(pattern).map_err(|why| format!("`{name}`: `pass_when`: {why}"))?;
        }
        if let Some(to) = &stage.loop_to {
            if stage.pass_when.is_none() {
                return Err(format!("`{name}`: `loop_to` needs a `pass_when`"));
            }
            if !stages.iter().any(|other| other.name == *to) {
                return Err(format!("`{name}`: `loop_to` names no stage: `{to}`"));
            }
        }
    }
    Ok(())
}

/// On the user's agent while a pipeline of it runs; saved with the
/// session, so a restart resumes it at its current stage.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub struct PipelineRun {
    /// The pipeline's name.
    pub name: String,
    /// What `/pipe` was given to do.
    pub task: String,
    /// Its stages, as they were when it started.
    pub stages: Vec<StageDef>,
    /// The stage running.
    pub at: usize,
    /// How often each stage sent the pipeline back.
    pub loops: Vec<u32>,
    /// Each stage's latest answer.
    pub answers: Vec<String>,
    /// The answer that last sent the pipeline back, until its stage passes.
    pub feedback: String,
    /// One line per stage run, in order.
    pub lines: Vec<String>,
    /// The latest answer.
    pub last: String,
    /// The request for the running stage's agent. Not saved: a restart
    /// starts the stage over.
    #[reflect(ignore)]
    request: Option<Entity>,
    /// The running stage's agent, once spawned.
    #[reflect(ignore)]
    agent: Option<Entity>,
}

/// How a stage run went on: the pipeline carries on, or it ended, passed
/// or failed.
#[derive(Debug, PartialEq, Eq)]
enum Next {
    Stage,
    Ended { passed: bool },
}

impl PipelineRun {
    fn new(name: String, task: String, stages: Vec<StageDef>) -> Self {
        let count = stages.len();
        Self {
            name,
            task,
            stages,
            loops: vec![0; count],
            answers: vec![String::new(); count],
            ..Self::default()
        }
    }

    /// The running stage's first message.
    fn prompt(&self, stage: &StageDef) -> String {
        let prev = self
            .at
            .checked_sub(1)
            .and_then(|before| self.answers.get(before));
        stage
            .prompt
            .replace("{{task}}", &self.task)
            .replace("{{prev}}", prev.map_or("", String::as_str))
            .replace("{{feedback}}", &self.feedback)
    }

    /// Records how the running stage ended, its answer or why it has none,
    /// and moves on.
    fn record(&mut self, answer: Result<String, String>) -> Next {
        let run = self.lines.len() + 1;
        let Some(stage) = self.stages.get(self.at).cloned() else {
            return Next::Ended { passed: false };
        };
        let name = &stage.name;
        let answer = match answer {
            Ok(answer) => answer,
            Err(why) => {
                self.lines.push(format!("{run}. {name}: {why}"));
                return Next::Ended { passed: false };
            }
        };
        let said = quote(&answer);
        self.last.clone_from(&answer);
        if let Some(slot) = self.answers.get_mut(self.at) {
            slot.clone_from(&answer);
        }
        let Some(pattern) = &stage.pass_when else {
            self.lines.push(format!("{run}. {name}: answered: {said}"));
            return self.forward();
        };
        let passed = match Regex::new(pattern) {
            Ok(pass) => pass.is_match(&answer),
            Err(why) => {
                self.lines
                    .push(format!("{run}. {name}: `pass_when`: {why}"));
                return Next::Ended { passed: false };
            }
        };
        if passed {
            self.lines.push(format!("{run}. {name}: passed: {said}"));
            self.feedback.clear();
            return self.forward();
        }
        let back = stage
            .loop_to
            .as_ref()
            .and_then(|to| self.stages.iter().position(|other| other.name == *to));
        let max = stage.max_loops.unwrap_or(MAX_LOOPS);
        match (back, self.loops.get_mut(self.at)) {
            (Some(back), Some(looped)) if *looped < max => {
                *looped += 1;
                let to = stage.loop_to.as_deref().unwrap_or_default();
                self.lines.push(format!(
                    "{run}. {name}: missed `{pattern}`, back to {to} (loop {looped} of {max}): {said}"
                ));
                self.feedback = answer;
                self.at = back;
                Next::Stage
            }
            _ => {
                self.lines
                    .push(format!("{run}. {name}: missed `{pattern}`: {said}"));
                Next::Ended { passed: false }
            }
        }
    }

    /// The next stage, or the end.
    fn forward(&mut self) -> Next {
        self.at += 1;
        if self.at < self.stages.len() {
            Next::Stage
        } else {
            Next::Ended { passed: true }
        }
    }

    /// Where the pipeline stands, for `/pipe status`.
    fn status(&self) -> String {
        let stage = self
            .stages
            .get(self.at)
            .map_or("", |stage| stage.name.as_str());
        let mut text = format!(
            "Pipeline `{}` on stage {stage} ({} of {}): {}",
            self.name,
            self.at + 1,
            self.stages.len(),
            self.task
        );
        for line in &self.lines {
            text.push('\n');
            text.push_str(line);
        }
        text
    }

    /// The message the user's agent gets at the end.
    fn report(&self, passed: bool) -> String {
        let how = if passed { "passed" } else { "failed" };
        let lines = self.lines.join("\n");
        format!(
            "Pipeline `{}` {how} on: {}\n\n{lines}\n\nThe last stage's answer:\n{}",
            self.name, self.task, self.last
        )
    }
}

/// The first line of `answer`, cut to [`QUOTE`] characters.
fn quote(answer: &str) -> String {
    let line = answer.trim().lines().next().unwrap_or_default();
    let mut said: String = line.chars().take(QUOTE).collect();
    if said.len() < line.len() || answer.trim().lines().nth(1).is_some() {
        said.push('…');
    }
    said
}

/// On the request for a stage's agent.
#[derive(Component)]
struct StageRequest;

/// Adds `/pipe`; see the [crate](self) docs.
#[derive(Default)]
pub struct RelayPipelinePlugin;

impl Plugin for RelayPipelinePlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<SubagentsPlugin>() {
            app.add_plugins(SubagentsPlugin);
        }
        app.init_resource::<RigHome>()
            .init_resource::<ProjectRoot>()
            .add_command(
                "pipe",
                "Run a pipeline of pipelines.toml: /pipe <name> <task>, /pipe status, /pipe stop",
                pipe,
            )
            .add_observer(start_stage)
            .add_observer(seat)
            .add_observer(step);
    }
}

/// `/pipe <name> <task>`, `/pipe status` and `/pipe stop`.
fn pipe(
    In(args): In<CommandArgs>,
    runs: Query<&PipelineRun>,
    working: Query<(), With<ActiveTurn>>,
    (home, root): (Res<RigHome>, Res<ProjectRoot>),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let host = args.agent;
    let running = runs.get(host).ok();
    let (name, task) = args
        .args
        .split_once(char::is_whitespace)
        .unwrap_or((&args.args, ""));
    let error = |why: String| Notice::error(host, why);
    match (name, running) {
        ("status", Some(run)) => {
            notices.write(Notice::info(host, run.status()));
        }
        ("stop", Some(run)) => {
            commands.entity(host).remove::<PipelineRun>();
            match (run.agent, run.request) {
                // A stage's agent at work stops its turn, and so retires.
                (Some(agent), _) if working.contains(agent) => {
                    let reason = Some("the pipeline stopped".to_owned());
                    commands.trigger(StopTurn {
                        entity: agent,
                        reason,
                    });
                }
                (Some(agent), _) => commands.trigger(Retire { entity: agent }),
                (None, Some(request)) => commands.entity(request).try_despawn(),
                (None, None) => {}
            }
            notices.write(Notice::info(
                host,
                format!("Pipeline `{}` stopped.", run.name),
            ));
        }
        ("status" | "stop", None) => {
            notices.write(error("No pipeline runs.".to_owned()));
        }
        (_, Some(run)) => {
            let why = format!("Pipeline `{}` runs already; /pipe stop stops it.", run.name);
            notices.write(error(why));
        }
        ("", None) => {
            notices.write(error(
                "/pipe needs a pipeline and a task: /pipe <name> <task>".to_owned(),
            ));
        }
        (name, None) => match load(home.0.root(), &root.0) {
            Err(why) => {
                notices.write(error(why));
            }
            Ok(mut pipelines) => {
                let task = task.trim();
                let Some(stages) = pipelines.remove(name) else {
                    let known: Vec<String> = pipelines.into_keys().collect();
                    let why = format!(
                        "No pipeline `{name}` in {FILE} (it has: {}).",
                        known.join(", ")
                    );
                    notices.write(error(why));
                    return;
                };
                if task.is_empty() {
                    notices.write(error(format!("/pipe {name} needs a task")));
                    return;
                }
                let names: Vec<&str> = stages.iter().map(|stage| stage.name.as_str()).collect();
                let text = format!("Pipeline `{name}` started: {}.", names.join(" → "));
                notices.write(Notice::info(host, text));
                let run = PipelineRun::new(name.to_owned(), task.to_owned(), stages);
                commands.entity(host).insert(run);
            }
        },
    }
}

/// Starts the running stage of a pipeline as it starts, and as a restart
/// restores it: a fresh agent, whatever ran before.
fn start_stage(
    inserted: On<Insert<PipelineRun>>,
    mut runs: Query<&mut PipelineRun>,
    mut commands: Commands,
) {
    if let Ok(mut run) = runs.get_mut(inserted.entity) {
        spawn_stage(&mut commands, inserted.entity, &mut run);
    }
}

/// Asks for the agent of `run`'s running stage, spawned by `host`.
fn spawn_stage(commands: &mut Commands, host: Entity, run: &mut PipelineRun) {
    let Some(stage) = run.stages.get(run.at) else {
        return;
    };
    let title = format!("{} {}", run.name, stage.name);
    let request = SpawnSubagent {
        model: stage.model.clone(),
        tools: stage.tools.clone(),
        ..SpawnSubagent::helper(title).brief(run.prompt(stage)).once()
    };
    run.request = Some(commands.spawn((request, ForAgent(host), StageRequest)).id());
    run.agent = None;
}

/// Notes the agent of a stage once spawned, or fails the pipeline when
/// none was.
fn seat(
    settled: On<Add<Settled>>,
    requests: Query<(&Outcome, Option<&SpawnedAgent>), With<StageRequest>>,
    mut runs: Query<(Entity, &mut PipelineRun)>,
    mut commands: Commands,
) {
    let Ok((outcome, spawned)) = requests.get(settled.entity) else {
        return;
    };
    let request = Some(settled.entity);
    let Some((host, mut run)) = runs.iter_mut().find(|(_, run)| run.request == request) else {
        return;
    };
    match (spawned, &outcome.0) {
        (Some(&SpawnedAgent(agent)), Ok(_)) => run.agent = Some(agent),
        (_, failed) => {
            let why = failed.as_ref().err().map(|why| why.message.clone());
            let why = why.unwrap_or_else(|| "no agent was spawned".to_owned());
            let next = run.record(Err(format!("no agent: {why}")));
            go_on(&mut commands, host, &mut run, next);
        }
    }
}

/// Records each stage's answer as its agent's one turn ends, and starts
/// the next stage or ends the pipeline.
fn step(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent), With<Turn>>,
    mut runs: Query<(Entity, &mut PipelineRun)>,
    mut commands: Commands,
) {
    let Ok((outcome, &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    let Some((host, mut run)) = runs.iter_mut().find(|(_, run)| run.agent == Some(agent)) else {
        return;
    };
    let answer = match outcome {
        TurnOutcome::Answered(message) => message
            .final_answer()
            .ok_or_else(|| "it gave no answer".to_owned()),
        TurnOutcome::Failed(why) => Err(format!("failed: {why}")),
        TurnOutcome::Stopped(why) => Err(format!(
            "stopped {}",
            why.as_deref().unwrap_or("by the user")
        )),
    };
    let next = run.record(answer);
    go_on(&mut commands, host, &mut run, next);
}

/// Starts the next stage of `host`'s `run`, or ends it with the message
/// to `host`.
fn go_on(commands: &mut Commands, host: Entity, run: &mut PipelineRun, next: Next) {
    match next {
        Next::Stage => spawn_stage(commands, host, run),
        Next::Ended { passed } => {
            let report = Deliver::new(host, run.report(passed), DeliveryMode::Queue);
            commands.trigger(report.with_author(Author::plugin("pipe")));
            commands.entity(host).remove::<PipelineRun>();
        }
    }
}

#[cfg(test)]
mod tests;
