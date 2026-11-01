//! Map-reduce over files: the `map_files` tool applies one instruction to
//! every file a glob matches, each in a small helper agent of its own that
//! may change only that file, at most `concurrency` at once, and answers
//! with one combined report: `path: answer` lines, or what one more helper
//! made of them when the call asks to `reduce`.
//!
//! The job is a [`MapJob`] on the `map_files` call's entity. Each worker is
//! asked for as a [`SpawnSubagent`] helper, a child of the call, and is
//! [`WorkerOf`] the call once spawned. A Gate system refuses a worker's
//! call that could change anything but its file, a Prepare system fails a
//! worker's turn once it asked its model more than
//! [`MapConfig::per_file_turn_limit`] times, and a worker, a one-job
//! helper ([`SpawnSubagent::once`]), hands its answer, or why there is
//! none, to the job as its turn ends, and retires. When the call settles,
//! interrupted or answered, the workers still running are retired with it.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use rig_basics::subagents::{SpawnSubagent, SpawnedAgent};
use rig_harness::prelude::*;
use rig_harness::rig_ecs::session_log::Never;
use schemars::JsonSchema;
use serde::Deserialize;

/// The tool's name.
pub const MAP_FILES: &str = "map_files";

/// The tools a worker gets, those of them its caller has.
const WORKER_TOOLS: [&str; 3] = ["read", "edit", "write"];

/// The tool the reducer gets, when its caller has it.
const REDUCER_TOOLS: [&str; 1] = ["read"];

/// Files worked on at once when a call does not say.
const DEFAULT_CONCURRENCY: usize = 4;

const DESCRIPTION: &str = "Apply one instruction to every file `glob` matches, each file in a \
    subagent of its own that sees only the instruction and its file's path and may change only \
    that file. At most `concurrency` subagents run at once. Returns a count of those that \
    succeeded with the reasons of those that failed, then one line per file, `path: answer`; \
    with `reduce`, one more subagent combines the answers as it says, and its answer comes \
    instead of the lines. Use it to make the same change to many files that do not depend on \
    each other, such as documenting every public function in each file of a folder. Write the \
    `instruction` as a full brief for one file: the subagent sees nothing else.";

/// What a map worker is told, besides its instruction.
const WORKER_ROLE: &str = "You are a map worker: another agent gave you one instruction and \
    one file. Apply the instruction to that file only; you may read other files, but changing \
    any other file is refused. Nobody answers questions while you work. Your last message is \
    your answer for that file: one or two lines saying what you changed.";

/// What the reducing worker is told.
const REDUCER_ROLE: &str = "You are a reducer: another agent sent you the answers of workers \
    that each handled one file. Combine them as asked. Your last message is the combined \
    answer.";

/// The `[map_reduce]` table of the settings:
///
/// ```toml
/// [map_reduce]
/// max_concurrency = 8
/// per_file_turn_limit = 20
/// ```
#[derive(Resource, Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MapConfig {
    /// The most workers one call runs at once, whatever it asks.
    pub max_concurrency: usize,
    /// The most model calls a worker makes before it counts as failed.
    pub per_file_turn_limit: usize,
}

impl Default for MapConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 8,
            per_file_turn_limit: 20,
        }
    }
}

/// Adds the `map_files` tool, keeps each worker to its file and its model
/// calls to the limit, and, with the `tui` feature, shows "map 6/10" in
/// the status line while a job runs.
#[derive(Default)]
pub struct MapReducePlugin;

impl Plugin for MapReducePlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<rig_basics::subagents::SubagentsPlugin>() {
            app.add_plugins(rig_basics::subagents::SubagentsPlugin);
        }
        let options = ToolOptions {
            footprint: Some(Touches::Writes(Scope::Anything)),
        };
        app.add_setting::<MapConfig>("map_reduce")
            .add_open_tool(MAP_FILES, DESCRIPTION, options, on_map)
            .add_observer(on_spawned)
            .add_observer(on_worker_end)
            .add_observer(on_job_end)
            .configure_sets(
                AgentPass,
                (
                    MapReduceSystems::Prepare.in_set(Stage::Prepare),
                    MapReduceSystems::Gate.in_set(Stage::Gate),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    limit_model_calls.in_set(MapReduceSystems::Prepare),
                    keep_to_file.in_set(MapReduceSystems::Gate),
                ),
            )
            .add_systems(Update, retire_orphans.before(AgentLoop));
        #[cfg(feature = "tui")]
        app.add_systems(
            PostUpdate,
            show_progress.in_set(rig_tui::StatusSystems::Show),
        );
    }
}

/// The plugin's stage systems, for other plugins to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MapReduceSystems {
    /// Prepare: fails a worker's turn past its model call limit.
    Prepare,
    /// Gate: refuses a worker's call that could change another file.
    Gate,
}

/// On a `map_files` call: its files, its workers' answers and what it does
/// with them.
#[derive(Component)]
pub struct MapJob {
    parent: Entity,
    instruction: String,
    reduce: Option<String>,
    concurrency: usize,
    /// The tools of a worker and of the reducer.
    tools: (Vec<String>, Vec<String>),
    /// The files no worker took yet.
    queue: VecDeque<PathBuf>,
    /// How many files the glob matched.
    pub total: usize,
    /// Each file a worker finished, with its answer or why it failed.
    pub results: Vec<(PathBuf, Result<String, String>)>,
    /// Workers asked for or working.
    running: usize,
    /// Whether the call has its output.
    pub finished: bool,
}

/// On a worker's [`SpawnSubagent`] request: the job, and its file, `None`
/// for the reducer.
#[derive(Component)]
struct MapRequest {
    job: Entity,
    path: Option<PathBuf>,
}

/// On a map worker: the `map_files` call whose job it does.
#[derive(Component, Debug)]
#[relationship(relationship_target = Workers)]
pub struct WorkerOf(pub Entity);

/// On a `map_files` call: its running workers.
#[derive(Component, Debug)]
#[relationship_target(relationship = WorkerOf)]
pub struct Workers(Vec<Entity>);

/// On a map worker: the one file it may change, `None` for the reducer,
/// and how many model calls it made.
#[derive(Component, Debug)]
pub struct MapWorker {
    /// The file as the glob matched it.
    pub path: Option<PathBuf>,
    /// The file as it is on disk, which its changes must name.
    allowed: Option<PathBuf>,
    model_calls: usize,
}

/// Saved on a map worker. A restart answers its `map_files` call as
/// interrupted, but brings back a worker the exit cut off mid-turn, which
/// would carry on with no job and no file rule, so this plugin retires it.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Saved(Never), Default)]
pub struct WasMapWorker;

/// The arguments of a `map_files` call.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct MapArgs {
    /// The files, as a glob from the working directory, such as
    /// `src/api/*.rs` or `src/**/*.rs`.
    glob: String,
    /// What to do to each file, in full: each file's subagent sees only this
    /// and its file's path.
    instruction: String,
    /// How many files are worked on at once; 4 when absent, capped by the
    /// app's limit.
    concurrency: Option<usize>,
    /// When given, one more subagent combines the per-file answers as this
    /// says, and its answer is the result.
    reduce: Option<String>,
}

/// The files `pattern` matches, in order.
fn matching(pattern: &str) -> Result<Vec<PathBuf>, String> {
    let paths = glob::glob(pattern).map_err(|error| format!("`{pattern}` is no glob: {error}"))?;
    let mut files: Vec<PathBuf> = paths.flatten().filter(|path| path.is_file()).collect();
    files.sort();
    Ok(files)
}

/// Starts the job of a `map_files` call: its first workers, the rest
/// queued. Refused when no file matches.
fn on_map(
    called: On<ToolCalled<MapArgs>>,
    access: Query<&ToolAccess>,
    tools: Query<&ToolDef>,
    mut calls: Query<&mut Decision, ToolCalls>,
    config: Res<MapConfig>,
    mut commands: Commands,
) {
    let (call, args) = (called.call, &called.args);
    let files = match matching(&args.glob) {
        Ok(files) if files.is_empty() => Err(format!("No file matches `{}`.", args.glob)),
        Ok(_) if args.instruction.trim().is_empty() => Err("`instruction` is empty.".to_owned()),
        files => files,
    };
    let files = match files {
        Ok(files) => files,
        Err(why) => {
            let why = format!("{why} Nothing ran; fix the call and send it again.");
            if let Ok(mut decision) = calls.get_mut(call) {
                decision.decide(Decider::plugin("map-reduce"), Outcome::refused(why));
            }
            return;
        }
    };
    let has = |name: &str| {
        tools.iter().any(|def| def.0.name.as_str() == name)
            && access
                .get(called.agent)
                .is_ok_and(|access| access.keeps(name))
    };
    let tools = |names: &[&str]| {
        let names = names.iter().filter(|name| has(name));
        names.map(|name| (*name).to_owned()).collect()
    };
    let mut job = MapJob {
        parent: called.agent,
        instruction: args.instruction.trim().to_owned(),
        reduce: args.reduce.clone().filter(|how| !how.trim().is_empty()),
        concurrency: args
            .concurrency
            .unwrap_or(DEFAULT_CONCURRENCY)
            .clamp(1, config.max_concurrency.max(1)),
        tools: (tools(&WORKER_TOOLS), tools(&REDUCER_TOOLS)),
        total: files.len(),
        queue: files.into(),
        results: Vec::new(),
        running: 0,
        finished: false,
    };
    job.fill(call, &mut commands);
    commands.entity(call).insert(job);
}

impl MapJob {
    /// Asks for workers on queued files until `concurrency` run.
    fn fill(&mut self, job: Entity, commands: &mut Commands) {
        while self.running < self.concurrency
            && let Some(path) = self.queue.pop_front()
        {
            let text = format!("{}\n\nYour file: {}", self.instruction, path.display());
            self.ask_for(job, Some(path), text, commands);
        }
    }

    /// Asks for a worker on `path`, or the reducer, briefed with `text`.
    fn ask_for(
        &mut self,
        job: Entity,
        path: Option<PathBuf>,
        text: String,
        commands: &mut Commands,
    ) {
        let (title, tools) = match &path {
            Some(path) => (format!("map: {}", path.display()), &self.tools.0),
            None => ("map: reduce".to_owned(), &self.tools.1),
        };
        let mut request = SpawnSubagent::helper(title).brief(text).once();
        request.tools = Some(tools.clone());
        commands.spawn((
            request,
            ForAgent(self.parent),
            ChildOf(job),
            MapRequest { job, path },
        ));
        self.running += 1;
    }

    /// Takes a worker's `result`: a file's, which may start the next
    /// worker or the reducer, or the reducer's, which ends the call. The
    /// call's answer once the job is done.
    fn settle(
        &mut self,
        job: Entity,
        path: Option<PathBuf>,
        result: Result<String, String>,
        commands: &mut Commands,
    ) -> Option<Outcome> {
        self.running = self.running.saturating_sub(1);
        let Some(path) = path else {
            let combined = match result {
                Ok(answer) => format!("{}\n\n{answer}", self.tally()),
                Err(why) => format!("{}\nThe reduce step failed: {why}", self.report()),
            };
            return Some(self.end(combined));
        };
        self.results.push((path, result));
        self.fill(job, commands);
        if self.running > 0 {
            return None;
        }
        let answers = self.answers();
        match self.reduce.clone() {
            Some(how) if !answers.is_empty() => {
                let text = format!("{how}\n\nThe answers, one line per file:\n{answers}");
                self.ask_for(job, None, text, commands);
                None
            }
            _ => {
                let report = self.report();
                Some(self.end(report))
            }
        }
    }

    /// The call's answer, `text`: a failure when no file succeeded.
    fn end(&mut self, text: String) -> Outcome {
        self.finished = true;
        if self.results.iter().any(|(_, result)| result.is_ok()) {
            Outcome::from(ToolResult::success(text.into()))
        } else {
            Outcome::refused(text)
        }
    }

    /// "4 of 5 succeeded; failed: c.txt (reason)".
    fn tally(&self) -> String {
        let ok = self.results.iter().filter(|(_, result)| result.is_ok());
        let mut tally = format!("{} of {} succeeded", ok.count(), self.total);
        let failed: Vec<String> = self
            .results
            .iter()
            .filter_map(|(path, result)| {
                let why = result.as_ref().err()?;
                Some(format!("{} ({why})", path.display()))
            })
            .collect();
        if !failed.is_empty() {
            tally.push_str(&format!("; failed: {}", failed.join(", ")));
        }
        tally
    }

    /// One `path: answer` line per file that succeeded.
    fn answers(&self) -> String {
        let lines = self.results.iter().filter_map(|(path, result)| {
            let answer = result.as_ref().ok()?;
            Some(format!("{}: {}", path.display(), answer.trim()))
        });
        lines.collect::<Vec<_>>().join("\n")
    }

    /// The tally, then the answers.
    fn report(&self) -> String {
        format!("{}\n\n{}", self.tally(), self.answers())
    }
}

/// The first line of `why`, for the tally.
fn short(why: &str) -> String {
    why.lines().next().unwrap_or_default().trim().to_owned()
}

/// A worker's request settled: the agent spawned joins its job, told its
/// role, or the job counts the file as failed. A worker spawned for a job
/// that ended is retired at once.
fn on_spawned(
    settled: On<Add<Settled>>,
    requests: Query<(&MapRequest, &Outcome, Option<&SpawnedAgent>)>,
    mut jobs: Query<(&mut MapJob, &mut Decision), ToolCalls>,
    mut commands: Commands,
) {
    let Ok((MapRequest { job, path }, outcome, spawned)) = requests.get(settled.entity) else {
        return;
    };
    let (job, path) = (*job, path.clone());
    let Some((mut map, mut decision)) = jobs.get_mut(job).ok().filter(|(map, _)| !map.finished)
    else {
        if let Some(&SpawnedAgent(agent)) = spawned {
            commands.trigger(Retire { entity: agent });
        }
        return;
    };
    let worker = match (spawned, &outcome.0) {
        (Some(&SpawnedAgent(agent)), Ok(_)) => agent,
        (_, outcome) => {
            let why = match outcome {
                Err(report) => format!("not started: {}", short(&report.message)),
                Ok(_) => "not started".to_owned(),
            };
            if let Some(answer) = map.settle(job, path, Err(why), &mut commands) {
                decision.decide(Decider::plugin("map-reduce"), answer);
            }
            return;
        }
    };
    let allowed = path
        .as_ref()
        .map(|path| std::fs::canonicalize(path).unwrap_or_else(|_| path.clone()));
    let (tag, role) = match path {
        Some(_) => ("map_worker_role", WORKER_ROLE),
        None => ("map_reducer_role", REDUCER_ROLE),
    };
    let section = PromptSection::new(PromptSection::ORDER_ROLE, tag, role);
    commands.spawn((Name::new(tag), section, SectionOf(worker)));
    commands.entity(worker).insert((
        MapWorker {
            path,
            allowed,
            model_calls: 0,
        },
        WorkerOf(job),
        WasMapWorker,
    ));
}

/// A worker's turn ended: its answer, or why there is none, goes to its
/// job, which may ask for the next worker. `rig_basics::subagents` retires the
/// worker, a one-job helper, as the same turn ends.
fn on_worker_end(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    workers: Query<(&MapWorker, &WorkerOf)>,
    mut jobs: Query<(&mut MapJob, &mut Decision), ToolCalls>,
    mut commands: Commands,
) {
    let Ok((outcome, &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    let Ok((MapWorker { path, .. }, &WorkerOf(job))) = workers.get(agent) else {
        return;
    };
    let Ok((mut map, mut decision)) = jobs.get_mut(job) else {
        return;
    };
    let result = match outcome {
        TurnOutcome::Answered(message) => message
            .final_answer()
            .ok_or_else(|| "ended without a final message".to_owned()),
        TurnOutcome::Failed(why) => Err(short(why)),
        TurnOutcome::Stopped(stop) => match stop.cause() {
            StopCause::User => Err("stopped".to_owned()),
            cause => Err(format!("stopped: {cause}")),
        },
    };
    if let Some(answer) = map.settle(job, path.clone(), result, &mut commands) {
        decision.decide(Decider::plugin("map-reduce"), answer);
    }
}

/// A `map_files` call settled, answered or interrupted: the workers it
/// still has are retired, and the requests for workers not spawned yet
/// cancelled.
fn on_job_end(
    settled: On<Add<Settled>>,
    mut jobs: Query<(&mut MapJob, Option<&Workers>, Option<&Children>)>,
    open: Query<(), (With<MapRequest>, Without<Settled>)>,
    mut commands: Commands,
) {
    let Ok((mut job, workers, children)) = jobs.get_mut(settled.entity) else {
        return;
    };
    job.finished = true;
    for worker in workers.into_iter().flat_map(Workers::iter) {
        commands.trigger(Retire { entity: worker });
    }
    let children = children.into_iter().flat_map(|children| children.iter());
    for request in children.filter(|&request| open.contains(request)) {
        commands.entity(request).despawn();
    }
}

/// Fails a worker's turn once it asks its model more than
/// [`MapConfig::per_file_turn_limit`] times: the call is never sent.
fn limit_model_calls(
    calls: Query<(&ForAgent, &CallOf), (Added<Admitted>, ModelCalls, With<TurnCall>)>,
    mut workers: Query<&mut MapWorker>,
    config: Res<MapConfig>,
    mut commands: Commands,
) {
    for (&ForAgent(agent), &CallOf(turn)) in &calls {
        let Ok(mut worker) = workers.get_mut(agent) else {
            continue;
        };
        worker.model_calls += 1;
        if worker.model_calls > config.per_file_turn_limit {
            let why = format!(
                "went past the limit of {} model calls",
                config.per_file_turn_limit
            );
            commands.entity(turn).insert(TurnOutcome::Failed(why));
        }
    }
}

/// Refuses a map worker's call that could change something, unless its
/// footprint writes only the worker's own file. Reading stays open.
fn keep_to_file(
    mut calls: Query<
        (&ToolRequest, &ForAgent, &Footprint, &mut Decision),
        (ReadyToStart, ToolCalls),
    >,
    workers: Query<&MapWorker>,
) {
    for (request, &ForAgent(agent), Footprint(touches), mut decision) in &mut calls {
        let Ok(worker) = workers.get(agent) else {
            continue;
        };
        if touches.reads_only() {
            continue;
        }
        let Some(allowed) = &worker.allowed else {
            decision.decide(
                Decider::plugin("map-reduce"),
                Outcome::refused("A reducer changes no file."),
            );
            continue;
        };
        let why = match touches {
            Touches::Writes(Scope::Paths(paths))
                if paths.iter().all(|path| same_file(path, allowed)) =>
            {
                continue;
            }
            Touches::Writes(Scope::Paths(paths)) => {
                let named: Vec<String> = paths
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect();
                format!(
                    "You may change only {}; `{}` was not changed.",
                    allowed.display(),
                    named.join("`, `")
                )
            }
            _ => format!(
                "You may change only {}; `{}` could change any file, so it did not run.",
                allowed.display(),
                request.call.function.name
            ),
        };
        decision.decide(Decider::plugin("map-reduce"), Outcome::refused(why));
    }
}

/// Whether `named`, from the working directory, is the file `allowed`.
fn same_file(named: &Path, allowed: &Path) -> bool {
    let named = std::fs::canonicalize(named)
        .or_else(|_| std::path::absolute(named))
        .unwrap_or_else(|_| named.to_path_buf());
    named == allowed
}

/// Retires each map worker a restart brought back: its call was answered
/// as interrupted, so nothing waits for its answer. Before the agent loop,
/// so its restored turn never goes on.
fn retire_orphans(
    restored: Query<Entity, (Added<WasMapWorker>, Without<WorkerOf>)>,
    mut commands: Commands,
) {
    for worker in &restored {
        commands.trigger(Retire { entity: worker });
    }
}

/// "map 6/10" on the caller's turn while its job runs.
#[cfg(feature = "tui")]
fn show_progress(
    jobs: Query<(&MapJob, &CallOf), Changed<MapJob>>,
    mut turns: Query<&mut rig_tui::StatusItems>,
) {
    use rig_tui::{Side, StatusItem, Tone};
    /// After the agent's subagents at work; gone just before them.
    const PROGRESS: StatusItem = StatusItem::new("map-reduce", Side::Left, 65, 14);
    for (job, &CallOf(turn)) in &jobs {
        if let Ok(mut items) = turns.get_mut(turn) {
            let text = match job.finished {
                true => String::new(),
                false => format!("map {}/{}", job.results.len(), job.total),
            };
            items.show(PROGRESS.says(text, Tone::Dim));
        }
    }
}

#[cfg(test)]
mod tests;
