//! Map-reduce over files: the `map_files` tool applies one instruction to
//! every file a glob matches, each in a small subagent of its own that may
//! change only that file, at most `concurrency` at once, and answers with
//! one combined report: `path: answer` lines, or what one more subagent
//! made of them when the call asks to `reduce`.
//!
//! The job is a [`MapJob`] on the `map_files` call's entity, and each
//! worker is an agent [`SpawnedBy`] the caller and a [`WorkerOf`] the call,
//! so interrupting the caller, which despawns its turn and the turn's
//! calls, despawns every worker with them. A worker's [`TurnEnded`] is its
//! answer or its failure; one that asks its model more than
//! [`MapConfig::per_file_turn_limit`] times is stopped and counted as
//! failed. The `edit` and `write` tools are taken over in
//! [`Plugin::finish`]: they run as before for every agent, and refuse a
//! map worker's call on any file but its own.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use bevy_tasks::{AsyncComputeTaskPool, TaskPool};
use rig_harness::prelude::*;
use rig_tools::{Edit, Write};
use schemars::JsonSchema;
use serde::Deserialize;
use serde::de::DeserializeOwned;

/// The tool's name.
pub const MAP_FILES: &str = "map_files";

/// The `read` tool, which every worker keeps.
const READ: &str = "read";

/// Files worked on at once when a call does not say.
const DEFAULT_CONCURRENCY: usize = 4;

const DESCRIPTION: &str = "Apply one instruction to every file `glob` matches, each file in a \
    subagent of its own that sees only the instruction and its file's path and may change only \
    that file. At most `concurrency` subagents run at once. Returns one line per file, `path: \
    answer`, after a count of those that succeeded and the reasons of those that failed; with \
    `reduce`, one more subagent combines the answers as it says, and its answer comes instead.";

const RULES: &[&str] = &[
    "Use `map_files` to make the same change to many files that do not depend on each other, \
     such as documenting every public function in each file of a folder. Write the \
     `instruction` as a full brief for one file: the subagent sees nothing else.",
];

/// What a map worker is told, besides its instruction.
const WORKER_ROLE: &str = "You are a map worker: another agent gave you one instruction and \
    one file. Apply the instruction to that file only; you may read other files, but changing \
    any other file is refused. Nobody answers questions while you work. Your last message is \
    your answer for that file: one or two lines saying what you changed.";

/// What the reducing worker is told.
const REDUCER_ROLE: &str = "You are a reducer: another agent sent you the answers of workers \
    that each handled one file. Combine them as asked. Your last message is the combined \
    answer.";

/// Limits on `map_files`, for the whole app.
#[derive(Resource, Clone, Debug)]
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

/// Adds the `map_files` tool, takes over `edit` and `write` so a worker
/// changes only its file, and, with the `tui` feature, shows "map 6/10" in
/// the status line while a job runs.
#[derive(Default)]
pub struct MapReducePlugin;

impl Plugin for MapReducePlugin {
    fn build(&self, app: &mut App) {
        let options = ToolOptions {
            rules: RULES,
            footprint: Footprint::Exclusive,
        };
        app.init_resource::<MapConfig>()
            .add_open_tool(MAP_FILES, DESCRIPTION, options, on_map)
            .add_observer(on_worker_end)
            .add_observer(limit_requests)
            .add_observer(drop_restored);
        #[cfg(feature = "tui")]
        app.add_systems(PostUpdate, show_progress.in_set(rig_tui::StatusSystems));
    }

    fn finish(&self, app: &mut App) {
        scope::<Edit>(app);
        scope::<Write>(app);
    }
}

/// On a `map_files` call: its files, its workers' answers and what it does
/// with them.
#[derive(Component)]
pub struct MapJob {
    instruction: String,
    reduce: Option<String>,
    concurrency: usize,
    /// The files no worker took yet.
    queue: VecDeque<PathBuf>,
    /// How many files the glob matched.
    pub total: usize,
    /// Each file a worker finished, with its answer or why it failed.
    pub results: Vec<(PathBuf, Result<String, String>)>,
    running: usize,
    /// Whether the call has its output.
    pub finished: bool,
    template: Template,
}

/// What every worker of a job takes from the calling agent.
struct Template {
    parent: Entity,
    caller: AgentId,
    request: RequestId,
    effect: rig_core::effect::EffectId,
    choice: Option<ModelChoice>,
    connection: Option<Connection>,
    effort: Effort,
    prompt: SystemPrompt,
    tools: Vec<String>,
}

/// On a map worker: the `map_files` call whose job it does. Despawning the
/// call despawns it.
#[derive(Component, Debug)]
#[relationship(relationship_target = Workers)]
pub struct WorkerOf(pub Entity);

/// On a `map_files` call: its running workers.
#[derive(Component, Debug)]
#[relationship_target(relationship = WorkerOf, linked_spawn)]
pub struct Workers(Vec<Entity>);

/// On a map worker: the one file it may change, `None` for the worker that
/// reduces, and how many model calls it made.
#[derive(Component, Debug)]
pub struct MapWorker {
    /// The file as the glob matched it.
    pub path: Option<PathBuf>,
    /// The file as it is on disk, which its `edit` and `write` calls must
    /// name.
    allowed: Option<PathBuf>,
    requests: usize,
}

/// Saved on a map worker, so a restart, which ends its job, drops it
/// instead of bringing back a worker nothing waits for.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Saved, Default)]
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

/// A call's answer.
fn answered(text: String) -> ToolOutput {
    ToolOutput(ToolResult::success(text.into()))
}

/// A call's refusal.
fn refused(why: String) -> ToolOutput {
    ToolOutput(ToolResult::failed(ToolExecutionError::other(why)))
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
    agents: Query<(
        Option<&ModelChoice>,
        Option<&Connection>,
        &Effort,
        &SystemPrompt,
        &ToolAccess,
    )>,
    config: Res<MapConfig>,
    mut commands: Commands,
) {
    let (call, args) = (called.call, &called.args);
    let Ok((choice, connection, effort, prompt, access)) = agents.get(called.agent) else {
        return;
    };
    let files = match matching(&args.glob) {
        Ok(files) if files.is_empty() => Err(format!("No file matches `{}`.", args.glob)),
        Ok(_) if args.instruction.trim().is_empty() => Err("`instruction` is empty.".to_owned()),
        files => files,
    };
    let files = match files {
        Ok(files) => files,
        Err(why) => {
            let why = format!("{why} Nothing ran; fix the call and send it again.");
            commands.entity(call).insert_if_new(refused(why));
            return;
        }
    };
    let tools = [READ, <Edit as Tool>::NAME, <Write as Tool>::NAME];
    let tools = tools.into_iter().filter(|name| access.allows(name));
    let mut job = MapJob {
        instruction: args.instruction.trim().to_owned(),
        reduce: args.reduce.clone().filter(|how| !how.trim().is_empty()),
        concurrency: args
            .concurrency
            .unwrap_or(DEFAULT_CONCURRENCY)
            .clamp(1, config.max_concurrency.max(1)),
        total: files.len(),
        queue: files.into(),
        results: Vec::new(),
        running: 0,
        finished: false,
        template: Template {
            parent: called.agent,
            caller: called.caller.clone(),
            request: RequestId(called.run.call.id.to_string()),
            effect: called.effect,
            choice: choice.cloned(),
            connection: connection.cloned(),
            effort: *effort,
            prompt: prompt.clone(),
            tools: tools.map(str::to_owned).collect(),
        },
    };
    job.fill(call, &mut commands);
    commands.entity(call).insert(job);
}

impl MapJob {
    /// Starts workers on queued files until `concurrency` run.
    fn fill(&mut self, job: Entity, commands: &mut Commands) {
        while self.running < self.concurrency
            && let Some(path) = self.queue.pop_front()
        {
            let text = format!("{}\n\nYour file: {}", self.instruction, path.display());
            self.spawn(job, Some(path), text, commands);
        }
    }

    /// Spawns a worker on `path`, or the reducer, and sends it `text`.
    fn spawn(&mut self, job: Entity, path: Option<PathBuf>, text: String, commands: &mut Commands) {
        let t = &self.template;
        let title = path.as_ref().map_or_else(
            || "map: reduce".to_owned(),
            |path| format!("map: {}", path.display()),
        );
        let (tools, tag, role) = match &path {
            Some(_) => (t.tools.clone(), "map_worker_role", WORKER_ROLE),
            None => (
                t.tools
                    .iter()
                    .filter(|name| *name == READ)
                    .cloned()
                    .collect(),
                "map_reducer_role",
                REDUCER_ROLE,
            ),
        };
        let allowed = path
            .as_ref()
            .map(|path| std::fs::canonicalize(path).unwrap_or_else(|_| path.clone()));
        let mut worker = commands.spawn_empty();
        // Before the model choice, so connecting it keeps this connection.
        if let Some(connection) = &t.connection {
            worker.insert(connection.clone());
        }
        worker.insert((
            Agent,
            Name::new(title.clone()),
            SpawnedBy(t.parent),
            WorkerOf(job),
            MapWorker {
                path,
                allowed,
                requests: 0,
            },
            WasMapWorker,
            t.effort,
            ToolAccess::Only(tools),
            t.prompt.clone(),
            EffectParent(t.effect),
        ));
        if let Some(choice) = &t.choice {
            worker.insert(choice.clone());
        }
        let worker = worker.id();
        let section = PromptSection::new(PromptSection::ORDER_ROLE, tag, role);
        commands.spawn((Name::new(tag), section, SectionOf(worker)));
        let origin = Origin::agent(t.caller.clone(), Some(t.request.clone())).titled(title);
        let brief = Deliver::new(worker, text, DeliveryMode::Queue).with_origin(origin);
        commands.trigger(brief);
        self.running += 1;
    }

    /// Takes a worker's `result`: a file's, which may start the next
    /// worker or the reducer, or the reducer's, which ends the call.
    fn settle(
        &mut self,
        job: Entity,
        path: Option<PathBuf>,
        result: Result<String, String>,
        commands: &mut Commands,
    ) {
        self.running = self.running.saturating_sub(1);
        let Some(path) = path else {
            let combined = match result {
                Ok(answer) => format!("{}\n\n{answer}", self.tally()),
                Err(why) => format!("{}\nThe reduce step failed: {why}", self.report()),
            };
            self.end(job, combined, commands);
            return;
        };
        self.results.push((path, result));
        self.fill(job, commands);
        if self.running > 0 || !self.queue.is_empty() {
            return;
        }
        let answers = self.answers();
        match self.reduce.clone() {
            Some(how) if !answers.is_empty() => {
                let text = format!("{how}\n\nThe answers, one line per file:\n{answers}");
                self.spawn(job, None, text, commands);
            }
            _ => {
                let report = self.report();
                self.end(job, report, commands);
            }
        }
    }

    /// Answers the call with `text`.
    fn end(&mut self, job: Entity, text: String, commands: &mut Commands) {
        self.finished = true;
        let output = if self.results.iter().any(|(_, result)| result.is_ok()) {
            answered(text)
        } else {
            refused(text)
        };
        commands.entity(job).try_insert_if_new(output);
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

/// A worker's turn ended: its answer, or why there is none, goes to its
/// job, and the worker goes away.
fn on_worker_end(
    end: On<TurnEnded>,
    workers: Query<(&MapWorker, &WorkerOf)>,
    mut jobs: Query<&mut MapJob>,
    mut commands: Commands,
) {
    let Ok((worker, &WorkerOf(job))) = workers.get(end.entity) else {
        return;
    };
    let result = match &end.outcome {
        TurnOutcome::Answered(message) => {
            final_answer(message).ok_or_else(|| "ended without a final message".to_owned())
        }
        TurnOutcome::Failed(why) => Err(why.clone()),
        TurnOutcome::Stopped => Err("stopped".to_owned()),
    };
    let path = worker.path.clone();
    commands.entity(end.entity).despawn();
    if let Ok(mut map) = jobs.get_mut(job) {
        map.settle(job, path, result, &mut commands);
    }
}

/// Counts a worker's model calls; one past the limit stops the worker,
/// which counts as failed.
fn limit_requests(
    prepare: On<PrepareRequest>,
    mut workers: Query<(&mut MapWorker, &WorkerOf)>,
    mut jobs: Query<&mut MapJob>,
    config: Res<MapConfig>,
    mut commands: Commands,
) {
    let Ok((mut worker, &WorkerOf(job))) = workers.get_mut(prepare.agent) else {
        return;
    };
    worker.requests += 1;
    if worker.requests <= config.per_file_turn_limit {
        return;
    }
    // Despawning the worker despawns its turn: the request is not sent.
    commands.entity(prepare.agent).despawn();
    let why = format!(
        "went past the limit of {} model calls",
        config.per_file_turn_limit
    );
    if let Ok(mut map) = jobs.get_mut(job) {
        map.settle(job, worker.path.clone(), Err(why), &mut commands);
    }
}

/// A map worker brought back by a restart has no job: its call was
/// answered as interrupted. It goes away.
fn drop_restored(
    mut restored: On<Restored>,
    workers: Query<(), With<WasMapWorker>>,
    mut commands: Commands,
) {
    if workers.contains(restored.entity) {
        restored.event_mut().resume = false;
        commands.entity(restored.entity).despawn();
    }
}

/// A file tool that a map worker may call only on its own file.
trait Scoped: Tool<Output = String, Error = ToolExecutionError> + Send + Sync + 'static {
    /// The tool.
    fn tool() -> Self;
}

impl Scoped for Edit {
    fn tool() -> Self {
        Edit
    }
}

impl Scoped for Write {
    fn tool() -> Self {
        Write
    }
}

/// Takes over the registered tool `T` with one that refuses a map
/// worker's call on any file but its own, keeping its description, rules
/// and footprint. Nothing when no plugin added `T`.
fn scope<T: Scoped>(app: &mut App)
where
    T::Args: DeserializeOwned + JsonSchema + Send + Sync + 'static,
{
    let world = app.world_mut();
    let mut tools = world.query::<(Entity, &ToolDef, &ToolRules, &Footprint)>();
    let found = tools
        .iter(world)
        .find(|(_, def, ..)| def.0.name.as_str() == <T as Tool>::NAME)
        .map(|(tool, def, rules, footprint)| {
            (tool, def.0.description.clone(), rules.0.clone(), *footprint)
        });
    let Some((tool, description, rules, footprint)) = found else {
        return;
    };
    world.entity_mut(tool).insert(Disabled);
    let rules: Vec<&str> = rules.iter().map(String::as_str).collect();
    let options = ToolOptions {
        rules: &rules,
        footprint,
    };
    app.add_open_tool(<T as Tool>::NAME, &description, options, run_scoped::<T>);
}

/// Runs a call of `T`, unless a map worker names a file not its own.
fn run_scoped<T: Scoped>(
    called: On<ToolCalled<T::Args>>,
    workers: Query<&MapWorker>,
    wake: Res<Wake>,
    mut commands: Commands,
) where
    T::Args: DeserializeOwned + Send + Sync + 'static,
{
    let (call, raw) = (called.call, &called.run.call.function.arguments);
    let named = raw.get("path").and_then(|path| path.as_str()).unwrap_or("");
    if let Ok(MapWorker {
        allowed: Some(allowed),
        ..
    }) = workers.get(called.agent)
        && !same_file(Path::new(named), allowed)
    {
        let why = format!(
            "You may change only {}; `{named}` was not changed.",
            allowed.display()
        );
        commands.entity(call).insert_if_new(refused(why));
        return;
    }
    // The call's arguments parsed again: the event holds them by reference.
    let args = match T::Args::deserialize(raw) {
        Ok(args) => args,
        Err(error) => {
            let why = format!("The arguments do not fit: {error}. Nothing ran.");
            commands.entity(call).insert_if_new(refused(why));
            return;
        }
    };
    let work = async move {
        let mut context = rig_harness::rig_core::tool::ToolContext::new();
        match T::tool().call(&mut context, args).await {
            Ok(output) => ToolResult::success(output.into()),
            Err(error) => ToolResult::failed(error),
        }
    };
    let pool = AsyncComputeTaskPool::get_or_init(TaskPool::default);
    let running = Running::spawn_into::<ToolOutput, _>(pool, &wake, work);
    commands.entity(call).try_insert(running);
}

/// Whether `named`, from the working directory, is the file `allowed`.
fn same_file(named: &Path, allowed: &Path) -> bool {
    let named = std::fs::canonicalize(named)
        .or_else(|_| std::path::absolute(named))
        .unwrap_or_else(|_| named.to_path_buf());
    named == allowed
}

/// "map 6/10" on the caller's turn while its job runs.
#[cfg(feature = "tui")]
fn show_progress(
    jobs: Query<(&MapJob, &CallOf), Changed<MapJob>>,
    mut turns: Query<&mut rig_tui::StatusItems>,
) {
    use rig_tui::{Side, StatusItem, Tone};
    /// After the agent's subagents at work; gone just before them.
    const PROGRESS: StatusItem = StatusItem::at(Side::Left, 65, 14);
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
