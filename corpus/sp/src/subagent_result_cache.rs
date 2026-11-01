//! SP-06: a subagent answer cache. A `task` whose subagent only read files
//! and listed or searched directories is recorded when its first turn
//! answers: its task text (whitespace-normalized), its kind, model and tool
//! set, its answer, the files it read with their content hashes and the
//! files under each directory it listed or searched. A later `task` of the
//! same agent with the same task, kind, model and tools is answered from
//! the record, with no subagent, model call or tool call, while every
//! recorded file has the same hash and every directory the same files.
//!
//! The records are agent state saved with the session. `/subcache` lists
//! them;
//! `/subcache off|on|clear` turns the cache off or on for the session, or
//! forgets the records.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use rig_basics::subagents::{Role, SpawnSubagent, SpawnedAgent};
use rig_harness::prelude::*;
use rig_harness::rig_core::tool::{FilesChanged, FilesRead, content_hash};
use rig_harness::rig_tools::workspace::{Listing as Files, Stat, Workspace};
use serde::{Deserialize, Serialize};

/// Who decides the requests and calls it answers from a record.
const NAME: &str = "subagent-result-cache";

/// Answers a repeated read-only `task` from its record, and records each
/// read-only subagent's first answer.
#[derive(Default)]
pub struct SubagentResultCachePlugin;

impl Plugin for SubagentResultCachePlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<rig_basics::subagents::SubagentsPlugin>() {
            app.add_plugins(rig_basics::subagents::SubagentsPlugin);
        }
        app.init_resource::<ProjectWorkspace>()
            .init_resource::<SubcacheSwitch>()
            .keeps_saved::<SubagentCache>()
            .add_systems(AgentPass, reuse.in_set(GateSystems::After))
            .add_observer(watch_spawned)
            .add_observer(watch_effect)
            .add_observer(record_first_turn)
            .add_command(
                "subcache",
                "The subagent answer cache: /subcache [off|on|clear]",
                subcache,
            );
    }
}

/// What makes two `task` requests the same task.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskKey {
    /// The task text, its whitespace runs made single spaces.
    pub task: String,
    /// Its kind: the arguments other plugins added to `task`, as JSON.
    pub kind: String,
    /// Its model and reasoning setting, as the subagent would get them.
    pub model: String,
    /// The tools it may use, sorted.
    pub tools: Vec<String>,
}

/// A subagent's answer and what it read to give it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Record {
    /// The task it answered.
    pub key: TaskKey,
    /// Its final answer.
    pub answer: String,
    /// The asking agent's turn that started it, from 1.
    pub turn: u64,
    /// Each file it read, by path, with the content hash it read (`None`:
    /// no such file).
    pub files: BTreeMap<String, Option<String>>,
    /// Each directory it listed or searched, by path, with the files under
    /// it then.
    pub dirs: BTreeMap<String, BTreeSet<String>>,
    /// The model calls it made.
    pub model_calls: u64,
    /// How often a `task` was answered from it, or from the records of the
    /// same task it replaced.
    pub hits: u64,
    /// The model calls those answers saved.
    pub saved: u64,
}

/// On an agent: the records of the subagents it started, one per task,
/// saved with the session.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SubagentCache(pub Vec<Record>);

/// Whether the cache answers and records in this session; saved with it.
#[derive(Resource, Reflect, Clone, Debug, Default)]
#[reflect(Resource, Saved, Default)]
pub struct SubcacheSwitch {
    /// `/subcache off` was the last word.
    pub off: bool,
}

/// On a `task` request the cache looked at.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Looked;

/// On a `task` request no record answered: what its subagent's answer is
/// recorded under.
#[derive(Component)]
struct Pending {
    key: TaskKey,
    turn: u64,
}

/// On a subagent whose first turn runs: what it read so far.
#[derive(Component)]
struct Watching {
    key: TaskKey,
    turn: u64,
    files: BTreeMap<String, Option<String>>,
    dirs: BTreeMap<String, BTreeSet<String>>,
    model_calls: u64,
    messages: u64,
    /// It changed something, ran a command, read what it cannot say, or
    /// was sent more than its task: never recorded.
    tainted: bool,
}

/// The agent a request asks for a subagent from.
type Parent<'a> = (
    Option<&'a ModelChoice>,
    &'a ReasoningSetting,
    &'a ToolAccess,
);

/// The key of `request` from `parent`'s settings, as `rig_basics::subagents` would
/// settle them, or `None` when it is no task the cache takes.
fn key_of(
    request: &SpawnSubagent,
    (model, &effort, access): Parent,
    tools: &Query<&ToolDef>,
    models: &Models,
) -> Option<TaskKey> {
    if request.role != Role::Task || request.peers {
        return None;
    }
    let inherited = ModelChoice::inherit(
        &models.0,
        (model, effort),
        request.model.as_deref(),
        request.effort.as_deref(),
    );
    let (Some(model), effort) = inherited.ok()? else {
        return None;
    };
    let mut tools: Vec<String> = match &request.tools {
        Some(asked) => asked.clone(),
        None => tools
            .iter()
            .map(|def| def.0.name.to_string())
            .filter(|name| access.keeps(name))
            .collect(),
    };
    tools.sort();
    tools.dedup();
    Some(TaskKey {
        task: request
            .brief
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" "),
        kind: serde_json::Value::Object(request.extra.clone()).to_string(),
        model: format!("{} ({effort})", model.0),
        tools,
    })
}

/// The content hash of the file `path`, `None` when there is none.
fn hash_of(workspace: &dyn Workspace, path: &Path) -> Option<String> {
    match workspace.stat(path).ok()? {
        Stat::File { .. } => {
            let contents = workspace.read(path, u64::MAX).ok()?;
            Some(content_hash(&contents.bytes).0)
        }
        _ => None,
    }
}

/// The files under the directory `path` the ignore rules keep, hidden ones
/// too: every file a `list` or `search` there could have seen, whatever
/// its glob or `hidden` argument.
fn files_under(workspace: &dyn Workspace, path: &Path) -> Option<BTreeSet<String>> {
    let files = workspace.files(path, None, Files::Hidden).ok()?.files;
    Some(
        files
            .iter()
            .map(|file| file.to_string_lossy().into_owned())
            .collect(),
    )
}

/// Whether every file and directory `record` read is as it was. A
/// workspace that is not this machine is never checked from the frame,
/// so it never matches.
fn unchanged(workspace: &dyn Workspace, record: &Record) -> bool {
    if workspace.host().is_some() {
        return false;
    }
    let files = record
        .files
        .iter()
        .all(|(path, hash)| hash_of(workspace, Path::new(path)) == *hash);
    files
        && record
            .dirs
            .iter()
            .all(|(path, listed)| files_under(workspace, Path::new(path)).as_ref() == Some(listed))
}

/// Answers each `task` request about to start whose record still holds
/// from the record: the request is decided, so no subagent starts, and its
/// `task` call gets the answer. One no record answers gets its key, for
/// its subagent's answer. After every gate, as a cached answer goes.
fn reuse(
    mut requests: Query<
        (Entity, &SpawnSubagent, &ForAgent, &ChildOf, &mut Decision),
        (
            ReadyToStart,
            PluginEffects,
            Without<Ask>,
            Without<Looked>,
            Without<Restored>,
        ),
    >,
    mut calls: Query<(&mut Decision, Option<&CallOf>), ToolCalls>,
    (parents, tools, turns): (Query<Parent>, Query<&ToolDef>, Query<&Turn>),
    mut caches: Query<&mut SubagentCache>,
    (models, workspace, switch): (Res<Models>, Res<ProjectWorkspace>, Res<SubcacheSwitch>),
    mut commands: Commands,
) {
    for (request_entity, request, &ForAgent(parent), of, mut decision) in &mut requests {
        commands.entity(request_entity).insert(Looked);
        if switch.off || decision.is_decided() {
            continue;
        }
        let Ok((mut call, call_of)) = calls.get_mut(of.parent()) else {
            continue;
        };
        let Some(key) = parents
            .get(parent)
            .ok()
            .and_then(|parent| key_of(request, parent, &tools, &models))
        else {
            continue;
        };
        let turn = call_of
            .and_then(|of| turns.get(of.0).ok())
            .map_or(0, |turn| turn.index + 1);
        let mut cache = caches.get_mut(parent).ok();
        let hit = cache.as_mut().and_then(|cache| {
            let record = cache.0.iter_mut().find(|record| record.key == key)?;
            unchanged(&*workspace.0, record).then_some(record)
        });
        let Some(record) = hit else {
            commands
                .entity(request_entity)
                .insert(Pending { key, turn });
            continue;
        };
        let text = format!(
            "{}\n(reused result from turn {}; inputs unchanged)",
            record.answer, record.turn
        );
        if call.decide(Decider::plugin(NAME), ToolResult::success(text.into())) {
            record.hits += 1;
            record.saved += record.model_calls;
            let reused = serde_json::json!({ "reused_from_turn": record.turn });
            decision.decide(Decider::plugin(NAME), Outcome::custom(reused));
        }
    }
}

/// Watches the first turn of a subagent started for a task no record
/// answered.
fn watch_spawned(
    spawned: On<Add<SpawnedAgent>>,
    requests: Query<(&SpawnedAgent, &Pending)>,
    mut commands: Commands,
) {
    let Ok((&SpawnedAgent(agent), pending)) = requests.get(spawned.entity) else {
        return;
    };
    commands.entity(agent).try_insert(Watching {
        key: pending.key.clone(),
        turn: pending.turn,
        files: BTreeMap::new(),
        dirs: BTreeMap::new(),
        model_calls: 0,
        messages: 0,
        tainted: false,
    });
}

/// An effect of a watched subagent, settled.
type Settling<'a> = (
    &'a ForAgent,
    Has<ModelRequest>,
    Has<Incoming>,
    Has<ToolRequest>,
    Option<&'a Footprint>,
    Option<&'a Published>,
);

/// Adds what each settled effect of a watched subagent read to what it
/// watches: a model call is counted, a second message taints it, and a
/// tool call that only read named paths adds the files it read with the
/// hashes it read (rig-tools' `FilesRead`), the hash of each other file it
/// named, and the files under each directory it named. Any other tool
/// call taints it.
fn watch_effect(
    settled: On<Add<Settled>>,
    effects: Query<Settling>,
    mut watched: Query<&mut Watching>,
    workspace: Res<ProjectWorkspace>,
) {
    let Ok((&ForAgent(agent), model, incoming, tool, footprint, published)) =
        effects.get(settled.entity)
    else {
        return;
    };
    let Ok(mut watching) = watched.get_mut(agent) else {
        return;
    };
    if model {
        watching.model_calls += 1;
    }
    if incoming {
        watching.messages += 1;
        watching.tainted |= watching.messages > 1;
    }
    if !tool || watching.tainted {
        return;
    }
    let read = published.and_then(|published| published.0.get::<FilesRead>().ok().flatten());
    let changed = published.and_then(|published| published.0.get::<FilesChanged>().ok().flatten());
    let workspace = &*workspace.0;
    let paths = match footprint.map(|footprint| &footprint.0) {
        Some(Touches::Reads(Scope::Paths(paths))) if workspace.host().is_none() => paths,
        _ => {
            watching.tainted = true;
            return;
        }
    };
    if changed.is_some_and(|changed| !changed.0.is_empty()) {
        watching.tainted = true;
        return;
    }
    for file in read.iter().flat_map(|read| &read.0) {
        let hash = Some(file.hash.0.clone());
        watching.files.insert(file.path.clone(), hash);
    }
    for path in paths {
        let name = path.to_string_lossy().into_owned();
        if matches!(workspace.stat(path), Ok(Stat::Dir)) {
            match files_under(workspace, path) {
                Some(files) => {
                    watching.dirs.insert(name, files);
                }
                None => watching.tainted = true,
            }
        } else {
            watching
                .files
                .entry(name)
                .or_insert_with(|| hash_of(workspace, path));
        }
    }
}

/// Records a watched subagent's first turn once it ends with an answer,
/// replacing the record of the same task and keeping its counts; a turn
/// that failed or was stopped, or a tainted subagent, is not recorded.
fn record_first_turn(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent), With<Turn>>,
    watched: Query<(&Watching, &SpawnedBy)>,
    mut caches: Query<&mut SubagentCache>,
    switch: Res<SubcacheSwitch>,
    mut commands: Commands,
) {
    let Ok((outcome, &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    let Ok((watching, &SpawnedBy(parent))) = watched.get(agent) else {
        return;
    };
    commands.entity(agent).remove::<Watching>();
    let TurnOutcome::Answered(message) = outcome else {
        return;
    };
    let Some(answer) = message
        .final_answer()
        .filter(|_| !watching.tainted && !switch.off)
    else {
        return;
    };
    let mut record = Record {
        key: watching.key.clone(),
        answer,
        turn: watching.turn,
        files: watching.files.clone(),
        dirs: watching.dirs.clone(),
        model_calls: watching.model_calls,
        hits: 0,
        saved: 0,
    };
    let Ok(mut cache) = caches.get_mut(parent) else {
        commands
            .entity(parent)
            .try_insert(SubagentCache(vec![record]));
        return;
    };
    match cache.0.iter_mut().find(|old| old.key == record.key) {
        Some(old) => {
            (record.hits, record.saved) = (old.hits, old.saved);
            *old = record;
        }
        None => cache.0.push(record),
    }
}

/// `/subcache`: the records with their hits and the model calls they
/// saved; `/subcache off|on` turns the cache off or on for the session,
/// `/subcache clear` forgets the records.
fn subcache(
    In(args): In<CommandArgs>,
    caches: Query<(Entity, &SubagentCache)>,
    mut switch: ResMut<SubcacheSwitch>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let agent = args.agent;
    match args.args.as_str() {
        "" => {
            let records: Vec<&Record> = caches.iter().flat_map(|(_, cache)| &cache.0).collect();
            let saved: u64 = records.iter().map(|record| record.saved).sum();
            let state = if switch.off { "off" } else { "on" };
            let mut text = format!(
                "Subagent cache {state}: {} record(s), {saved} model calls saved.",
                records.len()
            );
            for record in records {
                text.push_str(&format!(
                    "\n- \"{}\" (turn {}): {} hit(s), {} model calls saved",
                    record.key.task, record.turn, record.hits, record.saved
                ));
            }
            notices.write(Notice::info(agent, text));
        }
        "off" | "on" => {
            switch.off = args.args == "off";
            notices.write(Notice::info(
                agent,
                format!("Subagent cache {}.", args.args),
            ));
        }
        "clear" => {
            for (cached, _) in &caches {
                commands.entity(cached).remove::<SubagentCache>();
            }
            notices.write(Notice::info(agent, "Subagent cache cleared."));
        }
        _ => {
            notices.write(Notice::error(agent, "Usage: /subcache [off|on|clear]"));
        }
    }
}

#[cfg(test)]
mod tests;
