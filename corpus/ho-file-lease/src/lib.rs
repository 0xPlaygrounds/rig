//! Plugin corpus (held-out): H02: File leases across agents (spec `H02-file-lease.md`).
//!
//! A call of a file-changing tool (`[file_lease] tools`, by default `edit`
//! and `write`) takes a lease on the file its `path` names while it runs.
//! A call on a file another call holds waits, first come first served,
//! held by an open `file_lease_wait` effect of its turn named "waiting for
//! `<agent>`" (so the agent's status says so and the effect log records the
//! wait), until the lease is its own; after `max_wait_secs` it is refused.
//! Each agent that changed a file owns it until its turn ends, and is told
//! with its next model request when another agent changed it meanwhile.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};

use rig_harness::prelude::*;
use rig_harness::rig_core::effect::EffectKind;
use rig_harness::rig_core::tool::FilesChanged;
use serde::Deserialize;

/// The kind of the effect that holds a call waiting for a lease.
pub const WAIT: &str = "file_lease_wait";

/// The `[file_lease]` settings table.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct FileLease {
    /// The tools whose calls take a lease on the file their `path` names.
    pub tools: Vec<String>,
    /// How long a call waits for a lease before it is refused.
    pub max_wait_secs: u64,
}

impl Default for FileLease {
    fn default() -> Self {
        Self {
            tools: vec!["edit".to_owned(), "write".to_owned()],
            max_wait_secs: 120,
        }
    }
}

/// The lease gate, in `Stage::Gate`, for other gates to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LeaseSystems;

#[derive(Default)]
pub struct FileLeasePlugin;

impl Plugin for FileLeasePlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<FileLease>("file_lease")
            .init_resource::<ProjectRoot>()
            .init_resource::<Leases>()
            .configure_sets(AgentPass, LeaseSystems.in_set(Stage::Gate))
            .add_systems(AgentPass, lease.in_set(LeaseSystems))
            .add_observer(settled)
            .add_observer(release);
    }
}

/// Every leased file: who holds it, who waits, who owns it.
#[derive(Resource, Default)]
struct Leases(HashMap<PathBuf, File>);

#[derive(Default)]
struct File {
    /// The call that holds the lease.
    holder: Option<Entity>,
    /// The calls waiting for it, first come first, with their wait effect.
    queue: VecDeque<(Entity, Entity)>,
    /// The agents that changed it, with the turn they did it in.
    owners: Vec<(Entity, Entity)>,
}

/// On a call that holds or waits for the lease on this file.
#[derive(Component)]
struct LeaseCall(PathBuf);

/// Gives each call about to run the lease on its file, or queues it behind
/// the holder, held by a wait effect, with a deadline.
fn lease(
    calls: Query<(Entity, &ToolRequest, &Effect, &ForAgent, Option<&CallOf>), Undecided>,
    agents: Query<&ForAgent>,
    names: Query<&Name>,
    settings: Res<FileLease>,
    root: Res<ProjectRoot>,
    mut leases: ResMut<Leases>,
    mut commands: Commands,
) {
    let mut ready: Vec<_> = calls
        .iter()
        .filter(|(_, request, ..)| {
            let tool = request.call.function.name.as_str();
            settings.tools.iter().any(|leased| leased == tool)
        })
        .filter_map(|(call, request, effect, &ForAgent(agent), turn)| {
            let path = request.call.function.arguments.get("path")?.as_str()?;
            Some((effect.id(), call, canonical(&root.0, path), agent, turn))
        })
        .collect();
    // First come, first served: in the order the calls were made.
    ready.sort_by_key(|(id, ..)| *id);
    for (id, call, path, agent, turn) in ready {
        let file = leases.0.entry(path.clone()).or_default();
        let Some(holder) = file.holder else {
            file.holder = Some(call);
            commands.entity(call).insert(LeaseCall(path));
            continue;
        };
        if holder == call || file.queue.iter().any(|&(queued, _)| queued == call) {
            continue;
        }
        let holder = agent_name(holder, &agents, &names);
        let kind = EffectKind::Custom {
            kind: WAIT.into(),
            payload: serde_json::json!({ "path": shown(&root.0, &path), "holder": holder }),
        };
        let mut wait = commands.spawn((
            EffectRequest {
                kind,
                handler: None,
            },
            Effect::child_of(Some(id)),
            Name::new(format!("waiting for {holder}")),
            ForAgent(agent),
            HoldOf(call),
        ));
        if let Some(&CallOf(turn)) = turn {
            wait.insert(CallOf(turn));
        }
        let wait = wait.id();
        file.queue.push_back((call, wait));
        commands.entity(call).insert(LeaseCall(path));
        commands
            .delayed()
            .duration(Duration::from_secs(settings.max_wait_secs))
            .queue(move |world: &mut World| time_out(world, call, wait));
    }
}

/// Refuses a call still waiting for its lease when its time is up.
fn time_out(world: &mut World, call: Entity, wait: Entity) {
    if !world
        .get_entity(wait)
        .is_ok_and(|wait| wait.contains::<HoldOf>())
    {
        return;
    }
    let Some(LeaseCall(path)) = world.get::<LeaseCall>(call) else {
        return;
    };
    let path = path.clone();
    let root = world.resource::<ProjectRoot>().0.clone();
    let holder = world
        .resource::<Leases>()
        .0
        .get(&path)
        .and_then(|file| file.holder);
    let holder = holder
        .and_then(|holder| world.get::<ForAgent>(holder))
        .and_then(|&ForAgent(agent)| world.get::<Name>(agent))
        .map_or_else(
            || "another agent".to_owned(),
            |name| name.as_str().to_owned(),
        );
    if let Some(file) = world.resource_mut::<Leases>().0.get_mut(&path) {
        file.queue.retain(|&(queued, _)| queued != call);
    }
    let why = format!("{} is busy (held by {holder}'s call)", shown(&root, &path));
    if let Ok(mut call) = world.get_entity_mut(call) {
        call.insert_if_new(Outcome::refused(why));
    }
    if let Ok(mut wait) = world.get_entity_mut(wait) {
        wait.insert_if_new(Outcome::custom(serde_json::json!({ "timed_out": true })))
            .remove::<HoldOf>();
    }
}

/// As a leased call settles: if it changed its file, tells the agents that
/// own the file, and owns it until its turn ends. Then it lets go.
fn settled(
    settled: On<Add<Settled>>,
    calls: Query<(
        &LeaseCall,
        &ForAgent,
        Option<&CallOf>,
        Option<&Outcome>,
        Option<&Published>,
    )>,
    turns: Query<(), (With<Turn>, Without<Settled>)>,
    agents: Query<&ForAgent>,
    names: Query<&Name>,
    root: Res<ProjectRoot>,
    mut leases: ResMut<Leases>,
    mut commands: Commands,
) {
    let call = settled.entity;
    let Ok((LeaseCall(path), &ForAgent(agent), turn, outcome, published)) = calls.get(call) else {
        return;
    };
    commands.entity(call).remove::<LeaseCall>();
    let changed = outcome.is_some_and(|outcome| outcome.tool_result().is_success());
    let Some(file) = leases
        .0
        .get_mut(path)
        .filter(|file| file.holder == Some(call))
    else {
        return;
    };
    if !changed {
        return;
    }
    file.owners.retain(|&(_, turn)| turns.contains(turn));
    let name = agent_name(call, &agents, &names);
    let counts = published
        .and_then(|published| published.0.get::<FilesChanged>().ok().flatten())
        .and_then(|changed| {
            let change = changed
                .0
                .into_iter()
                .find(|change| canonical(&root.0, &change.path) == *path)?;
            Some(format!(" (+{} −{})", change.added, change.removed))
        })
        .unwrap_or_default();
    let shown = shown(&root.0, path);
    let note = format!("agent {name} changed {shown}{counts} while you were working on it");
    for &(owner, _) in file.owners.iter().filter(|(owner, _)| *owner != agent) {
        let note = Deliver::new(owner, note.clone(), DeliveryMode::Note);
        commands.trigger(note.with_origin(Origin::plugin("file-lease")));
    }
    if let Some(&CallOf(turn)) = turn
        && !file.owners.iter().any(|&(owner, _)| owner == agent)
    {
        file.owners.push((agent, turn));
    }
}

/// A call that held a lease lets go of it, settled or gone: the next live
/// waiter gets it, its wait answered. A waiter that goes leaves the queue.
fn release(
    removed: On<Remove<LeaseCall>>,
    calls: Query<&LeaseCall>,
    live: Query<(), (With<Effect>, Without<Settled>)>,
    turns: Query<(), (With<Turn>, Without<Settled>)>,
    mut leases: ResMut<Leases>,
    mut commands: Commands,
) {
    let call = removed.entity;
    let Ok(LeaseCall(path)) = calls.get(call) else {
        return;
    };
    let Some(file) = leases.0.get_mut(path) else {
        return;
    };
    file.queue.retain(|&(queued, _)| queued != call);
    if file.holder == Some(call) {
        file.holder = None;
        while let Some((next, wait)) = file.queue.pop_front() {
            if !live.contains(next) {
                continue;
            }
            file.holder = Some(next);
            commands
                .entity(wait)
                .try_insert(Outcome::custom(serde_json::json!({ "granted": true })))
                .try_remove::<HoldOf>();
            break;
        }
    }
    file.owners.retain(|&(_, turn)| turns.contains(turn));
    if file.holder.is_none() && file.queue.is_empty() && file.owners.is_empty() {
        leases.0.remove(&path.clone());
    }
}

/// The name of the agent `effect` works for.
fn agent_name(effect: Entity, agents: &Query<&ForAgent>, names: &Query<&Name>) -> String {
    agents
        .get(effect)
        .and_then(|&ForAgent(agent)| names.get(agent))
        .map_or_else(
            |_| "another agent".to_owned(),
            |name| name.as_str().to_owned(),
        )
}

/// `path` as the tools resolve it, against the project, with links and
/// `..` resolved as far as the file or its directory exists.
fn canonical(root: &Path, path: &str) -> PathBuf {
    let path = root.join(path);
    if let Ok(canonical) = path.canonicalize() {
        return canonical;
    }
    match (path.parent(), path.file_name()) {
        (Some(dir), Some(name)) => dir
            .canonicalize()
            .map_or_else(|_| path.clone(), |dir| dir.join(name)),
        _ => path,
    }
}

/// `path` as the user knows it: relative to the project when inside it.
fn shown(root: &Path, path: &Path) -> String {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_owned());
    path.strip_prefix(&root)
        .unwrap_or(path)
        .display()
        .to_string()
}

#[cfg(test)]
mod tests;
