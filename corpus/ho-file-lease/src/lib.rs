//! Plugin corpus (held-out): H02: File leases across agents (spec `H02-file-lease.md`).
//!
//! A call of a file-changing tool (`[file_lease] tools`, by default `edit`
//! and `write`) takes a lease on the files its `Footprint` says it writes
//! while it runs. A call on a file another call holds waits, first come
//! first served, held by a hold named "waiting for `<agent>`" (which the
//! front shows on the call), until the lease is its own; after
//! `max_wait_secs` it is refused. Each agent that changed a file owns it
//! until its turn ends, and is told with its next model request when
//! another agent changed it meanwhile.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use bevy_ecs::system::SystemParam;
use rig_harness::prelude::*;
use rig_harness::rig_core::tool::FilesChanged;
use serde::Deserialize;

/// The `[file_lease]` settings table.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct FileLease {
    /// The tools whose calls take a lease on the files they write.
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

/// The lease gate, in `Stage::Gate`. A gate that holds or asks about a
/// call orders `.before` it, so a call takes its lease once let through.
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
            .add_observer(time_out)
            .add_observer(settled)
            .add_observer(release);
    }
}

/// Who holds which files, who waits for them, and who changed them.
#[derive(Resource, Default)]
struct Leases {
    /// The calls holding a lease, with the files they write.
    held: Vec<(Entity, Scope)>,
    /// The calls waiting for one, first come first.
    queue: Vec<Waiter>,
    /// The agents that changed each file, with the turn they did it in.
    owners: HashMap<PathBuf, Vec<(Entity, Entity)>>,
}

struct Waiter {
    call: Entity,
    files: Scope,
    /// The hold that keeps the call from starting.
    hold: Entity,
}

impl Leases {
    /// The call that `call` writing `files` must wait for: one holding an
    /// overlapping lease, else one waiting for it ahead of `call`.
    fn blocker(&self, call: Entity, files: &Scope) -> Option<Entity> {
        let held = self.held.iter().map(|(other, scope)| (*other, scope));
        let queued = self.queue.iter().map(|waiter| (waiter.call, &waiter.files));
        held.chain(queued)
            .find(|(other, scope)| *other != call && scope.overlaps(files))
            .map(|(other, _)| other)
    }
}

/// On a call that holds or waits for a lease on these files.
#[derive(Component)]
struct LeaseCall(Scope);

/// On a hold that keeps a call waiting for a lease.
#[derive(Component)]
struct LeaseWait;

/// Names agents and the files they work on.
#[derive(SystemParam)]
struct Who<'w, 's> {
    agents: Query<'w, 's, &'static ForAgent>,
    names: Query<'w, 's, &'static Name>,
    dirs: Query<'w, 's, &'static WorkDir>,
    root: Res<'w, ProjectRoot>,
}

impl Who<'_, '_> {
    /// The name of the agent `effect` works for.
    fn agent(&self, effect: Entity) -> String {
        self.agents
            .get(effect)
            .and_then(|&ForAgent(agent)| self.names.get(agent))
            .map_or_else(|_| "another agent".to_owned(), |name| name.to_string())
    }

    /// `path` as `agent` knows it: relative to the directory it works in,
    /// when inside it.
    fn shown(&self, agent: Entity, path: &Path) -> String {
        let dir = self
            .dirs
            .get(agent)
            .map_or(&self.root.0, |WorkDir(dir)| dir);
        let dir = dir.canonicalize().unwrap_or_else(|_| dir.clone());
        path.strip_prefix(&dir)
            .unwrap_or(path)
            .display()
            .to_string()
    }
}

/// The files a call of a leased tool writes, if it names them.
fn written<'a>(
    settings: &FileLease,
    request: &ToolRequest,
    footprint: &'a Footprint,
) -> Option<&'a Scope> {
    let tool = request.call.function.name.as_str();
    match &footprint.0 {
        Touches::Writes(scope @ Scope::Paths(_)) if settings.tools.iter().any(|t| t == tool) => {
            Some(scope)
        }
        _ => None,
    }
}

/// Gives each call about to run the lease on the files it writes, or
/// queues it behind the call it must wait for, held, with a deadline.
fn lease(
    calls: Query<
        (Entity, &Effect, &ToolRequest, &Footprint, &Decision),
        (ReadyToStart, ToolCalls, Without<LeaseCall>),
    >,
    settings: Res<FileLease>,
    who: Who,
    mut leases: ResMut<Leases>,
    mut commands: Commands,
) {
    let mut ready: Vec<_> = calls
        .iter()
        .filter(|(.., decision)| !decision.is_decided())
        .filter_map(|(call, effect, request, footprint, _)| {
            Some((
                effect.id(),
                call,
                written(&settings, request, footprint)?.clone(),
            ))
        })
        .collect();
    // First come, first served: in the order the calls were made.
    ready.sort_by_key(|(id, ..)| *id);
    let max_wait = Duration::from_secs(settings.max_wait_secs);
    for (_, call, files) in ready {
        commands.entity(call).insert(LeaseCall(files.clone()));
        let Some(blocker) = leases.blocker(call, &files) else {
            leases.held.push((call, files));
            continue;
        };
        let hold = commands
            .spawn((
                LeaseWait,
                HoldOf(call),
                Name::new(format!("waiting for {}", who.agent(blocker))),
                WakeAt::after(max_wait),
            ))
            .id();
        leases.queue.push(Waiter { call, files, hold });
    }
}

/// Refuses a call still waiting for its lease when its time is up.
fn time_out(
    due: On<Add<Due>>,
    waits: Query<&HoldOf, With<LeaseWait>>,
    mut calls: Query<(&LeaseCall, &mut Decision, &ForAgent)>,
    who: Who,
    leases: Res<Leases>,
    mut commands: Commands,
) {
    let Ok(&HoldOf(call)) = waits.get(due.entity) else {
        return;
    };
    let Ok((LeaseCall(files), mut decision, &ForAgent(agent))) = calls.get_mut(call) else {
        return;
    };
    let file = match files {
        Scope::Paths(paths) => paths.first().map(|path| who.shown(agent, path)),
        Scope::Anything => None,
    };
    let holder = leases
        .blocker(call, files)
        .map(|blocker| who.agent(blocker));
    decision.decide(
        Decider::plugin("ho-file-lease"),
        Outcome::refused(format!(
            "{} is busy (held by {}'s call)",
            file.unwrap_or_default(),
            holder.unwrap_or_else(|| "another agent".to_owned()),
        )),
    );
    commands.entity(call).remove::<LeaseCall>();
}

/// As a leased call settles: if it changed its files, tells the agents
/// that own them, and owns them until its turn ends. Then it lets go.
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
    who: Who,
    mut leases: ResMut<Leases>,
    mut commands: Commands,
) {
    let call = settled.entity;
    let Ok((LeaseCall(files), &ForAgent(agent), turn, outcome, published)) = calls.get(call) else {
        return;
    };
    commands.entity(call).remove::<LeaseCall>();
    let (Scope::Paths(paths), Some(&CallOf(turn))) = (files, turn) else {
        return;
    };
    if !outcome.is_some_and(|outcome| outcome.tool_result().is_success()) {
        return;
    }
    let changes = published
        .and_then(|published| published.0.get::<FilesChanged>().ok().flatten())
        .map(|FilesChanged(changes)| changes)
        .unwrap_or_default();
    let name = who.agent(call);
    for path in paths {
        let counts = changes
            .iter()
            .find(|change| Path::new(&change.path) == path)
            .map(|change| format!(" (+{} −{})", change.added, change.removed))
            .unwrap_or_default();
        let owners = leases.owners.entry(path.clone()).or_default();
        owners.retain(|&(_, turn)| turns.contains(turn));
        for &(owner, _) in owners.iter().filter(|(owner, _)| *owner != agent) {
            let file = who.shown(owner, path);
            let note = format!("agent {name} changed {file}{counts} while you were working on it");
            let note = Deliver::new(owner, note, DeliveryMode::Inform);
            commands.trigger(note.with_author(Author::plugin("file-lease")));
        }
        if !owners.iter().any(|&(owner, _)| owner == agent) {
            owners.push((agent, turn));
        }
    }
}

/// A call lets go of its lease or leaves the queue, settled, timed out or
/// gone; each waiter, first come first, gets its lease once nothing ahead
/// of it overlaps its files.
fn release(
    removed: On<Remove<LeaseCall>>,
    who: Who,
    mut leases: ResMut<Leases>,
    mut commands: Commands,
) {
    let call = removed.entity;
    leases.held.retain(|&(holder, _)| holder != call);
    let waiting = std::mem::take(&mut leases.queue);
    for waiter in waiting {
        if waiter.call == call {
            commands.entity(waiter.hold).try_despawn();
            continue;
        }
        match leases.blocker(waiter.call, &waiter.files) {
            None => {
                commands.entity(waiter.hold).try_despawn();
                leases.held.push((waiter.call, waiter.files));
            }
            Some(blocker) => {
                let name = Name::new(format!("waiting for {}", who.agent(blocker)));
                commands.entity(waiter.hold).try_insert(name);
                leases.queue.push(waiter);
            }
        }
    }
}

#[cfg(test)]
mod tests;
