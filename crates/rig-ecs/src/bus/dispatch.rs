//! `BusSet::Dispatch`: the one system that takes pending effects.

use std::collections::{HashMap, HashSet};

use bevy_ecs::prelude::*;
use bevy_tasks::IoTaskPool;
use futures::channel::{mpsc, oneshot};
use rig_core::{
    effect::{EffectId, HandlerKey},
    error::{ErrorKind, ErrorReport},
    serve::{Origin, OutcomeSink},
};

use super::{
    effect::{
        EffectOutcome, Held, IdCounter, InFlight, Issued, PendingEffect, Publishing, Reserved,
        Scope, Seq, Serving, Streamed, Streaming, ToolInputs,
    },
    handlers::{Bound, HandlerTable, Served},
    plugin::{Intake, Policy, Progress},
    record::Recording,
};

/// A pending effect `Dispatch` may take: not held, not answered, not yet
/// issued.
pub type Candidate = (
    Without<InFlight>,
    Without<EffectOutcome>,
    Without<Held>,
    Without<Issued>,
);

/// What `Dispatch` reads of a candidate: its order, its intent, a reserved
/// id, and a tool call's inputs.
pub type CandidateView = (
    Entity,
    &'static Seq,
    &'static PendingEffect,
    Option<&'static Reserved>,
    Option<&'static ToolInputs>,
);

/// The dispatch system. In one pass, in ascending [`Seq`]:
///
/// - stops at the tick's intake bound ([`Policy`]'s `command_capacity`);
///   the rest stay `PendingEffect`, nobody is blocked;
/// - under serial serving, leaves an effect whose key is in flight for a
///   later pass, and refuses — `Request`, no record — one whose ancestor
///   is in flight on its own key (it could only wait for itself);
/// - answers `HandlerUnavailable`, no record, when no handler entity is
///   bound to the key;
/// - otherwise issues the id ([`Reserved`] or minted), opens the record
///   (`parent` from the nearest issued ancestor, `scope` from the nearest
///   [`Scope`]; a tool call's [`ToolInputs`] and a [`Publishing`] slot on
///   the sink), and either spawns the handler's future on the task pool —
///   into [`Serving`] for a unary effect, [`Streaming`] plus an empty
///   [`Streamed`] for a stream — or, for a handler that is a system, puts
///   the effect on the entity as `Asked<E>` (an open key adds nothing: the
///   entity is the question); then marks it [`InFlight`].
#[allow(
    clippy::too_many_arguments,
    reason = "one system, one pass: every parameter is a distinct world access it needs"
)]
pub fn dispatch(
    mut commands: Commands,
    policy: Res<Policy>,
    table: NonSend<HandlerTable>,
    bound: Query<(Entity, &Bound)>,
    pending: Query<CandidateView, Candidate>,
    in_flight: Query<&InFlight>,
    parents: Query<&ChildOf>,
    issued: Query<&Issued>,
    scopes: Query<&Scope>,
    recording: Option<Res<Recording>>,
    mut ids: ResMut<IdCounter>,
    mut intake: ResMut<Intake>,
    mut progress: ResMut<Progress>,
) {
    let mut candidates: Vec<_> = pending.iter().collect();
    candidates.sort_by_key(|(_, seq, _, _, _)| **seq);

    let policy = policy.0;
    let serial = policy.serial_per_handler;
    // Ids issued in this pass: `Issued` lands when the commands apply, and a
    // child taken in the same pass as its parent must still name it.
    let mut issued_now: HashMap<Entity, EffectId> = HashMap::new();
    let mut busy: HashSet<HandlerKey> = if serial {
        in_flight.iter().map(|flight| flight.key.clone()).collect()
    } else {
        HashSet::new()
    };

    for (entity, _, effect, reserved, inputs) in candidates {
        if intake.0 >= policy.command_capacity {
            return;
        }
        let key = &effect.key;
        if serial && busy.contains(key) {
            if ancestor_in_flight_on(entity, key, &parents, &in_flight) {
                commands
                    .entity(entity)
                    .insert(EffectOutcome(Err(reentrant(key))));
                progress.mark();
            }
            continue;
        }
        let served = bound
            .iter()
            .find(|(_, bound)| &bound.key == key)
            .and_then(|(handler, _)| table.served(handler));
        let Some(served) = served else {
            commands
                .entity(entity)
                .insert(EffectOutcome(Err(handler_unavailable(key))));
            progress.mark();
            continue;
        };

        let id = match reserved {
            Some(Reserved(id)) => {
                ids.0 = ids.0.max(id.as_u64() + 1);
                *id
            }
            None => {
                let id = EffectId::from_raw(ids.0);
                ids.0 += 1;
                id
            }
        };
        let origin = Origin {
            parent: nearest_issued(entity, &parents, &issued, &issued_now),
            scope: nearest_scope(entity, &parents, &scopes)
                .map(|scope| std::sync::Arc::from(scope.as_str())),
        };

        let mut entity_commands = commands.entity(entity);
        match served {
            Served::Task(handler) => {
                if let Some(recording) = &recording {
                    recording.begin(id, key.clone(), effect.kind.clone(), origin);
                }
                let handler = handler.clone();
                let kind = effect.kind.clone();
                if effect.is_stream() {
                    let (events, receiver) = mpsc::channel(policy.stream_capacity);
                    let sink = OutcomeSink::stream(id, events);
                    let task = IoTaskPool::get().spawn(async move {
                        handler.handle(kind, sink).await;
                    });
                    entity_commands.insert((
                        Streaming {
                            task,
                            events: receiver,
                            fold: rig_core::serve::StreamTap::new(),
                        },
                        Streamed::default(),
                    ));
                } else {
                    let (reply, receiver) = oneshot::channel();
                    let mut sink = OutcomeSink::unary(id, reply);
                    // A tool call's context travels beside the effect
                    // (format 5): the inbound values on the sink, and the
                    // slot the tool publishes into, read by `Collect`.
                    if let rig_core::effect::EffectKind::ToolCall { .. } = &kind {
                        let inbound = inputs.map(|inputs| inputs.0.clone()).unwrap_or_default();
                        let published = rig_core::tool::PublishedContext::new();
                        sink = sink
                            .with_scope(std::sync::Arc::new(inbound))
                            .with_scope(std::sync::Arc::clone(&published)
                                as std::sync::Arc<dyn std::any::Any + Send + Sync>);
                        entity_commands.insert(Publishing(published));
                    }
                    let task = IoTaskPool::get().spawn(async move {
                        handler.handle(kind, sink).await;
                        match receiver.await {
                            Ok(outcome) => outcome,
                            Err(oneshot::Canceled) => Err(ErrorReport::new(
                                ErrorKind::Internal,
                                "the handler dropped its outcome sink without answering",
                            )),
                        }
                    });
                    entity_commands.insert(Serving(task));
                }
            }
            Served::World(world) => {
                if let Err(report) = (world.ask)(&mut entity_commands, &effect.kind) {
                    entity_commands.insert(EffectOutcome(Err(report)));
                    progress.mark();
                    continue;
                }
                if let Some(recording) = &recording {
                    recording.begin(id, key.clone(), effect.kind.clone(), origin);
                }
            }
        }
        entity_commands
            .insert((Issued(id), InFlight { key: key.clone() }))
            .remove::<Reserved>();
        issued_now.insert(entity, id);
        if serial {
            busy.insert(key.clone());
        }
        intake.0 += 1;
        progress.mark();
    }
}

/// Whether an ancestor of `entity` is in flight on `key`.
fn ancestor_in_flight_on(
    entity: Entity,
    key: &HandlerKey,
    parents: &Query<&ChildOf>,
    in_flight: &Query<&InFlight>,
) -> bool {
    let mut current = entity;
    while let Ok(parent) = parents.get(current) {
        current = parent.parent();
        if in_flight
            .get(current)
            .is_ok_and(|flight| &flight.key == key)
        {
            return true;
        }
    }
    false
}

/// The nearest issued ancestor's id: the record's `parent`.
fn nearest_issued(
    entity: Entity,
    parents: &Query<&ChildOf>,
    issued: &Query<&Issued>,
    issued_now: &HashMap<Entity, EffectId>,
) -> Option<EffectId> {
    let mut current = entity;
    while let Ok(parent) = parents.get(current) {
        current = parent.parent();
        if let Some(id) = issued_now.get(&current) {
            return Some(*id);
        }
        if let Ok(Issued(id)) = issued.get(current) {
            return Some(*id);
        }
    }
    None
}

/// The nearest [`Scope`], the entity's own first: the record's `scope`.
fn nearest_scope(
    entity: Entity,
    parents: &Query<&ChildOf>,
    scopes: &Query<&Scope>,
) -> Option<String> {
    let mut current = entity;
    loop {
        if let Ok(Scope(scope)) = scopes.get(current) {
            return Some(scope.clone());
        }
        match parents.get(current) {
            Ok(parent) => current = parent.parent(),
            Err(_) => return None,
        }
    }
}

/// The report for a dispatch to a key with no bound handler.
pub fn handler_unavailable(key: &HandlerKey) -> ErrorReport {
    ErrorReport::new(
        ErrorKind::HandlerUnavailable,
        format!("no handler is bound to `{key}`"),
    )
    .with_retryable(false)
}

/// The report for an effect whose ancestor is in flight on its own serial
/// key: served, it would wait for itself.
pub fn reentrant(key: &HandlerKey) -> ErrorReport {
    ErrorReport::new(
        ErrorKind::Request,
        format!(
            "`{key}` is served one at a time and an ancestor of this effect is in flight on it: served, it would wait for itself"
        ),
    )
    .with_retryable(false)
}
