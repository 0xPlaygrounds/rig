//! Ordered dispatch of pending effects with bounded intake and serial-key checks.
//!
//! ```
//! use rig_ecs::bus::dispatch::handler_unavailable;
//! let report = handler_unavailable(&"missing".into());
//! assert_eq!(report.kind, rig_core::error::ErrorKind::HandlerUnavailable);
//! ```

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use bevy_ecs::prelude::*;

use rig_core::serve::Reply;
use rig_core::{
    effect::{EffectId, HandlerKey},
    error::{ErrorKind, ErrorReport},
    serve::{Dispatch, Origin},
};

use super::{
    effect::{
        EffectOutcome, Held, IdCounter, InFlight, Issued, PendingEffect, Publishing, Reserved,
        Scope, Seq, Streamed, Tasks, ToolInputs,
    },
    handlers::{HandlerIndex, Registry, Served, ServedBy, Serves},
    plugin::{Policy, Wake},
    record::Recording,
    witness::{DispatchWitness, Refused, bus_emitter},
};
use rig_core::observe::{Action, Reason, Stage};

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
    Option<&'static super::AdapterOperation>,
    Option<&'static ServedBy>,
);

/// Dispatch eligible effects in [`Seq`] order up to the serving policy's intake
/// bound. Busy serial keys wait; reentrant effects, missing handlers, and exhausted
/// IDs receive errors without opening records.
///
/// Accepted effects use reserved or fresh IDs and inherit the nearest issued
/// parent and scope. Task-served effects own their execution; world-served effects
/// await system answers. Tool inputs and publication slots travel beside requests.
pub fn dispatch(
    mut commands: Commands,
    policy: Res<Policy>,
    wake: Res<Wake>,
    index: Res<HandlerIndex>,
    registry: Registry,
    mut tasks: Tasks,
    serves: Query<&Serves>,
    served_by: Query<&ServedBy>,
    pending: Query<CandidateView, Candidate>,
    in_flight: Query<&InFlight>,
    parents: Query<&ChildOf>,
    issued: Query<&Issued>,
    scopes: Query<&Scope>,
    recording: Option<Res<Recording>>,
    witnessing: DispatchWitness,
    mut ids: ResMut<IdCounter>,
) {
    let mut candidates: Vec<_> = pending.iter().collect();
    candidates.sort_by_key(|(_, seq, ..)| **seq);
    let mut intake = 0usize;
    let DispatchWitness { witness, subjects } = witnessing;

    let policy = policy.0;
    let serial = policy.serial_per_handler;
    // Ids issued in this pass: `Issued` lands when the commands apply, and a
    // child taken in the same pass as its parent must still name it.
    let mut issued_now: HashMap<Entity, EffectId> = HashMap::new();
    // Handler entities taken this pass: their `Serves` cannot see it yet.
    let mut busy_now: HashSet<Entity> = HashSet::new();

    for (entity, _, effect, reserved, inputs, operation, resolved) in candidates {
        // Clamped at the read as well as at install: a host may replace the
        // resource, and a tick still takes at least one effect.
        if intake >= policy.command_capacity.max(1) {
            return;
        }
        let key = &effect.key;
        let handler = resolved
            .map(|ServedBy(handler)| *handler)
            .or_else(|| index.entity(key));
        let served = handler.and_then(|handler| registry.served(handler));
        if let Some(handler) = handler
            && serial
            && (busy_now.contains(&handler)
                || serves
                    .get(handler)
                    .is_ok_and(|serves| serves.iter().any(|effect| in_flight.contains(effect))))
        {
            if ancestor_served_by(entity, handler, &parents, &in_flight, &served_by) {
                if let Some(witness) = &witness {
                    let mut subject = subjects.of(entity);
                    subject.parent = nearest_issued(entity, &parents, &issued, &issued_now);
                    witness.emit(
                        subject,
                        Stage::Dispatch,
                        bus_emitter(),
                        Action::Refused {
                            reason: Reason::with_detail("reentrant", reentrant(key).message),
                        },
                    );
                }
                commands
                    .entity(entity)
                    .insert((Refused, EffectOutcome(Err(reentrant(key)))));
            }
            continue;
        }
        let Some(served) = served else {
            if let Some(witness) = &witness {
                let mut subject = subjects.of(entity);
                subject.parent = nearest_issued(entity, &parents, &issued, &issued_now);
                witness.emit(
                    subject,
                    Stage::Dispatch,
                    bus_emitter(),
                    Action::Refused {
                        reason: Reason::with_detail(
                            "handler_unavailable",
                            handler_unavailable(key).message,
                        ),
                    },
                );
            }
            commands
                .entity(entity)
                .insert((Refused, EffectOutcome(Err(handler_unavailable(key)))));
            continue;
        };

        let raw_id = reserved.map_or(ids.0, |Reserved(id)| id.as_u64());
        let Some(next_id) = raw_id.checked_add(1) else {
            let report = ErrorReport::new(ErrorKind::Request, "effect ID allocator exhausted");
            if let Some(witness) = &witness {
                let mut subject = subjects.of(entity);
                subject.parent = nearest_issued(entity, &parents, &issued, &issued_now);
                witness.emit(
                    subject,
                    Stage::Dispatch,
                    bus_emitter(),
                    Action::Refused {
                        reason: Reason::with_detail("ids_exhausted", report.message.clone()),
                    },
                );
            }
            commands
                .entity(entity)
                .insert((Refused, EffectOutcome(Err(report))));
            continue;
        };
        ids.0 = ids.0.max(next_id);
        let id = EffectId::from_raw(raw_id);
        let origin = Origin {
            parent: nearest_issued(entity, &parents, &issued, &issued_now),
            scope: nearest_scope(entity, &parents, &scopes)
                .map(|scope| std::sync::Arc::from(scope.as_str())),
        };
        // Parent Issued writes can still be deferred in this dispatch pass.
        // Resolve once from issued_now for every observer/context surface.
        let mut subject = subjects.of(entity);
        subject.effect = Some(id);
        subject.parent = origin.parent;
        let adapter = operation.map(|operation| {
            operation
                .context
                .for_host_attempt(subject.clone(), operation.host_attempt)
        });

        if let Some(witness) = &witness {
            witness.emit(
                subject.clone(),
                Stage::Dispatch,
                bus_emitter(),
                Action::Issued,
            );
        }
        let mut entity_commands = commands.entity(entity);
        match served {
            Served::Task(handler) => {
                if let Some(recording) = &recording {
                    recording.begin(id, key.clone(), effect.kind.clone(), origin);
                }
                let handler = handler.clone();
                let kind = effect.kind.clone();
                let observed = Arc::new(super::record::ObservedState::default());
                entity_commands.insert(super::record::Observed(observed.clone()));
                let mut dispatch = Dispatch::new(id, effect.is_stream());
                if let rig_core::effect::EffectKind::ToolCall { .. } = &kind {
                    let inbound = inputs.map(|inputs| inputs.0.clone()).unwrap_or_default();
                    let published = rig_core::tool::PublishedContext::new();
                    dispatch = dispatch
                        .with_scope(Arc::new(inbound))
                        .with_scope(published.clone());
                    entity_commands.insert(Publishing(published));
                }
                let published = dispatch.scope::<rig_core::tool::PublishedContext>();
                entity_commands.insert(super::record::ReplacedBy(dispatch.replaced_by()));
                let dispatch = dispatch.with_observer(Box::new(super::record::WorldObserver {
                    adapter,
                    published,
                    id,
                    recording: recording.as_ref().map(|r| (**r).clone()),
                    observed,
                    witness: witness
                        .as_ref()
                        .map(|witness| ((**witness).clone(), subject.clone())),
                }));
                let streaming = effect.is_stream();
                let wake = wake.clone();
                let task = bevy_tasks::IoTaskPool::get().spawn(async move {
                    let reply = handler.handle(kind, dispatch).await;
                    let reply = if streaming {
                        reply
                    } else {
                        Reply::Outcome(reply.into_outcome().await)
                    };
                    wake.signal();
                    reply
                });
                entity_commands.insert(tasks.serving(entity, task));
                if effect.is_stream() {
                    entity_commands.insert(Streamed::default());
                }
            }
            Served::World(world) => {
                if let Err(report) = (world.ask)(&mut entity_commands, &effect.kind) {
                    entity_commands.insert(EffectOutcome(Err(report)));
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
        if let Some(handler) = handler {
            if resolved.is_none() {
                entity_commands.insert(ServedBy(handler));
            }
            busy_now.insert(handler);
        }
        issued_now.insert(entity, id);
        intake += 1;
    }
}

/// Whether an ancestor of `entity` is in flight on `handler`.
fn ancestor_served_by(
    entity: Entity,
    handler: Entity,
    parents: &Query<&ChildOf>,
    in_flight: &Query<&InFlight>,
    served_by: &Query<&ServedBy>,
) -> bool {
    let mut current = entity;
    while let Ok(parent) = parents.get(current) {
        current = parent.parent();
        if in_flight.contains(current)
            && served_by
                .get(current)
                .is_ok_and(|ServedBy(serving)| *serving == handler)
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
