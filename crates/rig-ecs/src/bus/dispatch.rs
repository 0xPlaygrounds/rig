//! `BusSet::Dispatch`: the one system that takes pending effects.

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
        EffectOutcome, Executions, Held, IdCounter, InFlight, Issued, PendingEffect, Publishing,
        Reserved, Scope, Seq, Serving, Streamed, ToolInputs,
    },
    handlers::{Bound, HandlerTable, Served},
    plugin::{Intake, Policy, Progress},
    record::Recording,
    witness::{Deferred, DispatchWitness, Refused, bus_emitter},
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
///   dispatch context), and starts one initial task in [`Executions`], with
///   a [`Serving`] marker and an empty [`Streamed`] for a streaming consumer;
///   for a handler that is a system, puts
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
    mut executions: NonSendMut<Executions>,
    bound: Query<(Entity, &Bound)>,
    pending: Query<CandidateView, Candidate>,
    in_flight: Query<&InFlight>,
    parents: Query<&ChildOf>,
    issued: Query<&Issued>,
    scopes: Query<&Scope>,
    recording: Option<Res<Recording>>,
    witnessing: DispatchWitness,
    mut ids: ResMut<IdCounter>,
    mut intake: ResMut<Intake>,
    mut progress: ResMut<Progress>,
) {
    let mut candidates: Vec<_> = pending.iter().collect();
    candidates.sort_by_key(|(_, seq, _, _, _)| **seq);
    let DispatchWitness {
        witness,
        subjects,
        deferred,
    } = witnessing;

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

    let candidates_len = candidates.len();
    for (index, (entity, _, effect, reserved, inputs)) in candidates.into_iter().enumerate() {
        if intake.0 >= policy.command_capacity {
            // Observed once per pass, on the first intent left behind, with
            // how many wait with it: a summary, not one fact per pass per
            // intent.
            if let Some(witness) = &witness
                && deferred.get(entity).is_err()
            {
                let mut subject = subjects.of(entity);
                subject.parent = nearest_issued(entity, &parents, &issued, &issued_now);
                witness.emit(
                    subject,
                    Stage::Dispatch,
                    bus_emitter(),
                    Action::Deferred {
                        reason: Reason::with_detail(
                            "intake_bound",
                            format!(
                                "{} of {} intents wait for the next tick (intake {})",
                                candidates_len - index,
                                candidates_len,
                                policy.command_capacity
                            ),
                        ),
                    },
                );
                commands.entity(entity).insert(Deferred);
            }
            return;
        }
        let key = &effect.key;
        if serial && busy.contains(key) {
            if ancestor_in_flight_on(entity, key, &parents, &in_flight) {
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
                progress.mark();
            } else if let Some(witness) = &witness
                && deferred.get(entity).is_err()
            {
                let mut subject = subjects.of(entity);
                subject.parent = nearest_issued(entity, &parents, &issued, &issued_now);
                witness.emit(
                    subject,
                    Stage::Dispatch,
                    bus_emitter(),
                    Action::Deferred {
                        reason: Reason::with_detail(
                            "serial_key_busy",
                            format!("`{key}` is served one at a time and is in flight"),
                        ),
                    },
                );
                commands.entity(entity).insert(Deferred);
            }
            continue;
        }
        let served = bound
            .iter()
            .find(|(_, bound)| &bound.key == key)
            .and_then(|(handler, _)| table.served(handler));
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
            progress.mark();
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
            progress.mark();
            continue;
        };
        ids.0 = ids.0.max(next_id);
        let id = EffectId::from_raw(raw_id);
        let origin = Origin {
            parent: nearest_issued(entity, &parents, &issued, &issued_now),
            scope: nearest_scope(entity, &parents, &scopes)
                .map(|scope| std::sync::Arc::from(scope.as_str())),
        };

        if let Some(witness) = &witness {
            let mut subject = subjects.of(entity);
            subject.effect = Some(id);
            subject.parent = origin.parent;
            witness.emit(subject, Stage::Dispatch, bus_emitter(), Action::Issued);
        }
        let mut entity_commands = commands.entity(entity);
        entity_commands.remove::<Deferred>();
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
                    published,
                    id,
                    recording: recording.as_ref().map(|r| (**r).clone()),
                    observed,
                    witness: witness.as_ref().map(|witness| {
                        (
                            (**witness).clone(),
                            subjects.of(entity),
                            effect.kind.clone(),
                        )
                    }),
                }));
                let streaming = effect.is_stream();

                let task = bevy_tasks::IoTaskPool::get().spawn(async move {
                    let reply = handler.handle(kind, dispatch).await;
                    if !streaming {
                        return Reply::Outcome(reply.into_outcome().await);
                    }
                    reply
                });
                executions.tasks.insert(entity, task);
                entity_commands.insert(Serving);
                if effect.is_stream() {
                    entity_commands.insert(Streamed::default());
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
