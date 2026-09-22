//! Bounded collection of handler replies, stream delivery, and effect settlement.
//!
//! ```
//! use rig_ecs::bus::collect::STREAM_WORK_PER_TICK;
//! assert!(STREAM_WORK_PER_TICK > 0);
//! ```

use bevy_ecs::prelude::*;
use rig_core::{
    effect::EffectId,
    serve::{Reply, stream_truncated},
    streaming::{Delta, StreamEvent},
};
use std::task::Poll;

use super::{
    effect::{
        EffectOutcome, InFlight, Issued, Publishing, Serving, Streamed, Streaming, Tasks,
        ToolOutputs, WorldOutcome,
    },
    plugin::Wake,
    record::{DeliveryBatch, Observed, Recording},
    witness::{SeenOutcome, Subjects, Witnessing, bus_emitter, fingerprint},
};
use rig_core::observe::{Action, Emitter, OutcomeSummary, Reason, Stage};

/// How many trailing stream events a truncation observation keeps.
pub const TRUNCATION_TAIL: usize = 8;

/// Maximum live streaming queue checks in one pass.
pub const STREAM_WORK_PER_TICK: usize = 4096;
const STREAM_ITEMS_PER_EFFECT: usize = 64;

/// The sequence cursor where the last pass ran out of its allowance:
/// service rotates from it so a pass that exhausts the allowance does not
/// starve the effects after it.
#[derive(Default)]
pub struct StreamCursor {
    last: Option<super::Seq>,
}

/// Marker for a library collector's outcome insertion. Removed by settlement;
/// direct in-flight insertions without it cannot establish policy replay.
#[derive(Component)]
pub struct CollectedOutcome;

/// Publish submitted world answers at the same boundary as task results.
/// Arrival order is independent of dispatch order and entity archetypes.
pub fn collect_world(
    mut commands: Commands,
    ready: Query<(Entity, &WorldOutcome), (With<InFlight>, Without<EffectOutcome>)>,
) {
    let mut ready: Vec<_> = ready.iter().collect();
    ready.sort_by_key(|(_, outcome)| outcome.order());
    for (entity, outcome) in ready {
        commands
            .entity(entity)
            .insert(CollectedOutcome)
            .insert(EffectOutcome(outcome.outcome.clone()))
            .remove::<WorldOutcome>();
    }
}

/// Collect ready handler tasks without awaiting them. Publish unary outcomes and
/// tool outputs together; transfer returned streams to effect-owned workers.
pub fn collect_tasks(
    mut commands: Commands,
    serving: Query<(Entity, Option<&Publishing>, &Serving), With<InFlight>>,
    policy: Res<super::Policy>,
    wake: Res<Wake>,
    mut tasks: Tasks,
) {
    for (entity, publishing, _) in &serving {
        let Some(reply) = tasks.poll(entity) else {
            continue;
        };
        let mut entity_commands = commands.entity(entity);
        entity_commands.remove::<Serving>();
        match reply {
            Reply::Outcome(outcome) => {
                if let Some(Publishing(published)) = publishing {
                    entity_commands.remove::<Publishing>();
                    if let Some(context) = published.take() {
                        entity_commands.insert(ToolOutputs(context));
                    }
                }
                entity_commands
                    .insert(CollectedOutcome)
                    .insert(EffectOutcome(outcome));
            }
            Reply::Stream(stream) => {
                let streaming =
                    tasks.streaming(entity, stream, policy.0.stream_capacity, wake.clone());
                entity_commands.insert(streaming);
            }
        }
    }
}

/// Drain bounded worker delivery and fold each item. The first folded
/// outcome is retained; streaming effects settle at EOF, including post-final
/// metadata and errors. Pending waits for the host's next Collect invocation.
pub fn collect_streams(
    mut commands: Commands,
    mut streaming: Query<StreamingView, With<InFlight>>,
    mut tasks: Tasks,
    recording: Option<Res<Recording>>,
    witness: Option<Res<Witnessing>>,
    subjects: Subjects,
    batch: Res<DeliveryBatch>,
    mut cursor: Local<StreamCursor>,
    mut order: Local<Vec<(super::Seq, Entity)>>,
) {
    let mut remaining = STREAM_WORK_PER_TICK;
    order.clear();
    order.extend(streaming.iter().map(|(entity, seq, ..)| (*seq, entity)));
    order.sort_unstable_by_key(|(seq, _)| *seq);
    let start = cursor
        .last
        .map_or(0, |last| order.partition_point(|(seq, _)| *seq <= last));
    for &(seq, entity) in order.iter().cycle().skip(start).take(order.len()) {
        if remaining == 0 {
            break;
        }
        let Ok((_, _, Issued(id), mut streaming, mut streamed, publishing)) =
            streaming.get_mut(entity)
        else {
            continue;
        };
        let mut delivered = 0;
        let start = streamed
            .as_ref()
            .map_or(0, |state| state.events.len() + state.errors.len());
        let mut items = Vec::new();
        for _ in 0..STREAM_ITEMS_PER_EFFECT {
            if remaining == 0 {
                break;
            }
            remaining -= 1;
            let polled = match streaming.events.try_recv() {
                Ok(item) => Poll::Ready(Some(item)),
                Err(futures::channel::mpsc::TryRecvError::Empty) => Poll::Pending,
                Err(futures::channel::mpsc::TryRecvError::Closed) => Poll::Ready(None),
            };
            let outcome = match polled {
                Poll::Pending => break,
                Poll::Ready(Some(item)) => {
                    streaming.delivered += 1;
                    if let Some(streamed) = &mut streamed {
                        items.push(item.clone());
                        if let Err(error) = &item {
                            let position = streamed.events.len() + streamed.errors.len();
                            streamed.errors.push((position, error.clone()));
                        }
                        if streamed.outcome.is_none()
                            && let Some(outcome) = streaming.fold.observe(&item)
                        {
                            streamed.outcome = Some(outcome);
                        }
                        if let Ok(event) = item {
                            if let StreamEvent::BlockDelta {
                                delta: Delta::Text { text },
                                ..
                            } = &event
                            {
                                streamed.text.push_str(text);
                            }
                            streamed.events.push(event);
                        }
                        delivered += 1;
                        continue;
                    }
                    // A unary request answered by a stream ends at the first fold.
                    let Some(outcome) = streaming.fold.observe(&item) else {
                        continue;
                    };
                    outcome
                }
                Poll::Ready(None) => match streamed
                    .as_ref()
                    .and_then(|streamed| streamed.outcome.clone())
                {
                    Some(outcome) => outcome,
                    None => {
                        // The stream closed with no terminal record: keep
                        // the last frames the consumer saw so the failure
                        // can be classified after the fact.
                        if let Some(witness) = &witness {
                            let (tail, errors) = streamed.as_ref().map_or_else(
                                || (Vec::new(), Vec::new()),
                                |streamed| {
                                    let skip =
                                        streamed.events.len().saturating_sub(TRUNCATION_TAIL);
                                    (
                                        streamed.events.iter().skip(skip).cloned().collect(),
                                        streamed
                                            .errors
                                            .iter()
                                            .map(|(_, error)| Reason::from_report(error))
                                            .collect(),
                                    )
                                },
                            );
                            witness.emit(
                                subjects.of(entity),
                                Stage::Collect,
                                bus_emitter(),
                                Action::stream_truncated(
                                    streamed.as_ref().map_or(streaming.delivered, |streamed| {
                                        streamed.events.len() + streamed.errors.len()
                                    }),
                                    tail,
                                    errors,
                                ),
                            );
                        }
                        Err(stream_truncated())
                    }
                },
            };
            tasks.forget_stream(entity);
            if !items.is_empty() {
                commands.trigger(super::StreamItemsDelivered {
                    effect: entity,
                    id: *id,
                    start,
                    items: std::mem::take(&mut items),
                });
            }
            let published = publishing.map(|Publishing(published)| published.take());
            // Delivery observers may remove the effect or its owning graph.
            // Preserve their cancellation instead of finalizing a missing entity.
            commands.queue(move |world: &mut World| {
                let Ok(mut effect) = world.get_entity_mut(entity) else {
                    return;
                };
                if let Some(context) = published {
                    effect.remove::<Publishing>();
                    if let Some(context) = context {
                        effect.insert(ToolOutputs(context));
                    }
                }
                effect
                    .remove::<Streaming>()
                    .insert(CollectedOutcome)
                    .insert(EffectOutcome(outcome));
            });
            break;
        }
        if !items.is_empty() {
            commands.trigger(super::StreamItemsDelivered {
                effect: entity,
                id: *id,
                start,
                items,
            });
        }
        if delivered != 0
            && let Some(recording) = &recording
        {
            recording.delivery(
                batch.0,
                *id,
                rig_core::effect::DeliveryKind::Stream { items: delivered },
            );
        }
        if remaining == 0 {
            // Advance the cursor only when a pass exhausts its allowance. Empty
            // setup polls must not reorder a later ready batch; complete passes
            // retain the previous cursor so partial passes share service fairly.
            cursor.last = Some(seq);
        }
    }
}

/// The effect's delivery fold and optional streamed consumer state.
pub type StreamingView = (
    Entity,
    &'static super::Seq,
    &'static Issued,
    &'static mut Streaming,
    Option<&'static mut Streamed>,
    Option<&'static Publishing>,
);

/// An outcome that landed on an effect still in flight.
pub type Landing = (Added<EffectOutcome>, With<InFlight>);

/// Settle newly answered in-flight effects, record the original handler answer,
/// release execution state, and trigger [`Landed`]. Discarded dispatches and gate
/// denials produce no record; later policy rewrites are not re-recorded.
pub fn settle(
    mut commands: Commands,
    landed: Query<
        (
            Entity,
            &Issued,
            &EffectOutcome,
            Option<&Observed>,
            Option<&ToolOutputs>,
        ),
        Landing,
    >,
    replaced: Query<&super::record::ReplacedBy>,
    recording: Option<Res<Recording>>,
    witness: Option<Res<Witnessing>>,
    subjects: Subjects,
) {
    for (entity, &Issued(id), outcome, observed, outputs) in &landed {
        // A layered handler: the record holds what the innermost handler
        // answered (the observer's), never a layer's verdict; a dispatch a
        // layer discarded is no record.
        let recorded = observed
            .and_then(|observed| observed.0.take_outcome())
            .unwrap_or_else(|| outcome.0.clone());
        let discarded = observed.is_some_and(|observed| observed.0.is_discarded());
        let replaced_by = replaced
            .get(entity)
            .ok()
            .and_then(super::record::ReplacedBy::layer);
        if let (Some(recording), false) = (&recording, discarded) {
            // Layered dispatches captured output at the inner handler's
            // terminal, before an outer verdict could change it.
            if observed.is_none()
                && let Some(outputs) = outputs
            {
                recording.tool_output(id, outputs.0.result_context());
            }
            recording.resolve(id, recorded.clone());
        }
        if let Some(witness) = &witness {
            let subject = subjects.of(entity);
            if discarded {
                // The layer's denial was observed at the handler side; the
                // consumer's outcome is what the layer decided.
            } else {
                let served = OutcomeSummary::of(&recorded);
                let seen = SeenOutcome::of(&outcome.0);
                let consumed = seen.summary.clone();
                let differs = seen.fingerprint != fingerprint(&recorded);
                commands.entity(entity).insert(seen);
                let observation = rig_core::observe::Observation::new(
                    subject.clone(),
                    Stage::Collect,
                    bus_emitter(),
                    Action::Landed {
                        outcome: served.clone(),
                    },
                );
                witness.observe(observation);
                if differs {
                    // Attribute the difference to its named layer when available;
                    // unclaimed replacements retain unknown attribution.
                    witness.emit(
                        subject,
                        Stage::Handler,
                        replaced_by.map_or_else(Emitter::unknown, Emitter::named),
                        Action::Replaced {
                            recorded: served,
                            consumed,
                        },
                    );
                }
            }
        }
        // Remove the collection marker first so InFlight removal observers can
        // distinguish settlement from despawn before an answered effect settles.
        commands.entity(entity).remove::<CollectedOutcome>();
        commands
            .entity(entity)
            .remove::<(InFlight, Observed, super::record::ReplacedBy)>();
        // Defer notification until the record is closed and flight state removed.
        commands.trigger(Landed { entity, id });
    }
}

/// Notification that an answered effect has settled and its record is closed.
/// Propagates through `ChildOf`; `original_event_target` identifies the effect.
/// Observers run during collection before later `Judge` systems, so read the
/// outcome after judgement when policy replacements are needed.
#[derive(EntityEvent, Debug, Clone, Copy)]
#[entity_event(propagate, auto_propagate)]
pub struct Landed {
    /// The effect.
    pub entity: Entity,
    /// Its issued id.
    pub id: EffectId,
}

#[cfg(test)]
mod tests;
