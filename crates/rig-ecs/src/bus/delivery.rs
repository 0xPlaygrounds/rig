//! Replay collection at recorded policy observation boundaries.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use bevy_ecs::prelude::*;
use bevy_tasks::futures::check_ready;
use futures::channel::mpsc::TryRecvError;
use rig_core::{
    effect::{Delivery, DeliveryKind, EffectId, Outcome},
    error::{ErrorKind, ErrorReport},
    streaming::{Delta, StreamEvent},
};
use rig_effect_log::EffectLog;

use super::{
    effect::{
        EffectOutcome, IdCounter, InFlight, Issued, PendingEffect, Publishing, Reserved, Serving,
        Streamed, Streaming, ToolOutputs,
    },
    plugin::Progress,
    record::{DeliveryBatch, Observed, Recording},
};

/// A delivery trace could not be followed. Also inserted as a resource when
/// the unchanged program fails to create an effect needed by its next batch.
#[derive(Resource, Debug, Clone)]
pub struct ReplayFailure(pub ErrorReport);

/// Shared delivery plan installed by `Replay::register`. Readiness remains on
/// effect entities; this resource holds only the remaining recorded trace.
#[derive(Resource)]
pub struct ReplayDelivery {
    pending: VecDeque<Delivery>,
    ids: BTreeSet<EffectId>,
    keys: BTreeMap<EffectId, rig_core::effect::HandlerKey>,
    folded: BTreeSet<EffectId>,
    cancelled: BTreeSet<EffectId>,
    policy_visible: bool,
    waiting_for: Option<(EffectId, u64)>,
    refusals: rig_effect_log::ReplayRefusals,
}

fn invalid(message: impl Into<String>) -> ErrorReport {
    ErrorReport::new(ErrorKind::Divergence, message)
}

impl ReplayDelivery {
    /// Shared refusal signal for the replayers serving this delivery plan.
    pub fn refusals(&self) -> rig_effect_log::ReplayRefusals {
        self.refusals.clone()
    }

    /// Validate a recording and construct its delivery plan.
    pub fn new(log: &EffectLog, required: bool) -> Result<Option<Self>, ErrorReport> {
        if required && !log.header.delivery_limitations.is_empty() {
            return Err(invalid(format!(
                "policy-visible replay refused: {}",
                log.header.delivery_limitations.join("; ")
            )));
        }
        let Some(deliveries) = &log.header.deliveries else {
            return if required && !log.records.is_empty() {
                Err(invalid(
                    "policy-visible replay requires recorded delivery boundaries",
                ))
            } else {
                Ok(None)
            };
        };
        let mut records = BTreeMap::new();
        for record in &log.records {
            if records.insert(record.id, record).is_some() {
                return Err(invalid(format!("duplicate replay effect id {}", record.id)));
            }
        }
        let mut last_batch = 0;
        let mut terminal = BTreeSet::new();
        let mut items = BTreeMap::<EffectId, usize>::new();
        let mut folded = BTreeSet::new();
        for delivery in deliveries {
            let record = records
                .get(&delivery.id)
                .ok_or_else(|| invalid(format!("delivery names absent record {}", delivery.id)))?;
            if delivery.batch < last_batch || terminal.contains(&delivery.id) {
                return Err(invalid(format!(
                    "inconsistent delivery order for {}",
                    delivery.id
                )));
            }
            last_batch = delivery.batch;
            match delivery.kind {
                DeliveryKind::Outcome => {
                    terminal.insert(delivery.id);
                }
                DeliveryKind::Stream { items: count } => {
                    if count == 0 || !record.kind.streams() {
                        return Err(invalid(format!(
                            "invalid stream delivery for {}",
                            delivery.id
                        )));
                    }
                    let total = items.entry(delivery.id).or_default();
                    *total = total
                        .checked_add(count)
                        .ok_or_else(|| invalid("stream delivery count overflow"))?;
                    if record.events.is_none() {
                        if required {
                            return Err(invalid(format!(
                                "policy-visible replay requires kept stream events for {}",
                                delivery.id
                            )));
                        }
                        folded.insert(delivery.id);
                    }
                }
            }
        }
        for record in &log.records {
            let cancelled = record
                .outcome
                .as_ref()
                .is_err_and(|error| error.kind == ErrorKind::Cancelled);
            if !terminal.contains(&record.id) && !cancelled {
                return Err(invalid(format!(
                    "missing outcome delivery for {}",
                    record.id
                )));
            }
            if let Some(events) = &record.events {
                let count = items.get(&record.id).copied().unwrap_or_default();
                let maximum = events.len().saturating_add(
                    log.header
                        .stream_errors
                        .get(&record.id)
                        .map_or(usize::from(record.outcome.is_err()), Vec::len),
                );
                let minimum =
                    events.len() + log.header.stream_errors.get(&record.id).map_or(0, Vec::len);
                if count > maximum || (terminal.contains(&record.id) && count < minimum) {
                    return Err(invalid(format!(
                        "stream delivery counts disagree with events for {}",
                        record.id
                    )));
                }
                if required && !cancelled {
                    let errors = log
                        .header
                        .stream_errors
                        .get(&record.id)
                        .map(Vec::as_slice)
                        .unwrap_or_default();
                    let mut errors = errors.iter().peekable();
                    let mut events = events.iter();
                    let mut tap = rig_core::serve::StreamTap::new();
                    let mut first = None;
                    for position in 0..minimum {
                        let item = if errors.peek().is_some_and(|error| error.item == position) {
                            errors.next().map(|error| Err(error.error.clone()))
                        } else {
                            events.next().cloned().map(Ok)
                        };
                        if first.is_none()
                            && let Some(item) = item
                        {
                            first = tap.observe(&item);
                        }
                    }
                    let folded = first.unwrap_or_else(|| Err(rig_core::serve::stream_truncated()));
                    if serde_json::to_value(&folded).map_err(|error| invalid(error.to_string()))?
                        != serde_json::to_value(&record.outcome)
                            .map_err(|error| invalid(error.to_string()))?
                    {
                        return Err(invalid(format!(
                            "policy-visible replay cannot reconstruct the first stream outcome for {}; error positions or event bytes are missing or inconsistent",
                            record.id
                        )));
                    }
                }
            }
        }
        Ok(Some(Self {
            keys: log
                .records
                .iter()
                .map(|record| (record.id, record.key.clone()))
                .collect(),
            pending: deliveries
                .iter()
                .filter(|delivery| {
                    !(folded.contains(&delivery.id)
                        && matches!(delivery.kind, DeliveryKind::Stream { .. }))
                })
                .cloned()
                .collect(),
            ids: records
                .keys()
                .filter(|id| required || terminal.contains(id) || items.contains_key(id))
                .copied()
                .collect(),
            folded,
            cancelled: records
                .keys()
                .filter(|id| !terminal.contains(id) && (required || items.contains_key(id)))
                .copied()
                .collect(),
            policy_visible: required,
            waiting_for: None,
            refusals: rig_effect_log::ReplayRefusals::default(),
        }))
    }
}

/// Readiness stays on the effect entity. Removing the ordinary task/channel
/// component prevents the unpaced collector from exposing replay data early.
#[derive(Component)]
enum Buffered {
    Unary {
        task: Serving,
        answer: Option<Result<Outcome, ErrorReport>>,
    },
    Stream {
        streaming: Streaming,
        items: VecDeque<Result<StreamEvent, ErrorReport>>,
        closed: bool,
    },
}

impl Buffered {
    fn poll(&mut self) {
        match self {
            Self::Unary { task, answer } => {
                if answer.is_none() {
                    *answer = check_ready(&mut task.0);
                }
            }
            Self::Stream {
                streaming,
                items,
                closed,
            } => {
                while !*closed {
                    match streaming.events.try_recv() {
                        Ok(item) => items.push_back(item),
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Closed) => *closed = true,
                    }
                }
            }
        }
    }
}

/// Buffer ready replay data, then expose one complete recorded batch. Every
/// policy system gets a pass between distinct batches, even if all handler
/// futures completed together. Live handlers keep their ordinary collector.
pub fn collect_replayed(world: &mut World) {
    if !world.contains_resource::<ReplayDelivery>() {
        return;
    }
    world.resource_scope(|world, mut replay: Mut<ReplayDelivery>| {
        replay.waiting_for = None;
        let entities: Vec<_> = world
            .query_filtered::<(Entity, &Issued), With<InFlight>>()
            .iter(world)
            .filter(|(_, issued)| replay.ids.contains(&issued.0))
            .map(|(entity, issued)| (issued.0, entity))
            .collect();
        for (_, entity) in &entities {
            let mut entity = world.entity_mut(*entity);
            if let Some(task) = entity.take::<Serving>() {
                entity.insert(Buffered::Unary { task, answer: None });
            } else if let Some(streaming) = entity.take::<Streaming>() {
                entity.insert(Buffered::Stream {
                    streaming,
                    items: VecDeque::new(),
                    closed: false,
                });
            }
            if let Some(mut buffered) = entity.get_mut::<Buffered>() {
                buffered.poll();
            }
        }
        // A refusal remains terminal for replay effects created by later
        // policy sets. Buffer them before ordinary collection can expose data.
        if let Some(failure) = world.get_resource::<ReplayFailure>().cloned() {
            fail(world, failure.0);
            return;
        }
        // A request mismatch can change unary/stream shape, so its diagnostic
        // cannot wait for deliveries that only the matching request produces.
        if let Some(report) = replay.refusals.take() {
            fail(world, report);
            return;
        }
        let by_id: BTreeMap<_, _> = world
            .query::<(Entity, Option<&Issued>, Option<&Reserved>)>()
            .iter(world)
            .filter_map(|(entity, issued, reserved)| {
                issued
                    .map(|issued| issued.0)
                    .or_else(|| reserved.map(|reserved| reserved.0))
                    .map(|id| (id, entity))
            })
            .collect();
        let queued = world.query_filtered::<Entity, (
            With<PendingEffect>, Without<Issued>, Without<EffectOutcome>,
        )>().iter(world).next().is_some();
        // Give policy its intervening pass to reproduce a recorded cancel.
        // A bare exchange consumer that remains receives the recorded error.
        for id in &replay.cancelled {
            if replay.pending.iter().any(|step| &step.id == id) {
                continue;
            }
            let Some(entity) = by_id.get(id).copied() else {
                continue;
            };
            if replay.policy_visible {
                // Judge and later sets may need multiple passes. Diagnose an
                // unreproduced cancellation only after full quiescence.
                continue;
            }
            let count = match world.get::<Buffered>(entity) {
                Some(Buffered::Stream {
                    items,
                    closed: true,
                    ..
                }) => items.len(),
                _ => continue,
            };
            deliver_stream(world, entity, *id, count);
            deliver_outcome(world, entity);
            world.resource_mut::<Progress>().mark();
        }
        let Some(first) = replay.pending.front() else {
            return;
        };
        let batch = first.batch;
        let steps: Vec<_> = replay
            .pending
            .iter()
            .take_while(|step| step.batch == batch)
            .cloned()
            .collect();
        let mut needed = BTreeMap::<EffectId, usize>::new();
        for step in &steps {
            let unissued = by_id.get(&step.id).is_none_or(|entity| world.get::<Issued>(*entity).is_none());
            if unissued && world.resource::<super::plugin::Policy>().0.serial_per_handler {
                let blocked = replay.keys.get(&step.id).is_some_and(|key| {
                    steps.iter().filter(|other| other.id != step.id).any(|other| {
                        by_id.get(&other.id).and_then(|entity| world.get::<InFlight>(*entity)).is_some_and(|flight| &flight.key == key)
                    })
                });
                if blocked {
                    fail(world, invalid(format!("replay batch {batch} cannot dispatch {} while another effect in the same batch occupies its serial key", step.id)));
                    return;
                }
            }
            let Some(entity) = by_id.get(&step.id).copied() else {
                // A restored subset or a discarded/cancelled entity has
                // already passed the id counter. It is not a gate to await.
                if step.id.as_u64() < world.resource::<IdCounter>().0 {
                    continue;
                }
                // Intake is bounded per Update, not per quiescence pass.
                // Pending unreserved effects may mint this id next Update;
                // a Held effect may also await the host's release.
                if queued || world.resource::<Progress>().0 {
                    return;
                }
                // Continuations may run after Collect. Diagnose a missing
                // request only after the entire schedule is quiescent.
                replay.waiting_for = Some((step.id, batch));
                return;
            };
            if world.get::<EffectOutcome>(entity).is_some() {
                continue;
            }
            let Some(buffered) = world.get::<Buffered>(entity) else {
                return;
            };
            match (&step.kind, buffered) {
                (DeliveryKind::Outcome, Buffered::Unary { answer, .. }) if answer.is_some() => {}
                (DeliveryKind::Outcome, Buffered::Stream { closed: true, .. }) => {}
                (DeliveryKind::Stream { items: count }, Buffered::Stream { items, closed, .. }) => {
                    let total = needed.entry(step.id).or_default();
                    *total += count;
                    if items.len() < *total {
                        if *closed {
                            fail(
                                world,
                                invalid(format!(
                                    "replay stream {} closed before its recorded batch",
                                    step.id
                                )),
                            );
                        }
                        return;
                    }
                }
                _ => return,
            }
        }
        for step in &steps {
            let Some(entity) = by_id.get(&step.id).copied() else {
                continue;
            };
            if world.get::<EffectOutcome>(entity).is_some() {
                continue;
            }
            match step.kind {
                DeliveryKind::Stream { items } => deliver_stream(world, entity, step.id, items),
                DeliveryKind::Outcome => {
                    if replay.folded.contains(&step.id) {
                        let count = match world.get::<Buffered>(entity) {
                            Some(Buffered::Stream { items, .. }) => items.len(),
                            _ => 0,
                        };
                        deliver_stream(world, entity, step.id, count);
                    }
                    deliver_outcome(world, entity);
                }
            }
        }
        replay.pending.drain(..steps.len());
        replay.waiting_for = None;
        world.resource_mut::<Progress>().mark();
    });
}

/// Diagnose absent requests and unreproduced cancellations after every policy
/// set has run. Called by the bus runner before it stops at quiescence.
pub fn diagnose_idle_replay(world: &mut World) {
    if world.resource::<Progress>().0 || world.contains_resource::<ReplayFailure>() {
        return;
    }
    let mut effects = world.query::<(Option<&Issued>, Option<&Reserved>, Option<&EffectOutcome>)>();
    let Some(replay) = world.get_resource::<ReplayDelivery>() else {
        return;
    };
    let waiting_for = replay.waiting_for;
    let uncancelled = if replay.policy_visible && replay.pending.is_empty() {
        // Read the final world, not Collect's entity map: deferred policy
        // commands may have removed the effect since that earlier snapshot.
        effects
            .iter(world)
            .filter_map(|(issued, reserved, outcome)| {
                let id = issued
                    .map(|issued| issued.0)
                    .or_else(|| reserved.map(|reserved| reserved.0))?;
                (outcome.is_none() && replay.cancelled.contains(&id)).then_some(id)
            })
            .min()
    } else {
        None
    };
    if let Some(id) = uncancelled {
        fail(
            world,
            invalid(format!(
                "policy replay did not reproduce cancellation of {id}"
            )),
        );
        return;
    }
    let Some((id, batch)) = waiting_for else {
        return;
    };
    // Late policy may have minted or queued the request during this pass.
    if id.as_u64() < world.resource::<IdCounter>().0
        || world.query_filtered::<Entity, (With<PendingEffect>, Without<Issued>, Without<EffectOutcome>)>().iter(world).next().is_some()
    {
        return;
    }
    fail(
        world,
        invalid(format!("replay batch {batch} requires undispatched {id}")),
    );
}

fn fail(world: &mut World, report: ErrorReport) {
    let entities: Vec<_> = world
        .query_filtered::<Entity, With<Buffered>>()
        .iter(world)
        .collect();
    let changed = !entities.is_empty() || !world.contains_resource::<ReplayFailure>();
    for entity in entities {
        world
            .entity_mut(entity)
            .remove::<Buffered>()
            .insert(EffectOutcome(Err(report.clone())));
    }
    world.insert_resource(ReplayFailure(report));
    if changed {
        world.resource_mut::<Progress>().mark();
    }
}

fn deliver_stream(world: &mut World, entity: Entity, id: EffectId, count: usize) {
    let Some(mut buffered) = world.entity_mut(entity).take::<Buffered>() else {
        return;
    };
    let Buffered::Stream {
        streaming, items, ..
    } = &mut buffered
    else {
        world.entity_mut(entity).insert(buffered);
        return;
    };
    let recording = world.get_resource::<Recording>().cloned();
    let observed = world.get::<Observed>(entity).is_some();
    let mut progress = false;
    if let Some(mut streamed) = world.get_mut::<Streamed>(entity) {
        for item in items.drain(..count) {
            if let (Some(recording), false) = (&recording, observed)
                && recording.keep_events()
            {
                match &item {
                    Ok(event) => recording.event(id, event),
                    Err(error) => recording.stream_error(id, error),
                }
            }
            if streamed.outcome.is_none()
                && let Some(outcome) = streaming.fold.observe(&item)
            {
                streamed.outcome = Some(outcome);
                progress = true;
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
        }
    }
    world.entity_mut(entity).insert(buffered);
    if count != 0
        && let Some(recording) = recording
    {
        recording.delivery(
            world.resource::<DeliveryBatch>().0,
            id,
            DeliveryKind::Stream { items: count },
        );
    }
    if progress {
        world.resource_mut::<Progress>().mark();
    }
}

fn deliver_outcome(world: &mut World, entity: Entity) {
    let Some(buffered) = world.entity_mut(entity).take::<Buffered>() else {
        return;
    };
    let outcome = match buffered {
        Buffered::Unary {
            answer: Some(answer),
            ..
        } => answer,
        Buffered::Stream { .. } => world
            .get::<Streamed>(entity)
            .and_then(|streamed| streamed.outcome.clone())
            .unwrap_or_else(|| Err(rig_core::serve::stream_truncated())),
        Buffered::Unary { answer: None, .. } => Err(invalid("replay outcome was not ready")),
    };
    if let Some(Publishing(published)) = world.entity_mut(entity).take::<Publishing>()
        && let Some(context) = published.take()
    {
        world.entity_mut(entity).insert(ToolOutputs(context));
    }
    world
        .entity_mut(entity)
        .insert(super::collect::CollectedOutcome)
        .insert(EffectOutcome(outcome));
}
