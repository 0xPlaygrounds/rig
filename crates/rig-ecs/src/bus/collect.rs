//! `BusSet::Collect`: land what finished, close the record.

use bevy_ecs::prelude::*;
use bevy_tasks::futures::check_ready;
use rig_core::{
    serve::{Reply, stream_truncated},
    streaming::{Delta, StreamEvent},
};
use std::task::Poll;

use super::{
    effect::{
        EffectOutcome, Executions, InFlight, Issued, Publishing, Serving, Streamed, Streaming,
        ToolOutputs, WorldOutcome,
    },
    plugin::Progress,
    record::{DeliveryBatch, Observed, Recording},
};

/// Maximum live streaming queue checks across a host tick.
pub const STREAM_WORK_PER_TICK: usize = 4096;
const STREAM_ITEMS_PER_EFFECT: usize = 64;

/// Streaming delivery allowance shared by all quiescence passes in one host tick.
/// The sequence cursor rotates service when the allowance runs out.
#[derive(Resource)]
pub struct CollectionBudget {
    /// Queue checks left in the current allowance.
    pub remaining: usize,
    /// Whether collection belongs to the shared quiescence loop.
    pub in_runner: bool,
    last: Option<super::Seq>,
}
impl Default for CollectionBudget {
    fn default() -> Self {
        Self {
            remaining: STREAM_WORK_PER_TICK,
            in_runner: false,
            last: None,
        }
    }
}

/// Marker for a library collector's outcome insertion. Removed by settlement;
/// direct in-flight insertions without it cannot establish policy replay.
#[derive(Component)]
pub struct CollectedOutcome;

/// Submitted world answers whose visible outcome has not landed.
pub type WorldAnswersReady = (With<InFlight>, Without<EffectOutcome>);

/// Publish submitted world answers at the same boundary as task results.
/// Arrival order is independent of dispatch order and entity archetypes.
pub fn collect_world(
    mut commands: Commands,
    ready: Query<(Entity, &WorldOutcome), WorldAnswersReady>,
    mut progress: ResMut<Progress>,
) {
    let mut ready: Vec<_> = ready.iter().collect();
    ready.sort_by_key(|(_, outcome)| outcome.order());
    for (entity, outcome) in ready {
        commands
            .entity(entity)
            .insert(CollectedOutcome)
            .insert(EffectOutcome(outcome.outcome.clone()))
            .remove::<WorldOutcome>();
        progress.mark();
    }
}

/// A unary handler's task finished: its outcome lands as [`EffectOutcome`]
/// and the task leaves the entity; a tool call's published context lands
/// beside it as [`ToolOutputs`]. A non-blocking check per in-flight task
/// (`check_ready`), no waker kept, nothing awaited.
pub fn collect_tasks(
    mut commands: Commands,
    serving: Query<(Entity, &Serving, Option<&Publishing>), With<InFlight>>,
    policy: Res<super::Policy>,
    mut executions: NonSendMut<Executions>,
    mut progress: ResMut<Progress>,
) {
    for (entity, _, publishing) in &serving {
        let Some(task) = executions.tasks.get_mut(&entity) else {
            continue;
        };
        let Some(reply) = check_ready(task) else {
            continue;
        };
        executions.tasks.remove(&entity);
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
                progress.mark();
            }
            Reply::Stream(stream) => {
                let (streaming, task) = Streaming::spawn(stream, policy.0.stream_capacity);
                executions.streams.insert(entity, task);
                entity_commands.insert(streaming);
            }
        }
    }
}

/// Drain bounded worker delivery and fold each item. The first folded
/// outcome is retained; streaming effects settle at EOF, including post-final
/// metadata and errors. Pending waits for the host's next Collect invocation.
#[allow(
    clippy::too_many_arguments,
    reason = "one collection pass shares driver state and its work allowance"
)]
pub fn collect_streams(
    mut commands: Commands,
    mut streaming: Query<StreamingView, With<InFlight>>,
    mut executions: NonSendMut<Executions>,
    recording: Option<Res<Recording>>,
    batch: Res<DeliveryBatch>,
    mut progress: ResMut<Progress>,
    mut budget: ResMut<CollectionBudget>,
    mut order: Local<Vec<(super::Seq, Entity)>>,
) {
    order.clear();
    order.extend(streaming.iter().map(|(entity, seq, ..)| (*seq, entity)));
    order.sort_unstable_by_key(|(seq, _)| *seq);
    let start = budget
        .last
        .map_or(0, |last| order.partition_point(|(seq, _)| *seq <= last));
    for &(seq, entity) in order.iter().cycle().skip(start).take(order.len()) {
        if budget.remaining == 0 {
            break;
        }
        budget.last = Some(seq);
        let Ok((_, _, Issued(id), mut streaming, mut streamed, publishing)) =
            streaming.get_mut(entity)
        else {
            continue;
        };
        let mut delivered = 0;
        for _ in 0..STREAM_ITEMS_PER_EFFECT {
            if budget.remaining == 0 {
                break;
            }
            budget.remaining -= 1;
            let polled = match streaming.events.try_recv() {
                Ok(item) => Poll::Ready(Some(item)),
                Err(futures::channel::mpsc::TryRecvError::Empty) => Poll::Pending,
                Err(futures::channel::mpsc::TryRecvError::Closed) => Poll::Ready(None),
            };
            let outcome = match polled {
                Poll::Pending => break,
                Poll::Ready(Some(item)) => {
                    if let Some(streamed) = &mut streamed {
                        if let Err(error) = &item {
                            let position = streamed.events.len() + streamed.errors.len();
                            streamed.errors.push((position, error.clone()));
                        }
                        if streamed.outcome.is_none()
                            && let Some(outcome) = streaming.fold.observe(&item)
                        {
                            streamed.outcome = Some(outcome);
                            progress.mark();
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
                Poll::Ready(None) => streamed
                    .as_ref()
                    .and_then(|streamed| streamed.outcome.clone())
                    .unwrap_or_else(|| Err(stream_truncated())),
            };
            executions.streams.remove(&entity);
            let mut entity_commands = commands.entity(entity);
            if let Some(Publishing(published)) = publishing {
                entity_commands.remove::<Publishing>();
                if let Some(context) = published.take() {
                    entity_commands.insert(ToolOutputs(context));
                }
            }
            entity_commands
                .remove::<Streaming>()
                .insert(CollectedOutcome)
                .insert(EffectOutcome(outcome));
            progress.mark();
            break;
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
pub type Landed = (Added<EffectOutcome>, With<InFlight>);

/// The outcome and durable tool output needed to close a dispatch's record.
pub type LandedView = (
    Entity,
    &'static Issued,
    &'static EffectOutcome,
    Option<&'static Observed>,
    Option<&'static ToolOutputs>,
);

/// An outcome landed on an in-flight effect: the record closes with it and
/// the effect leaves flight. The one place records close, so a `Gate`
/// denial (never in flight) is no record and a `Judge` rewrite (after this)
/// is not re-recorded: decisions are program, never record.
pub fn settle(
    mut commands: Commands,
    landed: Query<LandedView, Landed>,
    recording: Option<Res<Recording>>,
    mut progress: ResMut<Progress>,
) {
    for (entity, &Issued(id), outcome, observed, outputs) in &landed {
        // A layered handler: the record holds what the innermost handler
        // answered (the observer's), never a layer's verdict; a dispatch a
        // layer discarded is no record.
        let recorded = observed
            .and_then(|observed| observed.0.take_outcome())
            .unwrap_or_else(|| outcome.0.clone());
        let discarded = observed.is_some_and(|observed| observed.0.is_discarded());
        if let (Some(recording), false) = (&recording, discarded) {
            // Layered dispatches captured output at the inner handler's
            // terminal, before an outer verdict could change it.
            if observed.is_none()
                && let Some(outputs) = outputs
            {
                recording.tool_output(id, outputs.0.result_context());
            }
            recording.resolve(id, recorded);
        }
        commands
            .entity(entity)
            .remove::<(InFlight, Observed, CollectedOutcome)>();
        progress.mark();
    }
}

#[cfg(test)]
mod tests;
