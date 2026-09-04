//! `BusSet::Collect`: land what finished, close the record.

use bevy_ecs::prelude::*;
use bevy_tasks::futures::check_ready;
use futures::channel::mpsc::TryRecvError;
use rig_core::{
    serve::stream_truncated,
    streaming::{Delta, StreamEvent},
};

use super::{
    effect::{EffectOutcome, InFlight, Issued, Serving, Streamed, Streaming},
    plugin::Progress,
    record::Recording,
};

/// A unary handler's task finished: its outcome lands as [`EffectOutcome`]
/// and the task leaves the entity. A non-blocking check per in-flight task
/// (`check_ready`), no waker kept, nothing awaited.
pub fn collect_tasks(
    mut commands: Commands,
    mut serving: Query<(Entity, &mut Serving), With<InFlight>>,
    mut progress: ResMut<Progress>,
) {
    for (entity, mut serving) in &mut serving {
        if let Some(outcome) = check_ready(&mut serving.0) {
            commands
                .entity(entity)
                .remove::<Serving>()
                .insert(EffectOutcome(outcome));
            progress.mark();
        }
    }
}

/// A streaming handler sent: every item it has sent since the last pass is
/// folded into [`Streamed`] (the record keeps the events when it keeps
/// events; the fold yields the outcome at the terminal or at an error);
/// when the handler's channel closes the fold's outcome — or a truncation
/// report when no terminal came — lands as [`EffectOutcome`].
pub fn collect_streams(
    mut commands: Commands,
    mut streaming: Query<(Entity, &Issued, &mut Streaming, &mut Streamed), With<InFlight>>,
    recording: Option<Res<Recording>>,
    mut progress: ResMut<Progress>,
) {
    for (entity, Issued(id), mut streaming, mut streamed) in &mut streaming {
        loop {
            match streaming.events.try_recv() {
                Ok(item) => {
                    if let (Some(recording), Ok(event)) = (&recording, &item)
                        && recording.keep_events()
                    {
                        recording.event(*id, event);
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
                    progress.mark();
                }
                Err(TryRecvError::Closed) => {
                    let outcome = streamed
                        .outcome
                        .clone()
                        .unwrap_or_else(|| Err(stream_truncated()));
                    commands
                        .entity(entity)
                        .remove::<Streaming>()
                        .insert(EffectOutcome(outcome));
                    progress.mark();
                    break;
                }
                Err(TryRecvError::Empty) => break,
            }
        }
    }
}

/// An outcome that landed on an effect still in flight.
pub type Landed = (Added<EffectOutcome>, With<InFlight>);

/// An outcome landed on an in-flight effect: the record closes with it and
/// the effect leaves flight. The one place records close, so a `Gate`
/// denial (never in flight) is no record and a `Judge` rewrite (after this)
/// is not re-recorded: decisions are program, never record.
pub fn settle(
    mut commands: Commands,
    landed: Query<(Entity, &Issued, &EffectOutcome), Landed>,
    recording: Option<Res<Recording>>,
    mut progress: ResMut<Progress>,
) {
    for (entity, Issued(id), outcome) in &landed {
        if let Some(recording) = &recording {
            recording.resolve(*id, outcome.0.clone());
        }
        commands.entity(entity).remove::<InFlight>();
        progress.mark();
    }
}
