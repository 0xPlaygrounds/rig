//! `BusSet::Collect`: land what finished, close the record.

use bevy_ecs::prelude::*;
use bevy_tasks::futures::check_ready;
use futures::channel::mpsc::TryRecvError;
use rig_core::{
    serve::stream_truncated,
    streaming::{Delta, StreamEvent},
};

use super::{
    effect::{
        EffectOutcome, InFlight, Issued, Publishing, Serving, Streamed, Streaming, ToolOutputs,
    },
    plugin::Progress,
    record::{Observed, Recording},
};

/// A unary handler's task finished: its outcome lands as [`EffectOutcome`]
/// and the task leaves the entity; a tool call's published context lands
/// beside it as [`ToolOutputs`]. A non-blocking check per in-flight task
/// (`check_ready`), no waker kept, nothing awaited.
pub fn collect_tasks(
    mut commands: Commands,
    mut serving: Query<(Entity, &mut Serving, Option<&Publishing>), With<InFlight>>,
    mut progress: ResMut<Progress>,
) {
    for (entity, mut serving, publishing) in &mut serving {
        if let Some(outcome) = check_ready(&mut serving.0) {
            let mut entity_commands = commands.entity(entity);
            entity_commands
                .remove::<Serving>()
                .insert(EffectOutcome(outcome));
            if let Some(Publishing(published)) = publishing {
                entity_commands.remove::<Publishing>();
                if let Some(context) = published.take() {
                    entity_commands.insert(ToolOutputs(context));
                }
            }
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
    mut streaming: Query<StreamingView, With<InFlight>>,
    recording: Option<Res<Recording>>,
    mut progress: ResMut<Progress>,
) {
    for (entity, Issued(id), mut streaming, mut streamed, observed) in &mut streaming {
        loop {
            match streaming.events.try_recv() {
                Ok(item) => {
                    // A layered handler's events are the observer's to
                    // record, from the innermost hop.
                    if let (Some(recording), Ok(event), false) = (&recording, &item, observed)
                        && recording.keep_events()
                    {
                        recording.event(*id, event);
                    }
                    if streamed.outcome.is_none()
                        && let Some(outcome) = streaming.fold.observe(&item)
                    {
                        streamed.outcome = Some(outcome);
                        // The fold's outcome is a transition; a delta is not —
                        // a fast handler must not spin the quiescence loop.
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

/// What `collect_streams` reads of a streaming effect: its id, the task
/// and channel, the fold so far, and whether a layer's observer records.
pub type StreamingView = (
    Entity,
    &'static Issued,
    &'static mut Streaming,
    &'static mut Streamed,
    Has<Observed>,
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
        let discarded = observed.is_some_and(|observed| observed.0.is_discarded());
        let recorded = observed
            .and_then(|observed| observed.0.take_outcome())
            .unwrap_or_else(|| outcome.0.clone());
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
        commands.entity(entity).remove::<(InFlight, Observed)>();
        progress.mark();
    }
}
