//! The record as a fold over effect entities: the world's [`Recorder`],
//! fed by the plugin's systems.

use std::sync::Arc;

use bevy_ecs::{
    lifecycle::{Insert, Remove},
    prelude::*,
};
use rig_core::{
    effect::{EffectId, EffectKind, HandlerDescriptor, HandlerKey, Outcome},
    error::ErrorReport,
    serve::{Origin, Recorder, cancelled},
    streaming::StreamEvent,
};

use super::{
    effect::{EffectOutcome, InFlight, Issued},
    handlers::Bound,
};
use bevy_ecs::component::Component;

/// The world's recorder. `Dispatch` opens a record as it takes an effect,
/// `Collect` closes it as the outcome lands, a despawn in flight closes it
/// as cancelled, and every `Bound` insert describes its handler. Install
/// with [`Recording::install`] (which also describes the handlers bound so
/// far) or as a plain resource before any handler is bound.
#[derive(Resource, Clone)]
pub struct Recording(Arc<dyn Recorder + Send + Sync>);

impl Recording {
    /// A recording into `recorder`.
    pub fn new(recorder: impl Recorder + Send + Sync) -> Self {
        Self(Arc::new(recorder))
    }

    /// Install `recorder` on `world`, describing every handler bound so
    /// far to it first, as a bus driver's `record_to` does.
    pub fn install(world: &mut World, recorder: impl Recorder + Send + Sync) {
        let recording = Self::new(recorder);
        recording.0.begin_delivery_tracking();
        let mut bound: Vec<HandlerDescriptor> = world
            .query::<&Bound>()
            .iter(world)
            .map(|bound| bound.descriptor.clone())
            .collect();
        bound.sort_by(|a, b| a.key.cmp(&b.key));
        recording.handlers(bound);
        world.insert_resource(recording);
    }

    /// Handlers the world serves.
    pub fn handlers(&self, handlers: Vec<HandlerDescriptor>) {
        self.0.handlers(handlers);
    }

    /// A dispatch begins.
    pub fn begin(&self, id: EffectId, key: HandlerKey, kind: EffectKind, origin: Origin) {
        self.0.begin(id, key, kind, origin);
    }

    /// Whether streamed events are wanted verbatim.
    pub fn keep_events(&self) -> bool {
        self.0.keep_events()
    }

    /// One streamed event of `id`.
    pub fn event(&self, id: EffectId, event: &StreamEvent) {
        self.0.event(id, event);
    }

    /// An error item at its original position in a kept stream.
    pub fn stream_error(&self, id: EffectId, error: &ErrorReport) {
        self.0.stream_error(id, error);
    }

    /// A consumer-visible transition in the current schedule pass.
    pub fn delivery(&self, batch: u64, id: EffectId, kind: rig_core::effect::DeliveryKind) {
        self.0
            .delivery(rig_core::effect::Delivery { batch, id, kind });
    }

    /// The outcome of `id`.
    pub fn resolve(&self, id: EffectId, outcome: Result<Outcome, ErrorReport>) {
        self.0.resolve(id, outcome);
    }

    /// Durable output published before this dispatch's terminal outcome.
    pub fn tool_output(&self, id: EffectId, output: rig_core::tool::ToolResultContext) {
        self.0.tool_output(id, output);
    }

    /// A layer decided `id` before any handler served it: no record.
    pub fn discard(&self, id: EffectId) {
        self.0.discard(id);
    }

    /// A layer served `kind` in place of what began: the record's request.
    pub fn patch(&self, id: EffectId, kind: EffectKind) {
        self.0.patch(id, kind);
    }
}

/// The current delivery batch. Advances once per schedule pass, including
/// passes run directly by a host rather than through the Update runner.
#[derive(Resource, Default)]
pub struct DeliveryBatch(pub u64);

/// Begin the next pass's observation group.
pub fn begin_delivery_pass(mut batch: ResMut<DeliveryBatch>) {
    batch.0 += 1;
}

/// Record visibility when the outcome is inserted, not later when a query
/// happens to visit it. This also captures answers from world-served handlers.
pub fn record_outcome(
    added: On<Add, EffectOutcome>,
    issued: Query<(&Issued, Has<super::collect::CollectedOutcome>), With<InFlight>>,
    recording: Option<Res<Recording>>,
    batch: Res<DeliveryBatch>,
) {
    if let Some(recording) = recording
        && let Ok((Issued(id), collected)) = issued.get(added.event().entity)
    {
        if !collected {
            recording.0.unsupported_delivery("an in-flight EffectOutcome bypassed Collect; submit world answers with WorldOutcome or typed Answer instead");
        }
        recording.delivery(batch.0, *id, rig_core::effect::DeliveryKind::Outcome);
    }
}

/// What a layered handler's sink observer saw of one dispatch, shared
/// with the effect entity: the outcome the innermost handler answered —
/// what the record holds, whatever verdict a layer's `after` gave the
/// world — and whether a layer discarded the dispatch before any handler
/// served it. Never serialized: in-flight state.
#[derive(Component, Clone, Default)]
pub struct Observed(pub Arc<ObservedState>);

/// The observer's slots.
#[derive(Default)]
pub struct ObservedState {
    outcome: std::sync::Mutex<Option<Result<Outcome, ErrorReport>>>,
    discarded: std::sync::atomic::AtomicBool,
}

impl ObservedState {
    /// The handler's outcome, if the observer was told one.
    pub fn take_outcome(&self) -> Option<Result<Outcome, ErrorReport>> {
        self.outcome
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    }

    /// Whether a layer discarded the dispatch.
    pub fn is_discarded(&self) -> bool {
        self.discarded.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// The sink observer `Dispatch` installs for a handler whose descriptor
/// names layers: a layer's decisions reach the record only through it
/// (`Observe::discard`, `Observe::patch`), and the events and the outcome
/// it is told are the innermost handler's — what the record holds,
/// whatever verdict the outer channel carries to the world.
pub struct WorldObserver {
    /// Tool output shared with the caller, read without consuming it.
    pub published: Option<Arc<rig_core::tool::PublishedContext>>,
    /// The dispatch.
    pub id: EffectId,
    /// The world's recorder, if any.
    pub recording: Option<Recording>,
    /// The shared slots.
    pub observed: Arc<ObservedState>,
}

impl rig_core::serve::Observe for WorldObserver {
    fn outcome(&mut self, outcome: &Result<Outcome, ErrorReport>) {
        if let (Some(recording), Some(output)) = (
            &self.recording,
            self.published
                .as_ref()
                .and_then(|published| published.result_context()),
        ) {
            recording.tool_output(self.id, output);
        }
        *self
            .observed
            .outcome
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(outcome.clone());
    }

    // A stream's events are recorded from the innermost hop, as its
    // outcome is: a layer's verdict may replace what the outer channel
    // carries after them (its terminal record among it).
    fn keep_events(&self) -> bool {
        self.recording.as_ref().is_some_and(Recording::keep_events)
    }

    fn event(&mut self, event: &StreamEvent) {
        if let Some(recording) = &self.recording {
            recording.event(self.id, event);
        }
    }

    fn stream_error(&mut self, error: &ErrorReport) {
        if let Some(recording) = &self.recording {
            recording.stream_error(self.id, error);
        }
    }

    fn discard(&mut self) {
        self.observed
            .discarded
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(recording) = &self.recording {
            recording.discard(self.id);
        }
    }

    fn patch(&mut self, kind: &EffectKind) {
        if let Some(recording) = &self.recording {
            recording.patch(self.id, kind.clone());
        }
    }
}

/// State needed to record cancellation and any output published before it.
pub type CancellationView = (
    &'static Issued,
    Option<&'static EffectOutcome>,
    Option<&'static super::effect::Publishing>,
    Option<&'static super::effect::ToolOutputs>,
);

/// An in-flight effect losing `InFlight` without an outcome — a despawn,
/// its own or an ancestor's — is a cancelled dispatch: the record says so,
/// as it does when a consumer drops its `Pending` on rig-bus.
pub fn record_cancelled(
    removed: On<Remove, InFlight>,
    effects: Query<CancellationView>,
    recording: Option<Res<Recording>>,
) {
    let Some(recording) = recording else {
        return;
    };
    if let Ok((Issued(id), None, publishing, outputs)) = effects.get(removed.event().entity) {
        let output = publishing
            .and_then(|published| published.0.result_context())
            .or_else(|| outputs.map(|outputs| outputs.0.result_context()));
        if let Some(output) = output {
            recording.tool_output(*id, output);
        }
        recording.resolve(*id, Err(cancelled()));
    }
}

/// A handler bound (or re-bound) while recording: described to the
/// recorder, as a driver describes each handler installed after recording
/// started.
pub fn record_bound(
    inserted: On<Insert, Bound>,
    bound: Query<&Bound>,
    recording: Option<Res<Recording>>,
) {
    let Some(recording) = recording else {
        return;
    };
    if let Ok(bound) = bound.get(inserted.event().entity) {
        recording.handlers(vec![bound.descriptor.clone()]);
    }
}
