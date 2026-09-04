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

    /// The outcome of `id`.
    pub fn resolve(&self, id: EffectId, outcome: Result<Outcome, ErrorReport>) {
        self.0.resolve(id, outcome);
    }
}

/// An in-flight effect losing `InFlight` without an outcome — a despawn,
/// its own or an ancestor's — is a cancelled dispatch: the record says so,
/// as it does when a consumer drops its `Pending` on rig-bus.
pub fn record_cancelled(
    removed: On<Remove, InFlight>,
    effects: Query<(&Issued, Option<&EffectOutcome>)>,
    recording: Option<Res<Recording>>,
) {
    let Some(recording) = recording else {
        return;
    };
    if let Ok((Issued(id), None)) = effects.get(removed.event().entity) {
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
