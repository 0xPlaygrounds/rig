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
pub fn begin_delivery_pass(
    mut batch: ResMut<DeliveryBatch>,
    mut budget: ResMut<super::collect::CollectionBudget>,
) {
    batch.0 += 1;
    if !budget.in_runner {
        budget.remaining = super::collect::STREAM_WORK_PER_TICK;
    }
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

/// What the handler's recording observer saw of one dispatch, shared
/// with the effect entity: the outcome the innermost handler answered —
/// what the record holds, whatever verdict a layer's `after` gave the
/// world — and whether a layer discarded the dispatch before any handler
/// served it. Never serialized: in-flight state.
#[derive(Component, Clone, Default)]
pub struct Observed(pub Arc<ObservedState>);

/// The dispatch's replacement slot (`Dispatch::replaced_by`): the layer
/// whose verdict the consumer's answer is from, once one said so. Runtime-
/// only; removed with the in-flight markers at collection.
#[derive(Component)]
pub struct ReplacedBy(pub Arc<std::sync::Mutex<Option<String>>>);

impl ReplacedBy {
    /// The layer named, if any.
    pub fn layer(&self) -> Option<String> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

/// The observer's slots.
#[derive(Default)]
pub struct ObservedState {
    state: std::sync::Mutex<Observation>,
}

#[derive(Default)]
struct Observation {
    outcome: Option<Result<Outcome, ErrorReport>>,
    discarded: bool,
    closed: bool,
}

impl ObservedState {
    fn lock(&self) -> std::sync::MutexGuard<'_, Observation> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Close observation and take the original answer. No later worker callback
    /// can modify recording after this boundary, even if its poll was in progress.
    pub fn take_outcome(&self) -> Option<Result<Outcome, ErrorReport>> {
        let mut state = self.lock();
        state.closed = true;
        state.outcome.take()
    }

    /// Whether the original handler answer has been observed.
    pub fn has_outcome(&self) -> bool {
        self.lock().outcome.is_some()
    }

    /// Whether a layer discarded the dispatch.
    pub fn is_discarded(&self) -> bool {
        self.lock().discarded
    }
}

/// The observer installed on task-served dispatches: layer decisions and
/// original answers reach the record through it
/// (`Observe::discard`, `Observe::patch`), and the events and the outcome
/// it is told are the innermost handler's — what the record holds,
/// whatever verdict the outer reply carries to the world.
pub struct WorldObserver {
    /// Explicit host operation, already bound to this dispatch's subject.
    pub adapter: Option<rig_core::observe::AdapterContext>,
    /// Tool output shared with the caller, read without consuming it.
    pub published: Option<Arc<rig_core::tool::PublishedContext>>,
    /// The dispatch.
    pub id: EffectId,
    /// The world's recorder, if any.
    pub recording: Option<Recording>,
    /// The shared slots.
    pub observed: Arc<ObservedState>,
    /// The world's witness with the dispatch's subject and the kind that
    /// began, so a layer's patch or discard is observed with its before.
    pub witness: Option<(
        super::witness::Witnessing,
        rig_core::observe::Subject,
        EffectKind,
    )>,
}

impl WorldObserver {
    fn record_answer(&self, state: &mut Observation, outcome: &Result<Outcome, ErrorReport>) {
        if let (Some(recording), Some(output)) = (
            &self.recording,
            self.published
                .as_ref()
                .and_then(|published| published.result_context()),
        ) {
            recording.tool_output(self.id, output);
        }
        state.outcome = Some(outcome.clone());
    }

    fn record_item(&self, item: &Result<StreamEvent, ErrorReport>) {
        if let Some(recording) = &self.recording
            && recording.keep_events()
        {
            match item {
                Ok(event) => recording.event(self.id, event),
                Err(error) => recording.stream_error(self.id, error),
            }
        }
    }
}

impl rig_core::serve::Observe for WorldObserver {
    fn adapter_context(&self) -> Option<rig_core::observe::AdapterContext> {
        if let Some(context) = &self.adapter {
            return Some(context.clone());
        }
        let (witness, subject, _) = self.witness.as_ref()?;
        let mut subject = subject.clone();
        subject.effect = Some(self.id);
        Some(rig_core::observe::AdapterContext::new(
            witness.sink().clone(),
            subject,
            format!("effect/{}", self.id),
        ))
    }

    fn outcome(&mut self, outcome: &Result<Outcome, ErrorReport>) {
        let mut state = self.observed.lock();
        if !state.closed {
            self.record_answer(&mut state, outcome);
        }
    }

    fn keep_events(&self) -> bool {
        self.recording.as_ref().is_some_and(Recording::keep_events)
    }

    fn stream_item(
        &mut self,
        item: &Result<StreamEvent, ErrorReport>,
        outcome: Option<&Result<Outcome, ErrorReport>>,
    ) {
        // Source polling and folding happen before this lock. Only observation
        // is atomic with cancellation, so arbitrary handler work cannot hold it.
        let mut state = self.observed.lock();
        if state.closed {
            return;
        }
        self.record_item(item);
        if let Some(outcome) = outcome {
            self.record_answer(&mut state, outcome);
        }
    }

    fn event(&mut self, event: &StreamEvent) {
        let state = self.observed.lock();
        if !state.closed
            && let Some(recording) = &self.recording
        {
            recording.event(self.id, event);
        }
    }

    fn stream_error(&mut self, error: &ErrorReport) {
        let state = self.observed.lock();
        if !state.closed
            && let Some(recording) = &self.recording
        {
            recording.stream_error(self.id, error);
        }
    }

    fn discard(&mut self, layer: &str) {
        let mut state = self.observed.lock();
        if !state.closed {
            state.discarded = true;
            if let Some(recording) = &self.recording {
                recording.discard(self.id);
            }
            if let Some((witness, subject, _)) = &self.witness {
                let mut subject = subject.clone();
                subject.effect = Some(self.id);
                witness.emit(
                    subject,
                    rig_core::observe::Stage::Handler,
                    rig_core::observe::Emitter::named(layer),
                    rig_core::observe::Action::Denied {
                        reason: rig_core::observe::Reason::code("layer_discarded"),
                    },
                );
            }
        }
    }

    fn patch(&mut self, layer: &str, kind: &EffectKind) {
        let state = self.observed.lock();
        if !state.closed {
            if let Some(recording) = &self.recording {
                recording.patch(self.id, kind.clone());
            }
            if let Some((witness, subject, before)) = &mut self.witness {
                let mut subject = subject.clone();
                subject.effect = Some(self.id);
                match rig_core::observe::Action::patched(before, kind) {
                    Ok(action) => witness.emit(
                        subject,
                        rig_core::observe::Stage::Handler,
                        rig_core::observe::Emitter::named(layer),
                        action,
                    ),
                    Err(error) => log::warn!(
                        target: "rig_ecs::bus",
                        "a patched effect kind did not serialize for the witness: {error}"
                    ),
                }
                // A later patch is observed against what was just served.
                *before = kind.clone();
            }
        }
    }
}

/// State needed to record cancellation and any output published before it.
pub type CancellationView = (
    &'static Issued,
    Option<&'static EffectOutcome>,
    Option<&'static super::effect::Publishing>,
    Option<&'static super::effect::ToolOutputs>,
    Option<&'static Observed>,
);

/// An in-flight effect losing `InFlight` without an outcome — a despawn,
/// its own or an ancestor's — is a cancelled dispatch: the record says so,
/// as it does when a consumer drops its `Pending` on rig-bus.
pub fn record_cancelled(
    removed: On<Remove, InFlight>,
    effects: Query<CancellationView>,
    recording: Option<Res<Recording>>,
    batch: Res<DeliveryBatch>,
) {
    let Some(recording) = recording else {
        return;
    };
    if let Ok((Issued(id), None, publishing, outputs, observed)) =
        effects.get(removed.event().entity)
    {
        let original = observed.and_then(|observed| observed.0.take_outcome());
        if observed.is_some_and(|observed| observed.0.is_discarded()) {
            return;
        }
        let output = publishing
            .and_then(|published| published.0.result_context())
            .or_else(|| outputs.map(|outputs| outputs.0.result_context()));
        if let Some(output) = output {
            recording.tool_output(*id, output);
        }
        if original.as_ref().is_some_and(|answer| {
            !answer
                .as_ref()
                .is_err_and(|error| error.kind == rig_core::error::ErrorKind::Cancelled)
        }) {
            recording.delivery(batch.0, *id, rig_core::effect::DeliveryKind::Cancelled);
        }
        recording.resolve(*id, original.unwrap_or_else(|| Err(cancelled())));
    }
}

/// An in-flight effect losing `InFlight` without an outcome, seen by the
/// witness: a cancelled dispatch, whether or not a record is kept.
pub fn witness_cancelled(
    removed: On<Remove, InFlight>,
    effects: Query<(Has<EffectOutcome>, Option<&Observed>), With<Issued>>,
    subjects: super::witness::Subjects,
    witness: Option<Res<super::witness::Witnessing>>,
    mut timings: Query<&mut super::witness::HandlerTimer>,
) {
    let entity = removed.event().entity;
    let timing = timings
        .get_mut(entity)
        .ok()
        .and_then(|mut timer| timer.finish(false));
    let Some(witness) = witness else {
        return;
    };
    let Ok((answered, observed)) = effects.get(entity) else {
        return;
    };
    if answered || observed.is_some_and(|observed| observed.0.is_discarded()) {
        return;
    }
    let mut observation = rig_core::observe::Observation::new(
        subjects.of(entity),
        rig_core::observe::Stage::Collect,
        super::witness::bus_emitter(),
        rig_core::observe::Action::Cancelled {
            reason: rig_core::observe::Reason::from_report(&cancelled()),
        },
    );
    observation.handler_timing = timing;
    witness.observe(observation);
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

#[cfg(all(test, feature = "replay"))]
mod tests;
