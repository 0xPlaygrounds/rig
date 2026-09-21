//! Effect recording and cancellation observers for the world bus.
//!
//! ```
//! use rig_ecs::bus::record::DeliveryBatch;
//! let batch = DeliveryBatch::default();
//! assert_eq!(batch.0, 0);
//! ```

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

    /// Begin delivery tracking, describe currently bound handlers in key order,
    /// and install `recorder` as the world's recording resource.
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

/// The current delivery batch. Advances once per schedule pass.
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

/// Shared in-flight observation state containing the original handler answer
/// and whether a layer discarded dispatch. Not serialized; layer verdicts do not
/// replace the recorded answer.
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

/// Records original answers, stream items, and layer patches or discards for a
/// task-served dispatch. Once observation closes, worker callbacks cannot mutate
/// the recording.
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
    /// The world's witness with the current dispatch subject.
    pub witness: Option<(super::witness::Witnessing, rig_core::observe::Subject)>,
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
        let (witness, subject) = self.witness.as_ref()?;
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
            if let Some((witness, subject)) = &self.witness {
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

    fn patch(&mut self, kind: &EffectKind) {
        let state = self.observed.lock();
        if !state.closed
            && let Some(recording) = &self.recording
        {
            recording.patch(self.id, kind.clone());
        }
    }
}

/// Close recording when an effect leaves flight before settlement.
/// Preserve any original handler answer and tool output; otherwise record
/// cancellation. An observed answer cancelled before delivery gets a cancellation
/// boundary, not an invented outcome delivery. Discarded dispatches stay unrecorded.
pub fn record_cancelled(
    removed: On<Remove, InFlight>,
    effects: Query<(
        &Issued,
        Option<&EffectOutcome>,
        Has<super::collect::CollectedOutcome>,
        Option<&super::effect::Publishing>,
        Option<&super::effect::ToolOutputs>,
        Option<&Observed>,
    )>,
    recording: Option<Res<Recording>>,
    batch: Res<DeliveryBatch>,
) {
    let Some(recording) = recording else {
        return;
    };
    let Ok((Issued(id), outcome, collected, publishing, outputs, observed)) =
        effects.get(removed.event().entity)
    else {
        return;
    };
    if observed.is_some_and(|observed| observed.0.is_discarded()) {
        return;
    }
    match outcome {
        // Landed but not settled: `settle` retires `CollectedOutcome` before
        // it retires `InFlight`, so its presence here means settlement never
        // ran (the entity is being despawned with its answer on it).
        Some(EffectOutcome(landed)) if collected => {
            let original = observed.and_then(|observed| observed.0.take_outcome());
            if observed.is_none()
                && let Some(outputs) = outputs
            {
                recording.tool_output(*id, outputs.0.result_context());
            }
            recording.resolve(*id, original.unwrap_or_else(|| landed.clone()));
        }
        Some(_) => {}
        None => {
            let original = observed.and_then(|observed| observed.0.take_outcome());
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
}

/// An in-flight effect losing `InFlight` without an outcome, seen by the
/// witness: a cancelled dispatch, whether or not a record is kept.
pub fn witness_cancelled(
    removed: On<Remove, InFlight>,
    effects: Query<(Has<EffectOutcome>, Option<&Observed>), With<Issued>>,
    subjects: super::witness::Subjects,
    witness: Option<Res<super::witness::Witnessing>>,
) {
    let entity = removed.event().entity;
    let Some(witness) = witness else {
        return;
    };
    let Ok((answered, observed)) = effects.get(entity) else {
        return;
    };
    if answered || observed.is_some_and(|observed| observed.0.is_discarded()) {
        return;
    }
    let observation = rig_core::observe::Observation::new(
        subjects.of(entity),
        rig_core::observe::Stage::Collect,
        super::witness::bus_emitter(),
        rig_core::observe::Action::Cancelled {
            reason: rig_core::observe::Reason::from_report(&cancelled()),
        },
    );
    witness.observe(observation);
}

/// Describe an inserted or replaced [`Bound`] component to the installed recorder.
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

#[cfg(test)]
mod tests;
