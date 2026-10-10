//! Runtime-independent recording of handler dispatches and consumer delivery.
//!
//! ```
//! use rig_core::serve::Origin;
//!
//! let origin = Origin::default();
//! assert!(origin.parent.is_none());
//! ```

use std::sync::Arc;

use crate::{
    effect::{EffectId, EffectKind, HandlerDescriptor, HandlerKey, Outcome},
    error::ErrorReport,
    streaming::{Item, StreamEvent},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

/// Recorded parent dispatch and stable program scope identifier, without live
/// runtime handles.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Origin {
    /// The dispatch this one was made from, if a handler made it.
    pub parent: Option<EffectId>,
    /// The scope of the program that made it, if its dispatcher was scoped.
    pub scope: Option<std::sync::Arc<str>>,
}

/// What a driver tells about the dispatches it serves. A driver calls
/// [`handlers`](Self::handlers) once when recording starts,
/// [`begin`](Self::begin) as each dispatch is handed to its handler,
/// [`event`](Self::event) for every streamed event when
/// [`keep_events`](Self::keep_events) says so, and
/// [`resolve`](Self::resolve) when the outcome is known. A recorder is
/// shared between the driver and its owner, so every method takes `&self`;
/// it rides in the dispatch observer and uses the platform compatibility bounds.
pub trait Recorder: WasmCompatSend + WasmCompatSync + 'static {
    /// Optional provider-observation context for a dispatch after [`Self::begin`].
    /// Keep this runtime-only; observations do not belong in the effect log.
    /// Return the same logical context when asked again for the same dispatch.
    /// An explicit invocation context on the dispatch takes precedence.
    fn adapter_context(&self, _id: EffectId) -> Option<crate::observe::AdapterContext> {
        None
    }
    /// Declare that this runtime records consumer-visible delivery boundaries.
    /// Handler-only recorders may ignore this optional scheduling metadata.
    fn begin_delivery_tracking(&self) {}
    /// A transition at the consumer boundary, after collection rather than
    /// when a handler produces a value. Order within a batch is call order.
    fn delivery(&self, _delivery: crate::effect::Delivery) {}
    /// This recording includes visibility outside the runtime's supported
    /// observation boundary and cannot prove policy-visible replay.
    fn unsupported_delivery(&self, _reason: &str) {}
    /// Handlers the driver serves: those registered when recording started,
    /// then each one installed later, as it is installed. A key described
    /// again is the same handler re-registered; the latest description
    /// stands.
    fn handlers(&self, handlers: Vec<HandlerDescriptor>);
    /// A dispatch begins: its id, the key it was routed to, the effect, and
    /// where it came from (its parent and scope).
    fn begin(&self, id: EffectId, key: HandlerKey, kind: EffectKind, origin: Origin);
    /// Removes a begun dispatch decided by a layer before handler execution.
    /// Replay reruns layer decisions rather than recording them as handler outcomes.
    fn discard(&self, id: EffectId);
    /// Replaces the recorded request with a same-family layer patch, so it
    /// reflects the request served by the innermost handler.
    fn patch(&self, id: EffectId, kind: EffectKind);
    /// Whether streamed items are wanted verbatim ([`Self::event`]).
    fn keep_events(&self) -> bool;
    /// One streamed item of `id`.
    fn event(&self, id: EffectId, item: &Item<StreamEvent>);
    /// An error item at its original position in a kept stream. Unlike the
    /// folded outcome, this includes errors after an earlier terminal item.
    fn stream_error(&self, _id: EffectId, _error: &ErrorReport) {}
    /// Who the streamed reply of `id` is from, before its first item, when
    /// [`keep_events`](Self::keep_events) says so.
    fn origin(&self, id: EffectId, origin: &crate::message::Origin);
    /// Explicitly published tool output, delivered before `resolve`. A driver
    /// snapshots it without consuming the caller's published context.
    fn tool_output(&self, id: EffectId, output: crate::tool::ToolResultContext);
    /// The outcome of `id`.
    fn resolve(&self, id: EffectId, outcome: Result<Outcome, ErrorReport>);
}

/// The [`Observe`](super::Observe) a recording driver installs on each
/// dispatch ([`Dispatch::recorded_by`](super::Dispatch::recorded_by)): it
/// tells `recorder` everything about the dispatch `id`, and the tool output
/// its handler published, if any, before the outcome.
pub(crate) struct RecordingObserver {
    pub(crate) recorder: Arc<dyn Recorder + Send + Sync>,
    pub(crate) id: EffectId,
    pub(crate) published: Option<Arc<crate::tool::PublishedContext>>,
}

impl super::Observe for RecordingObserver {
    fn adapter_context(&self) -> Option<crate::observe::AdapterContext> {
        self.recorder.adapter_context(self.id)
    }

    fn outcome(&mut self, outcome: &Result<Outcome, ErrorReport>) {
        if let Some(output) = self
            .published
            .as_ref()
            .and_then(|published| published.result_context())
        {
            self.recorder.tool_output(self.id, output);
        }
        self.recorder.resolve(self.id, outcome.clone());
    }

    fn keep_events(&self) -> bool {
        self.recorder.keep_events()
    }

    fn event(&mut self, item: &Item<StreamEvent>) {
        self.recorder.event(self.id, item);
    }

    fn stream_error(&mut self, error: &ErrorReport) {
        self.recorder.stream_error(self.id, error);
    }

    fn origin(&mut self, origin: &crate::message::Origin) {
        self.recorder.origin(self.id, origin);
    }

    /// A layer decided the dispatch before any handler saw it: the record
    /// is dropped, since replay reruns the layers.
    fn discard(&mut self, _layer: &str) {
        self.recorder.discard(self.id);
    }

    fn patch(&mut self, kind: &EffectKind) {
        self.recorder.patch(self.id, kind.clone());
    }
}

/// The record of an effect no handler serves, such as a tool call a person
/// or another program answers: begun when it opens, resolved by
/// [`Self::settle`], and recorded as cancelled when dropped unsettled.
/// Without a recorder it only carries the effect's id.
///
/// ```
/// use std::sync::Arc;
/// use rig_core::effect::{EffectId, EffectKind, tool_key};
/// use rig_core::serve::{OpenRecord, Origin, Recorder};
/// # fn demo(recorder: Arc<dyn Recorder + Send + Sync>) {
/// let kind = EffectKind::ToolCall { name: "ask".into(), args: "{}".into() };
/// let mut open = OpenRecord::begin(Some(recorder), EffectId::from_raw(1), tool_key("ask"), kind, Origin::default());
/// open.settle(Err(rig_core::serve::cancelled()));
/// # }
/// ```
pub struct OpenRecord {
    recorder: Option<Arc<dyn Recorder + Send + Sync>>,
    id: EffectId,
}

impl OpenRecord {
    /// Records the start of the effect `id`, `kind` routed to `key`, with
    /// `recorder`, if any.
    pub fn begin(
        recorder: Option<Arc<dyn Recorder + Send + Sync>>,
        id: EffectId,
        key: HandlerKey,
        kind: EffectKind,
        origin: Origin,
    ) -> Self {
        if let Some(recorder) = &recorder {
            recorder.begin(id, key, kind, origin);
        }
        Self { recorder, id }
    }

    /// The effect's id.
    pub fn id(&self) -> EffectId {
        self.id
    }

    /// Records `outcome` as the effect's outcome; only the first counts.
    pub fn settle(&mut self, outcome: Result<Outcome, ErrorReport>) {
        if let Some(recorder) = self.recorder.take() {
            recorder.resolve(self.id, outcome);
        }
    }
}

impl Drop for OpenRecord {
    fn drop(&mut self) {
        self.settle(Err(super::cancelled()));
    }
}

/// `work`, the task that drives the dispatch `id`, with a panic in it (in
/// the handler or in its reply's stream) turned into an internal error that
/// `recorder`, if any, also records as the dispatch's outcome. Unwinding
/// drops the dispatch's observer, which records a cancellation; this
/// replaces it.
pub fn catch_panics<T>(
    recorder: Option<Arc<dyn Recorder + Send + Sync>>,
    id: EffectId,
    work: impl Future<Output = Result<T, ErrorReport>>,
) -> impl Future<Output = Result<T, ErrorReport>> {
    use futures::FutureExt;
    async move {
        std::panic::AssertUnwindSafe(work)
            .catch_unwind()
            .await
            .unwrap_or_else(|panic| {
                let message = panic
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
                    .unwrap_or("no message");
                let report = ErrorReport::new(
                    crate::error::ErrorKind::Internal,
                    format!("panicked: {message}"),
                );
                if let Some(recorder) = recorder {
                    recorder.resolve(id, Err(report.clone()));
                }
                Err(report)
            })
    }
}
