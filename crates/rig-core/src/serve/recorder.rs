//! Runtime-independent recording of handler dispatches.
//!
//! ```
//! use rig_core::serve::DispatchContext;
//!
//! let origin = DispatchContext::default();
//! assert!(origin.parent.is_none());
//! ```

use std::sync::Arc;

use crate::{
    effect::{Answered, EffectId, EffectKind, HandlerDescriptor, HandlerKey},
    error::ErrorReport,
    streaming::{Item, StreamEvent},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

/// Recorded parent dispatch, stable program scope identifier and purpose,
/// without live runtime handles.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DispatchContext {
    /// The dispatch this one was made from, if a handler made it.
    pub parent: Option<EffectId>,
    /// The scope of the program that made it, if its dispatcher was scoped.
    pub scope: Option<std::sync::Arc<str>>,
    /// What it was made for, such as `"turn"` or `"summary"`, if its
    /// dispatcher said.
    pub purpose: Option<std::borrow::Cow<'static, str>>,
}

/// What a driver tells about the dispatches it serves. A driver calls
/// [`handlers`](Self::handlers) once when recording starts,
/// [`begin`](Self::begin) as each dispatch is handed to its handler,
/// [`event`](Self::event) for every streamed event, kept verbatim when
/// [`keep_events`](Self::keep_events) says so, and
/// [`resolve`](Self::resolve) when the outcome is known. A recorder is
/// shared between the driver and its owner, so every method takes `&self`;
/// it rides in the dispatch observer and uses the platform compatibility bounds.
pub trait Recorder: WasmCompatSend + WasmCompatSync + 'static {
    /// Handlers the driver serves: those registered when recording started,
    /// then each one installed later, as it is installed. A key described
    /// again is the same handler re-registered; the latest description
    /// stands.
    fn handlers(&self, handlers: Vec<HandlerDescriptor>);
    /// A dispatch begins: its id, the key it was routed to, the effect, and
    /// where it came from (its parent and scope).
    fn begin(&self, id: EffectId, key: HandlerKey, kind: EffectKind, origin: DispatchContext);
    /// Removes a begun dispatch decided by a layer before handler execution.
    /// Replay reruns layer decisions rather than recording them as handler outcomes.
    fn discard(&self, id: EffectId);
    /// Replaces the recorded request with a same-family layer patch, so it
    /// reflects the request served by the innermost handler.
    fn patch(&self, id: EffectId, kind: EffectKind);
    /// Whether streamed items are wanted verbatim ([`Self::event`]).
    fn keep_events(&self) -> bool;
    /// One streamed item of `id`, kept verbatim only when
    /// [`keep_events`](Self::keep_events) says so.
    fn event(&self, id: EffectId, item: &Item<StreamEvent>);
    /// An error item at its original position in a kept stream. Unlike the
    /// folded outcome, this includes errors after an earlier terminal item.
    fn stream_error(&self, _id: EffectId, _error: &ErrorReport) {}
    /// Who the streamed reply of `id` is from, before its first item, when
    /// [`keep_events`](Self::keep_events) says so.
    fn origin(&self, id: EffectId, origin: &crate::message::Origin);
    /// Explicitly published output, a tool's or a layer's facts, delivered
    /// before `resolve`. A driver snapshots it without consuming the
    /// caller's published context.
    fn tool_output(&self, id: EffectId, output: crate::tool::ToolResultContext);
    /// The outcome of `id`.
    fn resolve(&self, id: EffectId, outcome: Result<Answered, ErrorReport>);
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
    fn outcome(&mut self, outcome: Result<Answered, ErrorReport>) {
        if let Some(output) = self
            .published
            .as_ref()
            .and_then(|published| published.result_context())
        {
            self.recorder.tool_output(self.id, output);
        }
        self.recorder.resolve(self.id, outcome);
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
