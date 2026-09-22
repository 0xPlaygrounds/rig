//! Runtime-independent recording of handler dispatches and consumer delivery.
//!
//! ```
//! use rig_core::serve::Origin;
//!
//! let origin = Origin::default();
//! assert!(origin.parent.is_none());
//! ```

use crate::{
    effect::{EffectId, EffectKind, HandlerDescriptor, HandlerKey, Outcome},
    error::ErrorReport,
    streaming::StreamEvent,
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
    /// Whether streamed events are wanted verbatim ([`Self::event`]).
    fn keep_events(&self) -> bool;
    /// One streamed event of `id`.
    fn event(&self, id: EffectId, event: &StreamEvent);
    /// An error item at its original position in a kept stream. Unlike the
    /// folded outcome, this includes errors after an earlier terminal item.
    fn stream_error(&self, _id: EffectId, _error: &ErrorReport) {}
    /// Explicitly published tool output, delivered before `resolve`. A driver
    /// snapshots it without consuming the caller's published context.
    fn tool_output(&self, id: EffectId, output: crate::tool::ToolResultContext);
    /// The outcome of `id`.
    fn resolve(&self, id: EffectId, outcome: Result<Outcome, ErrorReport>);
}
