//! What a driver tells about the dispatches it serves: the seam a log
//! recorder implements. Beside the sink's taps because it is the same kind
//! of thing — the handler side's view of a dispatch's life — and so that a
//! recorder needs no runtime crate: `rig-effect-log` implements it over
//! rig-core alone, and any driver (the bus's, an ECS schedule's) feeds one.

use crate::{
    effect::{EffectId, EffectKind, HandlerDescriptor, HandlerKey, Outcome},
    error::ErrorReport,
    streaming::StreamEvent,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

/// Where a dispatch came from, as data: the dispatch it was made from, if
/// a handler made it, and the scope of the program that made it — a
/// stable serde id of the dispatching run or agent (never a runtime
/// handle), stamped by a scoped dispatcher. Both ride the record.
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
/// it rides in the sink's taps, which are `Send + Sync` on every target, so
/// a recorder is too.
pub trait Recorder: WasmCompatSend + WasmCompatSync + 'static {
    /// Handlers the driver serves: those registered when recording started,
    /// then each one installed later, as it is installed. A key described
    /// again is the same handler re-registered; the latest description
    /// stands.
    fn handlers(&self, handlers: Vec<HandlerDescriptor>);
    /// A dispatch begins: its id, the key it was routed to, the effect, and
    /// where it came from (its parent and scope).
    fn begin(&self, id: EffectId, key: HandlerKey, kind: EffectKind, origin: Origin);
    /// A dispatch that began is not a record after all: a layer decided it
    /// before any handler served it (a denial, a patch of the wrong
    /// family). Decisions are program, never record — a replay re-makes
    /// them — so the recorder forgets the slot `begin` opened.
    fn discard(&self, id: EffectId);
    /// A layer serves `kind` in place of the effect that began — a patch of
    /// the same family — so the record's request is what the innermost
    /// handler served, never what a layer saw first.
    fn patch(&self, id: EffectId, kind: EffectKind);
    /// Whether streamed events are wanted verbatim ([`Self::event`]).
    fn keep_events(&self) -> bool;
    /// One streamed event of `id`.
    fn event(&self, id: EffectId, event: &StreamEvent);
    /// The outcome of `id`.
    fn resolve(&self, id: EffectId, outcome: Result<Outcome, ErrorReport>);
}
