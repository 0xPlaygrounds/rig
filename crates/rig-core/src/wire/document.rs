//! The reply document a reply's frames rebuild. Every [`Wire`](super::Wire)
//! names one [`Reassemble`] type. The driver hands it each frame of a reply
//! whose transport reported no whole document, before the decoder reads the
//! frame, and records what it finishes with as the reply's `raw`. A
//! completion decoder cannot write `raw` itself, so a streamed reply's
//! `raw` has one owner per wire. A wire names a reassembler only for an
//! operation it [`Serves`], and [`Unreassembled`], which records nothing,
//! serves no completion.
//!
//! ```
//! use rig_core::wire::document::{Reassemble, Unreassembled};
//!
//! let mut document = Unreassembled;
//! Reassemble::<String>::absorb(&mut document, &"frame".to_owned());
//! assert!(Reassemble::<String>::finish(document).is_null());
//! ```

use crate::wasm_compat::WasmCompatSend;
use crate::wire::{Free, Operation};

/// Rebuilds one reply's provider document from the frames it arrived in.
///
/// A fresh value per reply. `absorb` sees every frame the driver reads, in
/// arrival order, including frames the decoder classifies as unknown or
/// corrupt. `finish` runs once, when the reply ends, fails or is cut
/// short; a reply that did not end yields the document so far.
pub trait Reassemble<Frame>: Default + WasmCompatSend + 'static {
    /// Absorb one frame of the reply.
    fn absorb(&mut self, frame: &Frame);

    /// The document the absorbed frames add up to. `Null` records none.
    fn finish(self) -> serde_json::Value;
}

/// The reassembler of a wire whose decoders record `raw` themselves, which
/// only an operation with [`Free`] events can: it records
/// nothing.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Unreassembled;

impl<Frame> Reassemble<Frame> for Unreassembled {
    fn absorb(&mut self, _frame: &Frame) {}

    fn finish(self) -> serde_json::Value {
        serde_json::Value::Null
    }
}

/// The operations whose wires may name a reassembler: a wire's
/// [`Reassembler`](super::Wire::Reassembler) must serve its operation.
///
/// [`Unreassembled`] serves only an operation whose decoders record `raw`
/// themselves ([`Free`] events), so a completion wire cannot name it and
/// must name a reassembler that rebuilds its document:
///
/// ```compile_fail,E0271
/// use rig_core::operation::Completion;
/// use rig_core::wire::document::{Serves, Unreassembled};
///
/// fn completion_reassembler<R: Serves<Completion>>() {}
/// completion_reassembler::<Unreassembled>();
/// ```
pub trait Serves<Op: Operation> {}

impl<Op: Operation<Emit = Free>> Serves<Op> for Unreassembled {}
