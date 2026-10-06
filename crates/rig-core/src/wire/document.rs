//! The reply document a reply's frames rebuild. Every [`Wire`](super::Wire)
//! names one [`Reassemble`] type. The driver hands it each frame of a reply
//! whose transport reported no whole document, before the decoder reads the
//! frame, and records what it finishes with as the reply's `raw`. A
//! completion decoder cannot write `raw` itself, so a streamed reply's
//! `raw` has one owner per wire.
//!
//! ```
//! use rig_core::wire::document::{Reassemble, Unreassembled};
//!
//! let mut document = Unreassembled;
//! Reassemble::<String>::absorb(&mut document, &"frame".to_owned());
//! assert!(Reassemble::<String>::finish(document).is_null());
//! ```

use crate::wasm_compat::WasmCompatSend;

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
/// only an operation with [`Free`](super::Free) events can: it records
/// nothing.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Unreassembled;

impl<Frame> Reassemble<Frame> for Unreassembled {
    fn absorb(&mut self, _frame: &Frame) {}

    fn finish(self) -> serde_json::Value {
        serde_json::Value::Null
    }
}
