//! The reply document a reply's events rebuild. Every [`Wire`](super::Wire)
//! names one [`Reassemble`] type. The driver hands it each event its decoder
//! classifies as known, of a reply whose transport reported no whole
//! document, before the decoder decodes the event, and records what it
//! finishes with as the reply's `raw`. A
//! completion decoder cannot write `raw` itself, so a streamed reply's
//! `raw` has one owner per wire. A wire names a reassembler only for an
//! operation it [`Serves`], and [`Unreassembled`], which records nothing,
//! serves no completion.
//!
//! ```
//! use rig_core::wire::document::{Reassemble, Unreassembled};
//!
//! let mut document = Unreassembled;
//! Reassemble::<String>::absorb(&mut document, &"event".to_owned());
//! assert!(Reassemble::<String>::finish(document).is_null());
//! ```

use crate::wasm_compat::WasmCompatSend;
use crate::wire::{Free, Operation};

/// Rebuilds one reply's provider document from its decoder's events.
///
/// A fresh value per reply. `absorb` sees every known event of the reply,
/// in arrival order, before the decoder decodes it; unknown and corrupt
/// frames never reach it. `finish` runs once, when the reply ends, fails or
/// is cut short; a reply that did not end yields the document so far.
pub trait Reassemble<Event>: Default + WasmCompatSend + 'static {
    /// Absorb one known event of the reply.
    fn absorb(&mut self, event: &Event);

    /// The document the absorbed events add up to. `Null` records none.
    fn finish(self) -> serde_json::Value;
}

/// Keep `value` at `key` of a document being rebuilt: a non-null value
/// replaces, and `null` only fills an absent field. A value equal to the
/// one held, as most fields restate in every event, is not copied again.
pub(crate) fn keep(
    map: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    value: &serde_json::Value,
) {
    match map.get_mut(key) {
        Some(held) if !value.is_null() && held != value => *held = value.clone(),
        Some(_) => {}
        None => {
            map.insert(key.to_owned(), value.clone());
        }
    }
}

/// The value at `key`, made an empty `kind` (a list or an object) when it is
/// absent or of another type.
pub(crate) fn slot<'a, T: Default>(
    map: &'a mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    kind: fn(T) -> serde_json::Value,
) -> &'a mut serde_json::Value {
    let slot = map.entry(key).or_insert(serde_json::Value::Null);
    if std::mem::discriminant(slot) != std::mem::discriminant(&kind(T::default())) {
        *slot = kind(T::default());
    }
    slot
}

/// The reassembler of a wire whose decoders record `raw` themselves, which
/// only an operation with [`Free`] events can: it records
/// nothing.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Unreassembled;

impl<Event> Reassemble<Event> for Unreassembled {
    fn absorb(&mut self, _event: &Event) {}

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

/// The document `frames` rebuild: each frame `decoder` classifies as known
/// is handed to `reassembler`, as the driver hands it.
#[cfg(test)]
pub(crate) fn rebuild<'id, Op, F, D, R>(
    decoder: &D,
    mut reassembler: R,
    frames: impl IntoIterator<Item = F>,
) -> serde_json::Value
where
    Op: Operation,
    D: super::Decoder<'id, Op, F>,
    R: Reassemble<D::Event>,
{
    for frame in frames {
        if let super::WireEvent::Known(event) = decoder.classify(frame) {
            reassembler.absorb(&event);
        }
    }
    reassembler.finish()
}
