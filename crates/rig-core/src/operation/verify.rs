//! Status-based credential verification.
//!
//! ```
//! use rig_core::operation::{Verify, VerifyDecoder};
//! use rig_core::wire::Decoder;
//!
//! fn decodes<D: Decoder<Verify>>(_: D) {}
//! decodes(VerifyDecoder);
//! ```

use super::Take;
use crate::wire::{Call, Decoder, End, Operation, Out, WireEvent, WireFrame};

/// Checks credentials using response status. HTTP 401/403 indicate invalid
/// authentication; status-only decoders do not interpret the response body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Verify;

impl Operation for Verify {
    type Request = ();
    type Event = ();
    type Response = ();
    type Fold = Take<Self>;

    fn is_terminal(_event: &Self::Event) -> bool {
        true
    }

    fn fold(_request: &Self::Request, _call: &mut Call<'_>) -> Self::Fold {
        Take::default()
    }
}

/// Accepts any body, including an empty one, after driver status validation.
/// Endpoints requiring body validation must use another decoder.
#[derive(Debug, Default)]
pub struct VerifyDecoder;

impl Decoder<Verify> for VerifyDecoder {
    type Event = ();

    fn classify(&self, _frame: WireFrame) -> WireEvent<Self::Event> {
        WireEvent::Known(())
    }

    fn interpret(&mut self, _event: Self::Event, out: &mut Out<'_, Verify>) {
        out.push(Ok(()));
    }

    /// A 2xx with an empty body still verifies: the driver only reaches EOF
    /// when nothing framed, and the status already said yes.
    fn end(&mut self, out: &mut Out<'_, Verify>, end: End) {
        if end == End::Eof {
            out.push(Ok(()));
        }
    }
}
