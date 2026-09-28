//! Status-based credential verification.
//!
//! ```
//! use rig_core::operation::{Verify, VerifyDecoder};
//! use rig_core::wire::Decoder;
//!
//! fn decodes<'id, D: Decoder<'id, Verify>>(_: D) {}
//! decodes(VerifyDecoder);
//! ```

use super::Whole;
use crate::error::ProviderError;
use crate::wire::{Call, Decoder, Flow, Free, Operation, Out, WireEvent, WireFrame};

/// Checks credentials using response status. HTTP 401/403 indicate invalid
/// authentication; status-only decoders do not interpret the response body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Verify;

impl Operation for Verify {
    type Request = ();
    type Event = std::convert::Infallible;
    type End = ();
    type Response = ();
    type Fold = Whole<Self>;
    type Emit = Free;

    fn fold(_request: &Self::Request, _call: &mut Call<'_>) -> Self::Fold {
        Whole::new()
    }
}

/// Accepts any body, including an empty one, after driver status validation.
/// Endpoints requiring body validation must use another decoder.
#[derive(Debug, Default)]
pub struct VerifyDecoder;

impl<'id> Decoder<'id, Verify> for VerifyDecoder {
    type Event = ();

    fn classify(&self, _frame: WireFrame) -> WireEvent<Self::Event> {
        WireEvent::Known(())
    }

    fn decode(&mut self, _event: (), out: Out<'id, Verify>) -> Result<Flow, ProviderError> {
        Ok(out.end(()))
    }

    /// A 2xx with an empty body still verifies: the driver only reaches EOF
    /// when nothing framed, and the status already said yes.
    fn eof(&mut self, out: Out<'id, Verify>) -> Result<Flow, ProviderError> {
        Ok(out.end(()))
    }
}
