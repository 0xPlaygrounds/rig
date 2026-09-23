//! Status-based credential verification.
//!
//! ```
//! use rig_core::{operation::Verify, wire::Operation};
//!
//! assert_eq!(Verify::NAME, "verify");
//! ```

use super::{One, Take};
use crate::wire::{Decoder, Operation, Output, Sink, WireEvent, WireFrame};

/// Checks credentials using response status. HTTP 401/403 indicate invalid
/// authentication; status-only decoders do not interpret the response body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Verify;

impl Operation for Verify {
    type Request = ();
    type Event = ();
    type Response = ();
    type Capabilities = ();
    type Output = One<Self>;
    type Fold = Take<Self>;
    type Telemetry = ();

    const NAME: &'static str = "verify";

    fn is_terminal(_event: &Self::Event) -> bool {
        true
    }

    fn telemetry(_streaming: bool) -> Self::Telemetry {}
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

    fn interpret(&mut self, _event: Self::Event, out: &mut Output<Verify>) {
        out.push(Ok(()));
    }

    /// A 2xx with an empty body still verifies: the driver only reaches
    /// `finish` when nothing framed, and the status already said yes.
    fn finish(&mut self, out: &mut Output<Verify>) {
        if out.items().is_empty() {
            out.push(Ok(()));
        }
    }
}
