//! The verification operation: a request whose only answer is its status.

use super::{One, Take};
use crate::client::VerifyError;
use crate::wire::{Decoder, Operation, Output, Sink, WireEvent, WireFrame};

/// Checking that a provider accepts the configured credentials.
///
/// The reply body is not read for meaning: a success status is the answer,
/// and the rejection classification (401/403 as invalid authentication)
/// lives in [`VerifyError`]'s [`WireError`](crate::wire::WireError) impl, so
/// no wire restates it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Verify;

impl Operation for Verify {
    type Request = ();
    type Event = ();
    type Response = ();
    type Error = VerifyError;
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

/// The decoder of every status-only credential check: a success status is
/// the whole answer, and the body is not read for meaning.
///
/// Shared by the wires whose verification endpoint answers with an
/// arbitrary document (Anthropic and every OpenAI-shaped dialect); a wire
/// that validates the body's shape names a decoder of its own.
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
