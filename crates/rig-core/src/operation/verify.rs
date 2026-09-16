//! The verification operation: a request whose only answer is its status.

use super::{One, Take};
use crate::client::VerifyError;
use crate::wire::Operation;

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
