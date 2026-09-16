use crate::client::verify::VerifyError;

use super::{Operation, TakeOne};

/// The provider verification operation (status-only check).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Verify;

impl Operation for Verify {
    type Request = ();
    type Event = ();
    type Response = ();
    type Error = VerifyError;
    type Capabilities = ();
    type Fold = TakeOne<(), VerifyError>;
}
