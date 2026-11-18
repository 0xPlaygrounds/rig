//! The context window a model's server serves it with.

use super::Whole;
use crate::wire::{Call, Free, Operation};

/// Asks a local server for the context window it serves a model with: the
/// most tokens one request and its reply may hold there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ContextWindow;

impl Operation for ContextWindow {
    type Request = ();
    type Event = std::convert::Infallible;
    type End = u32;
    type Response = u32;
    type Fold = Whole<Self>;
    type Emit = Free;

    fn fold(_request: &Self::Request, _call: &mut Call<'_>) -> Self::Fold {
        Whole::new()
    }
}
