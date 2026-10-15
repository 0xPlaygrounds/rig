//! One call to a provider's batch API. The provider-neutral surface built
//! on it is [`completion::batch`](crate::completion::batch).
//!
//! ```
//! use rig_core::operation::BatchCall;
//! use rig_core::wire::Operation;
//!
//! fn answers<Op: Operation<Response = String>>() {}
//! answers::<BatchCall<(), String>>();
//! ```

use std::marker::PhantomData;

use super::Whole;
use crate::wasm_compat::WasmCompatSend;
use crate::wire::{Call, Free, Operation};

/// One call to a provider's batch API: `Request` is the provider's call and
/// `Reply` its decoded answer, one whole document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchCall<Request, Reply>(PhantomData<fn(Request) -> Reply>);

impl<Request, Reply> Operation for BatchCall<Request, Reply>
where
    Request: WasmCompatSend + 'static,
    Reply: WasmCompatSend + 'static,
{
    type Request = Request;
    type Event = std::convert::Infallible;
    type End = Reply;
    type Response = Reply;
    type Fold = Whole<Self>;
    type Emit = Free;

    fn fold(_request: &Self::Request, _call: &mut Call<'_>) -> Self::Fold {
        Whole::new()
    }
}
