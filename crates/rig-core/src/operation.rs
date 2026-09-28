//! The operations a [`Wire`](crate::wire::Wire) can perform.
//!
//! Each operation declares its request, event, end and response types and
//! the fold one reply goes through. [`Completion`] streams events; most
//! others answer with one whole document, folded by [`Whole`]. Listings
//! follow their cursors above the driver: each page is one call.
//!
//! ```
//! use rig_core::operation::{Completion, Embedding};
//! use rig_core::wire::Operation;
//!
//! fn streams<Op: Operation>() {}
//! streams::<Completion>();
//! streams::<Embedding>();
//! ```

use crate::error::ProviderError;
use crate::wire::{Fold, Operation, Reply};

mod cached_content;
pub(crate) mod completion;
mod listing;
mod modality;
mod verify;

pub use cached_content::{CachedContentFold, ContextCache};
pub use completion::{
    CallFragment, CallPart, Completion, Finish, IfMalformed, ReasoningPart, Seal, TextPart, Turn,
};
pub use listing::{ModelListing, ModelPage};
#[cfg(feature = "audio")]
pub use modality::AudioGeneration;
#[cfg(feature = "image")]
pub use modality::ImageGeneration;
pub use modality::{Embedding, ImageEmbedding, Rerank, RerankRequest, Transcription};
pub use verify::{Verify, VerifyDecoder};

/// The fold of an operation whose reply is one whole answer: it has no
/// events, and the provider's end is the response. The stamp writes what
/// the driver learned about the reply onto it.
///
/// ```
/// use std::convert::Infallible;
/// use rig_core::operation::Whole;
/// use rig_core::wire::{Call, Free, Operation};
///
/// struct Echo;
///
/// impl Operation for Echo {
///     type Request = String;
///     type Event = Infallible;
///     type End = String;
///     type Response = String;
///     type Fold = Whole<Self>;
///     type Emit = Free;
///
///     fn fold(_request: &String, _call: &mut Call<'_>) -> Whole<Self> {
///         Whole::<Self>::stamping(|response, reply| response.push_str(&reply.provider))
///     }
/// }
/// ```
pub struct Whole<Op: Operation> {
    stamp: fn(&mut Op::Response, &Reply),
}

impl<Op: Operation> Whole<Op> {
    /// A fold that stamps nothing.
    pub fn new() -> Self {
        Self::stamping(|_, _| {})
    }

    /// A fold that writes what the driver learned about the reply onto the
    /// response with `stamp`.
    pub fn stamping(stamp: fn(&mut Op::Response, &Reply)) -> Self {
        Self { stamp }
    }
}

impl<Op: Operation> Default for Whole<Op> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Op> Fold<Op> for Whole<Op>
where
    Op: Operation<Event = std::convert::Infallible>,
    Op::End: Into<Op::Response>,
{
    fn absorb(&mut self, event: &std::convert::Infallible) -> Result<(), ProviderError> {
        match *event {}
    }

    fn finish(self, end: Op::End, reply: Reply) -> Result<Op::Response, ProviderError> {
        let mut response = end.into();
        (self.stamp)(&mut response, &reply);
        Ok(response)
    }
}
