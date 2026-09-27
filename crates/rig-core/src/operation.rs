//! The operations a [`Wire`](crate::wire::Wire) can perform.
//!
//! Each operation declares request, event and response types and the fold
//! one reply's events go through. [`Completion`] streams events; the others
//! fold buffered replies. Listings follow their cursors above the driver:
//! each page is one call.
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
mod completion;
mod listing;
mod modality;
mod verify;

pub use cached_content::{CachedContentFold, ContextCache};
pub(crate) use completion::Canonical;
pub use completion::{AdapterOutput, Completion, CompletionFold, CompletionReply, ImagePart};
pub use listing::{ModelListing, ModelPage};
#[cfg(feature = "audio")]
pub use modality::AudioGeneration;
#[cfg(feature = "image")]
pub use modality::ImageGeneration;
pub use modality::{Embedding, ImageEmbedding, Rerank, RerankRequest, Transcription};
pub use verify::{Verify, VerifyDecoder};

/// Retains the first event as the response and ignores later events.
/// Finishing without an event returns a decode error; otherwise the stamp
/// writes what the driver learned about the reply onto the response.
///
/// The fold sees events by reference, so it clones the one event it keeps,
/// once per reply.
pub struct Take<Op: Operation> {
    value: Option<Op::Event>,
    stamp: fn(&mut Op::Response, &Reply),
}

impl<Op: Operation> Take<Op> {
    /// A fold that writes what the driver learned about the reply onto the
    /// response with `stamp`.
    pub fn stamping(stamp: fn(&mut Op::Response, &Reply)) -> Self {
        Self { value: None, stamp }
    }
}

/// A fold that stamps nothing.
impl<Op: Operation> Default for Take<Op> {
    fn default() -> Self {
        Self::stamping(|_, _| {})
    }
}

impl<Op> Fold<Op> for Take<Op>
where
    Op: Operation,
    Op::Event: Clone + Into<Op::Response>,
{
    fn absorb(&mut self, event: &Op::Event) -> Result<(), ProviderError> {
        if self.value.is_none() {
            self.value = Some(event.clone());
        }
        Ok(())
    }

    fn finish(self, reply: Reply) -> Result<Op::Response, ProviderError> {
        let mut response: Op::Response = self
            .value
            .ok_or_else(|| ProviderError::Response("the reply carried no payload".to_owned()))?
            .into();
        (self.stamp)(&mut response, &reply);
        Ok(response)
    }
}
