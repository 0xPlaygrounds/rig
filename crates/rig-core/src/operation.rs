//! The operations a [`Wire`](crate::wire::Wire) can perform.
//!
//! Each operation declares request, event, response, and fold types.
//! [`Completion`] streams events; other operations fold buffered replies,
//! including pages for [`ModelListing`] and [`ContextCache`].
//!
//! ```
//! use rig_core::{operation::Completion, wire::Operation};
//!
//! assert_eq!(Completion::NAME, "completion");
//! ```

use crate::error::ProviderError;
use crate::wire::{Fold, Operation, Reply, Sink};

mod cached_content;
mod completion;
mod listing;
mod modality;
mod verify;

pub use cached_content::{CachedContentFold, ContextCache};
pub use completion::{AdapterOutput, Completion, CompletionFold};
pub use listing::{ModelListing, ModelListingFold};
#[cfg(feature = "audio")]
pub use modality::AudioGeneration;
#[cfg(feature = "image")]
pub use modality::ImageGeneration;
pub use modality::{
    Embedding, EmbeddingCapabilities, ImageEmbedding, Rerank, RerankRequest, Transcription,
};
pub use verify::{Verify, VerifyDecoder};

/// The sink of an operation whose reply is one event.
pub struct One<Op: Operation> {
    items: Vec<Result<Op::Event, ProviderError>>,
}

impl<Op: Operation> Default for One<Op> {
    fn default() -> Self {
        Self { items: Vec::new() }
    }
}

impl<Op: Operation> Sink<Op> for One<Op> {
    type Laws = ();

    fn push(&mut self, item: Result<Op::Event, ProviderError>) {
        self.items.push(item);
    }

    fn drain(&mut self) -> std::vec::Drain<'_, Result<Op::Event, ProviderError>> {
        self.items.drain(..)
    }

    fn items(&self) -> &[Result<Op::Event, ProviderError>] {
        &self.items
    }
}

/// Retains the first event and ignores later events. Finishing without an
/// event returns a decode error; otherwise reply metadata is stamped on the response.
pub struct Take<Op: Operation> {
    value: Option<Op::Event>,
}

impl<Op: Operation> Default for Take<Op> {
    fn default() -> Self {
        Self { value: None }
    }
}

impl<Op> Fold<Op> for Take<Op>
where
    Op: Operation,
    Op::Event: Into<Op::Response>,
{
    fn absorb(&mut self, event: Op::Event) -> Result<(), ProviderError> {
        if self.value.is_none() {
            self.value = Some(event);
        }
        Ok(())
    }

    fn finish(self, reply: Reply) -> Result<Op::Response, ProviderError> {
        let mut response: Op::Response = self
            .value
            .ok_or_else(|| {
                ProviderError::Response(format!("{} reply carried no payload", Op::NAME))
            })?
            .into();
        Op::stamp_reply(&mut response, reply);
        Ok(response)
    }
}
