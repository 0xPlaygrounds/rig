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
use crate::response::{Reported, Response};
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

/// Retains the first reply and ignores later ones. Finishing without a
/// reply returns a decode error; otherwise the driver's metadata completes
/// the reply into the response.
pub struct Take<T> {
    value: Option<Reported<T>>,
}

impl<T> Default for Take<T> {
    fn default() -> Self {
        Self { value: None }
    }
}

impl<Op, T> Fold<Op> for Take<T>
where
    Op: Operation<Event = Reported<T>, Response = Response<T>>,
{
    fn absorb(&mut self, event: Reported<T>) -> Result<(), ProviderError> {
        if self.value.is_none() {
            self.value = Some(event);
        }
        Ok(())
    }

    fn finish(self, reply: Reply) -> Result<Response<T>, ProviderError> {
        let Reported {
            output,
            model,
            response_id,
            usage,
        } = self.value.ok_or_else(|| {
            ProviderError::Response(format!("{} reply carried no payload", Op::NAME))
        })?;
        Ok(Response {
            output,
            meta: reply.meta(model, response_id, usage),
        })
    }
}
