//! Operations that provider wires execute.
//!
//! An operation defines its input request, output event stream, folded response,
//! and error type.

#[cfg(feature = "audio")]
pub mod audio_generation;
pub mod completion;
pub mod embedding;
pub mod image_embedding;
#[cfg(feature = "image")]
pub mod image_generation;
pub mod model_listing;
pub mod rerank;
pub mod transcription;
pub mod verify;

#[cfg(feature = "audio")]
pub use audio_generation::AudioGeneration;
pub use completion::{Completion, CompletionFold};
pub use embedding::Embedding;
pub use image_embedding::ImageEmbedding;
#[cfg(feature = "image")]
pub use image_generation::ImageGeneration;
pub use model_listing::{ModelListing, ModelListingFold};
pub use rerank::Rerank;
pub use transcription::Transcription;
pub use verify::Verify;
/// An operation: what goes in, what comes out event by event, and how events fold to a response.
pub trait Operation: 'static {
    type Request: 'static;
    type Event: 'static;
    type Response: 'static;
    type Error: 'static;
    type Capabilities: Default + 'static;
    type Fold: Fold<Self::Event, Response = Self::Response, Error = Self::Error> + Default;
}

/// Accumulator folding events into a final response.
pub trait Fold<Event>: Default {
    type Response;
    type Error;

    fn fold(&mut self, event: Event);
    fn finish(self) -> Result<Self::Response, Self::Error>;
}

/// A fold for unary operations that takes a single event.
#[derive(Debug)]
pub struct TakeOne<T, E> {
    item: Option<T>,
    _phantom: std::marker::PhantomData<E>,
}

impl<T, E> Default for TakeOne<T, E> {
    fn default() -> Self {
        Self {
            item: None,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, E: From<crate::http_client::Error>> Fold<T> for TakeOne<T, E> {
    type Response = T;
    type Error = E;

    fn fold(&mut self, event: T) {
        self.item = Some(event);
    }

    fn finish(self) -> Result<T, E> {
        self.item
            .ok_or_else(|| E::from(crate::http_client::Error::StreamEnded))
    }
}
