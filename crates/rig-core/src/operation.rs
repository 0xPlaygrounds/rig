//! The operations a [`Wire`](crate::wire::Wire) can perform.
//!
//! One unit struct per operation, each declaring what goes in, what comes
//! out event by event, and how those events fold into one response. A
//! provider never implements an operation — it picks one.
//!
//! [`Completion`] is the only operation whose replies stream; the rest are
//! unary, so their `Event` *is* their `Response` and the fold takes the one
//! event ([`Take`]). [`ModelListing`] and [`ContextCache`] are unary per
//! page and fold pages.

use crate::wire::{Fold, Operation, Reply, Sink, WireError};

mod cached_content;
mod completion;
mod listing;
mod modality;
mod verify;

pub use cached_content::{CachedContentFold, ContextCache};
pub use completion::{AdapterOutput, Completion, CompletionEvent, CompletionFold};
pub use listing::{ModelListing, ModelListingFold};
#[cfg(feature = "audio")]
pub use modality::AudioGeneration;
#[cfg(feature = "image")]
pub use modality::ImageGeneration;
pub use modality::{
    Embedding, EmbeddingCapabilities, ImageEmbedding, Rerank, RerankRequest, Transcription,
};
pub use verify::Verify;

/// The sink of an operation whose reply is one event.
pub struct One<Op: Operation> {
    items: Vec<Result<Op::Event, Op::Error>>,
}

impl<Op: Operation> Default for One<Op> {
    fn default() -> Self {
        Self { items: Vec::new() }
    }
}

impl<Op: Operation> Sink<Op> for One<Op> {
    type Laws = ();

    fn push(&mut self, item: Result<Op::Event, Op::Error>) {
        self.items.push(item);
    }

    fn drain(&mut self) -> std::vec::Drain<'_, Result<Op::Event, Op::Error>> {
        self.items.drain(..)
    }

    fn items(&self) -> &[Result<Op::Event, Op::Error>] {
        &self.items
    }
}

/// The fold of an operation whose reply is one event: take it.
///
/// A second event is a provider defect and the first one latches, matching
/// the streaming rule for a repeated terminal.
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
    fn absorb(&mut self, event: Op::Event) -> Result<(), Op::Error> {
        if self.value.is_none() {
            self.value = Some(event);
        }
        Ok(())
    }

    fn finish(self, reply: Reply) -> Result<Op::Response, Op::Error> {
        let mut response: Op::Response = self
            .value
            .ok_or_else(|| Op::Error::decode(format!("{} reply carried no payload", Op::NAME)))?
            .into();
        Op::stamp_reply(&mut response, reply);
        Ok(response)
    }
}
