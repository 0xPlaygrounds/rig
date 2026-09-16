//! The model-listing operation: unary per page, folded across pages.

use super::One;
use crate::model::{Model, ModelList, ModelListingError};
use crate::wire::{Fold, Operation, Reply};

/// Listing the models a provider offers.
///
/// The only paged operation: a decoder that read a next-page cursor off the
/// reply returns it from [`Decoder::continuation`](crate::wire::Decoder::continuation)
/// and the driver sends it, folding every page into one list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelListing;

impl Operation for ModelListing {
    type Request = ();
    type Event = ModelList;
    type Response = ModelList;
    type Error = ModelListingError;
    type Capabilities = ();
    type Output = One<Self>;
    type Fold = ModelListingFold;
    type Telemetry = ();

    const NAME: &'static str = "model_listing";

    fn is_terminal(_event: &Self::Event) -> bool {
        true
    }

    fn telemetry(_streaming: bool) -> Self::Telemetry {}
}

/// Concatenates the pages of a model listing, in arrival order.
#[derive(Default)]
pub struct ModelListingFold {
    models: Vec<Model>,
}

impl Fold<ModelListing> for ModelListingFold {
    fn absorb(&mut self, page: ModelList) -> Result<(), ModelListingError> {
        self.models.extend(page);
        Ok(())
    }

    fn finish(self, _reply: Reply) -> Result<ModelList, ModelListingError> {
        Ok(ModelList::new(self.models))
    }
}
