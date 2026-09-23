//! Model-listing operation and ordered page aggregation.
//!
//! ```
//! use rig_core::{operation::ModelListing, wire::Operation};
//!
//! assert_eq!(ModelListing::NAME, "model_listing");
//! ```

use super::One;
use crate::error::ProviderError;
use crate::model::{Model, ModelList};
use crate::wire::{Fold, Operation, Reply};

/// Lists provider models, concatenating pages requested through
/// [`Decoder::continuation`](crate::wire::Decoder::continuation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelListing;

impl Operation for ModelListing {
    type Request = ();
    type Event = ModelList;
    type Response = ModelList;
    type Capabilities = ();
    type Output = One<Self>;
    type Fold = ModelListingFold;
    type Telemetry = ();

    const NAME: &'static str = "model_listing";

    fn is_terminal(_event: &Self::Event) -> bool {
        true
    }

    fn telemetry(_streaming: bool) -> Self::Telemetry {}

    fn with_route(error: ProviderError, provider: &str, path: &str) -> ProviderError {
        crate::model::listing::with_route(error, provider, path)
    }
}

/// Concatenates the pages of a model listing, in arrival order.
#[derive(Default)]
pub struct ModelListingFold {
    models: Vec<Model>,
}

impl Fold<ModelListing> for ModelListingFold {
    fn absorb(&mut self, page: ModelList) -> Result<(), ProviderError> {
        self.models.extend(page);
        Ok(())
    }

    fn finish(self, _reply: Reply) -> Result<ModelList, ProviderError> {
        Ok(ModelList::new(self.models))
    }
}
