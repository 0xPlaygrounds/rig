use crate::model::Model;
use crate::model::listing::ModelListingError;

use super::{Fold, Operation};

/// The model listing operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ModelListing;

impl Operation for ModelListing {
    type Request = ();
    type Event = Vec<Model>;
    type Response = Vec<Model>;
    type Error = ModelListingError;
    type Capabilities = ();
    type Fold = ModelListingFold;
}

/// Accumulates paged model lists into a single collection of models.
#[derive(Default)]
pub struct ModelListingFold {
    models: Vec<Model>,
}

impl Fold<Vec<Model>> for ModelListingFold {
    type Response = Vec<Model>;
    type Error = ModelListingError;

    fn fold(&mut self, event: Vec<Model>) {
        self.models.extend(event);
    }

    fn finish(self) -> Result<Vec<Model>, ModelListingError> {
        Ok(self.models)
    }
}
