//! Model-listing operation and ordered page aggregation.
//!
//! ```
//! use rig_core::{operation::ModelListing, wire::Operation};
//!
//! assert_eq!(ModelListing::NAME, "model_listing");
//! ```

use super::{One, Take};
use crate::model::ModelPage;
use crate::wire::Operation;

/// Listing a provider's models, one page per call. The request is the
/// page's cursor, `None` for the first page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelListing;

impl Operation for ModelListing {
    type Request = Option<String>;
    type Event = ModelPage;
    type Response = ModelPage;
    type Capabilities = ();
    type Output = One<Self>;
    type Fold = Take<Self>;

    const NAME: &'static str = "model_listing";

    fn is_terminal(_event: &Self::Event) -> bool {
        true
    }
}
