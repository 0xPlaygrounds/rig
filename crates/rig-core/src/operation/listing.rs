//! Model listing: one call per page, and [`Model::list`] to follow the
//! cursors.
//!
//! ```
//! use rig_core::model::ModelList;
//! use rig_core::operation::ModelPage;
//!
//! let page = ModelPage { models: ModelList::new(Vec::new()), next: None };
//! assert!(page.next.is_none());
//! ```

use super::Take;
use crate::driver::{Model, Transport};
use crate::error::{ProviderError, RigError};
use crate::model::ModelList;
use crate::wire::{Call, Operation, Wire};

/// Lists provider models, one page per call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelListing;

/// One page of a model listing, and the cursor of the page after it.
#[derive(Debug, Clone)]
pub struct ModelPage {
    /// The page's models, in the order the provider listed them.
    pub models: ModelList,
    /// The next page's cursor. `None` on the last page.
    pub next: Option<String>,
}

impl Operation for ModelListing {
    /// The cursor of the page to read: `None` for the first.
    type Request = Option<String>;
    type Event = ModelPage;
    type Response = ModelPage;
    type Fold = Take<Self>;

    fn is_terminal(_event: &Self::Event) -> bool {
        true
    }

    fn fold(_request: &Self::Request, _call: &mut Call<'_>) -> Self::Fold {
        Take::default()
    }
}

impl<W, T> Model<W, T>
where
    W: Wire<Op = ModelListing>,
    T: Transport<W>,
{
    /// Every model the provider lists, every page followed in order.
    ///
    /// A failed page names the provider and its request path. Paging stops
    /// at a cursor that repeats the one just answered, and after a bounded
    /// number of pages.
    pub async fn list(&self) -> Result<ModelList, RigError> {
        let provider = self.name();
        let pages = crate::driver::follow_cursors::<_, ProviderError, _>(
            provider,
            "model_listing",
            |cursor| async move {
                let page = self.call_routed(cursor).await.map_err(|(error, path)| {
                    crate::model::listing::with_route(error, provider, &path)
                })?;
                Ok((page.models, page.next))
            },
        )
        .await?;
        Ok(ModelList::new(pages.into_iter().flatten().collect()))
    }
}
