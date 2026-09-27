//! The explicit context-cache verbs of a Gemini [`CachedContents`] model.
//! Each is one call through the driver; a request that targets an existing
//! handle reads HTTP 403 and 404 as [`ErrorDetail::CacheExpired`](crate::error::ErrorDetail::CacheExpired).

use crate::driver::{Model, Transport};
use crate::error::{ProviderError, RigError};
use crate::providers::gemini::cached_content::{
    CacheExpiry, CachedContent, CachedContentRequest, CachedContents, NewCachedContent, on_handle,
};

/// Explicit context-cache operations. Requests targeting an existing handle
/// read HTTP 403 and 404 as [`ErrorDetail::CacheExpired`](crate::error::ErrorDetail::CacheExpired).
impl<T> Model<CachedContents, T>
where
    T: Transport<CachedContents>,
{
    /// Creates cached content and returns its handle and storage usage metadata.
    pub async fn create(&self, request: NewCachedContent) -> Result<CachedContent, RigError> {
        self.call(CachedContentRequest::Create(request))
            .await?
            .resource()
    }

    /// Fetch one cached content by handle.
    pub async fn get(&self, name: &str) -> Result<CachedContent, RigError> {
        self.drained::<ProviderError>(CachedContentRequest::Get(name.to_owned()), None)
            .await
            .map_err(|error| on_handle(error, name))?
            .resource()
    }

    /// Every cached content this API key can see, following pagination at
    /// the wire's page size.
    pub async fn list(&self) -> Result<Vec<CachedContent>, RigError> {
        let pages = crate::driver::follow_cursors::<_, RigError, _>(
            self.name(),
            "cached_content",
            |page_token| async move {
                let reply = self.call(CachedContentRequest::List { page_token }).await?;
                let next = reply.next_page_token();
                Ok((reply.entries()?, next))
            },
        )
        .await?;
        Ok(pages.into_iter().flatten().collect())
    }

    /// Changes cache expiry without modifying its immutable content.
    /// Returns the updated resource or an error, including [`ErrorDetail::CacheExpired`](crate::error::ErrorDetail::CacheExpired) for HTTP 403/404.
    pub async fn update_expiry(
        &self,
        name: &str,
        expiry: CacheExpiry,
    ) -> Result<CachedContent, RigError> {
        let request = CachedContentRequest::UpdateExpiry {
            name: name.to_owned(),
            expiry,
        };
        self.drained::<ProviderError>(request, None)
            .await
            .map_err(|error| on_handle(error, name))?
            .resource()
    }

    /// Deletes cached content before its expiry. Callers should also clean up
    /// task-scoped caches on failure to avoid continued storage charges.
    /// Handles other than `cachedContents/<id>` or bare `<id>` return
    /// [`ErrorKind::Request`](crate::error::ErrorKind::Request) before dispatch.
    pub async fn delete(&self, name: &str) -> Result<(), RigError> {
        self.drained::<ProviderError>(CachedContentRequest::Delete(name.to_owned()), None)
            .await
            .map_err(|error| on_handle(error, name))?;
        Ok(())
    }
}
