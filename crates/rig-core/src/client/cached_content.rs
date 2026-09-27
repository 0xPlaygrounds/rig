//! The explicit context-cache verbs of a Gemini [`CachedContents`] model.
//! Each is one call through the driver; a request that targets an existing
//! handle maps HTTP 403 and 404 to [`ProviderError::CacheExpired`].

use crate::driver::{Model, Transport};
use crate::error::ProviderError;
use crate::providers::gemini::cached_content::{
    CacheExpiry, CachedContent, CachedContentRequest, CachedContents, NewCachedContent, on_handle,
};

/// Explicit context-cache operations. Requests targeting an existing handle
/// map HTTP 403 and 404 to [`ProviderError::CacheExpired`].
impl<T> Model<CachedContents, T>
where
    T: Transport<CachedContents>,
{
    /// Creates cached content and returns its handle and storage usage metadata.
    pub async fn create(&self, request: NewCachedContent) -> Result<CachedContent, ProviderError> {
        self.call(CachedContentRequest::Create(request))
            .await?
            .resource()
    }

    /// Fetch one cached content by handle.
    pub async fn get(&self, name: &str) -> Result<CachedContent, ProviderError> {
        self.call(CachedContentRequest::Get(name.to_owned()))
            .await
            .map_err(|error| on_handle(error, name))?
            .resource()
    }

    /// Every cached content this API key can see, following pagination at
    /// the wire's page size.
    pub async fn list(&self) -> Result<Vec<CachedContent>, ProviderError> {
        self.call(CachedContentRequest::List).await?.entries()
    }

    /// Changes cache expiry without modifying its immutable content.
    /// Returns the updated resource or an error, including `Expired` for HTTP 403/404.
    pub async fn update_expiry(
        &self,
        name: &str,
        expiry: CacheExpiry,
    ) -> Result<CachedContent, ProviderError> {
        let request = CachedContentRequest::UpdateExpiry {
            name: name.to_owned(),
            expiry,
        };
        self.call(request)
            .await
            .map_err(|error| on_handle(error, name))?
            .resource()
    }

    /// Deletes cached content before its expiry. Callers should also clean up
    /// task-scoped caches on failure to avoid continued storage charges.
    /// Handles other than `cachedContents/<id>` or bare `<id>` return
    /// [`ProviderError::Request`] before dispatch.
    pub async fn delete(&self, name: &str) -> Result<(), ProviderError> {
        self.call(CachedContentRequest::Delete(name.to_owned()))
            .await
            .map_err(|error| on_handle(error, name))?;
        Ok(())
    }
}
