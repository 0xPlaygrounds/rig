//! The explicit context-cache verbs of a Gemini [`CachedContents`] model.
//! Each is one call through the driver; a request that targets an existing
//! handle maps HTTP 403 and 404 to [`ProviderError::CacheExpired`].

use crate::driver::{Model, Transport};
use crate::error::ProviderError;
use crate::providers::gemini::api::CachedContent;
use crate::providers::gemini::cached_content::{
    CacheExpiry, CachedContentRequest, CachedContents, on_handle,
};
use crate::providers::gemini::{CachedPrefix, NewCachedContent};

/// Explicit context-cache operations. Requests targeting an existing handle
/// map HTTP 403 and 404 to [`ProviderError::CacheExpired`].
impl<T> Model<CachedContents, T>
where
    T: Transport<CachedContents>,
{
    /// Create a cache. The prefix it returns names the cache and keeps the
    /// body that created it.
    pub async fn create(&self, new: NewCachedContent) -> Result<CachedPrefix, ProviderError> {
        let request = new.render()?;
        self.recreate(request).await
    }

    /// `prefix` when its cache is alive, else a cache recreated from the body
    /// that created it. An absolute expiry is dropped on recreation, since it
    /// may have passed.
    pub async fn ensure(&self, prefix: &CachedPrefix) -> Result<CachedPrefix, ProviderError> {
        match self.get(prefix.name()).await {
            Ok(resource) => Ok(CachedPrefix {
                resource,
                request: prefix.request.clone(),
            }),
            Err(ProviderError::CacheExpired { .. }) => {
                let mut request = prefix.request.clone();
                request.expire_time = None;
                self.recreate(request).await
            }
            Err(error) => Err(error),
        }
    }

    async fn recreate(&self, request: CachedContent) -> Result<CachedPrefix, ProviderError> {
        let resource = self
            .call(CachedContentRequest::Create(Box::new(request.clone())))
            .await?
            .resource()?;
        Ok(CachedPrefix { resource, request })
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
        let pages =
            crate::driver::follow_cursors(self.name(), "cached_content", |page_token| async move {
                let reply = self.call(CachedContentRequest::List { page_token }).await?;
                let next = reply.next_page_token();
                Ok((reply.entries()?, next))
            })
            .await?;
        Ok(pages.into_iter().flatten().collect())
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
