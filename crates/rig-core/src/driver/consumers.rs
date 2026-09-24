//! The model traits a [`Model`] implements, one per wire operation, and the
//! inherent calls of operations that have no trait.

use std::future::Future;

use futures::StreamExt;

use super::{Model, Step, Transport};
use crate::completion::{
    CompletionModel, CompletionRequest, CompletionResponse, ProviderCapabilities,
};
use crate::embeddings::{
    EmbeddingModel, EmbeddingResponse, ImageEmbeddingModel, ImageEmbeddingResponse,
};
use crate::error::ProviderError;
use crate::model::{ModelList, ModelLister};
use crate::operation::{
    Completion, Embedding, ImageEmbedding, ModelListing, Rerank, RerankRequest, Transcription,
    Verify,
};
use crate::providers::gemini::cached_content::{
    CacheExpiry, CachedContent, CachedContentReply, CachedContentRequest, CachedContents,
    NewCachedContent, on_handle,
};
use crate::rerank::{RerankModel, RerankResponse};
use crate::streaming::CompletionStream;
use crate::transcription::{TranscriptionModel, TranscriptionRequest, TranscriptionResponse};
use crate::wasm_compat::WasmCompatSend;
use crate::wire::{Mode, Operation, Wire};

impl<W, T> CompletionModel for Model<W, T>
where
    W: Wire<Op = Completion> + Clone,
    T: Transport<Payload = W::Payload, Frame = W::Frame>,
{
    async fn complete(
        &self,
        mut request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        let extensions = std::mem::take(&mut request.extensions);
        self.unary(request, extensions, |error, _, _| error).await
    }

    async fn stream(
        &self,
        mut request: CompletionRequest,
    ) -> Result<CompletionStream, ProviderError> {
        let extensions = std::mem::take(&mut request.extensions);
        let span = self.span(&request, true);
        let steps = self.run(
            request,
            Mode::Streaming,
            extensions,
            span.clone(),
            |error, _, _| error,
        )?;
        let events = tracing_futures::Instrument::instrument(steps, span).filter_map(|step| {
            futures::future::ready(match step {
                Ok(Step::Event(event)) => Some(Ok(event)),
                Ok(Step::Done(_)) => None,
                Err(error) => Some(Err(error)),
            })
        });
        Ok(CompletionStream::new(self.wire.name(), events))
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.wire.capabilities()
    }
}

impl<W, T> EmbeddingModel for Model<W, T>
where
    W: Wire<Op = Embedding> + Clone,
    T: Transport<Payload = W::Payload, Frame = W::Frame>,
{
    fn max_documents(&self) -> usize {
        self.wire.capabilities().max_documents
    }

    fn ndims(&self) -> usize {
        self.wire.capabilities().ndims
    }

    async fn embed_texts_response(
        &self,
        texts: impl IntoIterator<Item = String> + WasmCompatSend,
    ) -> Result<EmbeddingResponse, ProviderError> {
        let response = self.call(texts.into_iter().collect()).await?;
        // Reject vectors whose width violates the model's declared dimensions.
        self.wire.capabilities().honour_declaration(
            self.wire.name(),
            response
                .embeddings
                .iter()
                .map(|embedding| embedding.vec.len()),
        )?;
        Ok(response)
    }
}

impl<W, T> ImageEmbeddingModel for Model<W, T>
where
    W: Wire<Op = ImageEmbedding> + Clone,
    T: Transport<Payload = W::Payload, Frame = W::Frame>,
{
    fn max_documents(&self) -> usize {
        self.wire.capabilities().max_documents
    }

    fn ndims(&self) -> usize {
        self.wire.capabilities().ndims
    }

    async fn embed_images_response(
        &self,
        images: impl IntoIterator<Item = Vec<u8>> + WasmCompatSend,
    ) -> Result<ImageEmbeddingResponse, ProviderError> {
        let response = self.call(images.into_iter().collect()).await?;
        // Image vectors must also honour the declared dimensions.
        self.wire.capabilities().honour_declaration(
            self.wire.name(),
            response
                .embeddings
                .iter()
                .map(|embedding| embedding.vec.len()),
        )?;
        Ok(response)
    }
}

impl<W, T> TranscriptionModel for Model<W, T>
where
    W: Wire<Op = Transcription> + Clone,
    T: Transport<Payload = W::Payload, Frame = W::Frame>,
{
    async fn transcription(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResponse, ProviderError> {
        self.call(request).await
    }
}

impl<W, T> RerankModel for Model<W, T>
where
    W: Wire<Op = Rerank> + Clone,
    T: Transport<Payload = W::Payload, Frame = W::Frame>,
{
    fn max_documents(&self) -> usize {
        self.wire.capabilities()
    }

    async fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
    ) -> Result<RerankResponse, ProviderError> {
        let request = RerankRequest {
            query: query.to_owned(),
            documents,
        };
        self.call(request).await
    }
}

#[cfg(feature = "image")]
impl<W, T> crate::image_generation::ImageGenerationModel for Model<W, T>
where
    W: Wire<Op = crate::operation::ImageGeneration> + Clone,
    T: Transport<Payload = W::Payload, Frame = W::Frame>,
{
    async fn image_generation(
        &self,
        request: crate::image_generation::ImageGenerationRequest,
    ) -> Result<crate::image_generation::ImageGenerationResponse, ProviderError> {
        self.call(request).await
    }
}

#[cfg(feature = "audio")]
impl<W, T> crate::audio_generation::AudioGenerationModel for Model<W, T>
where
    W: Wire<Op = crate::operation::AudioGeneration> + Clone,
    T: Transport<Payload = W::Payload, Frame = W::Frame>,
{
    async fn audio_generation(
        &self,
        request: crate::audio_generation::AudioGenerationRequest,
    ) -> Result<crate::audio_generation::AudioGenerationResponse, ProviderError> {
        self.call(request).await
    }
}

impl<W, T> ModelLister for Model<W, T>
where
    W: Wire<Op = ModelListing> + Clone,
    T: Transport<Payload = W::Payload, Frame = W::Frame>,
{
    async fn list_all(&self) -> Result<ModelList, ProviderError> {
        let pages = paginate(self.wire.name(), ModelListing::NAME, |cursor| async move {
            let page = self
                .unary(cursor, http::Extensions::new(), |error, provider, path| {
                    crate::model::listing::with_route(error, provider, path)
                })
                .await?;
            Ok((page.models, page.next))
        })
        .await?;
        Ok(ModelList::new(pages.into_iter().flatten().collect()))
    }
}

impl<W, T> Model<W, T>
where
    W: Wire<Op = Verify> + Clone,
    T: Transport<Payload = W::Payload, Frame = W::Frame>,
{
    /// Check that the provider accepts the configured credentials.
    pub async fn verify(&self) -> Result<(), ProviderError> {
        self.call(())
            .await
            .map_err(crate::client::verify::authentication)
    }
}

/// Explicit context-cache operations. Requests targeting an existing handle
/// map HTTP 403 and 404 to [`ProviderError::CacheExpired`].
impl<T> Model<CachedContents, T>
where
    T: Transport<Payload = crate::wire::Encoded, Frame = crate::wire::WireFrame>,
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
        let pages = paginate(
            self.wire.name(),
            "cached_content",
            |page_token| async move {
                match self.call(CachedContentRequest::List(page_token)).await? {
                    CachedContentReply::Page(page) => {
                        Ok((page.cached_contents, page.next_page_token))
                    }
                    other => Ok((other.entries()?, None)),
                }
            },
        )
        .await?;
        Ok(pages.into_iter().flatten().collect())
    }

    /// Lists cached content using an explicit page size.
    pub async fn list_with_page_size(
        &self,
        page_size: usize,
    ) -> Result<Vec<CachedContent>, ProviderError> {
        Model::new(
            self.wire.clone().with_page_size(page_size),
            self.transport.clone(),
        )
        .list()
        .await
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

/// Page count after which a listing stops following cursors, so a provider
/// that cycles its cursors cannot hold the caller forever.
pub(super) const MAX_PAGES: usize = 1000;

/// Fetch every page of a cursor-paged listing. `page` fetches the page after
/// a cursor (the first page for `None`) and names the next cursor. A cursor
/// that repeats the one just sent, or a listing past [`MAX_PAGES`], ends the
/// listing with the pages fetched so far.
async fn paginate<I, F, Fut>(
    provider: &str,
    operation: &str,
    mut page: F,
) -> Result<Vec<I>, ProviderError>
where
    F: FnMut(Option<String>) -> Fut,
    Fut: Future<Output = Result<(I, Option<String>), ProviderError>>,
{
    let mut pages = Vec::new();
    let mut cursor: Option<String> = None;
    loop {
        let (items, next) = page(cursor.clone()).await?;
        pages.push(items);
        let Some(next) = next else {
            break;
        };
        // Normal exhaustion is not truncation; a refused cursor is warned.
        if cursor.as_deref() == Some(next.as_str()) {
            tracing::warn!(
                provider,
                operation,
                pages = pages.len(),
                "listing repeated its pagination cursor; returning the pages fetched so far"
            );
            break;
        }
        if pages.len() >= MAX_PAGES {
            tracing::warn!(
                provider,
                operation,
                pages = pages.len(),
                "listing hit its page ceiling with a cursor still advancing; returning the \
                 pages fetched so far"
            );
            break;
        }
        cursor = Some(next);
    }
    Ok(pages)
}
