//! Consumer model implementations for [`Bound`] and traits for constructing
//! operation wires from provider configurations.
//!
//! ```no_run
//! use rig_core::driver::Bind;
//! use rig_core::providers::openai::{self, OpenAI};
//!
//! # fn example(http: impl rig_core::driver::Socket) -> Result<(), Box<dyn std::error::Error>> {
//! let provider = OpenAI::from_env()?.bind(http);
//! let model = provider.completion(openai::GPT_5_2);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

use super::{Bound, call, stream};
use crate::completion::{
    CompletionModel, CompletionRequest, CompletionResponse, ProviderCapabilities,
};
use crate::embeddings::{
    EmbeddingModel, EmbeddingResponse, ImageEmbeddingModel, ImageEmbeddingResponse,
};
use crate::error::ProviderError;
use crate::http_client::HttpClientExt;
use crate::model::ModelList;
use crate::observe::AdapterContext;
use crate::operation::{
    Completion, Embedding, ImageEmbedding, ModelListing, Rerank, RerankRequest, Transcription,
    Verify,
};
use crate::providers::gemini::cached_content::{
    CacheExpiry, CachedContent, CachedContentRequest, CachedContents, NewCachedContent, on_handle,
};
use crate::rerank::{RerankModel, RerankResponse};
use crate::streaming::StreamingCompletionResponse;
use crate::transcription::{TranscriptionModel, TranscriptionRequest, TranscriptionResponse};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use crate::wire::Wire;

/// The transport bound set every `Bound` model needs: `Clone` because a
/// streamed reply outlives the borrow that opened it, `'static` because the
/// stream owns its socket.
pub trait Socket: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static {}

impl<H> Socket for H where H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static {}

impl<W, H> CompletionModel for Bound<W, H>
where
    W: Wire<Op = Completion>,
    H: Socket,
{
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        call(&self.wire, &self.http, request, None).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse, ProviderError> {
        self.stream_with_context(request, None).await
    }

    async fn completion_with_context(
        &self,
        request: CompletionRequest,
        context: Option<AdapterContext>,
    ) -> Result<CompletionResponse, ProviderError> {
        call(&self.wire, &self.http, request, context).await
    }

    async fn stream_with_context(
        &self,
        request: CompletionRequest,
        context: Option<AdapterContext>,
    ) -> Result<StreamingCompletionResponse, ProviderError> {
        let provider = self.wire.name().to_owned();
        let frames = stream(&self.wire, &self.http, request, context)?;
        Ok(StreamingCompletionResponse::stream(
            provider,
            Box::pin(frames),
        ))
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.wire.capabilities()
    }
}

impl<W, H> EmbeddingModel for Bound<W, H>
where
    W: Wire<Op = Embedding>,
    H: Socket,
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
        let texts: Vec<String> = texts.into_iter().collect();
        let response = call(&self.wire, &self.http, texts, None).await?;
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

impl<W, H> ImageEmbeddingModel for Bound<W, H>
where
    W: Wire<Op = ImageEmbedding>,
    H: Socket,
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
        let images: Vec<Vec<u8>> = images.into_iter().collect();
        let response = call(&self.wire, &self.http, images, None).await?;
        // Image vectors must also honor the declared dimensions.
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

impl<W, H> TranscriptionModel for Bound<W, H>
where
    W: Wire<Op = Transcription>,
    H: Socket,
{
    async fn transcription(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResponse, ProviderError> {
        call(&self.wire, &self.http, request, None).await
    }
}

impl<W, H> RerankModel for Bound<W, H>
where
    W: Wire<Op = Rerank>,
    H: Socket,
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
        call(&self.wire, &self.http, request, None).await
    }
}

#[cfg(feature = "image")]
impl<W, H> crate::image_generation::ImageGenerationModel for Bound<W, H>
where
    W: Wire<Op = crate::operation::ImageGeneration>,
    H: Socket,
{
    async fn image_generation(
        &self,
        request: crate::image_generation::ImageGenerationRequest,
    ) -> Result<crate::image_generation::ImageGenerationResponse, crate::error::ProviderError> {
        call(&self.wire, &self.http, request, None).await
    }
}

#[cfg(feature = "audio")]
impl<W, H> crate::audio_generation::AudioGenerationModel for Bound<W, H>
where
    W: Wire<Op = crate::operation::AudioGeneration>,
    H: Socket,
{
    async fn audio_generation(
        &self,
        request: crate::audio_generation::AudioGenerationRequest,
    ) -> Result<crate::audio_generation::AudioGenerationResponse, crate::error::ProviderError> {
        call(&self.wire, &self.http, request, None).await
    }
}

impl<W, H> crate::model::ModelLister for Bound<W, H>
where
    W: Wire<Op = ModelListing>,
    H: Socket,
{
    async fn list_all(&self) -> Result<ModelList, ProviderError> {
        call(&self.wire, &self.http, (), None).await
    }
}

impl<P, H> Bound<P, H>
where
    P: HasVerify,
    H: Socket,
{
    /// Check that the provider accepts the configured credentials.
    pub async fn verify(&self) -> Result<(), ProviderError> {
        call(&self.wire.verify(), &self.http, (), None)
            .await
            .map_err(crate::client::verify::authentication)
    }
}

/// A provider config that has a completion wire.
pub use crate::wire::HasCompletion;

/// Constructs a completion model from a model name.
pub trait CompletionProvider {
    /// The model this provider builds.
    type Model: CompletionModel;

    /// The completion model for `model`.
    fn completion(&self, model: impl Into<String>) -> Self::Model;
}

impl<P, H> CompletionProvider for Bound<P, H>
where
    P: HasCompletion,
    H: Clone + Socket,
{
    type Model = Bound<P::Wire, H>;

    fn completion(&self, model: impl Into<String>) -> Self::Model {
        Bound::completion(self, model)
    }
}

/// A provider config that has an embedding wire.
pub trait HasEmbedding: WasmCompatSend + WasmCompatSync {
    /// The provider's embedding wire.
    type Wire: Wire<Op = Embedding>;

    /// Build the embedding wire for `model`, at `ndims` dimensions when the
    /// caller named one rather than taking the model's default.
    fn embedding(&self, model: impl Into<String>, ndims: Option<usize>) -> Self::Wire;
}

/// A provider config that has an image-embedding wire.
pub trait HasImageEmbedding: WasmCompatSend + WasmCompatSync {
    /// The provider's image-embedding wire.
    type Wire: Wire<Op = ImageEmbedding>;

    /// Build the image-embedding wire for `model`.
    fn image_embedding(&self, model: impl Into<String>, ndims: Option<usize>) -> Self::Wire;
}

/// A provider config that has a transcription wire.
pub trait HasTranscription: WasmCompatSend + WasmCompatSync {
    /// The provider's transcription wire.
    type Wire: Wire<Op = Transcription>;

    /// Build the transcription wire for `model`.
    fn transcription(&self, model: impl Into<String>) -> Self::Wire;
}

/// A provider config that has a rerank wire.
pub trait HasRerank: WasmCompatSend + WasmCompatSync {
    /// The provider's rerank wire.
    type Wire: Wire<Op = Rerank>;

    /// Build the rerank wire for `model`.
    fn rerank(&self, model: impl Into<String>) -> Self::Wire;
}

/// A provider config that has an image-generation wire.
#[cfg(feature = "image")]
pub trait HasImageGeneration: WasmCompatSend + WasmCompatSync {
    /// The provider's image-generation wire.
    type Wire: Wire<Op = crate::operation::ImageGeneration>;

    /// Build the image-generation wire for `model`.
    fn image_generation(&self, model: impl Into<String>) -> Self::Wire;
}

/// A provider config that has an audio-generation wire.
#[cfg(feature = "audio")]
pub trait HasAudioGeneration: WasmCompatSend + WasmCompatSync {
    /// The provider's audio-generation wire.
    type Wire: Wire<Op = crate::operation::AudioGeneration>;

    /// Build the audio-generation wire for `model`.
    fn audio_generation(&self, model: impl Into<String>) -> Self::Wire;
}

/// A provider config that has a model-listing wire.
pub trait HasModelListing: WasmCompatSend + WasmCompatSync {
    /// The provider's model-listing wire.
    type Wire: Wire<Op = ModelListing>;

    /// Build the model-listing wire.
    fn model_listing(&self) -> Self::Wire;
}

/// A provider config that has a verification wire.
pub trait HasVerify: WasmCompatSend + WasmCompatSync {
    /// The provider's verification wire.
    type Wire: Wire<Op = Verify>;

    /// Build the verification wire.
    fn verify(&self) -> Self::Wire;
}

/// Generates a bound wire constructor that clones the existing transport.
/// The bound method name can differ from the provider trait method.
macro_rules! bound_constructor {
    ($has:ident, $method:ident $(, $arg:ident : $ty:ty)*) => {
        bound_constructor!($has, $method => $method $(, $arg: $ty)*);
    };
    ($has:ident, $trait_method:ident => $method:ident $(, $arg:ident : $ty:ty)*) => {
        impl<P, H> Bound<P, H>
        where
            P: $has,
            H: Clone,
        {
            #[doc = concat!("The provider's `", stringify!($method), "` wire, on this socket.")]
            pub fn $method(&self $(, $arg: $ty)*) -> Bound<P::Wire, H> {
                Bound {
                    wire: P::$trait_method(&self.wire $(, $arg)*),
                    http: self.http.clone(),
                }
            }
        }
    };
}

bound_constructor!(HasCompletion, completion, model: impl Into<String>);
bound_constructor!(HasEmbedding, embedding, model: impl Into<String>, ndims: Option<usize>);
bound_constructor!(
    HasImageEmbedding,
    image_embedding,
    model: impl Into<String>,
    ndims: Option<usize>
);
bound_constructor!(HasTranscription, transcription, model: impl Into<String>);
bound_constructor!(HasRerank, rerank, model: impl Into<String>);
bound_constructor!(HasModelListing, model_listing => models);
#[cfg(feature = "image")]
bound_constructor!(HasImageGeneration, image_generation, model: impl Into<String>);
#[cfg(feature = "audio")]
bound_constructor!(HasAudioGeneration, audio_generation, model: impl Into<String>);

/// Document embedding builders using a bound provider's embedding wire.
impl<P, H> Bound<P, H>
where
    P: HasEmbedding,
    H: Clone + Socket,
    P::Wire: Wire<Op = Embedding>,
{
    /// An embedding builder over this provider's `model`.
    pub fn embeddings<D: crate::Embed>(
        &self,
        model: impl Into<String>,
    ) -> crate::embeddings::EmbeddingsBuilder<Bound<P::Wire, H>, D> {
        crate::embeddings::EmbeddingsBuilder::new(self.embedding(model, None))
    }

    /// An embedding builder over this provider's `model` at `ndims`
    /// dimensions.
    pub fn embeddings_with_ndims<D: crate::Embed>(
        &self,
        model: impl Into<String>,
        ndims: usize,
    ) -> crate::embeddings::EmbeddingsBuilder<Bound<P::Wire, H>, D> {
        crate::embeddings::EmbeddingsBuilder::new(self.embedding(model, Some(ndims)))
    }
}

/// Explicit context-cache operations. Requests targeting an existing handle
/// map HTTP 403 and 404 to [`ProviderError::CacheExpired`].
impl<H> Bound<CachedContents, H>
where
    H: Socket,
{
    /// Creates cached content and returns its handle and storage usage metadata.
    pub async fn create(&self, request: NewCachedContent) -> Result<CachedContent, ProviderError> {
        let request = CachedContentRequest::Create(request);
        call(&self.wire, &self.http, request, None)
            .await?
            .resource()
    }

    /// Fetch one cached content by handle.
    pub async fn get(&self, name: &str) -> Result<CachedContent, ProviderError> {
        let request = CachedContentRequest::Get(name.to_owned());
        call(&self.wire, &self.http, request, None)
            .await
            .map_err(|error| on_handle(error, name))?
            .resource()
    }

    /// Every cached content this API key can see, following pagination at
    /// the wire's page size.
    pub async fn list(&self) -> Result<Vec<CachedContent>, ProviderError> {
        call(&self.wire, &self.http, CachedContentRequest::List, None)
            .await?
            .entries()
    }

    /// Lists cached content using an explicit page size.
    pub async fn list_with_page_size(
        &self,
        page_size: usize,
    ) -> Result<Vec<CachedContent>, ProviderError> {
        let wire = self.wire.clone().with_page_size(page_size);
        call(&wire, &self.http, CachedContentRequest::List, None)
            .await?
            .entries()
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
        call(&self.wire, &self.http, request, None)
            .await
            .map_err(|error| on_handle(error, name))?
            .resource()
    }

    /// Deletes cached content before its expiry. Callers should also clean up
    /// task-scoped caches on failure to avoid continued storage charges.
    /// Handles other than `cachedContents/<id>` or bare `<id>` return
    /// [`ProviderError::Request`] before dispatch.
    pub async fn delete(&self, name: &str) -> Result<(), ProviderError> {
        let request = CachedContentRequest::Delete(name.to_owned());
        call(&self.wire, &self.http, request, None)
            .await
            .map_err(|error| on_handle(error, name))?;
        Ok(())
    }
}
