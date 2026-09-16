//! The consumer-facing traits, implemented once each for [`Bound`].
//!
//! Seven impls, total: one per trait. A provider contributes none of them —
//! it contributes a [`Wire`], and `Bound<W, H>` is the model.
//!
//! The `Has*` traits below are the other half: a provider *config* names the
//! wire it builds for an operation, so one `impl<P: Has…, H>` gives every
//! provider config the same construction methods on `Bound<P, H>` (and lets
//! rig-agent offer `agent(model)` without naming a provider).

use super::{Bound, call, stream};
use crate::completion::{
    CompletionError, CompletionModel, CompletionRequest, CompletionResponse, ProviderCapabilities,
};
use crate::embeddings::{
    EmbeddingError, EmbeddingModel, EmbeddingResponse, ImageEmbeddingModel, ImageEmbeddingResponse,
};
use crate::http_client::HttpClientExt;
use crate::model::{ModelList, ModelListingError};
use crate::observe::AdapterContext;
use crate::operation::{
    Completion, Embedding, ImageEmbedding, ModelListing, Rerank, RerankRequest, Transcription,
    Verify,
};
use crate::rerank::{RerankError, RerankModel, RerankResponse};
use crate::streaming::StreamingCompletionResponse;
use crate::transcription::{
    TranscriptionError, TranscriptionModel, TranscriptionRequest, TranscriptionResponse,
};
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
    ) -> Result<CompletionResponse, CompletionError> {
        call(&self.wire, &self.http, request, None).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        self.stream_with_context(request, None).await
    }

    async fn completion_with_context(
        &self,
        request: CompletionRequest,
        context: Option<AdapterContext>,
    ) -> Result<CompletionResponse, CompletionError> {
        call(&self.wire, &self.http, request, context).await
    }

    async fn stream_with_context(
        &self,
        request: CompletionRequest,
        context: Option<AdapterContext>,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
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
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        let texts: Vec<String> = texts.into_iter().collect();
        let response = call(&self.wire, &self.http, texts, None).await?;
        // This impl publishes `ndims()`, so this impl owes the caller
        // vectors of that width: the declaration is the wire's, the
        // vectors are the reply's, and nothing else holds both.
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
    ) -> Result<ImageEmbeddingResponse, EmbeddingError> {
        let images: Vec<Vec<u8>> = images.into_iter().collect();
        let response = call(&self.wire, &self.http, images, None).await?;
        // Same promise as the text wires, on the same value.
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
    ) -> Result<TranscriptionResponse, TranscriptionError> {
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
    ) -> Result<RerankResponse, RerankError> {
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
    ) -> Result<
        crate::image_generation::ImageGenerationResponse,
        crate::image_generation::ImageGenerationError,
    > {
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
    ) -> Result<
        crate::audio_generation::AudioGenerationResponse,
        crate::audio_generation::AudioGenerationError,
    > {
        call(&self.wire, &self.http, request, None).await
    }
}

impl<W, H> crate::model::ModelLister for Bound<W, H>
where
    W: Wire<Op = ModelListing>,
    H: Socket,
{
    async fn list_all(&self) -> Result<ModelList, ModelListingError> {
        call(&self.wire, &self.http, (), None).await
    }
}

impl<P, H> Bound<P, H>
where
    P: HasVerify,
    H: Socket,
{
    /// Check that the provider accepts the configured credentials.
    pub async fn verify(&self) -> Result<(), crate::client::VerifyError> {
        call(&self.wire.verify(), &self.http, (), None).await
    }
}

/// A provider config that has a completion wire.
pub use crate::wire::HasCompletion;

/// Anything that builds a completion model for a named model.
///
/// This is what the six deleted `*Client` traits were really for, reduced to
/// the one thing a caller wanted from them, and it is the seam the agent
/// sugar (`provider.agent(model)`) hangs on. `Bound<P, H>` satisfies it for
/// every wire-backed provider; the typed-transport providers — Bedrock's
/// Converse event stream, Vertex AI, gemini-grpc, in-process inference —
/// implement it directly, because their frames are an SDK's types rather
/// than bytes and they are not wires. Same spelling either way, which is the
/// point: a caller does not need to know which kind it has.
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

/// Build an operation's wire from a bound provider config, keeping the
/// socket. One impl per operation, never one per provider.
macro_rules! bound_constructor {
    ($has:ident, $method:ident $(, $arg:ident : $ty:ty)*) => {
        impl<P, H> Bound<P, H>
        where
            P: $has,
            H: Clone,
        {
            #[doc = concat!("The provider's `", stringify!($method), "` wire, on this socket.")]
            pub fn $method(&self $(, $arg: $ty)*) -> Bound<P::Wire, H> {
                Bound {
                    wire: P::$method(&self.wire $(, $arg)*),
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
bound_constructor!(HasModelListing, model_listing);
#[cfg(feature = "image")]
bound_constructor!(HasImageGeneration, image_generation, model: impl Into<String>);
#[cfg(feature = "audio")]
bound_constructor!(HasAudioGeneration, audio_generation, model: impl Into<String>);

/// An embedding builder over a bound embedding wire: it batches many documents
/// into one provider request, which is what a caller wants instead of an
/// `embed_text` call apiece.
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
