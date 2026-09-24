//! A provider configuration paired with the transport its models send
//! through, for tests that build several models from one recorded provider.
//! Each constructor pairs the provider's wire with a clone of the transport.

use rig_agent::agent::AgentBuilder;
use rig_agent::extractor::ExtractorBuilder;
use rig_core::Model;
use rig_core::driver::Transport;
use rig_core::embeddings::EmbeddingsBuilder;
use rig_core::http_client::BoxedHttpClient;
use rig_core::providers::{anthropic, cohere, copilot, gemini, ollama, openai, voyageai};

/// A provider configuration and its transport.
#[derive(Clone, Debug)]
pub struct Endpoint<P, H = BoxedHttpClient> {
    /// The provider configuration the wires are built from.
    pub wire: P,
    /// The transport every model built here sends through.
    pub http: H,
}

impl<P, H> Endpoint<P, H> {
    /// Pair `wire` with `http`.
    pub fn new(wire: P, http: H) -> Self {
        Self { wire, http }
    }

    /// This endpoint with its provider configuration transformed by `map`.
    pub fn map_wire<Q>(self, map: impl FnOnce(P) -> Q) -> Endpoint<Q, H> {
        Endpoint::new(map(self.wire), self.http)
    }

    /// `wire` over a clone of this endpoint's transport.
    fn pair<W>(&self, wire: W) -> Model<W, H>
    where
        H: Clone,
    {
        Model::new(wire, self.http.clone())
    }

    /// The model `wire` builds from this endpoint's provider configuration.
    pub fn model<W>(&self, wire: impl FnOnce(&P) -> W) -> Model<W, H>
    where
        H: Clone,
    {
        Model::new(wire(&self.wire), self.http.clone())
    }
}

impl<H: Clone> Endpoint<openai::wire::OpenAI, H> {
    /// The provider's `completion` model.
    pub fn completion(&self, model: impl Into<String>) -> Model<openai::wire::OpenAiWire, H> {
        self.pair(self.wire.completion(model))
    }

    /// The provider's `responses` model.
    pub fn responses(
        &self,
        model: impl Into<String>,
    ) -> Model<openai::responses_api::wire::Responses, H> {
        self.pair(self.wire.responses(model))
    }

    /// The provider's `chat` model.
    pub fn chat(&self, model: impl Into<String>) -> Model<openai::wire::Chat, H> {
        self.pair(self.wire.chat(model))
    }

    /// The provider's `embedding` model.
    pub fn embedding(
        &self,
        model: impl Into<String>,
        ndims: Option<usize>,
    ) -> Model<openai::wire::Embeddings, H> {
        self.pair(self.wire.embedding(model, ndims))
    }

    /// The provider's `rerank` model.
    pub fn rerank(&self, model: impl Into<String>) -> Model<openai::wire::Rerank, H> {
        self.pair(self.wire.rerank(model))
    }

    /// The provider's `transcription` model.
    pub fn transcription(
        &self,
        model: impl Into<String>,
    ) -> Model<openai::wire::Transcriptions, H> {
        self.pair(self.wire.transcription(model))
    }

    /// The provider's `models` model.
    pub fn models(&self) -> Model<openai::wire::Models, H> {
        self.pair(self.wire.models())
    }

    /// The provider's `verify` model.
    pub fn verify(&self) -> Model<openai::wire::Verify, H> {
        self.pair(self.wire.verify())
    }

    /// The provider's `image_generation` model.
    pub fn image_generation(&self, model: impl Into<String>) -> Model<openai::wire::Images, H> {
        self.pair(self.wire.image_generation(model))
    }

    /// The provider's `audio_generation` model.
    pub fn audio_generation(&self, model: impl Into<String>) -> Model<openai::wire::Speech, H> {
        self.pair(self.wire.audio_generation(model))
    }
}

impl<H> Endpoint<openai::wire::OpenAI, H>
where
    H: Transport<openai::wire::OpenAiWire>,
{
    /// An agent over the provider's completion model.
    pub fn agent(&self, model: impl Into<String>) -> AgentBuilder {
        AgentBuilder::new(self.completion(model))
    }

    /// A typed extractor over the provider's completion model.
    pub fn extractor<T>(&self, model: impl Into<String>) -> ExtractorBuilder<T>
    where
        T: schemars::JsonSchema
            + serde::de::DeserializeOwned
            + serde::Serialize
            + Send
            + Sync
            + 'static,
    {
        ExtractorBuilder::new(self.completion(model))
    }
}

impl<H> Endpoint<openai::wire::OpenAI, H>
where
    H: Transport<openai::wire::Embeddings>,
{
    /// An embeddings builder over the provider's `model`.
    pub fn embeddings<D: rig_core::Embed>(
        &self,
        model: impl Into<String>,
    ) -> EmbeddingsBuilder<Model<openai::wire::Embeddings, H>, D> {
        EmbeddingsBuilder::new(self.embedding(model, None))
    }

    /// An embeddings builder over the provider's `model` at `ndims`.
    pub fn embeddings_with_ndims<D: rig_core::Embed>(
        &self,
        model: impl Into<String>,
        ndims: usize,
    ) -> EmbeddingsBuilder<Model<openai::wire::Embeddings, H>, D> {
        EmbeddingsBuilder::new(self.embedding(model, Some(ndims)))
    }
}

impl<H: Clone> Endpoint<anthropic::wire::Anthropic, H> {
    /// The provider's `completion` model.
    pub fn completion(&self, model: impl Into<String>) -> Model<anthropic::wire::Messages, H> {
        self.pair(self.wire.completion(model))
    }

    /// The provider's `models` model.
    pub fn models(&self) -> Model<anthropic::wire::Models, H> {
        self.pair(self.wire.models())
    }

    /// The provider's `verify` model.
    pub fn verify(&self) -> Model<anthropic::wire::Verify, H> {
        self.pair(self.wire.verify())
    }
}

impl<H> Endpoint<anthropic::wire::Anthropic, H>
where
    H: Transport<anthropic::wire::Messages>,
{
    /// An agent over the provider's completion model.
    pub fn agent(&self, model: impl Into<String>) -> AgentBuilder {
        AgentBuilder::new(self.completion(model))
    }

    /// A typed extractor over the provider's completion model.
    pub fn extractor<T>(&self, model: impl Into<String>) -> ExtractorBuilder<T>
    where
        T: schemars::JsonSchema
            + serde::de::DeserializeOwned
            + serde::Serialize
            + Send
            + Sync
            + 'static,
    {
        ExtractorBuilder::new(self.completion(model))
    }
}

impl<H: Clone> Endpoint<gemini::Gemini, H> {
    /// The provider's `completion` model.
    pub fn completion(
        &self,
        model: impl Into<String>,
    ) -> Model<gemini::completion::GenerateContent, H> {
        self.pair(self.wire.completion(model))
    }

    /// The provider's `interactions` model.
    pub fn interactions(
        &self,
        model: impl Into<String>,
    ) -> Model<gemini::interactions_api::Interactions, H> {
        self.pair(self.wire.interactions(model))
    }

    /// The provider's `embedding` model.
    pub fn embedding(
        &self,
        model: impl Into<String>,
        ndims: Option<usize>,
    ) -> Model<gemini::embedding::Embeddings, H> {
        self.pair(self.wire.embedding(model, ndims))
    }

    /// The provider's `transcription` model.
    pub fn transcription(
        &self,
        model: impl Into<String>,
    ) -> Model<gemini::transcription::Transcriptions, H> {
        self.pair(self.wire.transcription(model))
    }

    /// The provider's `image_generation` model.
    pub fn image_generation(
        &self,
        model: impl Into<String>,
    ) -> Model<gemini::image_generation::Images, H> {
        self.pair(self.wire.image_generation(model))
    }

    /// The provider's `models` model.
    pub fn models(&self) -> Model<gemini::model_listing::Models, H> {
        self.pair(self.wire.models())
    }

    /// The provider's `verify` model.
    pub fn verify(&self) -> Model<gemini::model_listing::VerifyKey, H> {
        self.pair(self.wire.verify())
    }

    /// The provider's `interactions_models` model.
    pub fn interactions_models(&self) -> Model<gemini::model_listing::InteractionsModels, H> {
        self.pair(self.wire.interactions_models())
    }

    /// The provider's `cached_contents` model.
    pub fn cached_contents(&self) -> Model<gemini::cached_content::CachedContents, H> {
        self.pair(self.wire.cached_contents())
    }
}

impl<H> Endpoint<gemini::Gemini, H>
where
    H: Transport<gemini::completion::GenerateContent>,
{
    /// An agent over the provider's completion model.
    pub fn agent(&self, model: impl Into<String>) -> AgentBuilder {
        AgentBuilder::new(self.completion(model))
    }

    /// A typed extractor over the provider's completion model.
    pub fn extractor<T>(&self, model: impl Into<String>) -> ExtractorBuilder<T>
    where
        T: schemars::JsonSchema
            + serde::de::DeserializeOwned
            + serde::Serialize
            + Send
            + Sync
            + 'static,
    {
        ExtractorBuilder::new(self.completion(model))
    }
}

impl<H> Endpoint<gemini::Gemini, H>
where
    H: Transport<gemini::embedding::Embeddings>,
{
    /// An embeddings builder over the provider's `model`.
    pub fn embeddings<D: rig_core::Embed>(
        &self,
        model: impl Into<String>,
    ) -> EmbeddingsBuilder<Model<gemini::embedding::Embeddings, H>, D> {
        EmbeddingsBuilder::new(self.embedding(model, None))
    }

    /// An embeddings builder over the provider's `model` at `ndims`.
    pub fn embeddings_with_ndims<D: rig_core::Embed>(
        &self,
        model: impl Into<String>,
        ndims: usize,
    ) -> EmbeddingsBuilder<Model<gemini::embedding::Embeddings, H>, D> {
        EmbeddingsBuilder::new(self.embedding(model, Some(ndims)))
    }
}

impl<H: Clone> Endpoint<cohere::wire::Cohere, H> {
    /// The provider's `completion` model.
    pub fn completion(&self, model: impl Into<String>) -> Model<cohere::Chat, H> {
        self.pair(self.wire.completion(model))
    }

    /// The provider's `embedding` model.
    pub fn embedding(
        &self,
        model: impl Into<String>,
        ndims: Option<usize>,
    ) -> Model<cohere::wire::Embeddings, H> {
        self.pair(self.wire.embedding(model, ndims))
    }

    /// The provider's `image_embedding` model.
    pub fn image_embedding(&self) -> Model<cohere::wire::ImageEmbeddings, H> {
        self.pair(self.wire.image_embedding())
    }
}

impl<H> Endpoint<cohere::wire::Cohere, H>
where
    H: Transport<cohere::Chat>,
{
    /// An agent over the provider's completion model.
    pub fn agent(&self, model: impl Into<String>) -> AgentBuilder {
        AgentBuilder::new(self.completion(model))
    }

    /// A typed extractor over the provider's completion model.
    pub fn extractor<T>(&self, model: impl Into<String>) -> ExtractorBuilder<T>
    where
        T: schemars::JsonSchema
            + serde::de::DeserializeOwned
            + serde::Serialize
            + Send
            + Sync
            + 'static,
    {
        ExtractorBuilder::new(self.completion(model))
    }
}

impl<H> Endpoint<cohere::wire::Cohere, H>
where
    H: Transport<cohere::wire::Embeddings>,
{
    /// An embeddings builder over the provider's `model`.
    pub fn embeddings<D: rig_core::Embed>(
        &self,
        model: impl Into<String>,
    ) -> EmbeddingsBuilder<Model<cohere::wire::Embeddings, H>, D> {
        EmbeddingsBuilder::new(self.embedding(model, None))
    }

    /// An embeddings builder over the provider's `model` at `ndims`.
    pub fn embeddings_with_ndims<D: rig_core::Embed>(
        &self,
        model: impl Into<String>,
        ndims: usize,
    ) -> EmbeddingsBuilder<Model<cohere::wire::Embeddings, H>, D> {
        EmbeddingsBuilder::new(self.embedding(model, Some(ndims)))
    }
}

impl<H: Clone> Endpoint<ollama::wire::Ollama, H> {
    /// The provider's `completion` model.
    pub fn completion(&self, model: impl Into<String>) -> Model<ollama::wire::Chat, H> {
        self.pair(self.wire.completion(model))
    }

    /// The provider's `embedding` model.
    pub fn embedding(
        &self,
        model: impl Into<String>,
        ndims: Option<usize>,
    ) -> Model<ollama::wire::Embeddings, H> {
        self.pair(self.wire.embedding(model, ndims))
    }

    /// The provider's `models` model.
    pub fn models(&self) -> Model<ollama::wire::Models, H> {
        self.pair(self.wire.models())
    }
}

impl<H> Endpoint<ollama::wire::Ollama, H>
where
    H: Transport<ollama::wire::Chat>,
{
    /// An agent over the provider's completion model.
    pub fn agent(&self, model: impl Into<String>) -> AgentBuilder {
        AgentBuilder::new(self.completion(model))
    }

    /// A typed extractor over the provider's completion model.
    pub fn extractor<T>(&self, model: impl Into<String>) -> ExtractorBuilder<T>
    where
        T: schemars::JsonSchema
            + serde::de::DeserializeOwned
            + serde::Serialize
            + Send
            + Sync
            + 'static,
    {
        ExtractorBuilder::new(self.completion(model))
    }
}

impl<H> Endpoint<ollama::wire::Ollama, H>
where
    H: Transport<ollama::wire::Embeddings>,
{
    /// An embeddings builder over the provider's `model`.
    pub fn embeddings<D: rig_core::Embed>(
        &self,
        model: impl Into<String>,
    ) -> EmbeddingsBuilder<Model<ollama::wire::Embeddings, H>, D> {
        EmbeddingsBuilder::new(self.embedding(model, None))
    }

    /// An embeddings builder over the provider's `model` at `ndims`.
    pub fn embeddings_with_ndims<D: rig_core::Embed>(
        &self,
        model: impl Into<String>,
        ndims: usize,
    ) -> EmbeddingsBuilder<Model<ollama::wire::Embeddings, H>, D> {
        EmbeddingsBuilder::new(self.embedding(model, Some(ndims)))
    }
}

impl<H: Clone> Endpoint<copilot::wire::Copilot, H> {
    /// The provider's `completion` model.
    pub fn completion(&self, model: impl Into<String>) -> Model<copilot::wire::CopilotWire, H> {
        self.pair(self.wire.completion(model))
    }

    /// The provider's `embedding` model.
    pub fn embedding(
        &self,
        model: impl Into<String>,
        ndims: Option<usize>,
    ) -> Model<copilot::wire::Embeddings, H> {
        self.pair(self.wire.embedding(model, ndims))
    }

    /// The provider's `models` model.
    pub fn models(&self) -> Model<copilot::wire::Models, H> {
        self.pair(self.wire.models())
    }
}

impl<H> Endpoint<copilot::wire::Copilot, H>
where
    H: Transport<copilot::wire::CopilotWire>,
{
    /// An agent over the provider's completion model.
    pub fn agent(&self, model: impl Into<String>) -> AgentBuilder {
        AgentBuilder::new(self.completion(model))
    }

    /// A typed extractor over the provider's completion model.
    pub fn extractor<T>(&self, model: impl Into<String>) -> ExtractorBuilder<T>
    where
        T: schemars::JsonSchema
            + serde::de::DeserializeOwned
            + serde::Serialize
            + Send
            + Sync
            + 'static,
    {
        ExtractorBuilder::new(self.completion(model))
    }
}

impl<H> Endpoint<copilot::wire::Copilot, H>
where
    H: Transport<copilot::wire::Embeddings>,
{
    /// An embeddings builder over the provider's `model`.
    pub fn embeddings<D: rig_core::Embed>(
        &self,
        model: impl Into<String>,
    ) -> EmbeddingsBuilder<Model<copilot::wire::Embeddings, H>, D> {
        EmbeddingsBuilder::new(self.embedding(model, None))
    }

    /// An embeddings builder over the provider's `model` at `ndims`.
    pub fn embeddings_with_ndims<D: rig_core::Embed>(
        &self,
        model: impl Into<String>,
        ndims: usize,
    ) -> EmbeddingsBuilder<Model<copilot::wire::Embeddings, H>, D> {
        EmbeddingsBuilder::new(self.embedding(model, Some(ndims)))
    }
}

impl<H: Clone> Endpoint<voyageai::wire::VoyageAi, H> {
    /// The provider's `embedding` model.
    pub fn embedding(
        &self,
        model: impl Into<String>,
        ndims: Option<usize>,
    ) -> Model<voyageai::wire::Embeddings, H> {
        self.pair(self.wire.embedding(model, ndims))
    }

    /// The provider's `rerank` model.
    pub fn rerank(&self, model: impl Into<String>) -> Model<voyageai::wire::Rerank, H> {
        self.pair(self.wire.rerank(model))
    }
}

impl<H> Endpoint<voyageai::wire::VoyageAi, H>
where
    H: Transport<voyageai::wire::Embeddings>,
{
    /// An embeddings builder over the provider's `model`.
    pub fn embeddings<D: rig_core::Embed>(
        &self,
        model: impl Into<String>,
    ) -> EmbeddingsBuilder<Model<voyageai::wire::Embeddings, H>, D> {
        EmbeddingsBuilder::new(self.embedding(model, None))
    }

    /// An embeddings builder over the provider's `model` at `ndims`.
    pub fn embeddings_with_ndims<D: rig_core::Embed>(
        &self,
        model: impl Into<String>,
        ndims: usize,
    ) -> EmbeddingsBuilder<Model<voyageai::wire::Embeddings, H>, D> {
        EmbeddingsBuilder::new(self.embedding(model, Some(ndims)))
    }
}

/// `model` with its wire transformed by `map`, over the same transport.
pub fn map_wire<W, V, H>(model: Model<W, H>, map: impl FnOnce(W) -> V) -> Model<V, H> {
    Model::new(map(model.wire), model.transport)
}
