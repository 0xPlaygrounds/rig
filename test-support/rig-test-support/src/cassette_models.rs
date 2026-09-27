//! What a cassette suite builds its models from: one provider configuration
//! on the cassette's transport. A suite's tests build every model through
//! these types and never spell a provider constructor, so a change to how a
//! provider builds its models edits this file alone.

use rig_core::client::env::EnvError;
use rig_core::driver::Model;
use rig_core::error::ProviderError;
use rig_core::http_client::{DynHttpClient, HttpClientExt};
use rig_core::model::ModelList;
use rig_core::providers::{anthropic, cohere, copilot, gemini, ollama, openai};

/// An OpenAI-shaped provider's models, sent through one transport.
#[derive(Clone, Debug)]
pub struct OpenAiModels {
    /// The provider configuration.
    pub config: openai::OpenAI,
    /// The transport every model sends through.
    pub http: DynHttpClient,
}

impl OpenAiModels {
    /// `config`'s models, sent through `http`.
    pub fn new(config: openai::OpenAI, http: impl HttpClientExt + 'static) -> Self {
        Self {
            config,
            http: DynHttpClient::new(http),
        }
    }

    /// Official OpenAI from the environment, on the shared reqwest client:
    /// what a live cell calls.
    pub fn from_env() -> Result<Self, EnvError> {
        Self::from_env_with(&openai::wire::OPENAI)
    }

    /// `dialect` from the environment, on the shared reqwest client.
    pub fn from_env_with(dialect: &openai::wire::Dialect) -> Result<Self, EnvError> {
        Ok(Self::new(
            openai::OpenAI::from_env_with(dialect)?,
            rig_reqwest::shared(),
        ))
    }

    /// The same models with the configuration `f` returns.
    pub fn map_config(self, f: impl FnOnce(openai::OpenAI) -> openai::OpenAI) -> Self {
        Self {
            config: f(self.config),
            http: self.http,
        }
    }

    /// The completion model for `model`, on the configuration's route.
    pub fn completion(&self, model: impl Into<String>) -> Model<openai::wire::OpenAiWire> {
        Model::new(self.config.completion(model), self.http.clone())
    }

    /// The Responses model for `model`.
    pub fn responses(
        &self,
        model: impl Into<String>,
    ) -> Model<openai::responses_api::wire::Responses> {
        Model::new(self.config.responses(model), self.http.clone())
    }

    /// The Chat Completions model for `model`.
    pub fn chat(&self, model: impl Into<String>) -> Model<openai::wire::Chat> {
        Model::new(self.config.chat(model), self.http.clone())
    }

    /// The embedding model for `model`.
    pub fn embedding(
        &self,
        model: impl Into<String>,
        ndims: Option<usize>,
    ) -> Model<openai::wire::Embeddings> {
        Model::new(self.config.embedding(model, ndims), self.http.clone())
    }

    /// The rerank model for `model`.
    pub fn rerank(&self, model: impl Into<String>) -> Model<openai::wire::Rerank> {
        Model::new(self.config.rerank(model), self.http.clone())
    }

    /// The transcription model for `model`.
    pub fn transcription(&self, model: impl Into<String>) -> Model<openai::wire::Transcriptions> {
        Model::new(self.config.transcription(model), self.http.clone())
    }

    /// The image-generation model for `model`.
    pub fn image_generation(&self, model: impl Into<String>) -> Model<openai::wire::Images> {
        Model::new(self.config.image_generation(model), self.http.clone())
    }

    /// The speech model for `model`.
    pub fn audio_generation(&self, model: impl Into<String>) -> Model<openai::wire::Speech> {
        Model::new(self.config.audio_generation(model), self.http.clone())
    }

    /// The provider's model listing.
    pub async fn list_models(&self) -> Result<ModelList, ProviderError> {
        Model::new(self.config.models(), self.http.clone())
            .call(())
            .await
    }

    /// The provider's credential check.
    pub async fn verify(&self) -> Result<(), ProviderError> {
        Model::new(self.config.verify(), self.http.clone())
            .verify()
            .await
    }
}

/// A Messages-format provider's models, sent through one transport.
#[derive(Clone, Debug)]
pub struct AnthropicModels {
    /// The provider configuration.
    pub config: anthropic::wire::Anthropic,
    /// The transport every model sends through.
    pub http: DynHttpClient,
}

impl AnthropicModels {
    /// `config`'s models, sent through `http`.
    pub fn new(config: anthropic::wire::Anthropic, http: impl HttpClientExt + 'static) -> Self {
        Self {
            config,
            http: DynHttpClient::new(http),
        }
    }

    /// The provider from the environment, on the shared reqwest client:
    /// what a live cell calls.
    pub fn from_env() -> Result<Self, EnvError> {
        Ok(Self::new(
            anthropic::wire::Anthropic::from_env()?,
            rig_reqwest::shared(),
        ))
    }

    /// The same models with the configuration `f` returns.
    pub fn map_config(
        self,
        f: impl FnOnce(anthropic::wire::Anthropic) -> anthropic::wire::Anthropic,
    ) -> Self {
        Self {
            config: f(self.config),
            http: self.http,
        }
    }

    /// The Messages model for `model`.
    pub fn completion(&self, model: impl Into<String>) -> Model<anthropic::Messages> {
        Model::new(self.config.completion(model), self.http.clone())
    }

    /// The provider's model listing.
    pub async fn list_models(&self) -> Result<ModelList, ProviderError> {
        Model::new(self.config.models(), self.http.clone())
            .call(())
            .await
    }

    /// The provider's credential check.
    pub async fn verify(&self) -> Result<(), ProviderError> {
        Model::new(self.config.verify(), self.http.clone())
            .verify()
            .await
    }
}

/// Gemini's models, sent through one transport.
#[derive(Clone, Debug)]
pub struct GeminiModels {
    /// The provider configuration.
    pub config: gemini::Gemini,
    /// The transport every model sends through.
    pub http: DynHttpClient,
}

impl GeminiModels {
    /// `config`'s models, sent through `http`.
    pub fn new(config: gemini::Gemini, http: impl HttpClientExt + 'static) -> Self {
        Self {
            config,
            http: DynHttpClient::new(http),
        }
    }

    /// The provider from the environment, on the shared reqwest client:
    /// what a live cell calls.
    pub fn from_env() -> Result<Self, EnvError> {
        Ok(Self::new(
            gemini::Gemini::from_env()?,
            rig_reqwest::shared(),
        ))
    }

    /// The GenerateContent model for `model`.
    pub fn completion(
        &self,
        model: impl Into<String>,
    ) -> Model<gemini::completion::GenerateContent> {
        Model::new(self.config.completion(model), self.http.clone())
    }

    /// The Interactions model for `model`.
    pub fn interactions(
        &self,
        model: impl Into<String>,
    ) -> Model<gemini::interactions_api::Interactions> {
        Model::new(self.config.interactions(model), self.http.clone())
    }

    /// The model that reads the interaction `interaction_id`.
    pub fn interaction(
        &self,
        interaction_id: impl Into<String>,
    ) -> Model<gemini::interactions_api::InteractionResume> {
        Model::new(self.config.interaction(interaction_id), self.http.clone())
    }

    /// The model that resumes the interaction `interaction_id` after
    /// `last_event_id`.
    pub fn interaction_resumed(
        &self,
        interaction_id: impl Into<String>,
        last_event_id: Option<&str>,
    ) -> Model<gemini::interactions_api::InteractionResume> {
        Model::new(
            self.config
                .interaction_resumed(interaction_id, last_event_id),
            self.http.clone(),
        )
    }

    /// The embedding model for `model`.
    pub fn embedding(
        &self,
        model: impl Into<String>,
        ndims: Option<usize>,
    ) -> Model<gemini::embedding::Embeddings> {
        Model::new(self.config.embedding(model, ndims), self.http.clone())
    }

    /// The transcription model for `model`.
    pub fn transcription(
        &self,
        model: impl Into<String>,
    ) -> Model<gemini::transcription::Transcriptions> {
        Model::new(self.config.transcription(model), self.http.clone())
    }

    /// The image-generation model for `model`.
    pub fn image_generation(
        &self,
        model: impl Into<String>,
    ) -> Model<gemini::image_generation::Images> {
        Model::new(self.config.image_generation(model), self.http.clone())
    }

    /// The explicit context cache.
    pub fn cached_contents(&self) -> Model<gemini::CachedContents> {
        Model::new(self.config.cached_contents(), self.http.clone())
    }

    /// The provider's model listing.
    pub async fn list_models(&self) -> Result<ModelList, ProviderError> {
        Model::new(self.config.models(), self.http.clone())
            .call(())
            .await
    }

    /// The provider's credential check.
    pub async fn verify(&self) -> Result<(), ProviderError> {
        Model::new(self.config.verify(), self.http.clone())
            .verify()
            .await
    }
}

/// Cohere's models, sent through one transport.
#[derive(Clone, Debug)]
pub struct CohereModels {
    /// The provider configuration.
    pub config: cohere::Cohere,
    /// The transport every model sends through.
    pub http: DynHttpClient,
}

impl CohereModels {
    /// `config`'s models, sent through `http`.
    pub fn new(config: cohere::Cohere, http: impl HttpClientExt + 'static) -> Self {
        Self {
            config,
            http: DynHttpClient::new(http),
        }
    }

    /// The provider from the environment, on the shared reqwest client:
    /// what a live cell calls.
    pub fn from_env() -> Result<Self, EnvError> {
        Ok(Self::new(
            cohere::Cohere::from_env()?,
            rig_reqwest::shared(),
        ))
    }

    /// The chat model for `model`.
    pub fn completion(&self, model: impl Into<String>) -> Model<cohere::Chat> {
        Model::new(self.config.completion(model), self.http.clone())
    }

    /// The text-embedding model for `model`.
    pub fn embedding(
        &self,
        model: impl Into<String>,
        ndims: Option<usize>,
    ) -> Model<cohere::Embeddings> {
        Model::new(self.config.embedding(model, ndims), self.http.clone())
    }

    /// The image-embedding model.
    pub fn image_embedding(&self) -> Model<cohere::ImageEmbeddings> {
        Model::new(self.config.image_embedding(), self.http.clone())
    }
}

/// An Ollama daemon's models, sent through one transport.
#[derive(Clone, Debug)]
pub struct OllamaModels {
    /// The provider configuration.
    pub config: ollama::Ollama,
    /// The transport every model sends through.
    pub http: DynHttpClient,
}

impl OllamaModels {
    /// `config`'s models, sent through `http`.
    pub fn new(config: ollama::Ollama, http: impl HttpClientExt + 'static) -> Self {
        Self {
            config,
            http: DynHttpClient::new(http),
        }
    }

    /// The provider from the environment, on the shared reqwest client:
    /// what a live cell calls.
    pub fn from_env() -> Result<Self, EnvError> {
        Ok(Self::new(
            ollama::Ollama::from_env()?,
            rig_reqwest::shared(),
        ))
    }

    /// The chat model for `model`.
    pub fn completion(&self, model: impl Into<String>) -> Model<ollama::Chat> {
        Model::new(self.config.completion(model), self.http.clone())
    }

    /// The embedding model for `model`.
    pub fn embedding(
        &self,
        model: impl Into<String>,
        ndims: Option<usize>,
    ) -> Model<ollama::Embeddings> {
        Model::new(self.config.embedding(model, ndims), self.http.clone())
    }

    /// The daemon's model listing.
    pub async fn list_models(&self) -> Result<ModelList, ProviderError> {
        Model::new(self.config.models(), self.http.clone())
            .call(())
            .await
    }
}

/// Copilot's models, sent through one transport.
#[derive(Clone, Debug)]
pub struct CopilotModels {
    /// The provider configuration.
    pub config: copilot::wire::Copilot,
    /// The transport every model sends through.
    pub http: DynHttpClient,
}

impl CopilotModels {
    /// `config`'s models, sent through `http`.
    pub fn new(config: copilot::wire::Copilot, http: impl HttpClientExt + 'static) -> Self {
        Self {
            config,
            http: DynHttpClient::new(http),
        }
    }

    /// The provider from the environment, on the shared reqwest client:
    /// what a live cell calls.
    pub fn from_env() -> Result<Self, EnvError> {
        Ok(Self::new(
            copilot::wire::Copilot::from_env()?,
            rig_reqwest::shared(),
        ))
    }

    /// The completion model for `model`.
    pub fn completion(&self, model: impl Into<String>) -> Model<copilot::wire::CopilotWire> {
        Model::new(self.config.completion(model), self.http.clone())
    }

    /// The embedding model for `model`.
    pub fn embedding(
        &self,
        model: impl Into<String>,
        ndims: Option<usize>,
    ) -> Model<copilot::wire::Embeddings> {
        Model::new(self.config.embedding(model, ndims), self.http.clone())
    }

    /// The session's model listing.
    pub async fn list_models(&self) -> Result<ModelList, ProviderError> {
        Model::new(self.config.models(), self.http.clone())
            .call(())
            .await
    }
}

/// A model with its wire changed: how a test applies a wire option (strict
/// tools, prompt caching, a cached-content handle) to a model it built.
pub trait MapWire<W, T> {
    /// The same transport under the wire `f` returns.
    fn map_wire<V>(self, f: impl FnOnce(W) -> V) -> Model<V, T>;
}

impl<W, T> MapWire<W, T> for Model<W, T> {
    fn map_wire<V>(self, f: impl FnOnce(W) -> V) -> Model<V, T> {
        Model::new(f(self.wire), self.transport)
    }
}
