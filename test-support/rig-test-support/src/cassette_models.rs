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
    pub config: openai::OpenAIConfig,
    /// The transport every model sends through.
    pub http: DynHttpClient,
}

impl OpenAiModels {
    /// `config`'s models, sent through `http`.
    pub fn new(config: openai::OpenAIConfig, http: impl HttpClientExt + 'static) -> Self {
        Self {
            config,
            http: DynHttpClient::new(http),
        }
    }

    /// The client these models are built from.
    fn client(&self) -> openai::OpenAI {
        self.config.clone().connect(self.http.clone())
    }

    /// Official OpenAI from the environment, on the shared reqwest client:
    /// what a live cell calls.
    pub fn from_env() -> Result<Self, EnvError> {
        Self::from_env_for(&openai::wire::OPENAI)
    }

    /// `dialect` from the environment, on the shared reqwest client.
    pub fn from_env_for(dialect: &openai::wire::Dialect) -> Result<Self, EnvError> {
        Ok(Self::new(
            openai::OpenAIConfig::from_env_with(dialect)?,
            rig_reqwest::shared(),
        ))
    }

    /// The same models with the configuration `f` returns.
    pub fn map_config(self, f: impl FnOnce(openai::OpenAIConfig) -> openai::OpenAIConfig) -> Self {
        Self {
            config: f(self.config),
            http: self.http,
        }
    }

    /// The completion model for `model`, on the configuration's route.
    pub fn completion(&self, model: impl Into<String>) -> Model<openai::wire::OpenAiWire> {
        self.client().completion(model)
    }

    /// The Responses model for `model`.
    pub fn responses(
        &self,
        model: impl Into<String>,
    ) -> Model<openai::responses_api::wire::Responses> {
        self.client().responses(model)
    }

    /// The Chat Completions model for `model`.
    pub fn chat(&self, model: impl Into<String>) -> Model<openai::wire::Chat> {
        self.client().chat(model)
    }

    /// The embedding model for `model`.
    pub fn embedding(&self, model: impl Into<String>) -> Model<openai::wire::Embeddings> {
        self.client().embedding(model)
    }

    /// The rerank model for `model`.
    pub fn rerank(&self, model: impl Into<String>) -> Model<openai::wire::Rerank> {
        self.client().rerank(model)
    }

    /// The transcription model for `model`.
    pub fn transcription(&self, model: impl Into<String>) -> Model<openai::wire::Transcriptions> {
        self.client().transcription(model)
    }

    /// The image-generation model for `model`.
    pub fn image_generation(&self, model: impl Into<String>) -> Model<openai::wire::Images> {
        self.client().image_generation(model)
    }

    /// The speech model for `model`.
    pub fn audio_generation(&self, model: impl Into<String>) -> Model<openai::wire::Speech> {
        self.client().audio_generation(model)
    }

    /// The provider's model listing.
    pub async fn list_models(&self) -> Result<ModelList, ProviderError> {
        self.client().list_models().await
    }

    /// The provider's credential check.
    pub async fn verify(&self) -> Result<(), ProviderError> {
        self.client().verify().await
    }
}

/// A Messages-format provider's models, sent through one transport.
#[derive(Clone, Debug)]
pub struct AnthropicModels {
    /// The provider configuration.
    pub config: anthropic::AnthropicConfig,
    /// The transport every model sends through.
    pub http: DynHttpClient,
}

impl AnthropicModels {
    /// `config`'s models, sent through `http`.
    pub fn new(config: anthropic::AnthropicConfig, http: impl HttpClientExt + 'static) -> Self {
        Self {
            config,
            http: DynHttpClient::new(http),
        }
    }

    /// The client these models are built from.
    fn client(&self) -> anthropic::Anthropic {
        self.config.clone().connect(self.http.clone())
    }

    /// The provider from the environment, on the shared reqwest client:
    /// what a live cell calls.
    pub fn from_env() -> Result<Self, EnvError> {
        Ok(Self::new(
            anthropic::AnthropicConfig::from_env()?,
            rig_reqwest::shared(),
        ))
    }

    /// The same models with the configuration `f` returns.
    pub fn map_config(
        self,
        f: impl FnOnce(anthropic::AnthropicConfig) -> anthropic::AnthropicConfig,
    ) -> Self {
        Self {
            config: f(self.config),
            http: self.http,
        }
    }

    /// The Messages model for `model`.
    pub fn completion(&self, model: impl Into<String>) -> Model<anthropic::Messages> {
        self.client().completion(model)
    }

    /// The provider's model listing.
    pub async fn list_models(&self) -> Result<ModelList, ProviderError> {
        self.client().list_models().await
    }

    /// The provider's credential check.
    pub async fn verify(&self) -> Result<(), ProviderError> {
        self.client().verify().await
    }
}

/// Gemini's models, sent through one transport.
#[derive(Clone, Debug)]
pub struct GeminiModels {
    /// The provider configuration.
    pub config: gemini::GeminiConfig,
    /// The transport every model sends through.
    pub http: DynHttpClient,
}

impl GeminiModels {
    /// `config`'s models, sent through `http`.
    pub fn new(config: gemini::GeminiConfig, http: impl HttpClientExt + 'static) -> Self {
        Self {
            config,
            http: DynHttpClient::new(http),
        }
    }

    /// The client these models are built from.
    fn client(&self) -> gemini::Gemini {
        self.config.clone().connect(self.http.clone())
    }

    /// The provider from the environment, on the shared reqwest client:
    /// what a live cell calls.
    pub fn from_env() -> Result<Self, EnvError> {
        Ok(Self::new(
            gemini::GeminiConfig::from_env()?,
            rig_reqwest::shared(),
        ))
    }

    /// The GenerateContent model for `model`.
    pub fn completion(
        &self,
        model: impl Into<String>,
    ) -> Model<gemini::completion::GenerateContent> {
        self.client().completion(model)
    }

    /// The Interactions model for `model`.
    pub fn interactions(
        &self,
        model: impl Into<String>,
    ) -> Model<gemini::interactions_api::Interactions> {
        self.client().interactions(model)
    }

    /// The model that reads the interaction `interaction_id`.
    pub fn interaction(
        &self,
        interaction_id: impl Into<String>,
    ) -> Model<gemini::interactions_api::InteractionResume> {
        self.client().interaction(interaction_id)
    }

    /// The model that resumes the interaction `interaction_id` after
    /// `last_event_id`.
    pub fn interaction_resumed(
        &self,
        interaction_id: impl Into<String>,
        last_event_id: Option<&str>,
    ) -> Model<gemini::interactions_api::InteractionResume> {
        self.client()
            .interaction_resumed(interaction_id, last_event_id)
    }

    /// The embedding model for `model`.
    pub fn embedding(&self, model: impl Into<String>) -> Model<gemini::embedding::Embeddings> {
        self.client().embedding(model)
    }

    /// The transcription model for `model`.
    pub fn transcription(
        &self,
        model: impl Into<String>,
    ) -> Model<gemini::transcription::Transcriptions> {
        self.client().transcription(model)
    }

    /// The image-generation model for `model`.
    pub fn image_generation(
        &self,
        model: impl Into<String>,
    ) -> Model<gemini::image_generation::Images> {
        self.client().image_generation(model)
    }

    /// The explicit context cache.
    pub fn cached_contents(&self) -> Model<gemini::CachedContents> {
        self.client().cached_contents()
    }

    /// The provider's model listing.
    pub async fn list_models(&self) -> Result<ModelList, ProviderError> {
        self.client().list_models().await
    }

    /// The provider's credential check.
    pub async fn verify(&self) -> Result<(), ProviderError> {
        self.client().verify().await
    }
}

/// Cohere's models, sent through one transport.
#[derive(Clone, Debug)]
pub struct CohereModels {
    /// The provider configuration.
    pub config: cohere::CohereConfig,
    /// The transport every model sends through.
    pub http: DynHttpClient,
}

impl CohereModels {
    /// `config`'s models, sent through `http`.
    pub fn new(config: cohere::CohereConfig, http: impl HttpClientExt + 'static) -> Self {
        Self {
            config,
            http: DynHttpClient::new(http),
        }
    }

    /// The client these models are built from.
    fn client(&self) -> cohere::Cohere {
        self.config.clone().connect(self.http.clone())
    }

    /// The provider from the environment, on the shared reqwest client:
    /// what a live cell calls.
    pub fn from_env() -> Result<Self, EnvError> {
        Ok(Self::new(
            cohere::CohereConfig::from_env()?,
            rig_reqwest::shared(),
        ))
    }

    /// The chat model for `model`.
    pub fn completion(&self, model: impl Into<String>) -> Model<cohere::Chat> {
        self.client().completion(model)
    }

    /// The text-embedding model for `model`.
    pub fn embedding(&self, model: impl Into<String>) -> Model<cohere::Embeddings> {
        self.client().embedding(model)
    }

    /// The image-embedding model.
    pub fn image_embedding(&self) -> Model<cohere::ImageEmbeddings> {
        self.client().image_embedding()
    }
}

/// An Ollama daemon's models, sent through one transport.
#[derive(Clone, Debug)]
pub struct OllamaModels {
    /// The provider configuration.
    pub config: ollama::OllamaConfig,
    /// The transport every model sends through.
    pub http: DynHttpClient,
}

impl OllamaModels {
    /// `config`'s models, sent through `http`.
    pub fn new(config: ollama::OllamaConfig, http: impl HttpClientExt + 'static) -> Self {
        Self {
            config,
            http: DynHttpClient::new(http),
        }
    }

    /// The client these models are built from.
    fn client(&self) -> ollama::Ollama {
        self.config.clone().connect(self.http.clone())
    }

    /// The provider from the environment, on the shared reqwest client:
    /// what a live cell calls.
    pub fn from_env() -> Result<Self, EnvError> {
        Ok(Self::new(
            ollama::OllamaConfig::from_env()?,
            rig_reqwest::shared(),
        ))
    }

    /// The chat model for `model`.
    pub fn completion(&self, model: impl Into<String>) -> Model<ollama::Chat> {
        self.client().completion(model)
    }

    /// The embedding model for `model`.
    pub fn embedding(&self, model: impl Into<String>) -> Model<ollama::Embeddings> {
        self.client().embedding(model)
    }

    /// The daemon's model listing.
    pub async fn list_models(&self) -> Result<ModelList, ProviderError> {
        self.client().list_models().await
    }
}

/// Copilot's models, sent through one transport.
#[derive(Clone, Debug)]
pub struct CopilotModels {
    /// The provider configuration.
    pub config: copilot::CopilotConfig,
    /// The transport every model sends through.
    pub http: DynHttpClient,
}

impl CopilotModels {
    /// `config`'s models, sent through `http`.
    pub fn new(config: copilot::CopilotConfig, http: impl HttpClientExt + 'static) -> Self {
        Self {
            config,
            http: DynHttpClient::new(http),
        }
    }

    /// The client these models are built from.
    fn client(&self) -> copilot::Copilot {
        self.config.clone().connect(self.http.clone())
    }

    /// The provider from the environment, on the shared reqwest client:
    /// what a live cell calls.
    pub fn from_env() -> Result<Self, EnvError> {
        Ok(Self::new(
            copilot::CopilotConfig::from_env()?,
            rig_reqwest::shared(),
        ))
    }

    /// The completion model for `model`.
    pub fn completion(&self, model: impl Into<String>) -> Model<copilot::wire::CopilotWire> {
        self.client().completion(model)
    }

    /// The embedding model for `model`.
    pub fn embedding(&self, model: impl Into<String>) -> Model<copilot::wire::Embeddings> {
        self.client().embedding(model)
    }

    /// The session's model listing.
    pub async fn list_models(&self) -> Result<ModelList, ProviderError> {
        self.client().list_models().await
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
