use crate::agent::Agent;
use crate::client::ProviderClient;
use crate::completion::{CompletionRequest, GetTokenUsage, Message, Usage};
use crate::embeddings::embedding::EmbeddingModelDyn;
use crate::providers::{
    anthropic, azure, cohere, deepseek, galadriel, gemini, groq, huggingface, hyperbolic, mira,
    moonshot, ollama, openai, openrouter, perplexity, together, xai,
};
use crate::streaming::StreamingCompletionResponse;
use crate::transcription::TranscriptionModelDyn;
use rig::completion::CompletionModelDyn;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::panic::{RefUnwindSafe, UnwindSafe};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ClientBuildError {
    #[error("factory error: {}", .0)]
    FactoryError(String),
    #[error("invalid id string: {}", .0)]
    InvalidIdString(String),
    #[error("unsupported feature: {} for {}", .1, .0)]
    UnsupportedFeature(String, String),
    #[error("unknown provider")]
    UnknownProvider,
}

pub type BoxCompletionModel<'a> = Box<dyn CompletionModelDyn + 'a>;
pub type BoxAgentBuilder<'a> = AgentBuilder<CompletionModelHandle<'a>>;
pub type BoxAgent<'a> = Agent<CompletionModelHandle<'a>>;
pub type BoxEmbeddingModel<'a> = Box<dyn EmbeddingModelDyn + 'a>;
pub type BoxTranscriptionModel<'a> = Box<dyn TranscriptionModelDyn + 'a>;

/// A dynamic client builder.
/// Use this when you need to support creating any kind of client from a range of LLM providers (that Rig supports).
/// Usage:
/// ```rust
/// use rig::{
///     client::builder::DynClientBuilder, completion::Prompt, providers::anthropic::CLAUDE_3_7_SONNET,
/// };
/// #[tokio::main]
/// async fn main() {
///     let multi_client = DynClientBuilder::new();
///     // set up OpenAI client
///     let completion_openai = multi_client.agent("openai", "gpt-4o").unwrap();
///     let agent_openai = completion_openai
///         .preamble("You are a helpful assistant")
///         .build();
///     // set up Anthropic client
///     let completion_anthropic = multi_client.agent("anthropic", CLAUDE_3_7_SONNET).unwrap();
///     let agent_anthropic = completion_anthropic
///         .preamble("You are a helpful assistant")
///         .max_tokens(1024)
///         .build();
///     println!("Sending prompt: 'Hello world!'");
///     let res_openai = agent_openai.prompt("Hello world!").await.unwrap();
///     println!("Response from OpenAI (using gpt-4o): {res_openai}");
///     let res_anthropic = agent_anthropic.prompt("Hello world!").await.unwrap();
///     println!("Response from Anthropic (using Claude 3.7 Sonnet): {res_anthropic}");
/// }
/// ```
pub struct DynClientBuilder {
    registry: HashMap<String, ClientFactory>,
}

impl Default for DynClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> DynClientBuilder {
    /// Generate a new instance of `DynClientBuilder`.
    /// By default, every single possible client that can be registered
    /// will be registered to the client builder.
    pub fn new() -> Self {
        Self {
            registry: HashMap::new(),
        }
        .register_all(vec![
            ClientFactory::new(
                DefaultProviders::ANTHROPIC,
                anthropic::Client::<reqwest::Client>::from_env_boxed,
                anthropic::Client::<reqwest::Client>::from_val_boxed,
            ),
            ClientFactory::new(
                DefaultProviders::COHERE,
                cohere::Client::<reqwest::Client>::from_env_boxed,
                cohere::Client::<reqwest::Client>::from_val_boxed,
            ),
            ClientFactory::new(
                DefaultProviders::GEMINI,
                gemini::Client::<reqwest::Client>::from_env_boxed,
                gemini::Client::<reqwest::Client>::from_val_boxed,
            ),
            ClientFactory::new(
                DefaultProviders::HUGGINGFACE,
                huggingface::Client::<reqwest::Client>::from_env_boxed,
                huggingface::Client::<reqwest::Client>::from_val_boxed,
            ),
            ClientFactory::new(
                DefaultProviders::OPENAI,
                openai::Client::<reqwest::Client>::from_env_boxed,
                openai::Client::<reqwest::Client>::from_val_boxed,
            ),
            ClientFactory::new(
                DefaultProviders::OPENROUTER,
                openrouter::Client::<reqwest::Client>::from_env_boxed,
                openrouter::Client::<reqwest::Client>::from_val_boxed,
            ),
            ClientFactory::new(
                DefaultProviders::TOGETHER,
                together::Client::<reqwest::Client>::from_env_boxed,
                together::Client::<reqwest::Client>::from_val_boxed,
            ),
            ClientFactory::new(
                DefaultProviders::XAI,
                xai::Client::<reqwest::Client>::from_env_boxed,
                xai::Client::<reqwest::Client>::from_val_boxed,
            ),
            ClientFactory::new(
                DefaultProviders::AZURE,
                azure::Client::<reqwest::Client>::from_env_boxed,
                azure::Client::<reqwest::Client>::from_val_boxed,
            ),
            ClientFactory::new(
                DefaultProviders::DEEPSEEK,
                deepseek::Client::<reqwest::Client>::from_env_boxed,
                deepseek::Client::<reqwest::Client>::from_val_boxed,
            ),
            ClientFactory::new(
                DefaultProviders::GALADRIEL,
                galadriel::Client::<reqwest::Client>::from_env_boxed,
                galadriel::Client::<reqwest::Client>::from_val_boxed,
            ),
            ClientFactory::new(
                DefaultProviders::GROQ,
                groq::Client::<reqwest::Client>::from_env_boxed,
                groq::Client::<reqwest::Client>::from_val_boxed,
            ),
            ClientFactory::new(
                DefaultProviders::HYPERBOLIC,
                hyperbolic::Client::<reqwest::Client>::from_env_boxed,
                hyperbolic::Client::<reqwest::Client>::from_val_boxed,
            ),
            ClientFactory::new(
                DefaultProviders::MOONSHOT,
                moonshot::Client::<reqwest::Client>::from_env_boxed,
                moonshot::Client::<reqwest::Client>::from_val_boxed,
            ),
            ClientFactory::new(
                DefaultProviders::MIRA,
                mira::Client::<reqwest::Client>::from_env_boxed,
                mira::Client::<reqwest::Client>::from_val_boxed,
            ),
            ClientFactory::new(
                DefaultProviders::MISTRAL,
                mistral::Client::<reqwest::Client>::from_env_boxed,
                mistral::Client::<reqwest::Client>::from_val_boxed,
            ),
            ClientFactory::new(
                DefaultProviders::OLLAMA,
                ollama::Client::<reqwest::Client>::from_env_boxed,
                ollama::Client::<reqwest::Client>::from_val_boxed,
            ),
            ClientFactory::new(
                DefaultProviders::PERPLEXITY,
                perplexity::Client::<reqwest::Client>::from_env_boxed,
                perplexity::Client::<reqwest::Client>::from_val_boxed,
            ),
        ])
    }

    /// Generate a new instance of `DynClientBuilder` with no client factories registered.
    pub fn empty() -> Self {
        Self {
            registry: HashMap::new(),
        }
    }

    /// Register a new ClientFactory
    pub fn register(mut self, client_factory: ClientFactory) -> Self {
        self.registry
            .insert(client_factory.name.clone(), client_factory);
        self
    }

    /// Register multiple ClientFactories
    pub fn register_all(mut self, factories: impl IntoIterator<Item = ClientFactory>) -> Self {
        for factory in factories {
            self.registry.insert(factory.name.clone(), factory);
        }

        self
    }

    /// Returns a (boxed) specific provider based on the given provider.
    pub fn build(&self, provider: &str) -> Result<Box<dyn ProviderClient>, ClientBuildError> {
        let factory = self.get_factory(provider)?;
        factory.build()
    }

    /// Returns a (boxed) specific provider based on the given provider.
    pub fn build_val(
        &self,
        provider: &str,
        provider_value: ProviderValue,
    ) -> Result<Box<dyn ProviderClient>, ClientBuildError> {
        let factory = self.get_factory(provider)?;
        factory.build_from_val(provider_value)
    }

    /// Parses a provider:model string to the provider and the model separately.
    /// For example, `openai:gpt-4o` will return ("openai", "gpt-4o").
    pub fn parse(&self, id: &'a str) -> Result<(&'a str, &'a str), ClientBuildError> {
        let (provider, model) = id
            .split_once(":")
            .ok_or(ClientBuildError::InvalidIdString(id.to_string()))?;

        Ok((provider, model))
    }

    /// Returns a specific client factory (that exists in the registry).
    fn get_factory(&self, provider: &str) -> Result<&ClientFactory, ClientBuildError> {
        self.registry
            .get(provider)
            .ok_or(ClientBuildError::UnknownProvider)
    }

    /// Get a boxed completion model based on the provider and model.
    pub fn completion(
        &self,
        provider: &str,
        model: &str,
    ) -> Result<BoxCompletionModel<'a>, ClientBuildError> {
        let client = self.build(provider)?;

        let completion = client
            .as_completion()
            .ok_or(ClientBuildError::UnsupportedFeature(
                provider.to_string(),
                "completion".to_owned(),
            ))?;

        Ok(completion.completion_model(model))
    }

    /// Get a boxed agent based on the provider and model..
    pub fn agent(
        &self,
        provider: &str,
        model: &str,
    ) -> Result<BoxAgentBuilder<'a>, ClientBuildError> {
        let client = self.build(provider)?;

        let client = client
            .as_completion()
            .ok_or(ClientBuildError::UnsupportedFeature(
                provider.to_string(),
                "completion".to_string(),
            ))?;

        Ok(client.agent(model))
    }

    /// Get a boxed agent based on the provider and model, as well as an API key.
    pub fn agent_with_api_key_val<P>(
        &self,
        provider: &str,
        model: &str,
        provider_value: P,
    ) -> Result<BoxAgentBuilder<'a>, ClientBuildError>
    where
        P: Into<ProviderValue>,
    {
        let client = self.build_val(provider, provider_value.into())?;

        let client = client
            .as_completion()
            .ok_or(ClientBuildError::UnsupportedFeature(
                provider.to_string(),
                "completion".to_string(),
            ))?;

        Ok(client.agent(model))
    }

    /// Get a boxed embedding model based on the provider and model.
    pub fn embeddings(
        &self,
        provider: &str,
        model: &str,
    ) -> Result<Box<dyn EmbeddingModelDyn + 'a>, ClientBuildError> {
        let client = self.build(provider)?;

        let embeddings = client
            .as_embeddings()
            .ok_or(ClientBuildError::UnsupportedFeature(
                provider.to_string(),
                "embeddings".to_owned(),
            ))?;

        Ok(embeddings.embedding_model(model))
    }

    /// Get a boxed embedding model based on the provider and model.
    pub fn embeddings_with_api_key_val<P>(
        &self,
        provider: &str,
        model: &str,
        provider_value: P,
    ) -> Result<Box<dyn EmbeddingModelDyn + 'a>, ClientBuildError>
    where
        P: Into<ProviderValue>,
    {
        let client = self.build_val(provider, provider_value.into())?;

        let embeddings = client
            .as_embeddings()
            .ok_or(ClientBuildError::UnsupportedFeature(
                provider.to_string(),
                "embeddings".to_owned(),
            ))?;

        Ok(embeddings.embedding_model(model))
    }

    /// Get a boxed transcription model based on the provider and model.
    pub fn transcription(
        &self,
        provider: &str,
        model: &str,
    ) -> Result<Box<dyn TranscriptionModelDyn + 'a>, ClientBuildError> {
        let client = self.build(provider)?;
        let transcription =
            client
                .as_transcription()
                .ok_or(ClientBuildError::UnsupportedFeature(
                    provider.to_string(),
                    "transcription".to_owned(),
                ))?;

        Ok(transcription.transcription_model(model))
    }

    /// Get a boxed transcription model based on the provider and model.
    pub fn transcription_with_api_key_val<P>(
        &self,
        provider: &str,
        model: &str,
        provider_value: P,
    ) -> Result<Box<dyn TranscriptionModelDyn + 'a>, ClientBuildError>
    where
        P: Into<ProviderValue>,
    {
        let client = self.build_val(provider, provider_value.into())?;
        let transcription =
            client
                .as_transcription()
                .ok_or(ClientBuildError::UnsupportedFeature(
                    provider.to_string(),
                    "transcription".to_owned(),
                ))?;

        Ok(transcription.transcription_model(model))
    }

    /// Get the ID of a provider model based on a `provider:model` ID.
    pub fn id<'id>(&'a self, id: &'id str) -> Result<ProviderModelId<'a, 'id>, ClientBuildError> {
        let (provider, model) = self.parse(id)?;

        Ok(ProviderModelId {
            builder: self,
            provider,
            model,
        })
    }

    /// Stream a completion request to the specified provider and model.
    ///
    /// # Arguments
    /// * `provider` - The name of the provider (e.g., "openai", "anthropic")
    /// * `model` - The name of the model (e.g., "gpt-4o", "claude-3-sonnet")
    /// * `request` - The completion request containing prompt, parameters, etc.
    ///
    /// # Returns
    /// A future that resolves to a streaming completion response
    pub async fn stream_completion(
        &self,
        provider: &str,
        model: &str,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<FinalCompletionResponse>, ClientBuildError> {
        let client = self.build(provider)?;
        let completion = client
            .as_completion()
            .ok_or(ClientBuildError::UnsupportedFeature(
                provider.to_string(),
                "completion".to_string(),
            ))?;

        let model = completion.completion_model(model);
        model
            .stream(request)
            .await
            .map_err(|e| ClientBuildError::FactoryError(e.to_string()))
    }

    /// Stream a simple prompt to the specified provider and model.
    ///
    /// # Arguments
    /// * `provider` - The name of the provider (e.g., "openai", "anthropic")
    /// * `model` - The name of the model (e.g., "gpt-4o", "claude-3-sonnet")
    /// * `prompt` - The prompt to send to the model
    ///
    /// # Returns
    /// A future that resolves to a streaming completion response
    pub async fn stream_prompt(
        &self,
        provider: &str,
        model: &str,
        prompt: impl Into<Message> + Send,
    ) -> Result<StreamingCompletionResponse<FinalCompletionResponse>, ClientBuildError> {
        let client = self.build(provider)?;
        let completion = client
            .as_completion()
            .ok_or(ClientBuildError::UnsupportedFeature(
                provider.to_string(),
                "completion".to_string(),
            ))?;

        let model = completion.completion_model(model);
        let request = CompletionRequest {
            preamble: None,
            tools: vec![],
            documents: vec![],
            temperature: None,
            max_tokens: None,
            additional_params: None,
            tool_choice: None,
            chat_history: crate::OneOrMany::one(prompt.into()),
        };

        model
            .stream(request)
            .await
            .map_err(|e| ClientBuildError::FactoryError(e.to_string()))
    }

    /// Stream a chat with history to the specified provider and model.
    ///
    /// # Arguments
    /// * `provider` - The name of the provider (e.g., "openai", "anthropic")
    /// * `model` - The name of the model (e.g., "gpt-4o", "claude-3-sonnet")
    /// * `prompt` - The new prompt to send to the model
    /// * `chat_history` - The chat history to include with the request
    ///
    /// # Returns
    /// A future that resolves to a streaming completion response
    pub async fn stream_chat(
        &self,
        provider: &str,
        model: &str,
        prompt: impl Into<Message> + Send,
        chat_history: Vec<Message>,
    ) -> Result<StreamingCompletionResponse<FinalCompletionResponse>, ClientBuildError> {
        let client = self.build(provider)?;
        let completion = client
            .as_completion()
            .ok_or(ClientBuildError::UnsupportedFeature(
                provider.to_string(),
                "completion".to_string(),
            ))?;

        let model = completion.completion_model(model);
        let mut history = chat_history;
        history.push(prompt.into());

        let request = CompletionRequest {
            preamble: None,
            tools: vec![],
            documents: vec![],
            temperature: None,
            max_tokens: None,
            additional_params: None,
            tool_choice: None,
            chat_history: crate::OneOrMany::many(history)
                .unwrap_or_else(|_| crate::OneOrMany::one(Message::user(""))),
        };

        model
            .stream(request)
            .await
            .map_err(|e| ClientBuildError::FactoryError(e.to_string()))
    }
}

/// The final streaming response from a dynamic client.
#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct FinalCompletionResponse {
    pub usage: Option<Usage>,
}

impl GetTokenUsage for FinalCompletionResponse {
    fn token_usage(&self) -> Option<Usage> {
        self.usage
    }
}

pub struct ProviderModelId<'builder, 'id> {
    builder: &'builder DynClientBuilder,
    provider: &'id str,
    model: &'id str,
}

impl<'builder> ProviderModelId<'builder, '_> {
    pub fn completion(self) -> Result<BoxCompletionModel<'builder>, ClientBuildError> {
        self.builder.completion(self.provider, self.model)
    }

    pub fn agent(self) -> Result<BoxAgentBuilder<'builder>, ClientBuildError> {
        self.builder.agent(self.provider, self.model)
    }

    pub fn embedding(self) -> Result<BoxEmbeddingModel<'builder>, ClientBuildError> {
        self.builder.embeddings(self.provider, self.model)
    }

    pub fn transcription(self) -> Result<BoxTranscriptionModel<'builder>, ClientBuildError> {
        self.builder.transcription(self.provider, self.model)
    }

    /// Stream a completion request using this provider and model.
    ///
    /// # Arguments
    /// * `request` - The completion request containing prompt, parameters, etc.
    ///
    /// # Returns
    /// A future that resolves to a streaming completion response
    pub async fn stream_completion(
        self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<FinalCompletionResponse>, ClientBuildError> {
        self.builder
            .stream_completion(self.provider, self.model, request)
            .await
    }

    /// Stream a simple prompt using this provider and model.
    ///
    /// # Arguments
    /// * `prompt` - The prompt to send to the model
    ///
    /// # Returns
    /// A future that resolves to a streaming completion response
    pub async fn stream_prompt(
        self,
        prompt: impl Into<Message> + Send,
    ) -> Result<StreamingCompletionResponse<FinalCompletionResponse>, ClientBuildError> {
        self.builder
            .stream_prompt(self.provider, self.model, prompt)
            .await
    }

    /// Stream a chat with history using this provider and model.
    ///
    /// # Arguments
    /// * `prompt` - The new prompt to send to the model
    /// * `chat_history` - The chat history to include with the request
    ///
    /// # Returns
    /// A future that resolves to a streaming completion response
    pub async fn stream_chat(
        self,
        prompt: impl Into<Message> + Send,
        chat_history: Vec<Message>,
    ) -> Result<StreamingCompletionResponse<FinalCompletionResponse>, ClientBuildError> {
        self.builder
            .stream_chat(self.provider, self.model, prompt, chat_history)
            .await
    }
}

#[cfg(feature = "image")]
mod image {
    use crate::client::builder::ClientBuildError;
    use crate::image_generation::ImageGenerationModelDyn;
    use rig::client::builder::{DynClientBuilder, ProviderModelId};

    pub type BoxImageGenerationModel<'a> = Box<dyn ImageGenerationModelDyn + 'a>;

    impl DynClientBuilder {
        pub fn image_generation<'a>(
            &self,
            provider: &str,
            model: &str,
        ) -> Result<BoxImageGenerationModel<'a>, ClientBuildError> {
            let client = self.build(provider)?;
            let image =
                client
                    .as_image_generation()
                    .ok_or(ClientBuildError::UnsupportedFeature(
                        provider.to_string(),
                        "image_generation".to_string(),
                    ))?;

            Ok(image.image_generation_model(model))
        }
    }

    impl<'builder> ProviderModelId<'builder, '_> {
        pub fn image_generation(
            self,
        ) -> Result<Box<dyn ImageGenerationModelDyn + 'builder>, ClientBuildError> {
            self.builder.image_generation(self.provider, self.model)
        }
    }
}
#[cfg(feature = "image")]
pub use image::*;

#[cfg(feature = "audio")]
mod audio {
    use crate::audio_generation::AudioGenerationModelDyn;
    use crate::client::builder::DynClientBuilder;
    use crate::client::builder::{ClientBuildError, ProviderModelId};

    pub type BoxAudioGenerationModel<'a> = Box<dyn AudioGenerationModelDyn + 'a>;

    impl DynClientBuilder {
        pub fn audio_generation<'a>(
            &self,
            provider: &str,
            model: &str,
        ) -> Result<BoxAudioGenerationModel<'a>, ClientBuildError> {
            let client = self.build(provider)?;
            let audio =
                client
                    .as_audio_generation()
                    .ok_or(ClientBuildError::UnsupportedFeature(
                        provider.to_string(),
                        "audio_generation".to_owned(),
                    ))?;

            Ok(audio.audio_generation_model(model))
        }
    }

    impl<'builder> ProviderModelId<'builder, '_> {
        pub fn audio_generation(
            self,
        ) -> Result<Box<dyn AudioGenerationModelDyn + 'builder>, ClientBuildError> {
            self.builder.audio_generation(self.provider, self.model)
        }
    }
}
use crate::agent::AgentBuilder;
use crate::client::completion::CompletionModelHandle;
#[cfg(feature = "audio")]
pub use audio::*;
use rig::providers::mistral;

use super::ProviderValue;

pub struct ClientFactory {
    pub name: String,
    pub factory_env: Box<dyn Fn() -> Box<dyn ProviderClient>>,
    pub factory_val: Box<dyn Fn(ProviderValue) -> Box<dyn ProviderClient>>,
}

impl UnwindSafe for ClientFactory {}
impl RefUnwindSafe for ClientFactory {}

impl ClientFactory {
    pub fn new<F1, F2>(name: &str, func_env: F1, func_val: F2) -> Self
    where
        F1: 'static + Fn() -> Box<dyn ProviderClient>,
        F2: 'static + Fn(ProviderValue) -> Box<dyn ProviderClient>,
    {
        Self {
            name: name.to_string(),
            factory_env: Box::new(func_env),
            factory_val: Box::new(func_val),
        }
    }

    pub fn build(&self) -> Result<Box<dyn ProviderClient>, ClientBuildError> {
        std::panic::catch_unwind(|| (self.factory_env)())
            .map_err(|e| ClientBuildError::FactoryError(format!("{e:?}")))
    }

    pub fn build_from_val(
        &self,
        val: ProviderValue,
    ) -> Result<Box<dyn ProviderClient>, ClientBuildError> {
        std::panic::catch_unwind(|| (self.factory_val)(val))
            .map_err(|e| ClientBuildError::FactoryError(format!("{e:?}")))
    }
}

pub struct DefaultProviders;
impl DefaultProviders {
    pub const ANTHROPIC: &'static str = "anthropic";
    pub const COHERE: &'static str = "cohere";
    pub const GEMINI: &'static str = "gemini";
    pub const HUGGINGFACE: &'static str = "huggingface";
    pub const OPENAI: &'static str = "openai";
    pub const OPENROUTER: &'static str = "openrouter";
    pub const TOGETHER: &'static str = "together";
    pub const XAI: &'static str = "xai";
    pub const AZURE: &'static str = "azure";
    pub const DEEPSEEK: &'static str = "deepseek";
    pub const GALADRIEL: &'static str = "galadriel";
    pub const GROQ: &'static str = "groq";
    pub const HYPERBOLIC: &'static str = "hyperbolic";
    pub const MOONSHOT: &'static str = "moonshot";
    pub const MIRA: &'static str = "mira";
    pub const MISTRAL: &'static str = "mistral";
    pub const OLLAMA: &'static str = "ollama";
    pub const PERPLEXITY: &'static str = "perplexity";
}
