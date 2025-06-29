use crate::agent::Agent;
use crate::client::ProviderClient;
use crate::embeddings::embedding::EmbeddingModelDyn;
use crate::providers::{
    anthropic, azure, cohere, deepseek, galadriel, gemini, groq, huggingface, hyperbolic, mira,
    moonshot, ollama, openai, openrouter, perplexity, together, xai,
};
use crate::transcription::TranscriptionModelDyn;
use rig::completion::CompletionModelDyn;
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
                anthropic::Client::from_env_boxed,
            ),
            ClientFactory::new(DefaultProviders::COHERE, cohere::Client::from_env_boxed),
            ClientFactory::new(DefaultProviders::GEMINI, gemini::Client::from_env_boxed),
            ClientFactory::new(
                DefaultProviders::HUGGINGFACE,
                huggingface::Client::from_env_boxed,
            ),
            ClientFactory::new(DefaultProviders::OPENAI, openai::Client::from_env_boxed),
            ClientFactory::new(
                DefaultProviders::OPENROUTER,
                openrouter::Client::from_env_boxed,
            ),
            ClientFactory::new(DefaultProviders::TOGETHER, together::Client::from_env_boxed),
            ClientFactory::new(DefaultProviders::XAI, xai::Client::from_env_boxed),
            ClientFactory::new(DefaultProviders::AZURE, azure::Client::from_env_boxed),
            ClientFactory::new(DefaultProviders::DEEPSEEK, deepseek::Client::from_env_boxed),
            ClientFactory::new(
                DefaultProviders::GALADRIEL,
                galadriel::Client::from_env_boxed,
            ),
            ClientFactory::new(DefaultProviders::GROQ, groq::Client::from_env_boxed),
            ClientFactory::new(
                DefaultProviders::HYPERBOLIC,
                hyperbolic::Client::from_env_boxed,
            ),
            ClientFactory::new(DefaultProviders::MOONSHOT, moonshot::Client::from_env_boxed),
            ClientFactory::new(DefaultProviders::MIRA, mira::Client::from_env_boxed),
            ClientFactory::new(DefaultProviders::MISTRAL, mistral::Client::from_env_boxed),
            ClientFactory::new(DefaultProviders::OLLAMA, ollama::Client::from_env_boxed),
            ClientFactory::new(
                DefaultProviders::PERPLEXITY,
                perplexity::Client::from_env_boxed,
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

    /// Get the ID of a provider model based on a `provider:model` ID.
    pub fn id<'id>(&'a self, id: &'id str) -> Result<ProviderModelId<'a, 'id>, ClientBuildError> {
        let (provider, model) = self.parse(id)?;

        Ok(ProviderModelId {
            builder: self,
            provider,
            model,
        })
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

pub struct ClientFactory {
    pub name: String,
    pub factory: Box<dyn Fn() -> Box<dyn ProviderClient>>,
}

impl UnwindSafe for ClientFactory {}
impl RefUnwindSafe for ClientFactory {}

impl ClientFactory {
    pub fn new<F: 'static + Fn() -> Box<dyn ProviderClient>>(name: &str, func: F) -> Self {
        Self {
            name: name.to_string(),
            factory: Box::new(func),
        }
    }

    pub fn build(&self) -> Result<Box<dyn ProviderClient>, ClientBuildError> {
        std::panic::catch_unwind(|| (self.factory)())
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
