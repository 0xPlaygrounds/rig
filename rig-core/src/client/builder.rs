use crate::client::ProviderClient;
use crate::embeddings::embedding::EmbeddingModelDyn;
use crate::providers::{
    anthropic, azure, cohere, deepseek, galadriel, gemini, groq, huggingface, hyperbolic, mira,
    moonshot, ollama, openai, openrouter, perplexity, together, xai,
};
use crate::transcription::TranscriptionModelDyn;
use rig::completion::CompletionModelDyn;
use std::any::Any;
use std::collections::HashMap;
use std::panic::{RefUnwindSafe, UnwindSafe};

pub struct DynClientBuilder {
    registry: HashMap<String, ClientFactory>,
}

impl Default for DynClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub enum ClientBuildError {
    FactoryError(Box<dyn Any + Send>),
    InvalidIdString,
    UnsupportedFeature(String),
    UnknownProvider,
}

impl DynClientBuilder {
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
            ClientFactory::new(DefaultProviders::MIRA, mira::Client::from_env_boxed),
            ClientFactory::new(DefaultProviders::MOONSHOT, moonshot::Client::from_env_boxed),
            ClientFactory::new(DefaultProviders::OLLAMA, ollama::Client::from_env_boxed),
            ClientFactory::new(
                DefaultProviders::PERPLEXITY,
                perplexity::Client::from_env_boxed,
            ),
        ])
    }

    /// Register a new ClientFactory
    pub fn register(mut self, client_factory: ClientFactory) -> Self {
        self.registry
            .insert(client_factory.name.clone(), client_factory);
        self
    }

    /// Register multiple ClientFactory's
    pub fn register_all(mut self, factories: impl IntoIterator<Item = ClientFactory>) -> Self {
        for factory in factories {
            self.registry.insert(factory.name.clone(), factory);
        }

        self
    }

    pub fn build(&self, provider: &str) -> Result<Box<dyn ProviderClient>, ClientBuildError> {
        let factory = self.get_factory(provider)?;
        factory.build()
    }

    fn parse_id<'a>(&self, id: &'a str) -> Result<(&'a str, &'a str), ClientBuildError> {
        let (provider, model) = id
            .split_once(":")
            .ok_or(ClientBuildError::InvalidIdString)?;
        Ok((provider, model))
    }

    fn get_factory(&self, provider: &str) -> Result<&ClientFactory, ClientBuildError> {
        self.registry
            .get(provider)
            .ok_or(ClientBuildError::UnknownProvider)
    }

    pub fn completion<'a>(
        &self,
        provider: &str,
        model: &'a str,
    ) -> Result<Box<dyn CompletionModelDyn + 'a>, ClientBuildError> {
        let client = self.build(provider)?;

        let completion = client
            .as_completion()
            .ok_or(ClientBuildError::UnsupportedFeature(
                "completion".to_owned(),
            ))?;

        Ok(completion.completion_model(model))
    }

    pub fn embeddings<'a>(
        &self,
        provider: &str,
        model: &'a str,
    ) -> Result<Box<dyn EmbeddingModelDyn + 'a>, ClientBuildError> {
        let client = self.build(provider)?;

        let embeddings = client
            .as_embeddings()
            .ok_or(ClientBuildError::UnsupportedFeature(
                "embeddings".to_owned(),
            ))?;

        Ok(embeddings.embedding_model(model))
    }

    pub fn transcription<'a>(
        &self,
        provider: &str,
        model: &'a str,
    ) -> Result<Box<dyn TranscriptionModelDyn + 'a>, ClientBuildError> {
        let client = self.build(provider)?;
        let transcription =
            client
                .as_transcription()
                .ok_or(ClientBuildError::UnsupportedFeature(
                    "transcription".to_owned(),
                ))?;

        Ok(transcription.transcription_model(model))
    }

    pub fn id<'a>(&'a self, id: &'a str) -> Result<ProviderModelId<'a>, ClientBuildError> {
        let (provider, model) = self.parse_id(id)?;

        Ok(ProviderModelId {
            builder: self,
            provider,
            model,
        })
    }
}

pub struct ProviderModelId<'a> {
    builder: &'a DynClientBuilder,
    provider: &'a str,
    model: &'a str,
}

impl<'a> ProviderModelId<'a> {
    pub fn completion(self) -> Result<Box<dyn CompletionModelDyn + 'a>, ClientBuildError> {
        self.builder.completion(self.provider, self.model)
    }

    pub fn embedding(self) -> Result<Box<dyn EmbeddingModelDyn + 'a>, ClientBuildError> {
        self.builder.embeddings(self.provider, self.model)
    }

    pub fn transcription(self) -> Result<Box<dyn TranscriptionModelDyn + 'a>, ClientBuildError> {
        self.builder.transcription(self.provider, self.model)
    }
}

#[cfg(feature = "image")]
mod image {
    use crate::client::builder::ClientBuildError;
    use crate::image_generation::ImageGenerationModelDyn;
    use rig::client::builder::{DynClientBuilder, ProviderModelId};

    impl DynClientBuilder {
        pub fn image_generation<'a>(
            &self,
            provider: &str,
            model: &'a str,
        ) -> Result<Box<dyn ImageGenerationModelDyn + 'a>, ClientBuildError> {
            let client = self.build(provider)?;
            let image =
                client
                    .as_image_generation()
                    .ok_or(ClientBuildError::UnsupportedFeature(
                        "image_generation".to_string(),
                    ))?;

            Ok(image.image_generation_model(model))
        }
    }

    impl<'a> ProviderModelId<'a> {
        pub fn image_generation(
            self,
        ) -> Result<Box<dyn ImageGenerationModelDyn + 'a>, ClientBuildError> {
            self.builder.image_generation(self.provider, self.model)
        }
    }
}

#[cfg(feature = "audio")]
mod audio {
    use crate::audio_generation::AudioGenerationModelDyn;
    use crate::client::builder::DynClientBuilder;
    use crate::client::builder::{ClientBuildError, ProviderModelId};

    impl DynClientBuilder {
        pub fn audio_generation<'a>(
            &self,
            provider: &str,
            model: &'a str,
        ) -> Result<Box<dyn AudioGenerationModelDyn + 'a>, ClientBuildError> {
            let client = self.build(provider)?;
            let audio =
                client
                    .as_audio_generation()
                    .ok_or(ClientBuildError::UnsupportedFeature(
                        "audio_generation".to_owned(),
                    ))?;

            Ok(audio.audio_generation_model(model))
        }
    }

    impl<'a> ProviderModelId<'a> {
        pub fn audio_generation(
            self,
        ) -> Result<Box<dyn AudioGenerationModelDyn + 'a>, ClientBuildError> {
            self.builder.audio_generation(self.provider, self.model)
        }
    }
}

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
        std::panic::catch_unwind(|| (self.factory)()).map_err(ClientBuildError::FactoryError)
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
    pub const MIRA: &'static str = "mira";
    pub const MOONSHOT: &'static str = "moonshot";
    pub const OLLAMA: &'static str = "ollama";
    pub const PERPLEXITY: &'static str = "perplexity";
}
