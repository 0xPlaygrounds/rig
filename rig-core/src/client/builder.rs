#[allow(deprecated)]
#[cfg(feature = "audio")]
use super::audio_generation::AudioGenerationClientDyn;
#[cfg(feature = "image")]
#[allow(deprecated)]
use super::image_generation::ImageGenerationClientDyn;
#[allow(deprecated)]
#[cfg(feature = "audio")]
use crate::audio_generation::AudioGenerationModelDyn;
#[cfg(feature = "image")]
#[allow(deprecated)]
use crate::image_generation::ImageGenerationModelDyn;
#[allow(deprecated)]
use crate::{
    OneOrMany,
    agent::AgentBuilder,
    client::{
        Capabilities, Capability, Client, FinalCompletionResponse, Provider, ProviderClient,
        completion::{CompletionClientDyn, CompletionModelHandle},
        embeddings::EmbeddingsClientDyn,
        transcription::TranscriptionClientDyn,
    },
    completion::{CompletionError, CompletionModelDyn, CompletionRequest},
    embeddings::EmbeddingModelDyn,
    message::Message,
    providers::{
        anthropic, azure, cohere, deepseek, galadriel, gemini, groq, huggingface, hyperbolic, mira,
        mistral, moonshot, ollama, openai, openrouter, perplexity, together, xai,
    },
    streaming::StreamingCompletionResponse,
    transcription::TranscriptionModelDyn,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use std::{any::Any, collections::HashMap};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Provider '{0}' not found")]
    NotFound(String),
    #[error("Provider '{provider}' cannot be coerced to a '{role}'")]
    NotCapable { provider: String, role: String },
    #[error("Error generating response\n{0}")]
    Completion(#[from] CompletionError),
}

#[deprecated(
    since = "0.25.0",
    note = "`DynClientBuilder` and related features have been deprecated and will be removed in a future release."
)]
pub struct AnyClient {
    client: Box<dyn Any + 'static>,
    vtable: AnyClientVTable,
}

struct AnyClientVTable {
    #[allow(deprecated)]
    as_completion: fn(&dyn Any) -> Option<&&dyn CompletionClientDyn>,
    #[allow(deprecated)]
    as_embedding: fn(&dyn Any) -> Option<&&dyn EmbeddingsClientDyn>,
    #[allow(deprecated)]
    as_transcription: fn(&dyn Any) -> Option<&&dyn TranscriptionClientDyn>,
    #[allow(deprecated)]
    #[cfg(feature = "image")]
    as_image_generation: fn(&dyn Any) -> Option<&&dyn ImageGenerationClientDyn>,
    #[allow(deprecated)]
    #[cfg(feature = "audio")]
    as_audio_generation: fn(&dyn Any) -> Option<&&dyn AudioGenerationClientDyn>,
}

#[allow(deprecated)]
impl AnyClient {
    pub fn new<Ext, H>(client: Client<Ext, H>) -> Self
    where
        Ext: Provider + Capabilities + WasmCompatSend + WasmCompatSync + 'static,
        H: WasmCompatSend + WasmCompatSync + 'static,
        Client<Ext, H>: WasmCompatSend + WasmCompatSync + 'static,
    {
        Self {
            client: Box::new(client),
            vtable: AnyClientVTable {
                as_completion: if <<Ext as Capabilities>::Completion as Capability>::CAPABLE {
                    |any| any.downcast_ref()
                } else {
                    |_| None
                },

                as_embedding: if <<Ext as Capabilities>::Embeddings as Capability>::CAPABLE {
                    |any| any.downcast_ref()
                } else {
                    |_| None
                },

                as_transcription: if <<Ext as Capabilities>::Transcription as Capability>::CAPABLE {
                    |any| any.downcast_ref()
                } else {
                    |_| None
                },

                #[cfg(feature = "image")]
                as_image_generation:
                    if <<Ext as Capabilities>::ImageGeneration as Capability>::CAPABLE {
                        |any| any.downcast_ref()
                    } else {
                        |_| None
                    },

                #[cfg(feature = "audio")]
                as_audio_generation:
                    if <<Ext as Capabilities>::AudioGeneration as Capability>::CAPABLE {
                        |any| any.downcast_ref()
                    } else {
                        |_| None
                    },
            },
        }
    }

    pub fn as_completion(&self) -> Option<&dyn CompletionClientDyn> {
        (self.vtable.as_completion)(self.client.as_ref()).copied()
    }

    pub fn as_embedding(&self) -> Option<&dyn EmbeddingsClientDyn> {
        (self.vtable.as_embedding)(self.client.as_ref()).copied()
    }

    pub fn as_transcription(&self) -> Option<&dyn TranscriptionClientDyn> {
        (self.vtable.as_transcription)(self.client.as_ref()).copied()
    }

    #[cfg(feature = "image")]
    pub fn as_image_generation(&self) -> Option<&dyn ImageGenerationClientDyn> {
        (self.vtable.as_image_generation)(self.client.as_ref()).copied()
    }

    #[cfg(feature = "audio")]
    pub fn as_audio_generation(&self) -> Option<&dyn AudioGenerationClientDyn> {
        (self.vtable.as_audio_generation)(self.client.as_ref()).copied()
    }
}

#[deprecated(
    since = "0.25.0",
    note = "`DynClientBuilder` and related features have been deprecated and will be removed in a future release."
)]
#[derive(Debug, Clone)]
pub struct ProviderFactory {
    /// Create a client from environment variables
    #[allow(deprecated)]
    from_env: fn() -> Result<AnyClient, Error>,
}

#[allow(deprecated)]
#[deprecated(
    since = "0.25.0",
    note = "`DynClientBuilder` and related features have been deprecated and will be removed in a future release."
)]
#[derive(Debug, Clone)]
pub struct DynClientBuilder(HashMap<String, ProviderFactory>);

#[allow(deprecated)]
impl Default for DynClientBuilder {
    fn default() -> Self {
        // Give it a capacity ~the number of providers we have from the start
        Self(HashMap::with_capacity(32))
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum DefaultProviders {
    Anthropic,
    Cohere,
    Gemini,
    HuggingFace,
    OpenAI,
    OpenRouter,
    Together,
    XAI,
    Azure,
    DeepSeek,
    Galadriel,
    Groq,
    Hyperbolic,
    Moonshot,
    Mira,
    Mistral,
    Ollama,
    Perplexity,
}

impl From<DefaultProviders> for &'static str {
    fn from(value: DefaultProviders) -> Self {
        use DefaultProviders::*;

        match value {
            Anthropic => "anthropic",
            Cohere => "cohere",
            Gemini => "gemini",
            HuggingFace => "huggingface",
            OpenAI => "openai",
            OpenRouter => "openrouter",
            Together => "together",
            XAI => "xai",
            Azure => "azure",
            DeepSeek => "deepseek",
            Galadriel => "galadriel",
            Groq => "groq",
            Hyperbolic => "hyperbolic",
            Moonshot => "moonshot",
            Mira => "mira",
            Mistral => "mistral",
            Ollama => "ollama",
            Perplexity => "perplexity",
        }
    }
}
pub use DefaultProviders::*;

impl std::fmt::Display for DefaultProviders {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s: &str = (*self).into();
        f.write_str(s)
    }
}

impl DefaultProviders {
    fn all() -> impl Iterator<Item = Self> {
        use DefaultProviders::*;

        [
            Anthropic,
            Cohere,
            Gemini,
            HuggingFace,
            OpenAI,
            OpenRouter,
            Together,
            XAI,
            Azure,
            DeepSeek,
            Galadriel,
            Groq,
            Hyperbolic,
            Moonshot,
            Mira,
            Mistral,
            Ollama,
            Perplexity,
        ]
        .into_iter()
    }

    #[allow(deprecated)]
    fn get_env_fn(self) -> fn() -> Result<AnyClient, Error> {
        use DefaultProviders::*;

        match self {
            Anthropic => || Ok(AnyClient::new(anthropic::Client::from_env())),
            Cohere => || Ok(AnyClient::new(cohere::Client::from_env())),
            Gemini => || Ok(AnyClient::new(gemini::Client::from_env())),
            HuggingFace => || Ok(AnyClient::new(huggingface::Client::from_env())),
            OpenAI => || Ok(AnyClient::new(openai::Client::from_env())),
            OpenRouter => || Ok(AnyClient::new(openrouter::Client::from_env())),
            Together => || Ok(AnyClient::new(together::Client::from_env())),
            XAI => || Ok(AnyClient::new(xai::Client::from_env())),
            Azure => || Ok(AnyClient::new(azure::Client::from_env())),
            DeepSeek => || Ok(AnyClient::new(deepseek::Client::from_env())),
            Galadriel => || Ok(AnyClient::new(galadriel::Client::from_env())),
            Groq => || Ok(AnyClient::new(groq::Client::from_env())),
            Hyperbolic => || Ok(AnyClient::new(hyperbolic::Client::from_env())),
            Moonshot => || Ok(AnyClient::new(moonshot::Client::from_env())),
            Mira => || Ok(AnyClient::new(mira::Client::from_env())),
            Mistral => || Ok(AnyClient::new(mistral::Client::from_env())),
            Ollama => || Ok(AnyClient::new(ollama::Client::from_env())),
            Perplexity => || Ok(AnyClient::new(perplexity::Client::from_env())),
        }
    }
}

#[allow(deprecated)]
impl DynClientBuilder {
    pub fn new() -> Self {
        Self::default().register_all()
    }

    fn register_all(mut self) -> Self {
        for provider in DefaultProviders::all() {
            let from_env = provider.get_env_fn();
            self.0
                .insert(provider.to_string(), ProviderFactory { from_env });
        }

        self
    }

    fn to_key<Models>(provider_name: &'static str, model: &Models) -> String
    where
        Models: ToString,
    {
        format!("{provider_name}:{}", model.to_string())
    }

    pub fn register<Ext, H, Models>(mut self, provider_name: &'static str, model: Models) -> Self
    where
        Ext: Provider + Capabilities + WasmCompatSend + WasmCompatSync + 'static,
        H: Default + WasmCompatSend + WasmCompatSync + 'static,
        Client<Ext, H>: ProviderClient + WasmCompatSend + WasmCompatSync + 'static,
        Models: ToString,
    {
        let key = Self::to_key(provider_name, &model);

        let factory = ProviderFactory {
            from_env: || Ok(AnyClient::new(Client::<Ext, H>::from_env())),
        };

        self.0.insert(key, factory);

        self
    }

    pub fn from_env<T, Models>(
        &self,
        provider_name: &'static str,
        model: Models,
    ) -> Result<AnyClient, Error>
    where
        T: 'static,
        Models: ToString,
    {
        let key = Self::to_key(provider_name, &model);

        self.0
            .get(&key)
            .ok_or(Error::NotFound(key))
            .and_then(|factory| (factory.from_env)())
    }

    pub fn factory<Models>(
        &self,
        provider_name: &'static str,
        model: Models,
    ) -> Option<&ProviderFactory>
    where
        Models: ToString,
    {
        let key = Self::to_key(provider_name, &model);

        self.0.get(&key)
    }

    /// Get a boxed agent based on the provider and model, as well as an API key.
    pub fn agent<Models>(
        &self,
        provider_name: impl Into<&'static str>,
        model: Models,
    ) -> Result<AgentBuilder<CompletionModelHandle<'_>>, Error>
    where
        Models: ToString,
    {
        let key = Self::to_key(provider_name.into(), &model);

        let client = self
            .0
            .get(&key)
            .ok_or_else(|| Error::NotFound(key.clone()))
            .and_then(|factory| (factory.from_env)())?;

        let completion = client.as_completion().ok_or(Error::NotCapable {
            provider: key,
            role: "Completion".into(),
        })?;

        Ok(completion.agent(&model.to_string()))
    }

    /// Get a boxed completion model based on the provider and model.
    pub fn completion<Models>(
        &self,
        provider_name: &'static str,
        model: Models,
    ) -> Result<Box<dyn CompletionModelDyn>, Error>
    where
        Models: ToString,
    {
        let key = Self::to_key(provider_name, &model);

        let client = self
            .0
            .get(&key)
            .ok_or_else(|| Error::NotFound(key.clone()))
            .and_then(|factory| (factory.from_env)())?;

        let completion = client.as_completion().ok_or(Error::NotCapable {
            provider: key,
            role: "Embedding Model".into(),
        })?;

        Ok(completion.completion_model(&model.to_string()))
    }

    /// Get a boxed embedding model based on the provider and model.
    pub fn embeddings<Models>(
        &self,
        provider_name: &'static str,
        model: Models,
    ) -> Result<Box<dyn EmbeddingModelDyn>, Error>
    where
        Models: ToString,
    {
        let key = Self::to_key(provider_name, &model);

        let client = self
            .0
            .get(&key)
            .ok_or_else(|| Error::NotFound(key.clone()))
            .and_then(|factory| (factory.from_env)())?;

        let embeddings = client.as_embedding().ok_or(Error::NotCapable {
            provider: key,
            role: "Embedding Model".into(),
        })?;

        Ok(embeddings.embedding_model(&model.to_string()))
    }

    /// Get a boxed transcription model based on the provider and model.
    pub fn transcription<Models>(
        &self,
        provider_name: &'static str,
        model: Models,
    ) -> Result<Box<dyn TranscriptionModelDyn>, Error>
    where
        Models: ToString,
    {
        let key = Self::to_key(provider_name, &model);

        let client = self
            .0
            .get(&key)
            .ok_or_else(|| Error::NotFound(key.clone()))
            .and_then(|factory| (factory.from_env)())?;

        let transcription = client.as_transcription().ok_or(Error::NotCapable {
            provider: key,
            role: "transcription model".into(),
        })?;

        Ok(transcription.transcription_model(&model.to_string()))
    }

    #[cfg(feature = "image")]
    pub fn image_generation<Models>(
        &self,
        provider_name: &'static str,
        model: Models,
    ) -> Result<Box<dyn ImageGenerationModelDyn>, Error>
    where
        Models: ToString,
    {
        let key = Self::to_key(provider_name, &model);

        let client = self
            .0
            .get(&key)
            .ok_or_else(|| Error::NotFound(key.clone()))
            .and_then(|factory| (factory.from_env)())?;

        let image_generation = client.as_image_generation().ok_or(Error::NotCapable {
            provider: key,
            role: "Image generation".into(),
        })?;

        Ok(image_generation.image_generation_model(&model.to_string()))
    }

    #[cfg(feature = "audio")]
    pub fn audio_generation<Models>(
        &self,
        provider_name: &'static str,
        model: Models,
    ) -> Result<Box<dyn AudioGenerationModelDyn>, Error>
    where
        Models: ToString,
    {
        let key = Self::to_key(provider_name, &model);

        let client = self
            .0
            .get(&key)
            .ok_or_else(|| Error::NotFound(key.clone()))
            .and_then(|factory| (factory.from_env)())?;

        let audio_generation = client.as_audio_generation().ok_or(Error::NotCapable {
            provider: key,
            role: "Image generation".into(),
        })?;

        Ok(audio_generation.audio_generation_model(&model.to_string()))
    }

    /// Stream a completion request to the specified provider and model.
    pub async fn stream_completion<Models>(
        &self,
        provider_name: &'static str,
        model: Models,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<FinalCompletionResponse>, Error>
    where
        Models: ToString,
    {
        let completion = self.completion(provider_name, model)?;

        completion.stream(request).await.map_err(Error::Completion)
    }

    /// Stream a simple prompt to the specified provider and model.
    pub async fn stream_prompt<Models, Prompt>(
        &self,
        provider_name: impl Into<&'static str>,
        model: Models,
        prompt: Prompt,
    ) -> Result<StreamingCompletionResponse<FinalCompletionResponse>, Error>
    where
        Models: ToString,
        Prompt: Into<Message> + WasmCompatSend,
    {
        let completion = self.completion(provider_name.into(), model)?;

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

        completion.stream(request).await.map_err(Error::Completion)
    }

    /// Stream a chat with history to the specified provider and model.
    pub async fn stream_chat<Models, Prompt>(
        &self,
        provider_name: &'static str,
        model: Models,
        prompt: Prompt,
        mut history: Vec<Message>,
    ) -> Result<StreamingCompletionResponse<FinalCompletionResponse>, Error>
    where
        Models: ToString,
        Prompt: Into<Message> + WasmCompatSend,
    {
        let completion = self.completion(provider_name, model)?;

        history.push(prompt.into());
        let request = CompletionRequest {
            preamble: None,
            tools: vec![],
            documents: vec![],
            temperature: None,
            max_tokens: None,
            additional_params: None,
            tool_choice: None,
            chat_history: OneOrMany::many(history)
                .unwrap_or_else(|_| OneOrMany::one(Message::user(""))),
        };

        completion.stream(request).await.map_err(Error::Completion)
    }
}
