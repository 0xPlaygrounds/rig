//! This module provides traits for defining and creating provider clients.
//! Clients are used to create models for completion, embeddings, etc.
//! Dyn-compatible traits have been provided to allow for more provider-agnostic code.

pub mod audio_generation;
pub mod builder;
pub mod completion;
pub mod embeddings;
pub mod image_generation;
pub mod transcription;
pub mod verify;

#[cfg(feature = "derive")]
pub use rig_derive::ProviderClient;
use std::fmt::Debug;
use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ClientBuilderError {
    #[error("reqwest error: {0}")]
    HttpError(
        #[from]
        #[source]
        reqwest::Error,
    ),
    #[error("invalid property: {0}")]
    InvalidProperty(&'static str),
}

/// The base ProviderClient trait, facilitates conversion between client types
/// and creating a client from the environment.
///
/// All conversion traits must be implemented, they are automatically
/// implemented if the respective client trait is implemented.
pub trait ProviderClient:
    AsCompletion + AsTranscription + AsEmbeddings + AsImageGeneration + AsAudioGeneration + Debug
{
    /// Create a client from the process's environment.
    /// Panics if an environment is improperly configured.
    fn from_env() -> Self
    where
        Self: Sized;

    /// A helper method to box the client.
    fn boxed(self) -> Box<dyn ProviderClient>
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }

    /// Create a boxed client from the process's environment.
    /// Panics if an environment is improperly configured.
    fn from_env_boxed<'a>() -> Box<dyn ProviderClient + 'a>
    where
        Self: Sized,
        Self: 'a,
    {
        Box::new(Self::from_env())
    }

    fn from_val(input: ProviderValue) -> Self
    where
        Self: Sized;

    /// Create a boxed client from the process's environment.
    /// Panics if an environment is improperly configured.
    fn from_val_boxed<'a>(input: ProviderValue) -> Box<dyn ProviderClient + 'a>
    where
        Self: Sized,
        Self: 'a,
    {
        Box::new(Self::from_val(input))
    }
}

#[derive(Clone)]
pub enum ProviderValue {
    Simple(String),
    ApiKeyWithOptionalKey(String, Option<String>),
    ApiKeyWithVersionAndHeader(String, String, String),
}

impl From<&str> for ProviderValue {
    fn from(value: &str) -> Self {
        Self::Simple(value.to_string())
    }
}

impl From<String> for ProviderValue {
    fn from(value: String) -> Self {
        Self::Simple(value)
    }
}

impl<P> From<(P, Option<P>)> for ProviderValue
where
    P: AsRef<str>,
{
    fn from((api_key, optional_key): (P, Option<P>)) -> Self {
        Self::ApiKeyWithOptionalKey(
            api_key.as_ref().to_string(),
            optional_key.map(|x| x.as_ref().to_string()),
        )
    }
}

impl<P> From<(P, P, P)> for ProviderValue
where
    P: AsRef<str>,
{
    fn from((api_key, version, header): (P, P, P)) -> Self {
        Self::ApiKeyWithVersionAndHeader(
            api_key.as_ref().to_string(),
            version.as_ref().to_string(),
            header.as_ref().to_string(),
        )
    }
}

/// Attempt to convert a ProviderClient to a CompletionClient
pub trait AsCompletion {
    fn as_completion(&self) -> Option<Box<dyn CompletionClientDyn>> {
        None
    }
}

/// Attempt to convert a ProviderClient to a TranscriptionClient
pub trait AsTranscription {
    fn as_transcription(&self) -> Option<Box<dyn TranscriptionClientDyn>> {
        None
    }
}

/// Attempt to convert a ProviderClient to a EmbeddingsClient
pub trait AsEmbeddings {
    fn as_embeddings(&self) -> Option<Box<dyn EmbeddingsClientDyn>> {
        None
    }
}

/// Attempt to convert a ProviderClient to a AudioGenerationClient
pub trait AsAudioGeneration {
    #[cfg(feature = "audio")]
    fn as_audio_generation(&self) -> Option<Box<dyn AudioGenerationClientDyn>> {
        None
    }
}

/// Attempt to convert a ProviderClient to a ImageGenerationClient
pub trait AsImageGeneration {
    #[cfg(feature = "image")]
    fn as_image_generation(&self) -> Option<Box<dyn ImageGenerationClientDyn>> {
        None
    }
}

/// Attempt to convert a ProviderClient to a VerifyClient
pub trait AsVerify {
    fn as_verify(&self) -> Option<Box<dyn VerifyClientDyn>> {
        None
    }
}

#[cfg(not(feature = "audio"))]
impl<T: ProviderClient> AsAudioGeneration for T {}

#[cfg(not(feature = "image"))]
impl<T: ProviderClient> AsImageGeneration for T {}

/// Implements the conversion traits for a given struct
/// ```rust
/// pub struct Client;
/// impl ProviderClient for Client {
///     ...
/// }
/// impl_conversion_traits!(AsCompletion, AsEmbeddings for Client);
/// ```
#[macro_export]
macro_rules! impl_conversion_traits {
    ($( $trait_:ident ),* for $struct_:ident ) => {
        $(
            impl_conversion_traits!(@impl $trait_ for $struct_);
        )*
    };

    (@impl AsAudioGeneration for $struct_:ident ) => {
        rig::client::impl_audio_generation!($struct_);
    };

    (@impl AsImageGeneration for $struct_:ident ) => {
        rig::client::impl_image_generation!($struct_);
    };

    (@impl $trait_:ident for $struct_:ident) => {
        impl rig::client::$trait_ for $struct_ {}
    };
}

#[cfg(feature = "audio")]
#[macro_export]
macro_rules! impl_audio_generation {
    ($struct_:ident) => {
        impl rig::client::AsAudioGeneration for $struct_ {}
    };
}

#[cfg(not(feature = "audio"))]
#[macro_export]
macro_rules! impl_audio_generation {
    ($struct_:ident) => {};
}

#[cfg(feature = "image")]
#[macro_export]
macro_rules! impl_image_generation {
    ($struct_:ident) => {
        impl rig::client::AsImageGeneration for $struct_ {}
    };
}

#[cfg(not(feature = "image"))]
#[macro_export]
macro_rules! impl_image_generation {
    ($struct_:ident) => {};
}

pub use impl_audio_generation;
pub use impl_conversion_traits;
pub use impl_image_generation;

#[cfg(feature = "audio")]
use crate::client::audio_generation::AudioGenerationClientDyn;
use crate::client::completion::CompletionClientDyn;
use crate::client::embeddings::EmbeddingsClientDyn;
#[cfg(feature = "image")]
use crate::client::image_generation::ImageGenerationClientDyn;
use crate::client::transcription::TranscriptionClientDyn;
use crate::client::verify::VerifyClientDyn;

#[cfg(feature = "audio")]
pub use crate::client::audio_generation::AudioGenerationClient;
pub use crate::client::completion::CompletionClient;
pub use crate::client::embeddings::EmbeddingsClient;
#[cfg(feature = "image")]
pub use crate::client::image_generation::ImageGenerationClient;
pub use crate::client::transcription::TranscriptionClient;
pub use crate::client::verify::{VerifyClient, VerifyError};

#[cfg(test)]
mod tests {
    use crate::OneOrMany;
    use crate::client::ProviderClient;
    use crate::completion::{Completion, CompletionRequest, ToolDefinition};
    use crate::image_generation::ImageGenerationRequest;
    use crate::message::AssistantContent;
    use crate::providers::{
        anthropic, azure, cohere, deepseek, galadriel, gemini, huggingface, hyperbolic, mira,
        moonshot, openai, openrouter, together, xai,
    };
    use crate::streaming::StreamingCompletion;
    use crate::tool::Tool;
    use crate::transcription::TranscriptionRequest;
    use futures::StreamExt;
    use rig::message::Message;
    use rig::providers::{groq, ollama, perplexity};
    use serde::{Deserialize, Serialize};
    use serde_json::json;
    use std::fs::File;
    use std::io::Read;

    use super::ProviderValue;

    struct ClientConfig {
        name: &'static str,
        factory_env: Box<dyn Fn() -> Box<dyn ProviderClient>>,
        // Not sure where we're going to be using this but I've added it for completeness
        #[allow(dead_code)]
        factory_val: Box<dyn Fn(ProviderValue) -> Box<dyn ProviderClient>>,
        env_variable: &'static str,
        completion_model: Option<&'static str>,
        embeddings_model: Option<&'static str>,
        transcription_model: Option<&'static str>,
        image_generation_model: Option<&'static str>,
        audio_generation_model: Option<(&'static str, &'static str)>,
    }

    impl Default for ClientConfig {
        fn default() -> Self {
            Self {
                name: "",
                factory_env: Box::new(|| panic!("Not implemented")),
                factory_val: Box::new(|_| panic!("Not implemented")),
                env_variable: "",
                completion_model: None,
                embeddings_model: None,
                transcription_model: None,
                image_generation_model: None,
                audio_generation_model: None,
            }
        }
    }

    impl ClientConfig {
        fn is_env_var_set(&self) -> bool {
            self.env_variable.is_empty() || std::env::var(self.env_variable).is_ok()
        }

        fn factory_env(&self) -> Box<dyn ProviderClient + '_> {
            self.factory_env.as_ref()()
        }
    }

    fn providers() -> Vec<ClientConfig> {
        vec![
            ClientConfig {
                name: "Anthropic",
                factory_env: Box::new(anthropic::Client::from_env_boxed),
                factory_val: Box::new(anthropic::Client::from_val_boxed),
                env_variable: "ANTHROPIC_API_KEY",
                completion_model: Some(anthropic::CLAUDE_3_5_SONNET),
                ..Default::default()
            },
            ClientConfig {
                name: "Cohere",
                factory_env: Box::new(cohere::Client::from_env_boxed),
                factory_val: Box::new(cohere::Client::from_val_boxed),
                env_variable: "COHERE_API_KEY",
                completion_model: Some(cohere::COMMAND_R),
                embeddings_model: Some(cohere::EMBED_ENGLISH_LIGHT_V2),
                ..Default::default()
            },
            ClientConfig {
                name: "Gemini",
                factory_env: Box::new(gemini::Client::from_env_boxed),
                factory_val: Box::new(gemini::Client::from_val_boxed),
                env_variable: "GEMINI_API_KEY",
                completion_model: Some(gemini::completion::GEMINI_2_0_FLASH),
                embeddings_model: Some(gemini::embedding::EMBEDDING_001),
                transcription_model: Some(gemini::transcription::GEMINI_2_0_FLASH),
                ..Default::default()
            },
            ClientConfig {
                name: "Huggingface",
                factory_env: Box::new(huggingface::Client::from_env_boxed),
                factory_val: Box::new(huggingface::Client::from_val_boxed),
                env_variable: "HUGGINGFACE_API_KEY",
                completion_model: Some(huggingface::PHI_4),
                transcription_model: Some(huggingface::WHISPER_SMALL),
                image_generation_model: Some(huggingface::STABLE_DIFFUSION_3),
                ..Default::default()
            },
            ClientConfig {
                name: "OpenAI",
                factory_env: Box::new(openai::Client::from_env_boxed),
                factory_val: Box::new(openai::Client::from_val_boxed),
                env_variable: "OPENAI_API_KEY",
                completion_model: Some(openai::GPT_4O),
                embeddings_model: Some(openai::TEXT_EMBEDDING_ADA_002),
                transcription_model: Some(openai::WHISPER_1),
                image_generation_model: Some(openai::DALL_E_2),
                audio_generation_model: Some((openai::TTS_1, "onyx")),
            },
            ClientConfig {
                name: "OpenRouter",
                factory_env: Box::new(openrouter::Client::from_env_boxed),
                factory_val: Box::new(openrouter::Client::from_val_boxed),
                env_variable: "OPENROUTER_API_KEY",
                completion_model: Some(openrouter::CLAUDE_3_7_SONNET),
                ..Default::default()
            },
            ClientConfig {
                name: "Together",
                factory_env: Box::new(together::Client::from_env_boxed),
                factory_val: Box::new(together::Client::from_val_boxed),
                env_variable: "TOGETHER_API_KEY",
                completion_model: Some(together::ALPACA_7B),
                embeddings_model: Some(together::BERT_BASE_UNCASED),
                ..Default::default()
            },
            ClientConfig {
                name: "XAI",
                factory_env: Box::new(xai::Client::from_env_boxed),
                factory_val: Box::new(xai::Client::from_val_boxed),
                env_variable: "XAI_API_KEY",
                completion_model: Some(xai::GROK_3_MINI),
                embeddings_model: None,
                ..Default::default()
            },
            ClientConfig {
                name: "Azure",
                factory_env: Box::new(azure::Client::from_env_boxed),
                factory_val: Box::new(azure::Client::from_val_boxed),
                env_variable: "AZURE_API_KEY",
                completion_model: Some(azure::GPT_4O),
                embeddings_model: Some(azure::TEXT_EMBEDDING_ADA_002),
                transcription_model: Some("whisper-1"),
                image_generation_model: Some("dalle-2"),
                audio_generation_model: Some(("tts-1", "onyx")),
            },
            ClientConfig {
                name: "Deepseek",
                factory_env: Box::new(deepseek::Client::from_env_boxed),
                factory_val: Box::new(deepseek::Client::from_val_boxed),
                env_variable: "DEEPSEEK_API_KEY",
                completion_model: Some(deepseek::DEEPSEEK_CHAT),
                ..Default::default()
            },
            ClientConfig {
                name: "Galadriel",
                factory_env: Box::new(galadriel::Client::from_env_boxed),
                factory_val: Box::new(galadriel::Client::from_val_boxed),
                env_variable: "GALADRIEL_API_KEY",
                completion_model: Some(galadriel::GPT_4O),
                ..Default::default()
            },
            ClientConfig {
                name: "Groq",
                factory_env: Box::new(groq::Client::from_env_boxed),
                factory_val: Box::new(groq::Client::from_val_boxed),
                env_variable: "GROQ_API_KEY",
                completion_model: Some(groq::MIXTRAL_8X7B_32768),
                transcription_model: Some(groq::DISTIL_WHISPER_LARGE_V3),
                ..Default::default()
            },
            ClientConfig {
                name: "Hyperbolic",
                factory_env: Box::new(hyperbolic::Client::from_env_boxed),
                factory_val: Box::new(hyperbolic::Client::from_val_boxed),
                env_variable: "HYPERBOLIC_API_KEY",
                completion_model: Some(hyperbolic::LLAMA_3_1_8B),
                image_generation_model: Some(hyperbolic::SD1_5),
                audio_generation_model: Some(("EN", "EN-US")),
                ..Default::default()
            },
            ClientConfig {
                name: "Mira",
                factory_env: Box::new(mira::Client::from_env_boxed),
                factory_val: Box::new(mira::Client::from_val_boxed),
                env_variable: "MIRA_API_KEY",
                completion_model: Some("gpt-4o"),
                ..Default::default()
            },
            ClientConfig {
                name: "Moonshot",
                factory_env: Box::new(moonshot::Client::from_env_boxed),
                factory_val: Box::new(moonshot::Client::from_val_boxed),
                env_variable: "MOONSHOT_API_KEY",
                completion_model: Some(moonshot::MOONSHOT_CHAT),
                ..Default::default()
            },
            ClientConfig {
                name: "Ollama",
                factory_env: Box::new(ollama::Client::from_env_boxed),
                factory_val: Box::new(ollama::Client::from_val_boxed),
                env_variable: "OLLAMA_ENABLED",
                completion_model: Some("llama3.1:8b"),
                embeddings_model: Some(ollama::NOMIC_EMBED_TEXT),
                ..Default::default()
            },
            ClientConfig {
                name: "Perplexity",
                factory_env: Box::new(perplexity::Client::from_env_boxed),
                factory_val: Box::new(perplexity::Client::from_val_boxed),
                env_variable: "PERPLEXITY_API_KEY",
                completion_model: Some(perplexity::SONAR),
                ..Default::default()
            },
        ]
    }

    async fn test_completions_client(config: &ClientConfig) {
        let client = config.factory_env();

        let Some(client) = client.as_completion() else {
            return;
        };

        let model = config
            .completion_model
            .unwrap_or_else(|| panic!("{} does not have completion_model set", config.name));

        let model = client.completion_model(model);

        let resp = model
            .completion_request(Message::user("Whats the capital of France?"))
            .send()
            .await;

        assert!(
            resp.is_ok(),
            "[{}]: Error occurred when prompting, {}",
            config.name,
            resp.err().unwrap()
        );

        let resp = resp.unwrap();

        match resp.choice.first() {
            AssistantContent::Text(text) => {
                assert!(text.text.to_lowercase().contains("paris"));
            }
            _ => {
                unreachable!(
                    "[{}]: First choice wasn't a Text message, {:?}",
                    config.name,
                    resp.choice.first()
                );
            }
        }
    }

    #[tokio::test]
    #[ignore]
    async fn test_completions() {
        for p in providers().into_iter().filter(ClientConfig::is_env_var_set) {
            test_completions_client(&p).await;
        }
    }

    async fn test_tools_client(config: &ClientConfig) {
        let client = config.factory_env();
        let model = config
            .completion_model
            .unwrap_or_else(|| panic!("{} does not have the model set.", config.name));

        let Some(client) = client.as_completion() else {
            return;
        };

        let model = client.agent(model)
            .preamble("You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.")
            .max_tokens(1024)
            .tool(Adder)
            .tool(Subtract)
            .build();

        let request = model.completion("Calculate 2 - 5", vec![]).await;

        assert!(
            request.is_ok(),
            "[{}]: Error occurred when building prompt, {}",
            config.name,
            request.err().unwrap()
        );

        let resp = request.unwrap().send().await;

        assert!(
            resp.is_ok(),
            "[{}]: Error occurred when prompting, {}",
            config.name,
            resp.err().unwrap()
        );

        let resp = resp.unwrap();

        assert!(
            resp.choice.iter().any(|content| match content {
                AssistantContent::ToolCall(tc) => {
                    if tc.function.name != Subtract::NAME {
                        return false;
                    }

                    let arguments =
                        serde_json::from_value::<OperationArgs>((tc.function.arguments).clone())
                            .expect("Error parsing arguments");

                    arguments.x == 2.0 && arguments.y == 5.0
                }
                _ => false,
            }),
            "[{}]: Model did not use the Subtract tool.",
            config.name
        )
    }

    #[tokio::test]
    #[ignore]
    async fn test_tools() {
        for p in providers().into_iter().filter(ClientConfig::is_env_var_set) {
            test_tools_client(&p).await;
        }
    }

    async fn test_streaming_client(config: &ClientConfig) {
        let client = config.factory_env();

        let Some(client) = client.as_completion() else {
            return;
        };

        let model = config
            .completion_model
            .unwrap_or_else(|| panic!("{} does not have the model set.", config.name));

        let model = client.completion_model(model);

        let resp = model.stream(CompletionRequest {
            preamble: None,
            tools: vec![],
            documents: vec![],
            temperature: None,
            max_tokens: None,
            additional_params: None,
            chat_history: OneOrMany::one(Message::user("What is the capital of France?")),
        });

        let mut resp = resp.await.unwrap();

        let mut received_chunk = false;

        while let Some(chunk) = resp.next().await {
            received_chunk = true;
            assert!(chunk.is_ok());
        }

        assert!(
            received_chunk,
            "[{}]: Failed to receive a chunk from stream",
            config.name
        );

        for choice in resp.choice {
            match choice {
                AssistantContent::Text(text) => {
                    assert!(
                        text.text.to_lowercase().contains("paris"),
                        "[{}]: Did not answer with Paris",
                        config.name
                    );
                }
                AssistantContent::ToolCall(_) => {}
                AssistantContent::Reasoning(_) => {}
            }
        }
    }

    #[tokio::test]
    #[ignore]
    async fn test_streaming() {
        for provider in providers().into_iter().filter(ClientConfig::is_env_var_set) {
            test_streaming_client(&provider).await;
        }
    }

    async fn test_streaming_tools_client(config: &ClientConfig) {
        let client = config.factory_env();
        let model = config
            .completion_model
            .unwrap_or_else(|| panic!("{} does not have the model set.", config.name));

        let Some(client) = client.as_completion() else {
            return;
        };

        let model = client.agent(model)
            .preamble("You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.")
            .max_tokens(1024)
            .tool(Adder)
            .tool(Subtract)
            .build();

        let request = model.stream_completion("Calculate 2 - 5", vec![]).await;

        assert!(
            request.is_ok(),
            "[{}]: Error occurred when building prompt, {}",
            config.name,
            request.err().unwrap()
        );

        let resp = request.unwrap().stream().await;

        assert!(
            resp.is_ok(),
            "[{}]: Error occurred when prompting, {}",
            config.name,
            resp.err().unwrap()
        );

        let mut resp = resp.unwrap();

        let mut received_chunk = false;

        while let Some(chunk) = resp.next().await {
            received_chunk = true;
            assert!(chunk.is_ok());
        }

        assert!(
            received_chunk,
            "[{}]: Failed to receive a chunk from stream",
            config.name
        );

        assert!(
            resp.choice.iter().any(|content| match content {
                AssistantContent::ToolCall(tc) => {
                    if tc.function.name != Subtract::NAME {
                        return false;
                    }

                    let arguments =
                        serde_json::from_value::<OperationArgs>((tc.function.arguments).clone())
                            .expect("Error parsing arguments");

                    arguments.x == 2.0 && arguments.y == 5.0
                }
                _ => false,
            }),
            "[{}]: Model did not use the Subtract tool.",
            config.name
        )
    }

    #[tokio::test]
    #[ignore]
    async fn test_streaming_tools() {
        for p in providers().into_iter().filter(ClientConfig::is_env_var_set) {
            test_streaming_tools_client(&p).await;
        }
    }

    async fn test_audio_generation_client(config: &ClientConfig) {
        let client = config.factory_env();

        let Some(client) = client.as_audio_generation() else {
            return;
        };

        let (model, voice) = config
            .audio_generation_model
            .unwrap_or_else(|| panic!("{} doesn't have the model set", config.name));

        let model = client.audio_generation_model(model);

        let request = model
            .audio_generation_request()
            .text("Hello world!")
            .voice(voice);

        let resp = request.send().await;

        assert!(
            resp.is_ok(),
            "[{}]: Error occurred when sending request, {}",
            config.name,
            resp.err().unwrap()
        );

        let resp = resp.unwrap();

        assert!(
            !resp.audio.is_empty(),
            "[{}]: Returned audio was empty",
            config.name
        );
    }

    #[tokio::test]
    #[ignore]
    async fn test_audio_generation() {
        for p in providers().into_iter().filter(ClientConfig::is_env_var_set) {
            test_audio_generation_client(&p).await;
        }
    }

    fn assert_feature<F, M>(
        name: &str,
        feature_name: &str,
        model_name: &str,
        feature: Option<F>,
        model: Option<M>,
    ) {
        assert_eq!(
            feature.is_some(),
            model.is_some(),
            "{} has{} implemented {} but config.{} is {}.",
            name,
            if feature.is_some() { "" } else { "n't" },
            feature_name,
            model_name,
            if model.is_some() { "some" } else { "none" }
        );
    }

    #[test]
    #[ignore]
    pub fn test_polymorphism() {
        for config in providers().into_iter().filter(ClientConfig::is_env_var_set) {
            let client = config.factory_env();
            assert_feature(
                config.name,
                "AsCompletion",
                "completion_model",
                client.as_completion(),
                config.completion_model,
            );

            assert_feature(
                config.name,
                "AsEmbeddings",
                "embeddings_model",
                client.as_embeddings(),
                config.embeddings_model,
            );

            assert_feature(
                config.name,
                "AsTranscription",
                "transcription_model",
                client.as_transcription(),
                config.transcription_model,
            );

            assert_feature(
                config.name,
                "AsImageGeneration",
                "image_generation_model",
                client.as_image_generation(),
                config.image_generation_model,
            );

            assert_feature(
                config.name,
                "AsAudioGeneration",
                "audio_generation_model",
                client.as_audio_generation(),
                config.audio_generation_model,
            )
        }
    }

    async fn test_embed_client(config: &ClientConfig) {
        const TEST: &str = "Hello world.";

        let client = config.factory_env();

        let Some(client) = client.as_embeddings() else {
            return;
        };

        let model = config.embeddings_model.unwrap();

        let model = client.embedding_model(model);

        let resp = model.embed_text(TEST).await;

        assert!(
            resp.is_ok(),
            "[{}]: Error occurred when sending request, {}",
            config.name,
            resp.err().unwrap()
        );

        let resp = resp.unwrap();

        assert_eq!(resp.document, TEST);

        assert!(
            !resp.vec.is_empty(),
            "[{}]: Returned embed was empty",
            config.name
        );
    }

    #[tokio::test]
    #[ignore]
    async fn test_embed() {
        for config in providers().into_iter().filter(ClientConfig::is_env_var_set) {
            test_embed_client(&config).await;
        }
    }

    async fn test_image_generation_client(config: &ClientConfig) {
        let client = config.factory_env();
        let Some(client) = client.as_image_generation() else {
            return;
        };

        let model = config.image_generation_model.unwrap();

        let model = client.image_generation_model(model);

        let resp = model
            .image_generation(ImageGenerationRequest {
                prompt: "A castle sitting on a large hill.".to_string(),
                width: 256,
                height: 256,
                additional_params: None,
            })
            .await;

        assert!(
            resp.is_ok(),
            "[{}]: Error occurred when sending request, {}",
            config.name,
            resp.err().unwrap()
        );

        let resp = resp.unwrap();

        assert!(
            !resp.image.is_empty(),
            "[{}]: Generated image was empty",
            config.name
        );
    }

    #[tokio::test]
    #[ignore]
    async fn test_image_generation() {
        for config in providers().into_iter().filter(ClientConfig::is_env_var_set) {
            test_image_generation_client(&config).await;
        }
    }

    async fn test_transcription_client(config: &ClientConfig, data: Vec<u8>) {
        let client = config.factory_env();
        let Some(client) = client.as_transcription() else {
            return;
        };

        let model = config.image_generation_model.unwrap();

        let model = client.transcription_model(model);

        let resp = model
            .transcription(TranscriptionRequest {
                data,
                filename: "audio.mp3".to_string(),
                language: "en".to_string(),
                prompt: None,
                temperature: None,
                additional_params: None,
            })
            .await;

        assert!(
            resp.is_ok(),
            "[{}]: Error occurred when sending request, {}",
            config.name,
            resp.err().unwrap()
        );

        let resp = resp.unwrap();

        assert!(
            !resp.text.is_empty(),
            "[{}]: Returned transcription was empty",
            config.name
        );
    }

    #[tokio::test]
    #[ignore]
    async fn test_transcription() {
        let mut file = File::open("examples/audio/en-us-natural-speech.mp3").unwrap();

        let mut data = Vec::new();
        let _ = file.read(&mut data);

        for config in providers().into_iter().filter(ClientConfig::is_env_var_set) {
            test_transcription_client(&config, data.clone()).await;
        }
    }

    #[derive(Deserialize)]
    struct OperationArgs {
        x: f32,
        y: f32,
    }

    #[derive(Debug, thiserror::Error)]
    #[error("Math error")]
    struct MathError;

    #[derive(Deserialize, Serialize)]
    struct Adder;
    impl Tool for Adder {
        const NAME: &'static str = "add";

        type Error = MathError;
        type Args = OperationArgs;
        type Output = f32;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition {
                name: "add".to_string(),
                description: "Add x and y together".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "number",
                            "description": "The first number to add"
                        },
                        "y": {
                            "type": "number",
                            "description": "The second number to add"
                        }
                    }
                }),
            }
        }

        async fn call(&self, args: Self::Args) -> anyhow::Result<Self::Output, Self::Error> {
            println!("[tool-call] Adding {} and {}", args.x, args.y);
            let result = args.x + args.y;
            Ok(result)
        }
    }

    #[derive(Deserialize, Serialize)]
    struct Subtract;
    impl Tool for Subtract {
        const NAME: &'static str = "subtract";

        type Error = MathError;
        type Args = OperationArgs;
        type Output = f32;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            serde_json::from_value(json!({
                "name": "subtract",
                "description": "Subtract y from x (i.e.: x - y)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "number",
                            "description": "The number to subtract from"
                        },
                        "y": {
                            "type": "number",
                            "description": "The number to subtract"
                        }
                    }
                }
            }))
            .expect("Tool Definition")
        }

        async fn call(&self, args: Self::Args) -> anyhow::Result<Self::Output, Self::Error> {
            println!("[tool-call] Subtracting {} from {}", args.y, args.x);
            let result = args.x - args.y;
            Ok(result)
        }
    }
}
