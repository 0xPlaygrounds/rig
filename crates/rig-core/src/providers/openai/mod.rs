//! OpenAI: the client and its configuration, the Responses and Chat
//! Completions wires, and every OpenAI-shaped dialect.
//!
//! ```no_run
//! use rig_core::providers::openai;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = openai::OpenAI::from_env()?;
//!
//! let gpt_5_2 = provider.responses(openai::GPT_5_2);
//! let chat = provider.chat(openai::GPT_5_2);
//! let embeddings = provider.embedding(openai::TEXT_EMBEDDING_3_SMALL, None);
//! # let _ = (gpt_5_2, chat, embeddings);
//! # Ok(())
//! # }
//! ```
//!
//! Each model pairs a wire, which says what to send and how to read the reply,
//! with the client's transport. A vendor that speaks this format builds its
//! client from its own module, such as [`crate::providers::deepseek::from_env`].

pub mod completion;
pub mod embedding;
pub mod responses_api;

/// The OpenAI wires: the configuration, the chat-completions wire and one
/// `Dialect` constant per OpenAI-shaped provider.
pub mod wire;

pub use wire::{OpenAIConfig, Route};

use crate::client::macros::http_client;
use crate::driver::Model;
use crate::error::ProviderError;
use crate::model::ModelList;
use crate::providers::chatgpt::auth::{AuthError, Authenticator};
use responses_api::wire::Responses;
#[cfg(feature = "image")]
use wire::Images;
#[cfg(feature = "audio")]
use wire::Speech;
use wire::{Chat, Embeddings, OpenAiWire, Rerank, Transcriptions};

#[cfg(feature = "audio")]
#[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
pub mod audio_generation;

#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub mod image_generation;
#[cfg(feature = "image")]
pub use image_generation::*;

pub mod transcription;

pub use completion::*;
pub use embedding::*;

/// Sanitize nested schema definitions, properties, items, and combinators.
/// Require all properties, supply missing object properties and
/// `additionalProperties: false`, remove `$ref` siblings, and merge `oneOf`
/// into `anyOf`.
pub(crate) fn sanitize_schema(schema: &mut serde_json::Value) {
    crate::providers::internal::schema::sanitize_schema(
        schema,
        crate::providers::internal::schema::SanitizeOptions {
            strip_ref_siblings: true,
            inject_empty_properties: true,
            strip_numeric_constraints: false,
        },
    );
}

/// Return the schema title, or `response_schema` when absent, and a sanitized
/// schema for either structured-output endpoint.
pub(crate) fn structured_output_schema(schema: schemars::Schema) -> (String, serde_json::Value) {
    let name = schema
        .as_object()
        .and_then(|object| object.get("title"))
        .and_then(|title| title.as_str())
        .unwrap_or("response_schema")
        .to_string();
    let mut value = schema.to_value();
    sanitize_schema(&mut value);
    (name, value)
}

#[cfg(feature = "audio")]
pub use audio_generation::{TTS_1, TTS_1_HD};

pub use transcription::*;

http_client!(
    /// An OpenAI-shaped provider: its [`OpenAIConfig`] on a transport. Every
    /// model it builds sends through that transport.
    ///
    /// ```no_run
    /// use rig_core::completion::CompletionRequest;
    /// use rig_core::providers::openai::{self, OpenAIConfig};
    ///
    /// # async fn run(http: rig_core::http_client::DynHttpClient) -> Result<(), Box<dyn std::error::Error>> {
    /// let openai = OpenAIConfig::from_env()?.connect(http);
    /// let model = openai.completion(openai::GPT_5_2);
    /// let response = model.call(CompletionRequest::new("Capital of France?")).await?;
    /// # let _ = response;
    /// # Ok(())
    /// # }
    /// ```
    OpenAI,
    OpenAIConfig
);

impl OpenAI {
    /// Official OpenAI with `api_key`, on the shared reqwest client.
    #[cfg(feature = "reqwest")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
    pub fn new(api_key: impl Into<crate::wire::Secret>) -> Self {
        OpenAIConfig::new(api_key).client()
    }

    /// Official OpenAI from `OPENAI_API_KEY` and the optional
    /// `OPENAI_BASE_URL`, on the shared reqwest client.
    #[cfg(feature = "reqwest")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
    pub fn from_env() -> Result<Self, crate::client::env::EnvError> {
        Ok(OpenAIConfig::from_env()?.client())
    }

    /// The completion model for `model` on the configuration's
    /// [`completion_route`](OpenAIConfig::completion_route).
    pub fn completion(&self, model: impl Into<String>) -> Model<OpenAiWire> {
        self.model(self.config.completion(model))
    }

    /// The Responses model for `model`: `POST /responses`.
    pub fn responses(&self, model: impl Into<String>) -> Model<Responses> {
        self.model(self.config.responses(model))
    }

    /// The Chat Completions model for `model`.
    pub fn chat(&self, model: impl Into<String>) -> Model<Chat> {
        self.model(self.config.chat(model))
    }

    /// The embedding model for `model`, `ndims` wide when set.
    pub fn embedding(&self, model: impl Into<String>, ndims: Option<usize>) -> Model<Embeddings> {
        self.model(self.config.embedding(model, ndims))
    }

    /// The rerank model for `model`.
    pub fn rerank(&self, model: impl Into<String>) -> Model<Rerank> {
        self.model(self.config.rerank(model))
    }

    /// The transcription model for `model`.
    pub fn transcription(&self, model: impl Into<String>) -> Model<Transcriptions> {
        self.model(self.config.transcription(model))
    }

    /// The image-generation model for `model`.
    #[cfg(feature = "image")]
    #[cfg_attr(docsrs, doc(cfg(feature = "image")))]
    pub fn image_generation(&self, model: impl Into<String>) -> Model<Images> {
        self.model(self.config.image_generation(model))
    }

    /// The speech model for `model`.
    #[cfg(feature = "audio")]
    #[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
    pub fn audio_generation(&self, model: impl Into<String>) -> Model<Speech> {
        self.model(self.config.audio_generation(model))
    }

    /// The models this provider serves, every page followed.
    pub async fn list_models(&self) -> Result<ModelList, ProviderError> {
        self.model(self.config.models()).list().await
    }

    /// Check that the provider accepts the configured credential. A 401 or
    /// 403 reply is [`ProviderError::InvalidAuthentication`].
    pub async fn verify(&self) -> Result<(), ProviderError> {
        self.model(self.config.verify()).verify().await
    }

    /// This client with the ChatGPT subscription credential `authenticator`
    /// resolves, reading or refreshing it through this client's transport:
    /// the access token, and the account it belongs to when one is named.
    /// Every other setting is kept.
    ///
    /// ```no_run
    /// use rig_core::providers::chatgpt::{self, auth::{AuthSource, Authenticator, DeviceCodeHandler}};
    /// use rig_core::providers::openai::OpenAIConfig;
    ///
    /// # async fn run(http: rig_core::http_client::DynHttpClient) -> Result<(), Box<dyn std::error::Error>> {
    /// let authenticator = Authenticator::new(AuthSource::OAuth, None, DeviceCodeHandler::default(), true);
    /// let chatgpt = OpenAIConfig::with_key(&chatgpt::DIALECT, "")
    ///     .connect(http)
    ///     .authenticate(&authenticator)
    ///     .await?;
    /// # let _ = chatgpt;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn authenticate(self, authenticator: &Authenticator) -> Result<Self, AuthError> {
        let context = authenticator.auth_context(&self.http).await?;
        let mut config = self.config;
        config.api_key = context.access_token;
        if let Some(account_id) = context.account_id {
            config.account_id = Some(account_id);
        }
        Ok(Self {
            config,
            http: self.http,
        })
    }
}

#[cfg(test)]
mod tests;
