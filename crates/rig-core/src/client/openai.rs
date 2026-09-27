//! The OpenAI client: an [`OpenAIConfig`] on a transport, and the models it
//! builds.

use crate::client::macros::http_client;
use crate::driver::Model;
use crate::error::ProviderError;
use crate::model::ModelList;
use crate::providers::chatgpt::auth::{AuthError, Authenticator};
use crate::providers::openai::responses_api::wire::Responses;

#[cfg(feature = "image")]
use crate::providers::openai::wire::Images;
#[cfg(feature = "audio")]
use crate::providers::openai::wire::Speech;
use crate::providers::openai::wire::{
    Chat, Embeddings, OpenAIConfig, OpenAiWire, Rerank, Transcriptions,
};

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
