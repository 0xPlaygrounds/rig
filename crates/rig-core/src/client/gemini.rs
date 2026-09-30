//! The Gemini client: a [`GeminiConfig`] on a transport, and the models it
//! builds.

use crate::client::macros::http_client;
use crate::driver::Model;
use crate::error::ProviderError;
use crate::model::ModelList;

use crate::providers::gemini::GeminiConfig;
use crate::providers::gemini::cached_content::CachedContents;
use crate::providers::gemini::completion::GenerateContent;
use crate::providers::gemini::embedding::Embeddings;
#[cfg(feature = "image")]
use crate::providers::gemini::image_generation::Images;
use crate::providers::gemini::interactions_api::{InteractionResume, Interactions};
use crate::providers::gemini::transcription::Transcriptions;

http_client!(
    /// Gemini: its [`GeminiConfig`] on a transport. Every model it builds
    /// sends through that transport.
    Gemini,
    GeminiConfig
);

impl Gemini {
    /// Gemini with `api_key` and the public base URL, on the shared reqwest
    /// client.
    #[cfg(feature = "reqwest")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
    pub fn new(api_key: impl Into<crate::wire::Secret>) -> Self {
        GeminiConfig::new(api_key).client()
    }

    /// Gemini from `GEMINI_API_KEY`, on the shared reqwest client.
    #[cfg(feature = "reqwest")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
    pub fn from_env() -> Result<Self, crate::client::env::EnvError> {
        Ok(GeminiConfig::from_env()?.client())
    }

    /// The `generateContent` / `streamGenerateContent` model for `model`.
    pub fn completion(&self, model: impl Into<String>) -> Model<GenerateContent> {
        self.model(self.config.completion(model))
    }

    /// The Interactions API model for `model`.
    pub fn interactions(&self, model: impl Into<String>) -> Model<Interactions> {
        self.model(self.config.interactions(model))
    }

    /// The embedding model for `model`, at the model's known width.
    pub fn embedding(&self, model: impl Into<String>) -> Model<Embeddings> {
        self.model(self.config.embedding(model))
    }

    /// The embedding model for `model`, asking for `ndims`-wide vectors
    /// ([`Embeddings::with_ndims`]).
    pub fn embedding_with_ndims(
        &self,
        model: impl Into<String>,
        ndims: usize,
    ) -> Model<Embeddings> {
        self.model(self.config.embedding(model).with_ndims(ndims))
    }

    /// The audio transcription model for `model`.
    pub fn transcription(&self, model: impl Into<String>) -> Model<Transcriptions> {
        self.model(self.config.transcription(model))
    }

    /// The image-generation model for `model`.
    #[cfg(feature = "image")]
    #[cfg_attr(docsrs, doc(cfg(feature = "image")))]
    pub fn image_generation(&self, model: impl Into<String>) -> Model<Images> {
        self.model(self.config.image_generation(model))
    }

    /// Gemini's explicit context cache (`cachedContents`).
    pub fn cached_contents(&self) -> Model<CachedContents> {
        self.model(self.config.cached_contents())
    }

    /// The model that retrieves the interaction `interaction_id`, or resumes
    /// its stream. A unary call fetches the current resource; the caller
    /// controls repeated polling.
    pub fn interaction(&self, interaction_id: impl Into<String>) -> Model<InteractionResume> {
        self.model(self.config.interaction(interaction_id))
    }

    /// [`Self::interaction`], resuming a streamed read after the last event
    /// the consumer saw.
    pub fn interaction_resumed(
        &self,
        interaction_id: impl Into<String>,
        last_event_id: Option<&str>,
    ) -> Model<InteractionResume> {
        self.model(
            self.config
                .interaction_resumed(interaction_id, last_event_id),
        )
    }

    /// The models this API key can use, every page followed.
    pub async fn list_models(&self) -> Result<ModelList, ProviderError> {
        self.model(self.config.models()).list().await
    }

    /// Check that the provider accepts the configured key. A 401 or 403
    /// reply is [`ProviderError::InvalidAuthentication`].
    pub async fn verify(&self) -> Result<(), ProviderError> {
        self.model(self.config.verify()).verify().await
    }
}
