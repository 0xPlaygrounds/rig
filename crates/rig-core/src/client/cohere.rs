//! The Cohere client: a [`CohereConfig`] on a transport, and the models it
//! builds.

use crate::client::macros::http_client;
use crate::driver::Model;

use crate::providers::cohere::{CohereConfig, Embeddings, ImageEmbeddings};
use crate::providers::openai::wire::Chat;

http_client!(
    /// Cohere: its [`CohereConfig`] on a transport. Every model it builds
    /// sends through that transport.
    Cohere,
    CohereConfig
);

impl Cohere {
    /// Cohere with `api_key` and default settings, on the shared reqwest
    /// client.
    #[cfg(feature = "reqwest")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
    pub fn new(api_key: impl Into<crate::wire::Secret>) -> Self {
        CohereConfig::new(api_key).client()
    }

    /// Cohere from `COHERE_API_KEY`, on the shared reqwest client.
    #[cfg(feature = "reqwest")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
    pub fn from_env() -> Result<Self, crate::client::env::EnvError> {
        Ok(CohereConfig::from_env()?.client())
    }

    /// The chat model for `model`, on Cohere's OpenAI Compatibility API.
    pub fn completion(&self, model: impl Into<String>) -> Model<Chat> {
        self.model(self.config.completion(model))
    }

    /// The text-embedding model for `model`. `ndims` is the width it
    /// reports, defaulting to the model's known width.
    pub fn embedding(&self, model: impl Into<String>, ndims: Option<usize>) -> Model<Embeddings> {
        self.model(self.config.embedding(model, ndims))
    }

    /// The image-embedding model. Cohere embeds images with one fixed model,
    /// so it names none.
    pub fn image_embedding(&self) -> Model<ImageEmbeddings> {
        self.model(self.config.image_embedding())
    }
}
