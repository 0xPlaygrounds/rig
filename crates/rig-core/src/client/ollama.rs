//! The Ollama client: an [`OllamaConfig`] on a transport, and the models it
//! builds.

use crate::client::macros::http_client;
use crate::driver::Model;
use crate::error::ProviderError;
use crate::model::ModelList;

use crate::providers::ollama::wire::{Chat, Embeddings, OllamaConfig};

http_client!(
    /// An Ollama daemon: its [`OllamaConfig`] on a transport. Every model it
    /// builds sends through that transport.
    Ollama,
    OllamaConfig
);

impl Ollama {
    /// The local daemon, unauthenticated, on the shared reqwest client.
    #[cfg(feature = "reqwest")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
    #[allow(
        clippy::new_without_default,
        reason = "the configuration is the default; a client also names its transport"
    )]
    pub fn new() -> Self {
        OllamaConfig::new().client()
    }

    /// The daemon `OLLAMA_API_BASE_URL` names, with the token
    /// `OLLAMA_API_KEY` carries, on the shared reqwest client. Both are
    /// optional.
    #[cfg(feature = "reqwest")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
    pub fn from_env() -> Result<Self, crate::client::env::EnvError> {
        Ok(OllamaConfig::from_env()?.client())
    }

    /// The chat model for `model`.
    pub fn completion(&self, model: impl Into<String>) -> Model<Chat> {
        self.model(self.config.completion(model))
    }

    /// The embedding model for `model`, at its native width. Declare the
    /// width of a model this build does not know with
    /// [`EmbeddingWidth`](crate::embeddings::EmbeddingWidth).
    pub fn embedding(&self, model: impl Into<String>) -> Model<Embeddings> {
        self.model(self.config.embedding(model))
    }

    /// The models the daemon serves.
    pub async fn list_models(&self) -> Result<ModelList, ProviderError> {
        self.model(self.config.models()).list().await
    }
}
