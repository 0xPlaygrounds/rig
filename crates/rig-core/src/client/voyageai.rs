//! The Voyage AI client: a [`VoyageAiConfig`] on a transport, and the models
//! it builds.

use crate::client::macros::http_client;
use crate::driver::Model;

use crate::providers::voyageai::wire::{Embeddings, Rerank, VoyageAiConfig};

http_client!(
    /// Voyage AI: its [`VoyageAiConfig`] on a transport. Every model it
    /// builds sends through that transport.
    VoyageAi,
    VoyageAiConfig
);

impl VoyageAi {
    /// Voyage AI with `api_key` and default settings, on the shared reqwest
    /// client.
    #[cfg(feature = "reqwest")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
    pub fn new(api_key: impl Into<crate::wire::Secret>) -> Self {
        VoyageAiConfig::new(api_key).client()
    }

    /// Voyage AI from `VOYAGE_API_KEY`, on the shared reqwest client.
    #[cfg(feature = "reqwest")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
    pub fn from_env() -> Result<Self, crate::client::env::EnvError> {
        Ok(VoyageAiConfig::from_env()?.client())
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

    /// The rerank model for `model`.
    pub fn rerank(&self, model: impl Into<String>) -> Model<Rerank> {
        self.model(self.config.rerank(model))
    }
}
