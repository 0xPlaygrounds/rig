//! The Anthropic client: an [`AnthropicConfig`] on a transport, and the
//! models it builds.

use crate::driver::Model;
use crate::error::ProviderError;
use crate::model::ModelList;
use crate::providers::internal::client::http_client;

use super::wire::{AnthropicConfig, Messages};

http_client!(
    /// A Messages-format provider: its [`AnthropicConfig`] on a transport.
    /// Every model it builds sends through that transport.
    Anthropic,
    AnthropicConfig
);

impl Anthropic {
    /// Anthropic with `api_key` and default settings, on the shared reqwest
    /// client.
    #[cfg(feature = "reqwest")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
    pub fn new(api_key: impl Into<crate::wire::Secret>) -> Self {
        AnthropicConfig::new(api_key).client()
    }

    /// Anthropic from `ANTHROPIC_API_KEY` and `ANTHROPIC_BASE_URL`, on the
    /// shared reqwest client.
    #[cfg(feature = "reqwest")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
    pub fn from_env() -> Result<Self, crate::client::env::EnvError> {
        Ok(AnthropicConfig::from_env()?.client())
    }

    /// The Messages model for `model`.
    pub fn completion(&self, model: impl Into<String>) -> Model<Messages> {
        self.model(self.config.completion(model))
    }

    /// The models this provider serves, every page followed.
    pub async fn list_models(&self) -> Result<ModelList, ProviderError> {
        self.model(self.config.models()).call(()).await
    }

    /// Check that the provider accepts the configured credential. A 401 or
    /// 403 reply is [`ProviderError::InvalidAuthentication`].
    pub async fn verify(&self) -> Result<(), ProviderError> {
        self.model(self.config.verify()).verify().await
    }
}
