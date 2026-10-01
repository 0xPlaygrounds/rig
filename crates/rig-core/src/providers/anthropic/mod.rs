//! Anthropic provider configuration and Messages-format endpoint wires.
//!
//! ```no_run
//! use rig_core::providers::anthropic;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = anthropic::Anthropic::from_env()?;
//!
//! let sonnet = provider.completion(anthropic::completion::CLAUDE_SONNET_4_6);
//! # Ok(())
//! # }
//! ```
//!
//! Pair a wire with a transport in a [`Model`] to send it.

pub mod completion;
pub mod streaming;
pub mod wire;

pub use completion::{
    CLAUDE_FABLE_5, CLAUDE_FABLE_5_1, CLAUDE_HAIKU_4_5, CLAUDE_OPUS_4_6, CLAUDE_OPUS_4_7,
    CLAUDE_OPUS_4_8, CLAUDE_OPUS_5, CLAUDE_OPUS_5_5, CLAUDE_SONNET_4_6, CLAUDE_SONNET_5,
    CLAUDE_SONNET_5_5,
};
pub use wire::{
    ANTHROPIC, AnthropicConfig, Dialect, MaxTokens, Messages, Models, Quirks, Verify, compatible,
};

use crate::client::macros::http_client;
use crate::driver::Model;
use crate::error::ProviderError;
use crate::model::ModelList;

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
        self.model(self.config.models()).list().await
    }

    /// Check that the provider accepts the configured credential. A 401 or
    /// 403 reply is [`ProviderError::InvalidAuthentication`].
    pub async fn verify(&self) -> Result<(), ProviderError> {
        self.model(self.config.verify()).verify().await
    }
}
