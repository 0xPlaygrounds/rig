//! Cohere's configuration, and its two chat wires: the OpenAI Compatibility
//! API and the native chat API.
//!
//! ```
//! use rig_core::providers::cohere::CohereConfig;
//! let config = CohereConfig::new("key").with_base_url("https://api.cohere.ai/");
//! assert_eq!(config.base_url, "https://api.cohere.ai");
//! let chat = config.native_completion("command-a-03-2025");
//! assert_eq!(chat.model, "command-a-03-2025");
//! ```

use crate::client::env::{self, EnvError};
use crate::providers::openai::wire::{COHERE, Chat, OpenAIConfig};
use crate::wire::Secret;
use serde::{Deserialize, Serialize};

use super::NativeChat;

/// Cohere's API root.
const BASE_URL: &str = "https://api.cohere.ai";

/// Where Cohere's OpenAI Compatibility API sits under the API root.
const COMPATIBILITY_PATH: &str = "/compatibility/v1";

/// The environment variable carrying the API key.
const API_KEY_ENV: &str = "COHERE_API_KEY";

/// The settings of a Cohere provider: serializable, and the credential is
/// never serialized. [`connect`](Self::connect) puts it on a transport as a
/// [`Cohere`](super::Cohere) client.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CohereConfig {
    /// The API key, sent as `Authorization: Bearer`.
    pub api_key: Secret,
    /// The API root, without a trailing slash.
    pub base_url: String,
}

impl CohereConfig {
    /// Cohere with default settings.
    pub fn new(api_key: impl Into<Secret>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: BASE_URL.to_owned(),
        }
    }

    /// Cohere from `COHERE_API_KEY`.
    pub fn from_env() -> Result<Self, EnvError> {
        Ok(Self::new(env::required(API_KEY_ENV)?))
    }

    /// Point the wires at another API root.
    pub fn with_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.base_url = base_url.as_ref().trim_end_matches('/').to_owned();
        self
    }

    /// The chat wire for `model` on the OpenAI Compatibility API under this
    /// API root. It sends a request's documents as text in the history.
    pub fn completion(&self, model: impl Into<String>) -> Chat {
        OpenAIConfig::with_key(&COHERE, self.api_key.clone())
            .with_base_url(format!("{}{COMPATIBILITY_PATH}", self.base_url))
            .chat(model)
    }

    /// The chat wire for `model` on the native chat API under this API root,
    /// which grounds the answer in a request's documents and cites them.
    pub fn native_completion(&self, model: impl Into<String>) -> NativeChat {
        NativeChat::new(self.clone(), model)
    }
}

#[cfg(test)]
mod tests;
