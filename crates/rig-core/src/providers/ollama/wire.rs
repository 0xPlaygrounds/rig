//! Ollama daemon configuration: its address and optional credential.
//!
//! ```
//! use rig_core::providers::ollama::OllamaConfig;
//! let ollama = OllamaConfig::new().client();
//! assert_eq!(ollama.completion("qwen3").wire.model, "qwen3");
//! ```

use crate::client::env::{self, EnvError};
use crate::providers::openai::wire::{Chat, OLLAMA, OpenAIConfig};
use crate::wire::Secret;
use serde::{Deserialize, Serialize};

use super::OLLAMA_API_BASE_URL;

/// The environment variable overriding the daemon's address.
const BASE_URL_ENV: &str = "OLLAMA_API_BASE_URL";

/// The environment variable carrying the bearer token a proxied daemon
/// requires. A local daemon needs none.
const API_KEY_ENV: &str = "OLLAMA_API_KEY";

/// Where the daemon serves its OpenAI-compatible API.
const OPENAI_COMPATIBLE_PATH: &str = "/v1";

/// The settings of an Ollama daemon: serializable, and the credential is
/// never serialized. [`connect`](Self::connect) puts it on a transport as an
/// [`Ollama`](super::Ollama) client.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OllamaConfig {
    /// The daemon's address.
    pub base_url: String,
    /// Bearer token for a secured daemon. Empty by default; no credential is
    /// sent when empty.
    pub api_key: Secret,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl OllamaConfig {
    /// The local daemon, unauthenticated.
    pub fn new() -> Self {
        Self {
            base_url: OLLAMA_API_BASE_URL.to_owned(),
            api_key: Secret::default(),
        }
    }

    /// The daemon `OLLAMA_API_BASE_URL` names, with the token
    /// `OLLAMA_API_KEY` carries. Both are optional: an unset base URL is the
    /// local daemon and an unset key is no credential.
    pub fn from_env() -> Result<Self, EnvError> {
        let mut provider = Self::new();
        if let Some(base_url) = env::optional(BASE_URL_ENV)? {
            provider = provider.with_base_url(base_url);
        }
        if let Some(api_key) = env::optional(API_KEY_ENV)? {
            provider.api_key = api_key.into();
        }
        Ok(provider)
    }

    /// Point the wires at another daemon.
    pub fn with_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.base_url = base_url.as_ref().trim_end_matches('/').to_owned();
        self
    }

    /// Authenticate against a proxied daemon.
    pub fn with_api_key(mut self, api_key: impl Into<Secret>) -> Self {
        self.api_key = api_key.into();
        self
    }

    /// The Chat Completions wire for `model`, on the daemon's
    /// OpenAI-compatible API. The credential is sent only when there is one.
    pub(crate) fn completion(&self, model: impl Into<String>) -> Chat {
        OpenAIConfig::with_key(&OLLAMA, self.api_key.clone())
            .with_base_url(format!("{}{OPENAI_COMPATIBLE_PATH}", self.base_url))
            .chat(model)
    }

    /// One request to `path`, with the credential only when there is one.
    pub(super) fn request(&self, method: http::Method, path: &str) -> http::request::Builder {
        let builder = http::Request::builder()
            .method(method)
            .uri(format!("{}{path}", self.base_url))
            .header(http::header::CONTENT_TYPE, "application/json");
        if self.api_key.is_empty() {
            builder
        } else {
            builder.header(
                http::header::AUTHORIZATION,
                format!("Bearer {}", self.api_key.expose()),
            )
        }
    }
}

#[cfg(test)]
mod tests;
