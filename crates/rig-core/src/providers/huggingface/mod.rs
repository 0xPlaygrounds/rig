//! Hugging Face inference-router integration.
//!
//! # Example
//! ```no_run
//! use rig_core::providers::huggingface;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! # let request = rig_core::completion::CompletionRequest::from_prompt("hello");
//! let cfg = huggingface::functions::Config::from_env(huggingface::completion::GEMMA_2)?;
//! let rt = rig_core::http_runtime::HttpRuntime::new();
//! let response = huggingface::functions::complete(&cfg, &rt, request).await?;
//! # Ok(())
//! # }
//! ```

pub mod completion;
pub mod functions;

crate::providers::client::define_http_client! {
    config = functions::Config,
    default_base_url = functions::DEFAULT_BASE_URL,
    api_key_required = true,
}

impl Client {
    /// Materialize transcription configuration sharing this connection.
    pub fn transcription_config(&self, model: impl Into<String>) -> functions::Config {
        self.config(model)
    }

    /// Materialize image-generation configuration sharing this connection.
    #[cfg(feature = "image")]
    pub fn image_generation_config(&self, model: impl Into<String>) -> functions::Config {
        self.config(model)
    }
}

#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub mod image_generation;
pub mod transcription;

#[cfg(feature = "image")]
pub use image_generation::image_generation_models::*;

#[cfg(feature = "image")]
use crate::image_generation::ImageGenerationError;
use crate::transcription::TranscriptionError;
use std::fmt::Display;

/// Default Hugging Face inference-router base URL.
pub const HUGGINGFACE_API_BASE_URL: &str = "https://router.huggingface.co";

/// A Hugging Face inference-router sub-provider.
///
/// Serialized by name (`"hf-inference"`, `"together"`, …);
/// [`SubProvider::Custom`] round-trips as `{"custom": "<route>"}`, so the
/// representation is lossless.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum SubProvider {
    #[default]
    #[serde(rename = "hf-inference")]
    HFInference,
    #[serde(rename = "together")]
    Together,
    #[serde(rename = "sambanova")]
    SambaNova,
    #[serde(rename = "fireworks")]
    Fireworks,
    #[serde(rename = "hyperbolic")]
    Hyperbolic,
    #[serde(rename = "nebius")]
    Nebius,
    #[serde(rename = "novita")]
    Novita,
    #[serde(rename = "custom")]
    Custom(String),
}

impl SubProvider {
    /// Get the chat completion endpoint for the SubProvider
    /// Required because Huggingface Inference requires the model
    /// in the url and in the request body.
    pub fn completion_endpoint(&self, _model: &str) -> String {
        "v1/chat/completions".to_string()
    }

    /// Get the transcription endpoint for the SubProvider
    /// Required because Huggingface Inference requires the model
    /// in the url and in the request body.
    pub fn transcription_endpoint(&self, model: &str) -> Result<String, TranscriptionError> {
        match self {
            SubProvider::HFInference => Ok(format!("/{model}")),
            _ => Err(TranscriptionError::ProviderError(format!(
                "transcription endpoint is not supported yet for {self}"
            ))),
        }
    }

    /// Get the image generation endpoint for the SubProvider
    /// Required because Huggingface Inference requires the model
    /// in the url and in the request body.
    #[cfg(feature = "image")]
    pub fn image_generation_endpoint(&self, model: &str) -> Result<String, ImageGenerationError> {
        match self {
            SubProvider::HFInference => Ok(format!("/{model}")),
            _ => Err(ImageGenerationError::ProviderError(format!(
                "image generation endpoint is not supported yet for {self}"
            ))),
        }
    }

    pub fn model_identifier(&self, model: &str) -> String {
        match self {
            // Fireworks addresses models by a fully-qualified id. Guard against
            // re-prefixing an already-qualified id (e.g. a per-request model
            // override that is already fully qualified) — the generic path
            // applies this to the resolved request model unconditionally, so
            // without the guard a qualified override would become an invalid
            // `accounts/fireworks/models/accounts/fireworks/models/...` id.
            SubProvider::Fireworks => {
                const FIREWORKS_PREFIX: &str = "accounts/fireworks/models/";
                if model.starts_with(FIREWORKS_PREFIX) {
                    model.to_string()
                } else {
                    format!("{FIREWORKS_PREFIX}{model}")
                }
            }
            _ => model.to_string(),
        }
    }
}

impl From<&str> for SubProvider {
    fn from(s: &str) -> Self {
        SubProvider::Custom(s.to_string())
    }
}

impl From<String> for SubProvider {
    fn from(value: String) -> Self {
        SubProvider::Custom(value)
    }
}

impl Display for SubProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let route = match self {
            SubProvider::HFInference => "hf-inference/models".to_string(),
            SubProvider::Together => "together".to_string(),
            SubProvider::SambaNova => "sambanova".to_string(),
            SubProvider::Fireworks => "fireworks-ai".to_string(),
            SubProvider::Hyperbolic => "hyperbolic".to_string(),
            SubProvider::Nebius => "nebius".to_string(),
            SubProvider::Novita => "novita".to_string(),
            SubProvider::Custom(route) => route.clone(),
        };

        write!(f, "{route}")
    }
}

#[cfg(test)]
mod tests {
    use super::SubProvider;

    #[test]
    fn fireworks_model_identifier_is_idempotent() {
        // A bare id is qualified once...
        assert_eq!(
            SubProvider::Fireworks.model_identifier("deepseek-v3"),
            "accounts/fireworks/models/deepseek-v3"
        );
        // ...and an already-qualified id (e.g. a per-request model override)
        // is left untouched rather than double-prefixed.
        assert_eq!(
            SubProvider::Fireworks.model_identifier("accounts/fireworks/models/deepseek-v3"),
            "accounts/fireworks/models/deepseek-v3"
        );
        // Other sub-providers pass the id through verbatim.
        assert_eq!(
            SubProvider::HFInference.model_identifier("meta-llama/Llama-3.1-8B"),
            "meta-llama/Llama-3.1-8B"
        );
    }
}
