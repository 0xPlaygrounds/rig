//! Mistral's typed request options and reply extras
//! (<https://docs.mistral.ai/api/endpoint/chat>).
//!
//! ```
//! use rig_core::completion::{CompletionRequest, ProviderOptions};
//! use rig_core::providers::mistral::extension::{MistralExt, MistralOptions, PromptMode};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let options = MistralOptions::new().prompt_mode(PromptMode::Reasoning).safe_prompt(true);
//! let request = CompletionRequest::new("hi")
//!     .provider_options(ProviderOptions::new().with::<MistralExt>(&options)?);
//! # let _ = request;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::completion::provider_options::reply_field;
use crate::completion::{ExtensionOptions, ProviderExtension, ReplyExtras};
use crate::message::Api;
use crate::providers::openai::extension::Prediction;

/// Mistral's extension marker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MistralExt;

impl ProviderExtension for MistralExt {
    const PROVIDER: &'static str = super::PROVIDER_NAME;
    type Options = MistralOptions;
    type Extras = MistralExtras;
}

/// Mistral's request options.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct MistralOptions {
    /// The fields every route takes.
    #[serde(rename = "*")]
    pub shared: MistralShared,
}

/// The fields Mistral takes.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct MistralShared {
    /// The system prompt Mistral adds for a reasoning model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_mode: Option<PromptMode>,
    /// Whether Mistral prepends its safety prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safe_prompt: Option<bool>,
    /// The prompt-cache routing key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    /// Penalizes tokens by how often they already appear.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    /// Penalizes tokens that already appear.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// Predicted output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<Prediction>,
}

/// The system prompt Mistral adds.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PromptMode {
    /// The reasoning system prompt.
    Reasoning,
}

impl MistralOptions {
    /// No option set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add Mistral's `mode` system prompt.
    pub fn prompt_mode(mut self, mode: PromptMode) -> Self {
        self.shared.prompt_mode = Some(mode);
        self
    }

    /// Whether Mistral prepends its safety prompt.
    pub fn safe_prompt(mut self, safe: bool) -> Self {
        self.shared.safe_prompt = Some(safe);
        self
    }

    /// Route the prompt cache by `key`.
    pub fn prompt_cache_key(mut self, key: impl Into<String>) -> Self {
        self.shared.prompt_cache_key = Some(key.into());
        self
    }

    /// Set the frequency penalty.
    pub fn frequency_penalty(mut self, penalty: f64) -> Self {
        self.shared.frequency_penalty = Some(penalty);
        self
    }

    /// Set the presence penalty.
    pub fn presence_penalty(mut self, penalty: f64) -> Self {
        self.shared.presence_penalty = Some(penalty);
        self
    }

    /// Predict the output as `content`.
    pub fn prediction(mut self, content: impl Into<String>) -> Self {
        self.shared.prediction = Some(Prediction::content(content));
        self
    }
}

impl ExtensionOptions for MistralOptions {
    type Ext = MistralExt;
}

/// Mistral's reply fields. Each is `None` when the reply lacks it.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MistralExtras {
    /// The service tier that served the request.
    pub service_tier: Option<String>,
    /// Seconds of audio in the prompt.
    pub prompt_audio_seconds: Option<u64>,
    /// Prompt tokens read from the cache, as older replies report them.
    pub num_cached_tokens: Option<u64>,
    /// Prompt token details.
    pub prompt_tokens_details: Option<MistralPromptTokens>,
}

/// Where a Mistral reply's prompt tokens went.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
pub struct MistralPromptTokens {
    /// Tokens read from the cache.
    #[serde(default)]
    pub cached_tokens: Option<u64>,
    /// Audio tokens.
    #[serde(default)]
    pub audio_tokens: Option<u64>,
}

impl ReplyExtras for MistralExtras {
    fn from_reply(_api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        Ok(Self {
            service_tier: reply_field(raw, "/usage/service_tier")?,
            prompt_audio_seconds: reply_field(raw, "/usage/prompt_audio_seconds")?,
            num_cached_tokens: reply_field(raw, "/usage/num_cached_tokens")?,
            prompt_tokens_details: reply_field(raw, "/usage/prompt_tokens_details")?,
        })
    }
}

#[cfg(test)]
mod tests;
