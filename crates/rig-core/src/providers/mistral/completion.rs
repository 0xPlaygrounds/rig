//! Mistral chat model identifiers and its reply type with service-tier and audio usage.
//!
//! ```no_run
//! use rig_core::providers::mistral;
//! let model = mistral::from_env()?.chat(mistral::MISTRAL_SMALL);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use serde::{Deserialize, Serialize};

use crate::providers::openai;

/// The latest version of the `codestral` Mistral model
pub const CODESTRAL: &str = "codestral-latest";
/// The latest version of the `mistral-large` Mistral model
pub const MISTRAL_LARGE: &str = "mistral-large-latest";
/// The latest version of the `mistral-3b` Mistral completions model
pub const MINISTRAL_3B: &str = "ministral-3b-latest";
/// The latest version of the `mistral-8b` Mistral completions model
pub const MINISTRAL_8B: &str = "ministral-8b-latest";

/// The latest version of the `mistral-small` Mistral completions model
pub const MISTRAL_SMALL: &str = "mistral-small-latest";

/// Mistral's reply, read back from
/// [`crate::completion::CompletionResponse::raw`].
pub type CompletionResponse = openai::completion::ChatCompletionResponse<Usage>;

/// Token usage returned by Mistral's chat completions endpoint.
///
/// See <https://docs.mistral.ai/api/> (`UsageInfo` schema). Mistral fills the
/// fields beyond the three counts on a best-effort basis. Its top-level
/// `num_cached_tokens` and `prompt_tokens_details.audio_tokens` are fields of
/// [`openai::Usage`].
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Usage {
    /// The OpenAI-compatible counters.
    #[serde(flatten)]
    pub openai: openai::Usage,
    /// Capacity tier that served the request, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    /// Duration in seconds of audio tokens in the prompt (audio-input models only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_audio_seconds: Option<u64>,
}

#[cfg(test)]
mod tests;
