//! llama.cpp chat response types, including optional server-side timings.
//!
//! ```
//! use rig_core::providers::llamacpp::Timings;
//! let timings: Timings = serde_json::from_str(r#"{"predicted_n": 10}"#)?;
//! assert_eq!(timings.predicted_n, Some(10));
//! # Ok::<(), serde_json::Error>(())
//! ```

use crate::providers::openai;
use serde::{Deserialize, Serialize};

/// Conventional identifier for a single-model server. For a multi-model router,
/// use an identifier from its model listing instead.
pub const LLAMA_CPP: &str = "LLaMA_CPP";

/// Server-reported timing and throughput fields. Missing fields deserialize as
/// `None`; these measurements are separate from normalized token usage.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct Timings {
    /// Server-reported prompt tokens served from the KV cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_n: Option<u64>,
    /// Prompt tokens actually evaluated this turn (total minus `cache_n`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_n: Option<u64>,
    /// Wall-clock milliseconds spent evaluating the prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_ms: Option<f64>,
    /// `prompt_ms / prompt_n`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_per_token_ms: Option<f64>,
    /// Prompt-evaluation throughput.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_per_second: Option<f64>,
    /// Tokens generated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicted_n: Option<u64>,
    /// Wall-clock milliseconds spent generating.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicted_ms: Option<f64>,
    /// `predicted_ms / predicted_n`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicted_per_token_ms: Option<f64>,
    /// Generation throughput in tokens per second.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicted_per_second: Option<f64>,
}

/// Typed chat-completions payload with optional timings. Deserialize from
/// [`crate::completion::CompletionResponse::raw`] to inspect server measurements;
/// this does not model the native `/completion` response.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
    /// The OpenAI-compatible half of the payload.
    #[serde(flatten)]
    pub openai: openai::CompletionResponse,
    /// llama.cpp's server-side timing accounting, when the server reported it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timings: Option<Timings>,
}

#[cfg(test)]
mod tests;
