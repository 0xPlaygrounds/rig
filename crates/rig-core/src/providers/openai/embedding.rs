use crate::json_utils;
use serde::{Deserialize, Serialize};
use std::fmt;

/// `text-embedding-3-large` embedding model
pub const TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";
/// `text-embedding-3-small` embedding model
pub const TEXT_EMBEDDING_3_SMALL: &str = "text-embedding-3-small";
/// `text-embedding-ada-002` embedding model
pub const TEXT_EMBEDDING_ADA_002: &str = "text-embedding-ada-002";

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

/// Typed raw response from [`Embeddings`](super::wire::Embeddings).
/// Missing object and model fields default to empty strings; usage is optional.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibleEmbeddingResponse {
    #[serde(default)]
    pub object: String,
    pub data: Vec<EmbeddingData>,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EncodingFormat {
    Float,
    Base64,
}

/// One embedded input. Missing object and index fields default to empty and zero.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    #[serde(default)]
    pub object: String,
    pub embedding: Vec<serde_json::Number>,
    #[serde(default)]
    pub index: usize,
}

/// Return default dimensions for a known model identifier, or `None`.
pub(crate) fn model_dimensions_from_identifier(identifier: &str) -> Option<usize> {
    match identifier {
        TEXT_EMBEDDING_3_LARGE => Some(3_072),
        TEXT_EMBEDDING_3_SMALL | TEXT_EMBEDDING_ADA_002 => Some(1_536),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Default)]
pub struct PromptTokensDetails {
    /// Cached tokens from prompt caching
    #[serde(default)]
    pub cached_tokens: usize,
    /// Audio input tokens, defaulting null or missing values to zero.
    /// Zero is omitted from serialization. [`Usage::to_normalized`] uses the
    /// reported total to determine whether audio is additional to prompt tokens.
    #[serde(
        default,
        deserialize_with = "json_utils::null_or_default",
        skip_serializing_if = "is_zero"
    )]
    pub audio_tokens: usize,
    /// Tokens written to cache on this call. `None` means unreported, not zero.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<usize>,
}

/// Whether a counter is absent-as-zero, for `skip_serializing_if`.
fn is_zero(value: &usize) -> bool {
    *value == 0
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Default)]
pub struct CompletionTokensDetails {
    /// Reasoning tokens reported by reasoning-capable providers.
    #[serde(default)]
    pub reasoning_tokens: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<usize>,
    pub total_tokens: usize,
    // Not aliased to Mistral's singular `prompt_token_details`: Mistral's
    // embeddings reply carries *both* keys (the singular always `null`), and
    // an alias makes serde reject the document as a duplicate field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    /// Mistral's top-level cached-prompt count, reported beside (or instead
    /// of) `prompt_tokens_details.cached_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_cached_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_time: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_time: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_time: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_time: Option<f64>,
}

impl Usage {
    pub fn new() -> Self {
        Self {
            prompt_tokens: 0,
            completion_tokens: None,
            total_tokens: 0,
            prompt_tokens_details: None,
            completion_tokens_details: None,
            num_cached_tokens: None,
            queue_time: None,
            prompt_time: None,
            completion_time: None,
            total_time: None,
        }
    }
}

impl Default for Usage {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Usage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Usage {
            prompt_tokens,
            total_tokens,
            ..
        } = self;
        write!(
            f,
            "Prompt tokens: {prompt_tokens} Total tokens: {total_tokens}"
        )
    }
}

impl From<&Usage> for crate::completion::Usage {
    fn from(value: &Usage) -> crate::completion::Usage {
        value.to_normalized()
    }
}

impl From<Usage> for crate::completion::Usage {
    fn from(value: Usage) -> crate::completion::Usage {
        value.to_normalized()
    }
}

impl Usage {
    /// Return prompt tokens plus audio only when that sum and output match the total.
    /// Missing output counts are treated as zero for this comparison.
    fn input_tokens(&self) -> usize {
        let audio = self
            .prompt_tokens_details
            .map_or(0, |details| details.audio_tokens);
        let beside = self.prompt_tokens.saturating_add(audio);
        let accounted = beside.saturating_add(self.completion_tokens.unwrap_or(0));
        if audio != 0 && accounted == self.total_tokens {
            beside
        } else {
            self.prompt_tokens
        }
    }

    /// Normalize token accounting, deriving absent output counts from the total.
    /// Cached input prefers prompt details and falls back to `num_cached_tokens`.
    pub fn to_normalized(&self) -> crate::completion::Usage {
        let input_tokens = self.input_tokens();
        let details = self.prompt_tokens_details.as_ref();
        crate::completion::Usage {
            input_tokens: Some(input_tokens as u64),
            // Gateways that omit `completion_tokens` still send the total, so
            // the completion count is the remainder.
            output_tokens: Some(
                self.completion_tokens
                    .unwrap_or_else(|| self.total_tokens.saturating_sub(input_tokens))
                    as u64,
            ),
            total_tokens: Some(self.total_tokens as u64),
            cached_input_tokens: details
                .map(|d| d.cached_tokens as u64)
                .or(self.num_cached_tokens),
            cache_creation_input_tokens: details
                .and_then(|d| d.cache_write_tokens)
                .map(|tokens| tokens as u64),
            reasoning_tokens: self
                .completion_tokens_details
                .as_ref()
                .map(|d| d.reasoning_tokens as u64),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests;
