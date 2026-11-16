use serde::{Deserialize, Serialize};

/// `text-embedding-3-large` embedding model
pub const TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";
/// `text-embedding-3-small` embedding model
pub const TEXT_EMBEDDING_3_SMALL: &str = "text-embedding-3-small";
/// `text-embedding-ada-002` embedding model
pub const TEXT_EMBEDDING_ADA_002: &str = "text-embedding-ada-002";

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
    pub usage: Option<EmbeddingUsage>,
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
    pub embedding: Vec<f64>,
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

/// The token counts an embeddings reply reports.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u64,
    pub total_tokens: u64,
    // Not aliased to Mistral's singular `prompt_token_details`: Mistral's
    // embeddings reply carries *both* keys (the singular always `null`), and
    // an alias makes serde reject the document as a duplicate field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<EmbeddingTokensDetails>,
    /// Mistral's top-level cached-prompt count, read when the details are absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_cached_tokens: Option<u64>,
}

/// Where an embeddings reply's prompt tokens went.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingTokensDetails {
    /// Tokens read from the prompt cache.
    #[serde(default)]
    pub cached_tokens: u64,
}

/// The provider's counts are passed through unchanged, so a reply whose
/// `prompt_tokens` exceeds its `total_tokens` keeps both as sent.
impl From<&EmbeddingUsage> for crate::completion::Usage {
    fn from(usage: &EmbeddingUsage) -> Self {
        Self {
            input_tokens: Some(usage.prompt_tokens),
            // What the total counts beyond the prompt, normally zero.
            output_tokens: Some(usage.total_tokens.saturating_sub(usage.prompt_tokens)),
            total_tokens: Some(usage.total_tokens),
            cached_input_tokens: usage
                .prompt_tokens_details
                .map(|details| details.cached_tokens)
                .or(usage.num_cached_tokens),
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests;
