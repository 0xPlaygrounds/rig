use super::completion::Usage;
use serde::{Deserialize, Serialize};

// ================================================================
// OpenAI Embedding API
// ================================================================
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

/// The OpenAI-compatible embeddings wire response as every provider on this
/// wire answers it: the typed parse of an [`Embeddings`](super::wire::Embeddings)
/// reply's `raw` document. `usage` is optional here because compatible
/// providers may omit it; the strict [`EmbeddingResponse`] above is OpenAI's
/// own contract.
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

/// One embedded input.
///
/// `object` and `index` carry no meaning past the envelope and a compatible
/// gateway may omit either, so neither is required: the vector is the datum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    #[serde(default)]
    pub object: String,
    pub embedding: Vec<serde_json::Number>,
    #[serde(default)]
    pub index: usize,
}

/// Default dimensions for OpenAI's known embedding models (also used by
/// Azure OpenAI, which deploys the same models).
pub(crate) fn model_dimensions_from_identifier(identifier: &str) -> Option<usize> {
    match identifier {
        TEXT_EMBEDDING_3_LARGE => Some(3_072),
        TEXT_EMBEDDING_3_SMALL | TEXT_EMBEDDING_ADA_002 => Some(1_536),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
