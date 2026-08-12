// ================================================================
//! Doubleword Embeddings Integration
//! From [Doubleword Inference API](https://docs.doubleword.ai/inference-api/models)
// ================================================================

use crate::{
    embeddings::EmbeddingError,
    providers::openai::embedding::{
        EmbeddingDimensions, GenericEmbeddingModel, OpenAIEmbeddingsCompatible,
    },
};

use super::client::DoublewordExt;

// ================================================================
// Doubleword Embedding API
// ================================================================
pub const QWEN3_EMBEDDING_8B: &str = "Qwen/Qwen3-Embedding-8B";

impl OpenAIEmbeddingsCompatible for DoublewordExt {
    const PROVIDER_NAME: &'static str = "doubleword";

    // Doubleword responses are not guaranteed to carry usage; usage is
    // reported when present and zero otherwise.
    const REQUIRES_USAGE: bool = false;
    const SUPPORTS_ENCODING_FORMAT: bool = false;
    const SUPPORTS_USER: bool = false;

    // The hand-rolled request never sent a dimensions field; that wire shape
    // is preserved.
    fn embedding_dimensions(
        &self,
        _model: &str,
        _dimensions: Option<usize>,
    ) -> Result<Option<EmbeddingDimensions>, EmbeddingError> {
        Ok(None)
    }
}

/// Doubleword embedding model, driven by the shared OpenAI-compatible
/// embeddings path.
pub type EmbeddingModel<T = reqwest::Client> = GenericEmbeddingModel<DoublewordExt, T>;
