// ================================================================
//! Venice Embeddings Integration
//! From [Venice's embeddings endpoint](https://docs.venice.ai/api-reference/endpoint/embeddings/generate)
// ================================================================

use crate::providers::openai::embedding::{GenericEmbeddingModel, OpenAIEmbeddingsCompatible};

use super::client::VeniceExt;

// ================================================================
// Venice Embedding API
// ================================================================
/// `text-embedding-bge-m3`
pub const TEXT_EMBEDDING_BGE_M3: &str = "text-embedding-bge-m3";
/// `text-embedding-bge-en-icl`
pub const TEXT_EMBEDDING_BGE_EN_ICL: &str = "text-embedding-bge-en-icl";
/// `text-embedding-qwen3-8b`
pub const TEXT_EMBEDDING_QWEN3_8B: &str = "text-embedding-qwen3-8b";
/// `text-embedding-qwen3-0-6b`
pub const TEXT_EMBEDDING_QWEN3_0_6B: &str = "text-embedding-qwen3-0-6b";
/// `text-embedding-multilingual-e5-large-instruct`
pub const TEXT_EMBEDDING_MULTILINGUAL_E5_LARGE_INSTRUCT: &str =
    "text-embedding-multilingual-e5-large-instruct";

// Venice's embeddings endpoint is OpenAI-compatible on every field Rig sends:
// `model`, `input`, `encoding_format`, `dimensions` (honored — a request for
// 256 dimensions returns 256), and `user` (accepted for compatibility), and it
// answers with `usage`.
impl OpenAIEmbeddingsCompatible for VeniceExt {
    const PROVIDER_NAME: &'static str = "venice";
}

/// Venice embedding model, driven by the shared OpenAI-compatible
/// embeddings path.
pub type EmbeddingModel<T = reqwest::Client> = GenericEmbeddingModel<VeniceExt, T>;
