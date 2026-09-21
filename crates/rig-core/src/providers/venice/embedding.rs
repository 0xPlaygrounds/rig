//! Venice's embedding model identifiers.
//!
//! ```
//! use rig_core::providers::venice::embedding::TEXT_EMBEDDING_BGE_M3;
//! assert_eq!(TEXT_EMBEDDING_BGE_M3, "text-embedding-bge-m3");
//! ```

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
