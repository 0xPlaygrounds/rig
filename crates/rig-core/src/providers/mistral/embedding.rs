//! Mistral embedding model identifiers.
//!
//! The request/response machinery lives on the data-oriented path:
//! [`functions::EmbeddingConfig`](super::functions::EmbeddingConfig) plus
//! [`functions::embed`](super::functions::embed).

pub const MISTRAL_EMBED: &str = "mistral-embed";
/// Codestral embedding model with configurable output dimensions.
pub const CODESTRAL_EMBED: &str = "codestral-embed";

pub const MAX_DOCUMENTS: usize = 1024;
