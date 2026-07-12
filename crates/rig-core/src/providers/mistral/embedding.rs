//! Mistral's embeddings endpoint is OpenAI-compatible, so [`EmbeddingModel`] is
//! a thin alias over the shared [`GenericEmbeddingModel`]. The `/v1/embeddings`
//! path is supplied by [`MistralExt`](super::client::MistralExt)'s
//! [`OpenAIEmbeddingsCompatible`](crate::providers::openai::embedding::OpenAIEmbeddingsCompatible)
//! implementation in `client.rs`.

use super::client::MistralExt;
use crate::providers::openai::embedding::GenericEmbeddingModel;

// ================================================================
// Mistral Embedding API
// ================================================================
pub const MISTRAL_EMBED: &str = "mistral-embed";

pub const MAX_DOCUMENTS: usize = 1024;

/// Mistral embedding model, backed by the shared OpenAI-compatible
/// [`GenericEmbeddingModel`].
pub type EmbeddingModel<H = reqwest::Client> = GenericEmbeddingModel<MistralExt, H>;
