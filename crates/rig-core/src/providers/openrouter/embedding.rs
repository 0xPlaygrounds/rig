//! OpenRouter's embeddings endpoint is OpenAI-compatible, so [`EmbeddingModel`]
//! is a thin alias over the shared [`GenericEmbeddingModel`]. OpenRouter's base
//! URL already carries the `/api/v1` segment, so the default `/embeddings` path
//! from the
//! [`OpenAIEmbeddingsCompatible`](crate::providers::openai::embedding::OpenAIEmbeddingsCompatible)
//! implementation on [`OpenRouterExt`](super::client::OpenRouterExt) is used.

use super::client::OpenRouterExt;
use crate::providers::openai::embedding::GenericEmbeddingModel;

/// OpenRouter embedding model, backed by the shared OpenAI-compatible
/// [`GenericEmbeddingModel`].
pub type EmbeddingModel<H = reqwest::Client> = GenericEmbeddingModel<OpenRouterExt, H>;
