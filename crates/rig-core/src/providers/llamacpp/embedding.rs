//! llama.cpp embedding models.
//!
//! `llama-server` serves `POST /v1/embeddings` only when started with
//! `--embeddings`; without it the route answers
//! `501 {"error":{"code":501,"message":"This server does not support
//! embeddings. Start it with `--embeddings`","type":"not_supported_error"}}`.
//! It additionally requires a pooling type the OpenAI wire can express:
//! `--pooling none` returns one vector *per token* and the server rejects the
//! request with a 500 rather than reshaping it.

use crate::embeddings::EmbeddingError;
use crate::providers::openai;
use crate::providers::openai::embedding::{EmbeddingDimensions, OpenAIEmbeddingsCompatible};

impl OpenAIEmbeddingsCompatible for super::client::LlamacppExt {
    const PROVIDER_NAME: &'static str = "llamacpp";

    /// Never send `dimensions`.
    ///
    /// `llama-server`'s embeddings handler reads no such field — the string
    /// does not appear anywhere in `tools/server/` — so a request carrying it
    /// answers 200 with the loaded model's native width and no diagnostic.
    /// Measured on b10499-6d05498 with `Qwen3-Embedding-0.6B`: asking for 128
    /// returns 1,024.
    ///
    /// Sending a field the server ignores puts a claim on the wire that the
    /// wire does not honour, so this hook drops it. The *caller's* declared
    /// width is not dropped, and is not silently believed either: the shared
    /// driver compares it against the width that came back and raises
    /// [`EmbeddingError::MismatchedDimensions`] when they differ, which is the
    /// only place the disagreement is visible at all.
    ///
    /// A caller who knows their GGUF's width may still state it — that is
    /// what `embedding_model_with_ndims` means here, a declaration rather than
    /// a request — and a wrong declaration now fails loudly instead of sizing
    /// a vector index that cannot hold its own vectors.
    fn embedding_dimensions(
        &self,
        _model: &str,
        _dimensions: Option<usize>,
    ) -> Result<Option<EmbeddingDimensions>, EmbeddingError> {
        Ok(None)
    }
}

/// llama.cpp embedding model, driven by the shared OpenAI embeddings path.
pub type EmbeddingModel<H = crate::http_client::BoxedHttpClient> =
    openai::embedding::GenericEmbeddingModel<super::client::LlamacppExt, H>;
