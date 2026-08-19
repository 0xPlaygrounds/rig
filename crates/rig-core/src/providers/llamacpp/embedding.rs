//! llama.cpp embedding models.
//!
//! `llama-server` serves `POST /v1/embeddings` only when started with
//! `--embeddings`; without it the route answers
//! `501 {"error":{"code":501,"message":"This server does not support
//! embeddings. Start it with `--embeddings`","type":"not_supported_error"}}`.
//! It additionally requires a pooling type the OpenAI wire can express:
//! `--pooling none` returns one vector *per token* and the server rejects the
//! request with a 500 rather than reshaping it.

use crate::providers::openai;

/// llama.cpp embedding model, driven by the shared OpenAI embeddings path.
pub type EmbeddingModel<H = reqwest::Client> =
    openai::embedding::GenericEmbeddingModel<super::client::LlamacppExt, H>;
