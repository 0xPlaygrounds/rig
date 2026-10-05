//! llama.cpp embeddings smoke coverage.
//!
//! **Server**: the `--embeddings --pooling mean` configuration, loading
//! `Qwen/Qwen3-Embedding-0.6B-GGUF` Q8_0 — a real embedding model rather than
//! a causal LM pooled into one, which is why this suite states its model.

use super::super::cassette_support::*;

use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

#[tokio::test]
async fn embeddings_smoke() {
    with_llamacpp_embeddings_cassette("embeddings/embeddings_smoke", |client| async move {
        let model = client.embedding(CASSETTE_EMBEDDING_MODEL, None);

        let embeddings = model
            .call(
                EMBEDDING_INPUTS
                    .iter()
                    .map(|input| (*input).to_string())
                    .collect::<Vec<_>>(),
            )
            .await
            .map(|response| response.embeddings)
            .expect("embedding request should succeed");

        assert_embeddings_nonempty_and_consistent(&embeddings, EMBEDDING_INPUTS.len());
    })
    .await;
}
