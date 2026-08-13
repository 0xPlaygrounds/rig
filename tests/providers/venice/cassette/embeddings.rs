//! Cassette-backed Venice embeddings coverage.

use rig::client::EmbeddingsClient;
use rig::embeddings::EmbeddingModel;
use rig::providers::venice;

use super::super::support::with_venice_cassette;
use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

#[tokio::test]
async fn embeddings_smoke() {
    with_venice_cassette("embeddings/embeddings_smoke", |client| async move {
        let model = client.embedding_model(venice::TEXT_EMBEDDING_QWEN3_0_6B);
        let embeddings = model
            .embed_texts(EMBEDDING_INPUTS.iter().map(|input| (*input).to_string()))
            .await
            .expect("embedding request should succeed");
        assert_embeddings_nonempty_and_consistent(&embeddings, EMBEDDING_INPUTS.len());
    })
    .await;
}

/// Venice honors OpenAI's `dimensions` field: the returned vectors must have
/// exactly the requested width, not the model's native one.
#[tokio::test]
async fn embeddings_honor_requested_dimensions() {
    with_venice_cassette("embeddings/requested_dimensions", |client| async move {
        let model = client.embedding_model_with_ndims(venice::TEXT_EMBEDDING_QWEN3_0_6B, 256);
        let embeddings = model
            .embed_texts(["dimensioned input".to_string()])
            .await
            .expect("embedding request should succeed");

        let embedding = embeddings.first().expect("one embedding");
        assert_eq!(
            embedding.vec.len(),
            256,
            "expected Venice to honor the requested dimensions"
        );
    })
    .await;
}
