//! Cassette-backed Venice embeddings coverage.

use rig::providers::venice;

use super::super::support::with_venice_cassette;
use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

#[tokio::test]
async fn embeddings_smoke() {
    with_venice_cassette("embeddings/embeddings_smoke", |client| async move {
        let model = rig::model(client.embedding(venice::TEXT_EMBEDDING_QWEN3_0_6B, None));
        let embeddings = model
            .call(
                EMBEDDING_INPUTS
                    .iter()
                    .map(|input| (*input).to_string())
                    .collect(),
                None,
            )
            .await
            .map(|response| response.embeddings)
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
        let model = rig::model(client.embedding(venice::TEXT_EMBEDDING_QWEN3_0_6B, Some(256)));
        let embeddings = model
            .call(vec!["dimensioned input".to_string()], None)
            .await
            .map(|response| response.embeddings)
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
