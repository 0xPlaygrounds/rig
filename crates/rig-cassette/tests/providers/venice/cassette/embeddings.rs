//! Cassette-backed Venice embeddings coverage.

use rig::providers::venice;

use super::super::support::with_venice_cassette;
use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

#[tokio::test]
async fn embeddings_smoke() {
    with_venice_cassette("embeddings/embeddings_smoke", |client| async move {
        let model = client.embedding(venice::TEXT_EMBEDDING_QWEN3_0_6B, None);
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
