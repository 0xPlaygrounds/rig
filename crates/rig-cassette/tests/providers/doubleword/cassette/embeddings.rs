//! Cassette-backed Doubleword embeddings coverage.

use rig::providers::doubleword;

use super::super::support::with_doubleword_cassette;
use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

#[tokio::test]
async fn embeddings_smoke() {
    with_doubleword_cassette("embeddings/embeddings_smoke", |client| async move {
        let model = rig::model(client.embedding(doubleword::QWEN3_EMBEDDING_8B, None));
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
