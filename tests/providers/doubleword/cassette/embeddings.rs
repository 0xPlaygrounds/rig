//! Cassette-backed Doubleword embeddings coverage.

use rig::client::EmbeddingsClient;
use rig::embeddings::EmbeddingModel;
use rig::providers::doubleword;

use super::super::support::with_doubleword_cassette;
use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

#[tokio::test]
async fn embeddings_smoke() {
    with_doubleword_cassette("embeddings/embeddings_smoke", |client| async move {
        let model = client.embedding_model(doubleword::QWEN3_EMBEDDING_8B);
        let embeddings = model
            .embed_texts(EMBEDDING_INPUTS.iter().map(|input| (*input).to_string()))
            .await
            .expect("embedding request should succeed");
        assert_embeddings_nonempty_and_consistent(&embeddings, EMBEDDING_INPUTS.len());
    })
    .await;
}
