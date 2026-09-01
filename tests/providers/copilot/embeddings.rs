//! Copilot embeddings smoke test.

use crate::copilot::{live_embedding_model, with_copilot_cassette};
use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};
use rig::client::EmbeddingsClient;
use rig::embeddings::EmbeddingModel;

#[tokio::test]
async fn embeddings_smoke() {
    with_copilot_cassette("embeddings/embeddings_smoke", |client| async move {
        let model = client.embedding_model(live_embedding_model());

        let embeddings = model
            .embed_texts(EMBEDDING_INPUTS.iter().map(|input| (*input).to_string()))
            .await
            .expect("embedding request should succeed");

        assert_embeddings_nonempty_and_consistent(&embeddings, EMBEDDING_INPUTS.len());
    })
    .await;
}
