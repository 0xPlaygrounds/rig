//! Copilot embeddings smoke test.

use crate::copilot::{live_embedding_model, with_copilot_cassette};
use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};
use rig::embeddings::EmbeddingModel;

#[tokio::test]
async fn embeddings_smoke() {
    with_copilot_cassette("embeddings/embeddings_smoke", |client| async move {
        let model = client.embedding(live_embedding_model(), None);

        let embeddings = model
            .embed_texts(EMBEDDING_INPUTS.iter().map(|input| (*input).to_string()))
            .await
            .expect("embedding request should succeed");

        assert_embeddings_nonempty_and_consistent(&embeddings, EMBEDDING_INPUTS.len());
    })
    .await;
}
