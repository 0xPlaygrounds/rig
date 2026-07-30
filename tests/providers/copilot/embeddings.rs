//! Copilot embeddings smoke test.

use crate::copilot::{live_embedding_model, with_copilot_cassette};
use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};
use rig::providers::copilot;

#[tokio::test]
async fn embeddings_smoke() {
    with_copilot_cassette("embeddings/embeddings_smoke", |client| async move {
        // The deleted `client.embedding_model(m)` sent the model's native
        // dimension count; `text-embedding-3-small` is 1536.
        let cfg = client
            .embedding_config(live_embedding_model())
            .with_dimensions(1536);

        let embeddings = copilot::functions::embed(
            &cfg,
            &client.http(),
            EMBEDDING_INPUTS
                .iter()
                .map(|input| (*input).to_string())
                .collect(),
        )
        .await
        .expect("embedding request should succeed")
        .embeddings;

        assert_embeddings_nonempty_and_consistent(&embeddings, EMBEDDING_INPUTS.len());
    })
    .await;
}
