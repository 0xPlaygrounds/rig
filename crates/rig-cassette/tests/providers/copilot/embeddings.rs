//! Copilot embeddings smoke test.

use crate::copilot::{live_embedding_model, with_copilot_cassette};
use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

#[tokio::test]
async fn embeddings_smoke() {
    with_copilot_cassette("embeddings/embeddings_smoke", |client| async move {
        let model = client.embedding(live_embedding_model(), None);

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
