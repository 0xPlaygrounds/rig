//! Copilot embeddings smoke test.

use crate::copilot::{live_client, live_embedding_model};
use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};
use rig_core::client::EmbeddingsClient;
use rig_core::embeddings::EmbeddingModel;

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn embeddings_smoke() {
    let model = live_client().embedding_model(live_embedding_model());

    let embeddings = model
        .embed_texts(EMBEDDING_INPUTS.iter().map(|input| (*input).to_string()))
        .await
        .expect("embedding request should succeed");

    assert_embeddings_nonempty_and_consistent(&embeddings, EMBEDDING_INPUTS.len());
}
