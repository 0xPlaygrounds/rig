//! Doubleword embeddings smoke test.

use rig::client::{EmbeddingsClient, ProviderClient};
use rig::embeddings::EmbeddingModel;
use rig::providers::doubleword;

use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

#[tokio::test]
#[ignore = "requires DOUBLEWORD_API_KEY"]
async fn embeddings_smoke() {
    let client = doubleword::Client::from_env().expect("client should build");
    let model = client.embedding_model(doubleword::QWEN3_EMBEDDING_8B);

    let embeddings = model
        .embed_texts(EMBEDDING_INPUTS.iter().map(|input| (*input).to_string()))
        .await
        .expect("embedding request should succeed");

    assert_embeddings_nonempty_and_consistent(&embeddings, EMBEDDING_INPUTS.len());
}
