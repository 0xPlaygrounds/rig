//! Together embeddings smoke test.

use rig_core::client::{EmbeddingsClient, ProviderClient};
use rig_core::embeddings::EmbeddingModel;
use rig_core::providers::together;

use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

#[tokio::test]
#[ignore = "requires TOGETHER_API_KEY"]
async fn embeddings_smoke() {
    let client = together::Client::from_env().expect("client should build");
    let model = client.embedding_model(together::embedding::M2_BERT_80M_8K_RETRIEVAL);

    let embeddings = model
        .embed_texts(EMBEDDING_INPUTS.iter().map(|input| (*input).to_string()))
        .await
        .expect("embedding request should succeed");

    assert_embeddings_nonempty_and_consistent(&embeddings, EMBEDDING_INPUTS.len());
}
