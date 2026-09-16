//! Together embeddings smoke test.

use rig::embeddings::EmbeddingModel;
use rig::prelude::*;
use rig::providers::openai::wire::{OpenAI, TOGETHER};
use rig::providers::together;

use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

#[tokio::test]
#[ignore = "requires TOGETHER_API_KEY"]
async fn embeddings_smoke() {
    let provider = OpenAI::from_env_with(&TOGETHER)
        .expect("config should build from env")
        .bound()
        .expect("transport should build");
    let model = provider.embedding(together::embedding::M2_BERT_80M_8K_RETRIEVAL, None);

    let embeddings = model
        .embed_texts(EMBEDDING_INPUTS.iter().map(|input| (*input).to_string()))
        .await
        .expect("embedding request should succeed");

    assert_embeddings_nonempty_and_consistent(&embeddings, EMBEDDING_INPUTS.len());
}
