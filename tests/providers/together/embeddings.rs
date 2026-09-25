//! Together embeddings smoke test.

use rig::providers::openai::wire::{OpenAI, TOGETHER};
use rig::providers::together;

use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

#[tokio::test]
#[ignore = "requires TOGETHER_API_KEY"]
async fn embeddings_smoke() {
    let provider = OpenAI::from_env_with(&TOGETHER).expect("config should build from env");
    let model = rig::model(provider.embedding(together::embedding::M2_BERT_80M_8K_RETRIEVAL, None));

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
}
