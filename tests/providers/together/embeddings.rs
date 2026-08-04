//! Together embeddings smoke test.

use rig::http_runtime::HttpRuntime;
use rig::providers::together;

use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

#[tokio::test]
#[ignore = "requires TOGETHER_API_KEY"]
async fn embeddings_smoke() {
    let cfg = together::functions::EmbeddingConfig::from_env(
        together::embedding::M2_BERT_80M_8K_RETRIEVAL,
    )
    .expect("config should build");
    let rt = HttpRuntime::new();

    let response = together::functions::embed(
        &cfg,
        &rt,
        EMBEDDING_INPUTS
            .iter()
            .map(|input| (*input).to_string())
            .collect(),
    )
    .await
    .expect("embedding request should succeed");

    assert_embeddings_nonempty_and_consistent(&response.embeddings, EMBEDDING_INPUTS.len());
}
