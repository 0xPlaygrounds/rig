//! VoyageAI embeddings smoke test.

use rig::http_runtime::HttpRuntime;
use rig::providers::voyageai;

use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

#[tokio::test]
#[ignore = "requires VOYAGE_API_KEY"]
async fn embeddings_smoke() {
    let cfg = voyageai::functions::EmbeddingConfig::from_env(voyageai::VOYAGE_3_LARGE)
        .expect("config should build");
    let rt = HttpRuntime::new();

    let response = voyageai::functions::embed(
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
