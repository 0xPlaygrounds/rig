//! Cassette-backed Doubleword embeddings coverage.

use rig::http_runtime::HttpRuntime;
use rig::providers::doubleword;

use super::super::support::with_doubleword_cassette;
use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

#[tokio::test]
async fn embeddings_smoke() {
    with_doubleword_cassette("embeddings/embeddings_smoke", |env| async move {
        let cfg = env.embedding_config(doubleword::QWEN3_EMBEDDING_8B);
        let rt = HttpRuntime::new();
        let embeddings = doubleword::functions::embed(
            &cfg,
            &rt,
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
