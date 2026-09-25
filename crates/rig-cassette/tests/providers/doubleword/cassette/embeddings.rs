//! Cassette-backed Doubleword embeddings coverage.

use rig::providers::doubleword;
use rig::wire::Wire as _;

use super::super::support::with_doubleword_cassette;
use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

#[tokio::test]
async fn embeddings_smoke() {
    with_doubleword_cassette("embeddings/embeddings_smoke", |client| async move {
        let model = client
            .embedding(doubleword::QWEN3_EMBEDDING_8B, None)
            .on(rig::transport());
        let embeddings = model
            .call(
                EMBEDDING_INPUTS
                    .iter()
                    .map(|input| (*input).to_string())
                    .collect(),
            )
            .await
            .map(|response| response.embeddings)
            .expect("embedding request should succeed");
        assert_embeddings_nonempty_and_consistent(&embeddings, EMBEDDING_INPUTS.len());
    })
    .await;
}
