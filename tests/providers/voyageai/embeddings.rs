//! VoyageAI embeddings smoke test.

use rig::providers::voyageai::{self, wire::VoyageAi};
use rig::wire::Wire as _;

use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

#[tokio::test]
#[ignore = "requires VOYAGE_API_KEY"]
async fn embeddings_smoke() {
    let provider = VoyageAi::from_env().expect("config should build from env");
    let model = provider
        .embedding(voyageai::VOYAGE_3_LARGE, None)
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
}
