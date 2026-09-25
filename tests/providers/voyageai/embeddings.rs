//! VoyageAI embeddings smoke test.

use rig::providers::voyageai::{self, wire::VoyageAi};
use rig_test_support::endpoint::Endpoint;

use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

#[tokio::test]
#[ignore = "requires VOYAGE_API_KEY"]
async fn embeddings_smoke() {
    let provider = Endpoint::new(
        VoyageAi::from_env().expect("config should build from env"),
        rig::rig_reqwest::shared(),
    );
    let model = provider.embedding(voyageai::VOYAGE_3_LARGE, None);

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
