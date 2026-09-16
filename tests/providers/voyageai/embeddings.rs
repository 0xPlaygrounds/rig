//! VoyageAI embeddings smoke test.

use rig::embeddings::EmbeddingModel;
use rig::prelude::*;
use rig::providers::voyageai::{self, wire::VoyageAi};

use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

#[tokio::test]
#[ignore = "requires VOYAGE_API_KEY"]
async fn embeddings_smoke() {
    let provider = VoyageAi::from_env()
        .expect("config should build from env")
        .bound()
        .expect("transport should build");
    let model = provider.embedding(voyageai::VOYAGE_3_LARGE, None);

    let embeddings = model
        .embed_texts(EMBEDDING_INPUTS.iter().map(|input| (*input).to_string()))
        .await
        .expect("embedding request should succeed");

    assert_embeddings_nonempty_and_consistent(&embeddings, EMBEDDING_INPUTS.len());
}
