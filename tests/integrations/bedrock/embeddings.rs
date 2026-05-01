//! AWS Bedrock embedding smoke test inspired by provider embedding coverage.

use rig::client::EmbeddingsClient;
use rig::embeddings::EmbeddingModel as _;

use super::{
    BEDROCK_EMBEDDING_MODEL, client,
    support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent},
};

#[tokio::test]
#[ignore = "requires AWS credentials and Bedrock embedding model access"]
async fn embeddings_smoke() {
    let model = client().embedding_model_with_ndims(BEDROCK_EMBEDDING_MODEL, 256);
    let embeddings = model
        .embed_texts(EMBEDDING_INPUTS.into_iter().map(str::to_string))
        .await
        .expect("embedding request should succeed");

    assert_embeddings_nonempty_and_consistent(&embeddings, EMBEDDING_INPUTS.len());
    assert!(
        embeddings
            .iter()
            .all(|embedding| embedding.vec.len() == 256),
        "Titan text embeddings v2 should return the requested 256 dimensions"
    );
}
