//! AWS Bedrock embedding smoke test inspired by provider embedding coverage.

use super::{
    BEDROCK_EMBEDDING_MODEL, aws_client,
    support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent},
};

#[tokio::test]
#[ignore = "requires AWS credentials and Bedrock embedding model access"]
async fn embeddings_smoke() {
    let response = rig::bedrock::functions::embed(
        &aws_client().await,
        BEDROCK_EMBEDDING_MODEL,
        Some(256),
        EMBEDDING_INPUTS.into_iter().map(str::to_string).collect(),
    )
    .await
    .expect("embedding request should succeed");
    let embeddings = response.embeddings;

    assert_embeddings_nonempty_and_consistent(&embeddings, EMBEDDING_INPUTS.len());
    assert!(
        embeddings
            .iter()
            .all(|embedding| embedding.vec.len() == 256),
        "Titan text embeddings v2 should return the requested 256 dimensions"
    );
}
