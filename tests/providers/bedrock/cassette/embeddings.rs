//! AWS Bedrock embeddings replay smoke test.

use rig::bedrock;
use rig::client::EmbeddingsClient;
use rig::embeddings::EmbeddingModel;

use super::super::support::with_bedrock_cassette;
use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

const EMBEDDING_INPUT: &str = "Rust cassette replay keeps Bedrock tests deterministic.";

#[tokio::test]
async fn embeddings_smoke() {
    with_bedrock_cassette("embeddings/embeddings_smoke", |client| async move {
        let model = client
            .embedding_model_with_ndims(bedrock::embedding::AMAZON_TITAN_EMBED_TEXT_V2_0, 256);

        let embeddings = model
            .embed_texts([EMBEDDING_INPUT.to_string()])
            .await
            .expect("embedding request should succeed");

        assert_eq!(embeddings.len(), 1);
        let embedding = &embeddings[0];
        assert_eq!(embedding.document, EMBEDDING_INPUT);
        assert!(
            !embedding.vec.is_empty(),
            "expected embedding vector to be non-empty"
        );
    })
    .await;
}

#[tokio::test]
async fn embeddings_batch_smoke() {
    with_bedrock_cassette("embeddings/embeddings_batch_smoke", |client| async move {
        let model = client
            .embedding_model_with_ndims(bedrock::embedding::AMAZON_TITAN_EMBED_TEXT_V2_0, 256);

        let embeddings = model
            .embed_texts(EMBEDDING_INPUTS.into_iter().map(str::to_string))
            .await
            .expect("batch embedding request should succeed");

        assert_embeddings_nonempty_and_consistent(&embeddings, EMBEDDING_INPUTS.len());
        assert!(
            embeddings
                .iter()
                .all(|embedding| embedding.vec.len() == 256),
            "Titan text embeddings v2 should return the requested 256 dimensions"
        );
    })
    .await;
}
