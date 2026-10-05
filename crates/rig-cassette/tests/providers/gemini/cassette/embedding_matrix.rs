//! Recorded embedding matrix for Gemini: the normalized response contract
//! pinned against live `batchEmbedContents` recordings.
//!
//! Gemini's embedding wire reports no usage, no model echo, no response id,
//! and sends no transport request-id header — every metadata axis lands on
//! its documented `None`/zero outcome, which is exactly what these cells
//! assert rather than skip. Dimensions ride `output_dimensionality`.

use rig::providers::gemini;

use super::super::support::with_gemini_cassette;
use crate::support::{
    EMBEDDING_INPUTS, EmbeddingMatrixExpectations, assert_normalized_embedding_response,
};

fn expectations() -> EmbeddingMatrixExpectations {
    EmbeddingMatrixExpectations {
        provider: "gcp.gemini",
        reports_usage: false,
        reports_model: false,
        reports_request_id: false,
    }
}

fn inputs() -> Vec<String> {
    EMBEDDING_INPUTS.iter().map(|s| (*s).to_string()).collect()
}

#[tokio::test]
async fn normalized_response_is_complete() {
    with_gemini_cassette(
        "embedding_matrix/normalized_response_is_complete",
        |client| async move {
            let model = client.embedding(gemini::embedding::EMBEDDING_001, None);
            let response = model
                .call(inputs())
                .await
                .expect("embedding request should succeed");
            assert_normalized_embedding_response(&response, &EMBEDDING_INPUTS, &expectations());
        },
    )
    .await;
}

/// `output_dimensionality` narrows the vector; the driver reports the width
/// it was asked for and the wire honors it.
#[tokio::test]
async fn dimensions_request() {
    with_gemini_cassette("embedding_matrix/dimensions_request", |client| async move {
        let model = client.embedding(gemini::embedding::EMBEDDING_001, Some(256));
        let response = model
            .call(inputs())
            .await
            .expect("dimension-constrained embedding should succeed");
        for embedding in &response.embeddings {
            assert_eq!(embedding.vec.len(), 256);
        }
    })
    .await;
}

#[tokio::test]
async fn error_preserves_provider_body() {
    with_gemini_cassette(
        "embedding_matrix/error_preserves_provider_body",
        |client| async move {
            let model = client.embedding("no-such-embedding-model", None);
            let error = model
                .call(inputs())
                .await
                .expect_err("a bogus model must be rejected");
            assert!(
                error.provider_response_status().is_some(),
                "the provider's HTTP status survives: {error:?}"
            );
            assert!(
                error
                    .provider_response_body()
                    .is_some_and(|body| !body.is_empty()),
                "the provider's raw body survives: {error:?}"
            );
        },
    )
    .await;
}
