//! Recorded embedding matrix for mistral: the normalized response contract
//! pinned against live wire recordings. Mistral reports usage and a model echo; `mistral-correlation-id` is its transport id (see the bug cell).
//!
//! Cells asserted from recordings, not assumptions: response completeness
//! (order, provider, usage/model/request-id exactly as the wire reports),
//! `raw` round-tripping to the provider's own type, raw-route parity,
//! the single-text convenience, and the error path preserving the body.

use super::support::with_mistral_embedding_cassette;
use rig::providers::mistral;

use crate::support::{
    EMBEDDING_INPUTS, EmbeddingMatrixExpectations, assert_normalized_embedding_response,
};

fn expectations() -> EmbeddingMatrixExpectations {
    EmbeddingMatrixExpectations {
        provider: "mistral",
        reports_usage: true,
        reports_model: true,
        reports_request_id: true,
    }
}

fn inputs() -> Vec<String> {
    EMBEDDING_INPUTS.iter().map(|s| (*s).to_string()).collect()
}

#[tokio::test]
async fn normalized_response_is_complete() {
    with_mistral_embedding_cassette(
        "embedding_matrix/normalized_response_is_complete",
        |client| async move {
            let model = client.embedding(mistral::embedding::MISTRAL_EMBED, None);
            let response = model
                .call(inputs())
                .await
                .expect("embedding request should succeed");
            assert_normalized_embedding_response(&response, &EMBEDDING_INPUTS, &expectations());
        },
    )
    .await;
}

/// `embedding(model, Some(n))` round-trips the requested width — the
/// provider either honors it or the driver errors honestly with
/// `MismatchedDimensions`; a silent mismatch is the bug this cell exists to
/// catch.
#[tokio::test]
async fn dimensions_request() {
    with_mistral_embedding_cassette("embedding_matrix/dimensions_request", |client| async move {
        // `mistral-embed` is fixed-width; `output_dimension` is a
        // codestral-embed capability, so the cell exercises that model.
        let ndims = 64;
        let model = client.embedding(mistral::embedding::CODESTRAL_EMBED, Some(ndims));
        let response = model
            .call(inputs())
            .await
            .expect("dimension-constrained embedding should succeed");
        for embedding in &response.embeddings {
            assert_eq!(embedding.vec.len(), ndims);
        }
    })
    .await;
}

/// A rejected request surfaces the provider's own error body, preserved raw.
#[tokio::test]
async fn error_preserves_provider_body() {
    with_mistral_embedding_cassette(
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
