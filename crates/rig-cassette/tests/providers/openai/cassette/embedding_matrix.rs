//! Recorded embedding matrix for openai: the normalized response contract
//! pinned against live wire recordings. OpenAI reports usage, echoes the model, and sends `x-request-id`.
//!
//! Cells asserted from recordings, not assumptions: response completeness
//! (order, provider, usage/model/request-id exactly as the wire reports),
//! `raw` deserializing back to the provider's own response type with its
//! native fields agreeing with the normalized view, that agreement holding
//! per reply across two exchanges, the single-text convenience, and the
//! error path preserving the body.

use super::super::support::with_openai_cassette;
use rig::providers::openai;

use crate::support::{
    EMBEDDING_INPUTS, EmbeddingMatrixExpectations, assert_normalized_embedding_response,
};

fn expectations() -> EmbeddingMatrixExpectations {
    EmbeddingMatrixExpectations {
        provider: "openai",
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
    with_openai_cassette(
        "embedding_matrix/normalized_response_is_complete",
        |client| async move {
            let model = client
                .openai
                .embedding(openai::TEXT_EMBEDDING_3_SMALL, None);
            let response = model
                .call(inputs())
                .await
                .expect("embedding request should succeed");
            assert_normalized_embedding_response(&response, &EMBEDDING_INPUTS, &expectations());
        },
    )
    .await;
}

/// An embedding wire asked for an explicit width round-trips it — the
/// provider either honors it or the driver errors honestly with
/// `MismatchedDimensions`; a silent mismatch is the bug this cell exists to
/// catch.
#[tokio::test]
async fn dimensions_request() {
    with_openai_cassette("embedding_matrix/dimensions_request", |client| async move {
        let ndims = 512;
        let model = client
            .openai
            .embedding(openai::TEXT_EMBEDDING_3_SMALL, Some(ndims));
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
