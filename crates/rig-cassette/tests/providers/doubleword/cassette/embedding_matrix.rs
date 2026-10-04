//! Recorded embedding matrix for doubleword: the normalized response contract
//! pinned against live wire recordings. Doubleword never sends `dimensions` (its wire ignores the field), so the dimensions cell is skipped.
//!
//! Cells asserted from recordings, not assumptions: response completeness
//! (order, provider, usage/model/request-id exactly as the wire reports),
//! `raw` reading back as the shared OpenAI-compatible embeddings payload,
//! `encode` determinism across two identical exchanges, the single-text
//! convenience, and the error path preserving the body.

use super::super::support::with_doubleword_cassette;
use rig::providers::doubleword;

use crate::support::{
    EMBEDDING_INPUTS, EmbeddingMatrixExpectations, assert_normalized_embedding_response,
};

fn expectations() -> EmbeddingMatrixExpectations {
    EmbeddingMatrixExpectations {
        provider: "doubleword",
        reports_usage: true,
        reports_model: true,
        reports_request_id: false,
    }
}

fn inputs() -> Vec<String> {
    EMBEDDING_INPUTS.iter().map(|s| (*s).to_string()).collect()
}

#[tokio::test]
async fn normalized_response_is_complete() {
    with_doubleword_cassette(
        "embedding_matrix/normalized_response_is_complete",
        |client| async move {
            let model = client.embedding(doubleword::QWEN3_EMBEDDING_8B, None);
            let response = model
                .call(inputs())
                .await
                .expect("embedding request should succeed");
            assert_normalized_embedding_response(&response, &EMBEDDING_INPUTS, &expectations());
        },
    )
    .await;
}
