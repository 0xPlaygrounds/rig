//! Recorded embedding matrix for Cohere: the normalized response contract
//! pinned against live wire recordings, on Cohere's own `/v1/embed` wire.
//!
//! Cohere reports usage as `meta.billed_units` and a response-scoped `id`,
//! but echoes no model and has no transport request-id header on this
//! endpoint — those axes are asserted as their documented `None`/zero
//! outcomes, not skipped. The dimensions cell is absent: Cohere's embed wire
//! takes no dimension parameter.

use rig::providers::cohere;
use rig_test_support::cassette_models::MapWire;

use super::super::support::with_cohere_cassette;
use crate::support::{
    EMBEDDING_INPUTS, EmbeddingMatrixExpectations, assert_normalized_embedding_response,
};

const INPUT_TYPE: &str = "search_document";

fn expectations() -> EmbeddingMatrixExpectations {
    EmbeddingMatrixExpectations {
        provider: "cohere",
        reports_usage: true,
        reports_model: false,
        reports_request_id: false,
    }
}

fn inputs() -> Vec<String> {
    EMBEDDING_INPUTS.iter().map(|s| (*s).to_string()).collect()
}

#[tokio::test]
async fn normalized_response_is_complete() {
    with_cohere_cassette(
        "embedding_matrix/normalized_response_is_complete",
        |client| async move {
            let model = client
                .embedding(cohere::EMBED_V4, None)
                .map_wire(|wire| wire.with_input_type(INPUT_TYPE));
            let response = model
                .call(inputs())
                .await
                .expect("embedding request should succeed");
            assert_normalized_embedding_response(&response, &EMBEDDING_INPUTS, &expectations());
            // Cohere's one extra identity axis: the response-scoped id.
            assert!(
                response.response_id.is_some(),
                "Cohere reports a response id: {response:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn error_preserves_provider_body() {
    with_cohere_cassette(
        "embedding_matrix/error_preserves_provider_body",
        |client| async move {
            let model = client
                .embedding("no-such-embedding-model", None)
                .map_wire(|wire| wire.with_input_type(INPUT_TYPE));
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
