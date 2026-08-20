//! Recorded embedding matrix for Gemini: the normalized response contract
//! pinned against live `batchEmbedContents` recordings.
//!
//! Gemini's embedding wire reports no usage, no model echo, no response id,
//! and sends no transport request-id header — every metadata axis lands on
//! its documented `None`/zero outcome, which is exactly what these cells
//! assert rather than skip. Dimensions ride `output_dimensionality`.

use rig::client::EmbeddingsClient;
use rig::embeddings::{EmbeddingModel as _, NormalizeEmbeddingResponse as _};
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
            let model = client.embedding_model(gemini::embedding::EMBEDDING_001);
            let response = model
                .embed_texts_response(inputs())
                .await
                .expect("embedding request should succeed");
            assert_normalized_embedding_response(&response, &EMBEDDING_INPUTS, &expectations());
        },
    )
    .await;
}

#[tokio::test]
async fn raw_round_trips() {
    with_gemini_cassette("embedding_matrix/raw_round_trips", |client| async move {
        let model = client.embedding_model(gemini::embedding::EMBEDDING_001);
        let response = model
            .embed_texts_response(inputs())
            .await
            .expect("embedding request should succeed");

        let raw: gemini::embedding::gemini_api_types::EmbeddingResponse =
            serde_json::from_value(response.raw.clone()).expect("raw round-trips");
        assert_eq!(raw.embeddings.len(), response.embeddings.len());

        let renormalized = raw
            .normalize("gcp.gemini", inputs())
            .expect("re-normalization succeeds");
        assert_eq!(renormalized.embeddings.len(), response.embeddings.len());
    })
    .await;
}

#[tokio::test]
async fn raw_route_parity() {
    with_gemini_cassette("embedding_matrix/raw_route_parity", |client| async move {
        let model = client.embedding_model(gemini::embedding::EMBEDDING_001);
        let normalized = model
            .embed_texts_response(inputs())
            .await
            .expect("normalized call should succeed");
        let raw = model
            .raw_embed_texts(inputs())
            .await
            .expect("raw call should succeed");
        assert_eq!(raw.embeddings.len(), normalized.embeddings.len());
    })
    .await;
}

#[tokio::test]
async fn single_text_convenience() {
    with_gemini_cassette(
        "embedding_matrix/single_text_convenience",
        |client| async move {
            let model = client.embedding_model(gemini::embedding::EMBEDDING_001);
            let response = model
                .embed_text_response(EMBEDDING_INPUTS[0])
                .await
                .expect("single-text embedding should succeed");
            assert_eq!(response.embeddings.len(), 1);
            assert_eq!(response.embeddings[0].document, EMBEDDING_INPUTS[0]);
            assert_eq!(response.provider, "gcp.gemini");
        },
    )
    .await;
}

/// `output_dimensionality` narrows the vector; the driver reports the width
/// it was asked for and the wire honors it.
#[tokio::test]
async fn dimensions_request() {
    with_gemini_cassette("embedding_matrix/dimensions_request", |client| async move {
        let model = client.embedding_model_with_ndims(gemini::embedding::EMBEDDING_001, 256);
        let response = model
            .embed_texts_response(inputs())
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
            let model = client.embedding_model("no-such-embedding-model");
            let error = model
                .embed_texts_response(inputs())
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
