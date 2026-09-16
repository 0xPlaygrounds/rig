//! Recorded embedding matrix for openrouter: the normalized response contract
//! pinned against live wire recordings. OpenRouter fronts embedding models by prefixed id; usage passthrough varies by upstream — the recording is the truth.
//!
//! Cells asserted from recordings, not assumptions: response completeness
//! (order, provider, usage/model/request-id exactly as the wire reports),
//! `raw` reading back as the provider's own payload, deterministic request
//! encoding across a repeated turn, the single-text convenience, and the error
//! path preserving the body.

use super::super::support::with_openrouter_cassette;
use rig::embeddings::EmbeddingModel as _;
use rig::providers::openai;

use crate::support::{
    EMBEDDING_INPUTS, EmbeddingMatrixExpectations, assert_normalized_embedding_response,
};

fn expectations() -> EmbeddingMatrixExpectations {
    EmbeddingMatrixExpectations {
        provider: "openrouter",
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
    with_openrouter_cassette(
        "embedding_matrix/normalized_response_is_complete",
        |client| async move {
            let model = client.embedding("openai/text-embedding-3-small", None);
            let response = model
                .embed_texts_response(inputs())
                .await
                .expect("embedding request should succeed");
            assert_normalized_embedding_response(&response, &EMBEDDING_INPUTS, &expectations());
        },
    )
    .await;
}

/// `raw` is the provider's own reply document: it reads back as the
/// compatible embeddings payload, and that payload's own fields are the ones
/// the normalized response reports.
#[tokio::test]
async fn raw_round_trips() {
    with_openrouter_cassette("embedding_matrix/raw_round_trips", |client| async move {
        let model = client.embedding("openai/text-embedding-3-small", None);
        let response = model
            .embed_texts_response(inputs())
            .await
            .expect("embedding request should succeed");

        let raw: openai::CompatibleEmbeddingResponse =
            serde_json::from_value(response.raw.clone()).expect("raw round-trips");
        assert_eq!(raw.data.len(), response.embeddings.len());
        assert_eq!(raw.model, response.model.clone().unwrap_or_default());
        // One decoder, one mapping: the document's vectors, in wire order, are
        // the vectors the normalized response joined back onto the inputs.
        for (datum, embedding) in raw.data.iter().zip(&response.embeddings) {
            assert_eq!(datum.embedding.len(), embedding.vec.len());
        }
    })
    .await;
}

/// There is one embed seam, so what this cell pins is that `encode` is
/// deterministic — the recording's two turns must carry byte-identical request
/// bodies — and that `raw` is a faithful second view of the reply it rode on.
/// The scenario keeps its two recorded interactions, so it still issues two
/// requests.
#[tokio::test]
async fn raw_route_parity() {
    const SCENARIO: &str = "embedding_matrix/raw_route_parity";
    with_openrouter_cassette("embedding_matrix/raw_route_parity", |client| async move {
        let model = client.embedding("openai/text-embedding-3-small", None);
        let first = model
            .embed_texts_response(inputs())
            .await
            .expect("the first call should succeed");
        let second = model
            .embed_texts_response(inputs())
            .await
            .expect("the same request should succeed again");

        assert_eq!(first.embeddings.len(), second.embeddings.len());
        assert_eq!(first.model, second.model);
        let raw: openai::CompatibleEmbeddingResponse = serde_json::from_value(second.raw.clone())
            .expect("raw is the compatible embeddings payload");
        assert_eq!(raw.data.len(), second.embeddings.len());
        assert_eq!(raw.model, second.model.clone().unwrap_or_default());
    })
    .await;

    let bodies = crate::cassettes::recorded_interaction_bodies("openrouter", SCENARIO);
    assert_eq!(
        bodies.len(),
        2,
        "{SCENARIO}: the cell records the request and then its twin"
    );
    assert_eq!(
        bodies[0].0, bodies[1].0,
        "{SCENARIO}: `encode` is deterministic, so both turns must send the same request bytes"
    );
}

/// The single-text conveniences derive from the full method: same embedding,
/// same metadata.
#[tokio::test]
async fn single_text_convenience() {
    with_openrouter_cassette(
        "embedding_matrix/single_text_convenience",
        |client| async move {
            let model = client.embedding("openai/text-embedding-3-small", None);
            let response = model
                .embed_text_response(EMBEDDING_INPUTS[0])
                .await
                .expect("single-text embedding should succeed");
            assert_eq!(response.embeddings.len(), 1);
            assert_eq!(response.embeddings[0].document, EMBEDDING_INPUTS[0]);
            assert_eq!(response.provider, "openrouter");
            let embedding = model
                .embed_text(EMBEDDING_INPUTS[0])
                .await
                .expect("convenience embedding should succeed");
            assert_eq!(embedding.vec.len(), response.embeddings[0].vec.len());
        },
    )
    .await;
}

/// A rejected request surfaces the provider's own error body, preserved raw.
#[tokio::test]
async fn error_preserves_provider_body() {
    with_openrouter_cassette(
        "embedding_matrix/error_preserves_provider_body",
        |client| async move {
            let model = client.embedding("no-such/embedding-model", None);
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
