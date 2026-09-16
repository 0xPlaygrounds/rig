//! Recorded embedding matrix for venice: the normalized response contract
//! pinned against live wire recordings. Venice speaks the OpenAI-compatible wire with no request-id header contract.
//!
//! Cells asserted from recordings, not assumptions: response completeness
//! (order, provider, usage/model/request-id exactly as the wire reports),
//! `raw` carrying the provider's verbatim payload, `encode` determinism
//! across two identical exchanges, the single-text convenience, and the
//! error path preserving the body.

use super::super::support::with_venice_cassette;
use rig::embeddings::EmbeddingModel as _;
use rig::providers::venice;

use crate::support::{
    EMBEDDING_INPUTS, EmbeddingMatrixExpectations, assert_normalized_embedding_response,
};

fn expectations() -> EmbeddingMatrixExpectations {
    EmbeddingMatrixExpectations {
        provider: "venice",
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
    with_venice_cassette(
        "embedding_matrix/normalized_response_is_complete",
        |client| async move {
            let model = client.embedding(venice::TEXT_EMBEDDING_QWEN3_0_6B, None);
            let response = model
                .embed_texts_response(inputs())
                .await
                .expect("embedding request should succeed");
            assert_normalized_embedding_response(&response, &EMBEDDING_INPUTS, &expectations());
        },
    )
    .await;
}

/// `raw` is the provider's verbatim payload: every field the normalized view
/// carries is readable from it, in the same order, plus the ones `Usage` and
/// `EmbeddingResponse` have no slot for.
#[tokio::test]
async fn raw_round_trips() {
    with_venice_cassette("embedding_matrix/raw_round_trips", |client| async move {
        let model = client.embedding(venice::TEXT_EMBEDDING_QWEN3_0_6B, None);
        let response = model
            .embed_texts_response(inputs())
            .await
            .expect("embedding request should succeed");

        let data = response.raw["data"]
            .as_array()
            .expect("raw carries the provider's `data` array");
        assert_eq!(data.len(), response.embeddings.len());
        for (index, (datum, embedding)) in data.iter().zip(&response.embeddings).enumerate() {
            assert_eq!(datum["index"].as_u64(), Some(index as u64));
            let vector = datum["embedding"]
                .as_array()
                .expect("each datum carries its vector");
            assert_eq!(vector.len(), embedding.vec.len());
            assert_eq!(vector[0].as_f64(), Some(embedding.vec[0]));
        }
        assert_eq!(
            response.raw["model"].as_str(),
            response.model.as_deref(),
            "the normalized model is the one the payload names"
        );
        assert_eq!(
            response.raw["usage"]["total_tokens"].as_u64(),
            response.usage.total_tokens,
            "the normalized usage is the one the payload reports"
        );
    })
    .await;
}

/// There is one embed seam, so the axis this cell pins is that `encode` is
/// deterministic — the same inputs produce byte-identical request bodies on
/// both exchanges — and that `raw` is a faithful second view of the reply it
/// rode on rather than a summary.
#[tokio::test]
async fn raw_route_parity() {
    const SCENARIO: &str = "embedding_matrix/raw_route_parity";

    with_venice_cassette("embedding_matrix/raw_route_parity", |client| async move {
        let model = client.embedding(venice::TEXT_EMBEDDING_QWEN3_0_6B, None);
        let normalized = model
            .embed_texts_response(inputs())
            .await
            .expect("normalized call should succeed");
        let again = model
            .embed_texts_response(inputs())
            .await
            .expect("the same request should succeed again");

        assert_eq!(again.embeddings.len(), normalized.embeddings.len());
        assert_eq!(again.model, normalized.model);
        assert_eq!(again.usage, normalized.usage);

        let data = again.raw["data"]
            .as_array()
            .expect("raw carries the provider's `data` array");
        assert_eq!(data.len(), normalized.embeddings.len());
        assert_eq!(again.raw["model"].as_str(), again.model.as_deref());
        assert_eq!(
            again.raw["object"].as_str(),
            Some("list"),
            "the payload's envelope kind is not normalized anywhere else"
        );
    })
    .await;

    let bodies = crate::cassettes::recorded_interaction_bodies("venice", SCENARIO);
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
    with_venice_cassette(
        "embedding_matrix/single_text_convenience",
        |client| async move {
            let model = client.embedding(venice::TEXT_EMBEDDING_QWEN3_0_6B, None);
            let response = model
                .embed_text_response(EMBEDDING_INPUTS[0])
                .await
                .expect("single-text embedding should succeed");
            assert_eq!(response.embeddings.len(), 1);
            assert_eq!(response.embeddings[0].document, EMBEDDING_INPUTS[0]);
            assert_eq!(response.provider, "venice");
            let embedding = model
                .embed_text(EMBEDDING_INPUTS[0])
                .await
                .expect("convenience embedding should succeed");
            assert_eq!(embedding.vec.len(), response.embeddings[0].vec.len());
        },
    )
    .await;
}

/// `embedding(model, Some(ndims))` round-trips the requested width — the
/// provider either honors it or the driver errors honestly with
/// `MismatchedDimensions`; a silent mismatch is the bug this cell exists to
/// catch.
#[tokio::test]
async fn dimensions_request() {
    with_venice_cassette("embedding_matrix/dimensions_request", |client| async move {
        let ndims = 256;
        let model = client.embedding(venice::TEXT_EMBEDDING_QWEN3_0_6B, Some(ndims));
        let response = model
            .embed_texts_response(inputs())
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
    with_venice_cassette(
        "embedding_matrix/error_preserves_provider_body",
        |client| async move {
            let model = client.embedding("no-such-embedding-model", None);
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
