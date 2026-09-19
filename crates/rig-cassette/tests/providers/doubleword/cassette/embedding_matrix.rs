//! Recorded embedding matrix for doubleword: the normalized response contract
//! pinned against live wire recordings. Doubleword never sends `dimensions` (its wire ignores the field), so the dimensions cell is skipped.
//!
//! Cells asserted from recordings, not assumptions: response completeness
//! (order, provider, usage/model/request-id exactly as the wire reports),
//! `raw` reading back as the shared OpenAI-compatible embeddings payload,
//! `encode` determinism across two identical exchanges, the single-text
//! convenience, and the error path preserving the body.

use super::super::support::with_doubleword_cassette;
use rig::embeddings::EmbeddingModel as _;
use rig::providers::{doubleword, openai};
use serde::Deserialize as _;

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
                .embed_texts_response(inputs())
                .await
                .expect("embedding request should succeed");
            assert_normalized_embedding_response(&response, &EMBEDDING_INPUTS, &expectations());
        },
    )
    .await;
}

/// `raw` is the provider's verbatim payload: it reads back as the shared
/// OpenAI-compatible embeddings response, and that value's provider-native
/// fields are the ones the decoder normalized.
#[tokio::test]
async fn raw_round_trips() {
    with_doubleword_cassette("embedding_matrix/raw_round_trips", |client| async move {
        let model = client.embedding(doubleword::QWEN3_EMBEDDING_8B, None);
        let response = model
            .embed_texts_response(inputs())
            .await
            .expect("embedding request should succeed");

        let reply = openai::CompatibleEmbeddingResponse::deserialize(&response.raw)
            .expect("raw is the shared OpenAI-compatible embeddings response");
        assert_eq!(reply.data.len(), response.embeddings.len());
        assert_eq!(Some(reply.model.as_str()), response.model.as_deref());
        let native_usage = reply.usage.as_ref().expect("Doubleword reports usage");
        assert_eq!(
            response.usage.total_tokens,
            Some(native_usage.total_tokens as u64)
        );
        assert_eq!(
            response.usage.input_tokens,
            Some(native_usage.prompt_tokens as u64)
        );
        for (datum, embedding) in reply.data.iter().zip(&response.embeddings) {
            assert_eq!(datum.embedding.len(), embedding.vec.len());
        }
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

    with_doubleword_cassette("embedding_matrix/raw_route_parity", |client| async move {
        let model = client.embedding(doubleword::QWEN3_EMBEDDING_8B, None);
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

        let reply = openai::CompatibleEmbeddingResponse::deserialize(&again.raw)
            .expect("raw is the shared OpenAI-compatible embeddings response");
        assert_eq!(reply.data.len(), normalized.embeddings.len());
        assert_eq!(Some(reply.model.as_str()), normalized.model.as_deref());
    })
    .await;

    let bodies = crate::cassettes::recorded_interaction_bodies("doubleword", SCENARIO);
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
    with_doubleword_cassette(
        "embedding_matrix/single_text_convenience",
        |client| async move {
            let model = client.embedding(doubleword::QWEN3_EMBEDDING_8B, None);
            let response = model
                .embed_text_response(EMBEDDING_INPUTS[0])
                .await
                .expect("single-text embedding should succeed");
            assert_eq!(response.embeddings.len(), 1);
            assert_eq!(response.embeddings[0].document, EMBEDDING_INPUTS[0]);
            assert_eq!(response.provider, "doubleword");
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
    with_doubleword_cassette(
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
