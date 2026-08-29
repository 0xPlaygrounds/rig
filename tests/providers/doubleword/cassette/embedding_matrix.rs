//! Recorded embedding matrix for doubleword: the normalized response contract
//! pinned against live wire recordings. Doubleword never sends `dimensions` (its wire ignores the field), so the dimensions cell is skipped.
//!
//! Cells asserted from recordings, not assumptions: response completeness
//! (order, provider, usage/model/request-id exactly as the wire reports),
//! `raw` round-tripping to the provider's own type, raw-route parity,
//! the single-text convenience, and the error path preserving the body.

use super::super::support::with_doubleword_cassette;
use rig::client::EmbeddingsClient;
use rig::embeddings::{EmbeddingModel as _, NormalizeEmbeddingResponse as _};
use rig::providers::{doubleword, openai_compatible as openai};

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
            let model = client.embedding_model(doubleword::QWEN3_EMBEDDING_8B);
            let response = model
                .embed_texts_response(inputs())
                .await
                .expect("embedding request should succeed");
            assert_normalized_embedding_response(&response, &EMBEDDING_INPUTS, &expectations());
        },
    )
    .await;
}

/// `raw` is the provider's own payload, serialized: it deserializes back to
/// the wire type and normalizing that value reproduces the normalized view.
#[tokio::test]
async fn raw_round_trips() {
    with_doubleword_cassette("embedding_matrix/raw_round_trips", |client| async move {
        let model = client.embedding_model(doubleword::QWEN3_EMBEDDING_8B);
        let response = model
            .embed_texts_response(inputs())
            .await
            .expect("embedding request should succeed");

        let raw: openai::CompatibleEmbeddingResponse =
            serde_json::from_value(response.raw.clone()).expect("raw round-trips");
        assert_eq!(raw.data.len(), response.embeddings.len());

        let renormalized = raw
            .normalize(response.provider.as_str(), inputs())
            .expect("re-normalization succeeds");
        assert_eq!(renormalized.embeddings.len(), response.embeddings.len());
        assert_eq!(renormalized.model, response.model);
        assert_eq!(renormalized.usage, response.usage);
    })
    .await;
}

/// The inherent raw route answers with the payload whose normalization agrees
/// with the normalized call — two live exchanges in one recording, following
/// the raw-parity matrices' shape.
#[tokio::test]
async fn raw_route_parity() {
    with_doubleword_cassette("embedding_matrix/raw_route_parity", |client| async move {
        let model = client.embedding_model(doubleword::QWEN3_EMBEDDING_8B);
        let normalized = model
            .embed_texts_response(inputs())
            .await
            .expect("normalized call should succeed");
        let raw = model
            .raw_embed_texts(inputs())
            .await
            .expect("raw call should succeed");

        assert_eq!(raw.data.len(), normalized.embeddings.len());
        let renormalized = raw
            .normalize(normalized.provider.as_str(), inputs())
            .expect("raw payload normalizes");
        assert_eq!(renormalized.model, normalized.model);
    })
    .await;
}

/// The single-text conveniences derive from the full method: same embedding,
/// same metadata.
#[tokio::test]
async fn single_text_convenience() {
    with_doubleword_cassette(
        "embedding_matrix/single_text_convenience",
        |client| async move {
            let model = client.embedding_model(doubleword::QWEN3_EMBEDDING_8B);
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
