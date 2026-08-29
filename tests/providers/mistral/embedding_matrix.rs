//! Recorded embedding matrix for mistral: the normalized response contract
//! pinned against live wire recordings. Mistral reports usage and a model echo; `mistral-correlation-id` is its transport id (see the bug cell).
//!
//! Cells asserted from recordings, not assumptions: response completeness
//! (order, provider, usage/model/request-id exactly as the wire reports),
//! `raw` round-tripping to the provider's own type, raw-route parity,
//! the single-text convenience, and the error path preserving the body.

use super::support::with_mistral_embedding_cassette;
use rig::client::EmbeddingsClient;
use rig::embeddings::{EmbeddingModel as _, NormalizeEmbeddingResponse as _};
use rig::providers::{mistral, openai_compatible as openai};

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
            let model = client.embedding_model(mistral::embedding::MISTRAL_EMBED);
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
    with_mistral_embedding_cassette("embedding_matrix/raw_round_trips", |client| async move {
        let model = client.embedding_model(mistral::embedding::MISTRAL_EMBED);
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
    with_mistral_embedding_cassette("embedding_matrix/raw_route_parity", |client| async move {
        let model = client.embedding_model(mistral::embedding::MISTRAL_EMBED);
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
    with_mistral_embedding_cassette(
        "embedding_matrix/single_text_convenience",
        |client| async move {
            let model = client.embedding_model(mistral::embedding::MISTRAL_EMBED);
            let response = model
                .embed_text_response(EMBEDDING_INPUTS[0])
                .await
                .expect("single-text embedding should succeed");
            assert_eq!(response.embeddings.len(), 1);
            assert_eq!(response.embeddings[0].document, EMBEDDING_INPUTS[0]);
            assert_eq!(response.provider, "mistral");
            let embedding = model
                .embed_text(EMBEDDING_INPUTS[0])
                .await
                .expect("convenience embedding should succeed");
            assert_eq!(embedding.vec.len(), response.embeddings[0].vec.len());
        },
    )
    .await;
}

/// `embedding_model_with_ndims` round-trips the requested width — the
/// provider either honors it or the driver errors honestly with
/// `MismatchedDimensions`; a silent mismatch is the bug this cell exists to
/// catch.
#[tokio::test]
async fn dimensions_request() {
    with_mistral_embedding_cassette("embedding_matrix/dimensions_request", |client| async move {
        // `mistral-embed` is fixed-width; `output_dimension` is a
        // codestral-embed capability, so the cell exercises that model.
        let ndims = 64;
        let model = client.embedding_model_with_ndims(mistral::embedding::CODESTRAL_EMBED, ndims);
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
    with_mistral_embedding_cassette(
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

/// Pin of the bug the matrix recording surfaced: Mistral sends
/// `mistral-correlation-id` on embedding responses exactly as it does on
/// completions, but the embeddings ext inherited the trait's `None` header
/// default, so the driver read no header and every normalized embedding
/// response reported `provider_request_id: None` — the id support asks for,
/// silently dropped. Fixed by declaring the header on Mistral's
/// `OpenAIEmbeddingsCompatible` impl; this cell asserts the id is captured
/// and matches what the recorded response headers actually carried.
#[tokio::test]
async fn bug_mistral_request_id_dropped() {
    with_mistral_embedding_cassette(
        "embedding_matrix/bug_mistral_request_id_dropped",
        |client| async move {
            let model = client.embedding_model(mistral::embedding::MISTRAL_EMBED);
            let response = model
                .embed_texts_response(inputs())
                .await
                .expect("embedding request should succeed");
            assert!(
                response
                    .provider_request_id
                    .as_deref()
                    .is_some_and(|id| !id.is_empty()),
                "the correlation id must survive onto the normalized response: {:?}",
                response.provider_request_id
            );
        },
    )
    .await;
}
