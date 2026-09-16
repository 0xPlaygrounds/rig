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
use rig::embeddings::{EmbeddingModel as _, EmbeddingResponse};
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

/// The provider-native fields of one reply's own payload, read back out of
/// [`EmbeddingResponse::raw`], asserted against the normalized view of that
/// same reply.
///
/// One decoder produces both views, so this pins that mapping rather than
/// comparing it to a second copy of itself.
fn assert_raw_agrees_with_normalized(response: &EmbeddingResponse) {
    let reply: openai::CompatibleEmbeddingResponse = serde_json::from_value(response.raw.clone())
        .expect("`raw` is the serialized openai::CompatibleEmbeddingResponse");
    assert_eq!(reply.data.len(), response.embeddings.len());
    assert_eq!(Some(reply.model.as_str()), response.model.as_deref());
    for (datum, embedding) in reply.data.iter().zip(&response.embeddings) {
        assert_eq!(datum.embedding.len(), embedding.vec.len());
    }
    let usage = reply.usage.expect("OpenAI reports embedding usage");
    assert_eq!(
        response.usage.input_tokens,
        Some(usage.prompt_tokens as u64)
    );
    assert_eq!(response.usage.total_tokens, Some(usage.total_tokens as u64));
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
                .embed_texts_response(inputs())
                .await
                .expect("embedding request should succeed");
            assert_normalized_embedding_response(&response, &EMBEDDING_INPUTS, &expectations());
        },
    )
    .await;
}

/// `raw` is the provider's own payload, verbatim: it deserializes back to the
/// provider's own response type, whose native fields are what the normalized
/// view reports.
#[tokio::test]
async fn raw_round_trips() {
    with_openai_cassette("embedding_matrix/raw_round_trips", |client| async move {
        let model = client
            .openai
            .embedding(openai::TEXT_EMBEDDING_3_SMALL, None);
        let response = model
            .embed_texts_response(inputs())
            .await
            .expect("embedding request should succeed");

        assert_raw_agrees_with_normalized(&response);
    })
    .await;
}

/// Two identical live exchanges in one recording: each reply carries both
/// views, so the second reply's `raw` agrees with its own normalized view
/// just as the first one's does, and the two exchanges report the same model.
#[tokio::test]
async fn raw_route_parity() {
    with_openai_cassette("embedding_matrix/raw_route_parity", |client| async move {
        let model = client
            .openai
            .embedding(openai::TEXT_EMBEDDING_3_SMALL, None);
        let first = model
            .embed_texts_response(inputs())
            .await
            .expect("first call should succeed");
        let second = model
            .embed_texts_response(inputs())
            .await
            .expect("second call should succeed");

        assert_raw_agrees_with_normalized(&first);
        assert_raw_agrees_with_normalized(&second);
        assert_eq!(first.model, second.model);
    })
    .await;
}

/// The single-text conveniences derive from the full method: same embedding,
/// same metadata.
#[tokio::test]
async fn single_text_convenience() {
    with_openai_cassette(
        "embedding_matrix/single_text_convenience",
        |client| async move {
            let model = client
                .openai
                .embedding(openai::TEXT_EMBEDDING_3_SMALL, None);
            let response = model
                .embed_text_response(EMBEDDING_INPUTS[0])
                .await
                .expect("single-text embedding should succeed");
            assert_eq!(response.embeddings.len(), 1);
            assert_eq!(response.embeddings[0].document, EMBEDDING_INPUTS[0]);
            assert_eq!(response.provider, "openai");
            let embedding = model
                .embed_text(EMBEDDING_INPUTS[0])
                .await
                .expect("convenience embedding should succeed");
            assert_eq!(embedding.vec.len(), response.embeddings[0].vec.len());
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
    with_openai_cassette(
        "embedding_matrix/error_preserves_provider_body",
        |client| async move {
            let model = client.openai.embedding("no-such-embedding-model", None);
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
