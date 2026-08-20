//! Recorded embedding matrix for Cohere: the normalized response contract
//! pinned against live wire recordings, on Cohere's own `/v1/embed` wire.
//!
//! Cohere reports usage as `meta.billed_units` and a response-scoped `id`,
//! but echoes no model and has no transport request-id header on this
//! endpoint — those axes are asserted as their documented `None`/zero
//! outcomes, not skipped. The dimensions cell is absent: Cohere's embed wire
//! takes no dimension parameter.

use rig::embeddings::{
    EmbeddingModel as _, ImageEmbeddingModel as _, NormalizeEmbeddingResponse as _,
};
use rig::providers::cohere;

use super::super::support::with_cohere_cassette;
use crate::support::{
    EMBEDDING_INPUTS, EmbeddingMatrixExpectations, assert_normalized_embedding_response,
};

const INPUT_TYPE: &str = "search_document";

/// A 2x2 red PNG, the same fixture the existing image-embedding smoke uses.
const PNG_2X2: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACAQMAAABIeJ9nAAAAA1BMVEX/AAAZ4gk3AAAADElEQVQI12NgYGAAAAAEAAEnNCcKAAAAAElFTkSuQmCC";

fn decode_image(encoded: &str) -> Vec<u8> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .expect("fixture should be valid base64")
}

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
            let model = client.embedding_model(cohere::EMBED_V4, INPUT_TYPE);
            let response = model
                .embed_texts_response(inputs())
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
async fn raw_round_trips() {
    with_cohere_cassette("embedding_matrix/raw_round_trips", |client| async move {
        let model = client.embedding_model(cohere::EMBED_V4, INPUT_TYPE);
        let response = model
            .embed_texts_response(inputs())
            .await
            .expect("embedding request should succeed");

        let raw: cohere::embeddings::EmbeddingResponse =
            serde_json::from_value(response.raw.clone()).expect("raw round-trips");
        assert_eq!(raw.embeddings.len(), response.embeddings.len());

        let renormalized = raw
            .normalize("cohere", inputs())
            .expect("re-normalization succeeds");
        assert_eq!(renormalized.embeddings.len(), response.embeddings.len());
        assert_eq!(renormalized.response_id, response.response_id);
        assert_eq!(renormalized.usage, response.usage);
    })
    .await;
}

#[tokio::test]
async fn raw_route_parity() {
    with_cohere_cassette("embedding_matrix/raw_route_parity", |client| async move {
        let model = client.embedding_model(cohere::EMBED_V4, INPUT_TYPE);
        let normalized = model
            .embed_texts_response(inputs())
            .await
            .expect("normalized call should succeed");
        let raw = model
            .raw_embed_texts(inputs())
            .await
            .expect("raw call should succeed");
        assert_eq!(raw.embeddings.len(), normalized.embeddings.len());
        assert!(raw.meta.is_some(), "billed units ride the raw payload");
    })
    .await;
}

#[tokio::test]
async fn single_text_convenience() {
    with_cohere_cassette(
        "embedding_matrix/single_text_convenience",
        |client| async move {
            let model = client.embedding_model(cohere::EMBED_V4, INPUT_TYPE);
            let response = model
                .embed_text_response(EMBEDDING_INPUTS[0])
                .await
                .expect("single-text embedding should succeed");
            assert_eq!(response.embeddings.len(), 1);
            assert_eq!(response.embeddings[0].document, EMBEDDING_INPUTS[0]);
            assert_eq!(response.provider, "cohere");
        },
    )
    .await;
}

#[tokio::test]
async fn error_preserves_provider_body() {
    with_cohere_cassette(
        "embedding_matrix/error_preserves_provider_body",
        |client| async move {
            let model = client.embedding_model("no-such-embedding-model", INPUT_TYPE);
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

/// The image half: one Cohere answer per image, the normalized response
/// aggregating them and `raw` carrying the per-image array.
#[tokio::test]
async fn image_normalized_and_raw_round_trip() {
    with_cohere_cassette(
        "embedding_matrix/image_normalized_and_raw_round_trip",
        |client| async move {
            let model = client.image_embedding_model();
            let response = model
                .embed_images_response(vec![decode_image(PNG_2X2)])
                .await
                .expect("image embedding should succeed");

            assert_eq!(response.embeddings.len(), 1);
            assert_eq!(response.provider, "cohere");
            assert_eq!(response.embeddings[0].vec.len(), model.ndims());
            // Cohere bills image embeds as `billed_units.images`, not tokens
            // — `Usage` is token-denominated, so the normalized usage is
            // honestly zero and the image count is read off the raw payload.
            assert!(
                !response.usage.has_values(),
                "no token usage exists to report for an image embed: {:?}",
                response.usage
            );

            let raw: Vec<cohere::embeddings::ImageEmbeddingResponse> =
                serde_json::from_value(response.raw.clone()).expect("raw is the per-image array");
            assert_eq!(raw.len(), 1);
            assert_eq!(
                raw[0].embeddings.values.len(),
                1,
                "one embedding per per-image answer"
            );
            assert_eq!(
                raw[0].meta.as_ref().map(|meta| meta.billed_units.images),
                Some(1),
                "the image count rides the raw payload's billed_units"
            );

            let typed = model
                .raw_embed_images(vec![decode_image(PNG_2X2)])
                .await
                .expect("raw image route should succeed");
            assert_eq!(typed.len(), 1);
        },
    )
    .await;
}
