//! Cassette-backed Cohere embeddings coverage.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use rig::embeddings::{
    EmbeddingModel as TextEmbeddingModel, ImageEmbeddingModel as ImageEmbeddingModelTrait,
};
use rig::providers::cohere;

use super::super::support::with_cohere_cassette;
use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

const PNG_2X2: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACAQMAAABIeJ9nAAAAA1BMVEX/AAAZ4gk3AAAADElEQVQI12NgYGAAAAAEAAEnNCcKAAAAAElFTkSuQmCC";
const GIF_2X2: &str = "R0lGODlhAgACAPAAAAAA/wAAACH5BAAAAAAALAAAAAACAAIAAAIChFEAOw==";

fn decode_image(encoded: &str) -> Vec<u8> {
    STANDARD
        .decode(encoded)
        .expect("embedded cassette image should be valid base64")
}

#[tokio::test]
async fn embed_texts_smoke() {
    with_cohere_cassette("embeddings/embed_texts_smoke", |client| async move {
        let model = client
            .embedding(cohere::EMBED_V4, None)
            .map_wire(|wire| wire.with_input_type("search_document"));
        assert_eq!(model.ndims(), 1536);

        let embeddings = model
            .embed_texts(EMBEDDING_INPUTS.iter().map(|input| (*input).to_string()))
            .await
            .expect("embedding request should succeed");

        assert_embeddings_nonempty_and_consistent(&embeddings, EMBEDDING_INPUTS.len());
    })
    .await;
}

#[tokio::test]
async fn embed_search_query_smoke() {
    with_cohere_cassette("embeddings/embed_search_query_smoke", |client| async move {
        let model = client
            .embedding(cohere::EMBED_ENGLISH_LIGHT_V3, None)
            .map_wire(|wire| wire.with_input_type("search_query"));
        assert_eq!(model.ndims(), 384);

        let embeddings = model
            .embed_texts(["Where can I find coffee near the office?".to_string()])
            .await
            .expect("search query embedding should succeed");

        assert_embeddings_nonempty_and_consistent(&embeddings, 1);
    })
    .await;
}

#[tokio::test]
async fn embed_classification_smoke() {
    with_cohere_cassette(
        "embeddings/embed_classification_smoke",
        |client| async move {
            let model = client
                .embedding(cohere::EMBED_ENGLISH_LIGHT_V3, None)
                .map_wire(|wire| wire.with_input_type("classification"));
            assert_eq!(model.ndims(), 384);

            let embeddings = model
                .embed_texts(["The package arrived early and in perfect condition.".to_string()])
                .await
                .expect("classification embedding should succeed");

            assert_embeddings_nonempty_and_consistent(&embeddings, 1);
        },
    )
    .await;
}

#[tokio::test]
async fn embed_image_smoke() {
    with_cohere_cassette("embeddings/embed_image_smoke", |client| async move {
        let model = client.map_wire(|cohere| cohere.image_embeddings());
        assert_eq!(ImageEmbeddingModelTrait::ndims(&model), 1024);
        assert_eq!(ImageEmbeddingModelTrait::max_documents(&model), 1);

        let embedding = model
            .embed_image(&decode_image(PNG_2X2))
            .await
            .expect("image embedding request should succeed");

        assert_eq!(embedding.vec.len(), 1024);
        assert!(embedding.document.starts_with("image/png;sha256="));
    })
    .await;
}

#[tokio::test]
async fn embed_images_preserves_batch_order() {
    with_cohere_cassette(
        "embeddings/embed_images_preserves_batch_order",
        |client| async move {
            let model = client.map_wire(|cohere| cohere.image_embeddings());
            let response = model
                .embed_images_response([decode_image(PNG_2X2), decode_image(GIF_2X2)])
                .await
                .expect("image embedding batch should succeed");
            let embeddings = response.embeddings.clone();

            assert_embeddings_nonempty_and_consistent(&embeddings, 2);
            // Two images are two requests on this wire, so the operation has
            // two pages and `raw` is the per-image sequence in input order —
            // what the per-request escape hatch used to return. Each page
            // carries its own `meta.billed_units.images`, the only route to an
            // image count: `Usage` is token-denominated and has no slot for it.
            let raw: Vec<cohere::embeddings::ImageEmbeddingResponse> =
                serde_json::from_value(response.raw.clone()).expect("raw is the per-image array");
            assert_eq!(raw.len(), 2, "one answer per input image: {raw:?}");
            for page in &raw {
                assert_eq!(
                    page.meta.as_ref().map(|meta| meta.billed_units.images),
                    Some(1),
                    "each page bills its own image: {page:?}"
                );
                assert_eq!(
                    page.embeddings.values.len(),
                    1,
                    "one embedding per per-image answer: {page:?}"
                );
            }
            // Not byte equality between the pages: Cohere mints a fresh
            // generation id per call, so the two recorded replies differ in
            // `id` alone. What repeats is the shape and the billing.
            assert_ne!(
                raw.first().and_then(|page| page.id.clone()),
                raw.get(1).and_then(|page| page.id.clone()),
                "each call is its own generation, so the ids differ"
            );
            assert_eq!(embeddings.first().map(|item| item.vec.len()), Some(1024));
            assert_eq!(embeddings.get(1).map(|item| item.vec.len()), Some(1024));
            assert!(
                embeddings
                    .first()
                    .is_some_and(|item| item.document.starts_with("image/png;sha256="))
            );
            assert!(
                embeddings
                    .get(1)
                    .is_some_and(|item| item.document.starts_with("image/gif;sha256="))
            );
        },
    )
    .await;
}
