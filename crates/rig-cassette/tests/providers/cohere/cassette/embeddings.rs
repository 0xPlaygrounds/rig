//! Cassette-backed Cohere embeddings coverage.

use base64::{Engine as _, engine::general_purpose::STANDARD};
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
        assert_eq!(model.capabilities().ndims, 1536);

        let embeddings = model
            .call(
                EMBEDDING_INPUTS
                    .iter()
                    .map(|input| (*input).to_string())
                    .collect::<Vec<_>>(),
            )
            .await
            .map(|response| response.embeddings)
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
        assert_eq!(model.capabilities().ndims, 384);

        let embeddings = model
            .call(vec!["Where can I find coffee near the office?".to_string()])
            .await
            .map(|response| response.embeddings)
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
            assert_eq!(model.capabilities().ndims, 384);

            let embeddings = model
                .call(vec![
                    "The package arrived early and in perfect condition.".to_string(),
                ])
                .await
                .map(|response| response.embeddings)
                .expect("classification embedding should succeed");

            assert_embeddings_nonempty_and_consistent(&embeddings, 1);
        },
    )
    .await;
}

#[tokio::test]
async fn embed_image_smoke() {
    with_cohere_cassette("embeddings/embed_image_smoke", |client| async move {
        let model = client.image_embedding();
        assert_eq!(model.capabilities().ndims, 1024);
        assert_eq!(model.capabilities().max_documents, 1);

        let response = model
            .call(vec![decode_image(PNG_2X2)])
            .await
            .expect("image embedding request should succeed");
        let embedding = response
            .embeddings
            .into_iter()
            .next()
            .expect("one image embeds to one vector");

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
            // Cohere embeds one image per request, so a batch is one call
            // per image, in input order.
            let model = client.image_embedding();
            let mut embeddings = Vec::new();
            let mut raw = Vec::new();
            for image in [decode_image(PNG_2X2), decode_image(GIF_2X2)] {
                let response = model
                    .call(vec![image])
                    .await
                    .expect("image embedding should succeed");
                embeddings.extend(response.embeddings.clone());
                raw.push(
                    serde_json::from_value::<cohere::embeddings::ImageEmbeddingResponse>(
                        response.raw.clone(),
                    )
                    .expect("raw is Cohere's reply"),
                );
            }

            assert_embeddings_nonempty_and_consistent(&embeddings, 2);
            // Each reply carries its own `meta.billed_units.images`, the only
            // route to an image count: `Usage` is token-denominated and has no
            // slot for it.
            for page in &raw {
                assert_eq!(
                    page.meta.as_ref().map(|meta| meta.billed_units.images),
                    Some(1),
                    "each reply bills its own image: {page:?}"
                );
                assert_eq!(
                    page.embeddings.values.len(),
                    1,
                    "one embedding per image: {page:?}"
                );
            }
            // Not byte equality between the replies: Cohere mints a fresh
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
