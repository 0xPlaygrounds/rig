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
        let model = client.embedding_model(cohere::EMBED_V4, "search_document");
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
        let model = client.embedding_model(cohere::EMBED_ENGLISH_LIGHT_V3, "search_query");
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
            let model = client.embedding_model(cohere::EMBED_ENGLISH_LIGHT_V3, "classification");
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
        let model = client.image_embedding_model();
        assert_eq!(ImageEmbeddingModelTrait::ndims(&model), 1024);
        assert_eq!(
            <cohere::ImageEmbeddingModel as ImageEmbeddingModelTrait>::MAX_DOCUMENTS,
            1
        );

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
            let model = client.image_embedding_model();
            let embeddings = model
                .embed_images([decode_image(PNG_2X2), decode_image(GIF_2X2)])
                .await
                .expect("image embedding batch should succeed");

            assert_embeddings_nonempty_and_consistent(&embeddings, 2);
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
