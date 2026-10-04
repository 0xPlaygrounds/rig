//! Cassette-backed Cohere embeddings coverage.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use rig::providers::cohere;
use rig_test_support::cassette_models::MapWire;

use super::super::support::with_cohere_cassette;
use crate::support::assert_embeddings_nonempty_and_consistent;

const PNG_2X2: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACAQMAAABIeJ9nAAAAA1BMVEX/AAAZ4gk3AAAADElEQVQI12NgYGAAAAAEAAEnNCcKAAAAAElFTkSuQmCC";

fn decode_image(encoded: &str) -> Vec<u8> {
    STANDARD
        .decode(encoded)
        .expect("embedded cassette image should be valid base64")
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
