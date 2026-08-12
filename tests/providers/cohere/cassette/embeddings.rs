//! Cassette-backed Cohere embeddings coverage.

use rig::embeddings::EmbeddingModel;
use rig::providers::cohere;

use super::super::support::with_cohere_cassette;
use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

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
