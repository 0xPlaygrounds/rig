//! Cassette-backed Cohere embeddings coverage.

use rig::embeddings::EmbeddingModel;
use rig::providers::cohere;

use super::super::support::with_cohere_cassette;
use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

/// Cohere's embed endpoint takes an explicit `input_type`, and the dimension count
/// is derived from the model identifier rather than reported by the response.
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
