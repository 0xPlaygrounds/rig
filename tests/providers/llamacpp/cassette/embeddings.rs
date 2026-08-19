//! llama.cpp embeddings smoke test.
//!
//! Replays by default; set `RIG_PROVIDER_TEST_MODE=record` to record against a
//! local OpenAI-compatible llama.cpp-family server (see `cassette_support`).

use rig::client::EmbeddingsClient;
use rig::embeddings::EmbeddingModel;

use super::super::cassette_support::*;

use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};
use rig::Embed;

#[cfg(feature = "derive")]
#[derive(Embed, Debug)]
struct Greetings {
    #[embed]
    message: String,
}

#[tokio::test]
async fn embeddings_smoke() {
    with_llamacpp_cassette("embeddings/embeddings_smoke", |client| async move {
        let model = client.embedding_model(CASSETTE_EMBEDDING_MODEL);

        let embeddings = model
            .embed_texts(EMBEDDING_INPUTS.iter().map(|input| (*input).to_string()))
            .await
            .expect("embedding request should succeed");

        assert_embeddings_nonempty_and_consistent(&embeddings, EMBEDDING_INPUTS.len());
    })
    .await;
}

#[tokio::test]
async fn derive_document_embeddings() {
    with_llamacpp_cassette(
        "embeddings/derive_document_embeddings",
        |client| async move {
            let embeddings = client
                .embeddings(CASSETTE_MODEL)
                .document(Greetings {
                    message: "Hello, world!".to_string(),
                })
                .expect("first document should build")
                .document(Greetings {
                    message: "Goodbye, world!".to_string(),
                })
                .expect("second document should build")
                .build()
                .await
                .expect("embedding request should succeed");

            assert_eq!(embeddings.len(), 2);
            for (_document, embeddings_for_document) in embeddings {
                let mut dims = None;
                for embedding in embeddings_for_document {
                    assert!(
                        !embedding.vec.is_empty(),
                        "expected each embedding vector to be non-empty"
                    );

                    match dims {
                        Some(expected_dims) => assert_eq!(embedding.vec.len(), expected_dims),
                        None => dims = Some(embedding.vec.len()),
                    }
                }
            }
        },
    )
    .await;
}
