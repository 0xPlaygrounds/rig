//! llama.cpp embeddings smoke coverage.
//!
//! **Server**: the `--embeddings --pooling mean` configuration, loading
//! `Qwen/Qwen3-Embedding-0.6B-GGUF` Q8_0 — a real embedding model rather than
//! a causal LM pooled into one. The pre-merge fixtures for these two cells
//! were recorded against Ollama's `all-minilm`; what a causal LM under
//! `--pooling mean` actually returns is now its own cell in
//! `embedding_matrix.rs`, and the difference is why this suite states its
//! model.

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
    with_llamacpp_embeddings_cassette("embeddings/embeddings_smoke", |client| async move {
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
    with_llamacpp_embeddings_cassette(
        "embeddings/derive_document_embeddings",
        |client| async move {
            let embeddings = client
                .embeddings(CASSETTE_EMBEDDING_MODEL)
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
