//! Llamafile embeddings smoke tests.

#[cfg(feature = "derive")]
use rig_core as rig;
#[cfg(feature = "derive")]
use rig_core::Embed;
use rig_core::client::EmbeddingsClient;
use rig_core::embeddings::EmbeddingModel;

use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

use super::support;

#[cfg(feature = "derive")]
#[derive(Embed, Debug)]
struct Greetings {
    #[embed]
    message: String,
}

#[tokio::test]
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn embeddings_smoke() {
    if support::skip_if_server_unavailable() {
        return;
    }

    let client = support::client();
    let model = client.embedding_model(support::model_name());

    let embeddings = model
        .embed_texts(EMBEDDING_INPUTS.iter().map(|input| (*input).to_string()))
        .await
        .expect("embedding request should succeed");

    assert_embeddings_nonempty_and_consistent(&embeddings, EMBEDDING_INPUTS.len());
}

#[cfg(feature = "derive")]
#[tokio::test]
#[ignore = "requires a local llamafile server at http://localhost:8080 and --features derive"]
async fn derive_document_embeddings() {
    if support::skip_if_server_unavailable() {
        return;
    }

    let client = support::client();
    let embeddings = client
        .embeddings(support::model_name())
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
}
