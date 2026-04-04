//! Gemini embeddings smoke test.

#[cfg(feature = "derive")]
use rig::Embed;
use rig::client::{EmbeddingsClient, ProviderClient};
use rig::embeddings::EmbeddingModel;
use rig::providers::gemini;

use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

#[cfg(feature = "derive")]
#[derive(Embed, Debug)]
struct Greetings {
    #[embed]
    message: String,
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn embeddings_smoke() {
    let client = gemini::Client::from_env();
    let model = client.embedding_model(gemini::embedding::EMBEDDING_001);

    let embeddings = model
        .embed_texts(EMBEDDING_INPUTS.iter().map(|input| (*input).to_string()))
        .await
        .expect("embedding request should succeed");

    assert_embeddings_nonempty_and_consistent(&embeddings, EMBEDDING_INPUTS.len());
}

#[cfg(feature = "derive")]
#[tokio::test]
#[ignore = "requires GEMINI_API_KEY and --features derive"]
async fn derive_document_embeddings() {
    let client = gemini::Client::from_env();
    let embeddings = client
        .embeddings(gemini::embedding::EMBEDDING_001)
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
