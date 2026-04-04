//! Migrated from `examples/gemini_embeddings.rs`.

use rig::Embed;
use rig::client::{EmbeddingsClient, ProviderClient};
use rig::providers::gemini;

#[derive(Embed, Debug)]
struct Greetings {
    #[embed]
    message: String,
}

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
