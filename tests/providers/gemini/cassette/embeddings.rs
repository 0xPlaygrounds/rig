//! Gemini embeddings smoke test.

#[cfg(feature = "derive")]
use rig::Embed;
use rig::providers::gemini;

use crate::support::{EMBEDDING_INPUTS, assert_embeddings_nonempty_and_consistent};

#[cfg(feature = "derive")]
#[derive(Embed, Debug)]
struct Greetings {
    #[embed]
    message: String,
}

#[tokio::test]
async fn embeddings_smoke() {
    super::super::support::with_gemini_cassette(
        "embeddings/embeddings_smoke",
        |client| async move {
            let cfg = client.embedding_config(gemini::embedding::EMBEDDING_001);
            let rt = client.http();

            let embeddings = gemini::functions::embed(
                &cfg,
                &rt,
                EMBEDDING_INPUTS
                    .iter()
                    .map(|input| (*input).to_string())
                    .collect(),
            )
            .await
            .expect("embedding request should succeed")
            .embeddings;

            assert_embeddings_nonempty_and_consistent(&embeddings, EMBEDDING_INPUTS.len());
        },
    )
    .await;
}

#[cfg(feature = "derive")]
#[tokio::test]
async fn derive_document_embeddings() {
    super::super::support::with_gemini_cassette(
        "embeddings/derive_document_embeddings",
        |client| async move {
            let cfg = client.embedding_config(gemini::embedding::EMBEDDING_001);
            let rt = client.http();

            let embeddings = rig::embeddings::embed_documents(
                vec![
                    Greetings {
                        message: "Hello, world!".to_string(),
                    },
                    Greetings {
                        message: "Goodbye, world!".to_string(),
                    },
                ],
                gemini::functions::DESCRIPTOR
                    .max_embedding_documents
                    .expect("gemini declares a batch limit"),
                1,
                |texts| gemini::functions::embed(&cfg, &rt, texts),
            )
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
