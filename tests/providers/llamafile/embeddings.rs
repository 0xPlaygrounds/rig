//! Llamafile embeddings smoke tests.

#[cfg(feature = "derive")]
use rig::Embed;
use rig::http_runtime::HttpRuntime;
use rig::providers::llamafile;

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

    let cfg = support::client().embedding_config(support::model_name());
    let rt = HttpRuntime::new();

    let embeddings = llamafile::functions::embed(
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
}

#[cfg(feature = "derive")]
#[tokio::test]
#[ignore = "requires a local llamafile server at http://localhost:8080 and --features derive"]
async fn derive_document_embeddings() {
    if support::skip_if_server_unavailable() {
        return;
    }

    let cfg = support::client().embedding_config(support::model_name());
    let rt = HttpRuntime::new();
    let embeddings = rig::embeddings::embed_documents(
        vec![
            Greetings {
                message: "Hello, world!".to_string(),
            },
            Greetings {
                message: "Goodbye, world!".to_string(),
            },
        ],
        llamafile::functions::DESCRIPTOR
            .max_embedding_documents
            .unwrap_or(usize::MAX),
        1,
        |texts| llamafile::functions::embed(&cfg, &rt, texts),
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
}
