//! Migrated from `examples/mistral_embeddings.rs`. Vector search was removed with
//! the core vector-store abstraction, so this now covers embedding generation
//! only (retrieval is a user-land concern — see the `tool_active_rag` /
//! `hook_passive_rag` examples).

use rig::Embed;
use rig::client::{EmbeddingsClient, ProviderClient};
use rig::embeddings::EmbeddingsBuilder;
use rig::providers::mistral;
use serde::{Deserialize, Serialize};

#[derive(Embed, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct Greetings {
    #[embed]
    message: String,
}

#[tokio::test]
#[ignore = "requires MISTRAL_API_KEY and --features derive"]
async fn derive_embeddings() {
    let client = mistral::Client::from_env().expect("client should build");
    let embedding_model = client.embedding_model(mistral::embedding::MISTRAL_EMBED);
    let embeddings = EmbeddingsBuilder::new(embedding_model)
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

    assert_eq!(embeddings.len(), 2, "one embedding set per document");
    for (doc, embeds) in &embeddings {
        assert!(
            !embeds.first().vec.is_empty(),
            "embedding vector for {doc:?} should be non-empty"
        );
    }
}
