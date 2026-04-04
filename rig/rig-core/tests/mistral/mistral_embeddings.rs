//! Migrated from `examples/mistral_embeddings.rs`.

use rig::Embed;
use rig::client::{EmbeddingsClient, ProviderClient};
use rig::embeddings::EmbeddingsBuilder;
use rig::providers::mistral;
use rig::vector_store::VectorStoreIndex;
use rig::vector_store::in_memory_store::InMemoryVectorStore;
use rig::vector_store::request::VectorSearchRequest;
use serde::{Deserialize, Serialize};

#[derive(Embed, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct Greetings {
    #[embed]
    message: String,
}

#[tokio::test]
#[ignore = "requires MISTRAL_API_KEY and --features derive"]
async fn derive_embeddings_and_vector_search() {
    let client = mistral::Client::from_env();
    let embedding_model = client.embedding_model(mistral::embedding::MISTRAL_EMBED);
    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
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

    let vector_store = InMemoryVectorStore::from_documents(embeddings);
    let index = vector_store.index(embedding_model);
    let request = VectorSearchRequest::builder()
        .query("Hello world")
        .samples(1)
        .build()
        .expect("request should build");
    let results = index
        .top_n::<Greetings>(request)
        .await
        .expect("vector search should succeed");

    assert_eq!(results.len(), 1);
    assert!(
        results[0].2.message.contains("Hello"),
        "expected the hello document to be the closest match"
    );
}
