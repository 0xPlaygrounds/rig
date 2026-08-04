//! Migrated from `examples/mistral_embeddings.rs`.

use rig::Embed;
use rig::OneOrMany;
use rig::embeddings::batching::default_concurrency;
use rig::embeddings::embed_documents;
use rig::http_runtime::HttpRuntime;
use rig::providers::mistral;
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
    let cfg = mistral::functions::EmbeddingConfig::from_env(mistral::embedding::MISTRAL_EMBED)
        .expect("embedding config should build");
    let rt = HttpRuntime::new();
    let max_documents = mistral::functions::DESCRIPTOR
        .max_embedding_documents
        .unwrap_or(1);

    let embeddings = embed_documents(
        vec![
            Greetings {
                message: "Hello, world!".to_string(),
            },
            Greetings {
                message: "Goodbye, world!".to_string(),
            },
        ],
        max_documents,
        default_concurrency(max_documents),
        |texts| mistral::functions::embed(&cfg, &rt, texts),
    )
    .await
    .expect("embedding request should succeed");

    let vector_store =
        InMemoryVectorStore::from_documents(embeddings).expect("documents should serialize");
    let query = mistral::functions::embed(&cfg, &rt, vec!["Hello world".to_string()])
        .await
        .expect("query embedding should succeed")
        .embeddings
        .into_iter()
        .next()
        .expect("query embedding should return one vector");
    let request = VectorSearchRequest::new(OneOrMany::one(query), 1);
    let results = vector_store
        .top_n_as::<Greetings>(request)
        .await
        .expect("vector search should succeed");

    assert_eq!(results.len(), 1);
    assert!(
        results[0].2.message.contains("Hello"),
        "expected the hello document to be the closest match"
    );
}
