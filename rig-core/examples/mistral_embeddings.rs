use rig::Embed;
use rig::client::EmbeddingsClient;
use rig::embeddings::EmbeddingsBuilder;
use rig::providers::mistral;
use rig::vector_store::VectorStoreIndex;
use rig::vector_store::in_memory_store::InMemoryVectorStore;
use rig::vector_store::request::VectorSearchRequest;
use serde::{Deserialize, Serialize};

#[derive(Embed, Debug, Serialize, Deserialize, Eq, PartialEq)]
struct Greetings {
    #[embed]
    message: String,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize the Mistral client
    let client = mistral::Client::from_env();
    let embedding_model = client.embedding_model(mistral::embedding::MISTRAL_EMBED);
    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .document(Greetings {
            message: "Hello, world!".to_string(),
        })?
        .document(Greetings {
            message: "Goodbye, world!".to_string(),
        })?
        .build()
        .await
        .expect("Failed to embed documents");

    // Create vector store with the embeddings
    let vector_store = InMemoryVectorStore::from_documents(embeddings);

    // Create vector store index
    let index = vector_store.index(embedding_model);

    let req = VectorSearchRequest::builder()
        .query("Hello world")
        .samples(1)
        .build()?;

    let results = index.top_n::<Greetings>(req).await?;

    println!("{results:?}");

    Ok(())
}
