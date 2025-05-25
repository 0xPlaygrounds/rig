use rig::embeddings::EmbeddingsBuilder;
use rig::providers::aliyun;
use rig::vector_store::in_memory_store::InMemoryVectorStore;
use rig::vector_store::VectorStoreIndex;
use rig::Embed;
use serde::{Deserialize, Serialize};

#[derive(Embed, Debug, Serialize, Deserialize, Eq, PartialEq)]
struct Greetings {
    id: String,

    #[embed]
    message: String,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize the Aliyun client
    let client = aliyun::Client::from_env();

    let embedding_model = client.embedding_model(aliyun::embedding::EMBEDDING_V1);

    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .documents(vec![
            Greetings {
                id: "1".to_string(),
                message: "Hello, world!".to_string(),
            },
            Greetings {
                id: "2".to_string(),
                message: "Goodbye, world!".to_string(),
            },
            Greetings {
                id: "3".to_string(),
                message: "The meaning of life is exploration and discovery".to_string(),
            },
            Greetings {
                id: "4".to_string(),
                message: "The purpose of life is to find happiness".to_string(),
            },
        ])?
        .build()
        .await
        .expect("Failed to embed documents");

    // Create vector store with the embeddings
    let vector_store = InMemoryVectorStore::from_documents(embeddings);

    // Create vector store index
    let index = vector_store.index(embedding_model);

    let results = index
        .top_n::<Greetings>("What is the meaning of life?", 2)
        .await?;

    println!("{:?}", results);

    Ok(())
}
