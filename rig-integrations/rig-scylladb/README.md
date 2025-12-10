# Rig-ScyllaDB

Vector store implementation for [ScyllaDB](https://www.scylladb.com/). This integration provides vector storage and similarity search using ScyllaDB as the backend.

## Usage

```rust
use rig::{providers::openai, vector_store::VectorStoreIndex, Embed};
use rig_scylladb::{ScyllaDbVectorStore, create_session};

#[derive(Embed, serde::Deserialize, serde::Serialize, Debug)]
struct Document {
    id: String,
    #[embed]
    text: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create ScyllaDB session
    let session = create_session("127.0.0.1:9042").await?;
    
    // Create OpenAI client and embedding model
    let openai_client = openai::Client::from_env();
    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);
    
    // Create vector store
    let vector_store = ScyllaDbVectorStore::new(
        model,
        session,
        "vector_db",    // keyspace
        "documents",    // table
        1536,          // embedding dimensions
    ).await?;
    
    // Query the store
    let results = vector_store
        .top_n::<Document>("search query", 5)
        .await?;
    
    for (score, id, doc) in results {
        println!("Score: {}, ID: {}, Document: {:?}", score, id, doc);
    }
    
    Ok(())
}
```

See the [`/examples`](./examples) folder for usage examples.

## Notes

- Uses application-level cosine similarity search (similar to SQLite and SurrealDB implementations)
- Suitable for small to medium datasets (< 100k vectors)
- Provides ScyllaDB's operational benefits: high availability, horizontal scaling, low latency
- Future-ready for ScyllaDB's native vector search capabilities 
