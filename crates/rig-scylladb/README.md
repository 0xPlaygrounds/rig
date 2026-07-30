# Rig-ScyllaDB

Vector store implementation for [ScyllaDB](https://www.scylladb.com/). This integration provides vector storage and similarity search using ScyllaDB as the backend.

## Usage

```rust
use rig::{
    http_runtime::HttpRuntime, providers::openai, vector_store::request::VectorSearchRequest,
    Embed, OneOrMany,
};
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

    // Embedding configuration is plain data plus a shared HTTP runtime.
    let embed_cfg = openai::functions::EmbeddingConfig::from_env(openai::TEXT_EMBEDDING_ADA_002)?;
    let rt = HttpRuntime::new();

    // Create vector store; queries arrive pre-embedded
    let vector_store = ScyllaDbVectorStore::new(
        session,
        "vector_db",    // keyspace
        "documents",    // table
        1536,          // embedding dimensions
    ).await?;

    // Embed the query, then query the store
    let query_embedding =
        openai::functions::embed(&embed_cfg, &rt, vec!["search query".to_string()])
            .await?
            .embeddings
            .into_iter()
            .next()
            .expect("one embedding per input text");
    let req = VectorSearchRequest::new(OneOrMany::one(query_embedding), 5);
    let results = vector_store.top_n_as::<Document>(req).await?;

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
