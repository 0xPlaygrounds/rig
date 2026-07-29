//! Example: Implementing a custom vector store backend
//!
//! This demonstrates how to implement a vector store backend over rig's shared
//! vector store data vocabulary (`VectorSearchRequest`, `SearchHit`,
//! `StoreRecord`). There is no store trait: a backend exposes concrete
//! inherent async methods (`top_n`, `top_n_ids`, `insert`, ...) and receives
//! pre-embedded queries. Use this as a template for your own backend.
use redis::{
    AsyncCommands, Client,
    aio::MultiplexedConnection,
    vector_sets::{VAddOptions, VSimOptions, VectorAddInput, VectorSimilaritySearchInput},
};
use rig::{
    OneOrMany,
    client::{EmbeddingsClient, ProviderClient},
    embeddings::{Embedding, EmbeddingModel},
    providers::openai,
    vector_store::{SearchHit, StoreRecord, VectorSearchRequest, VectorStoreError},
};
use serde::{Deserialize, Serialize};

// This is the struct representing our vector store backend.
//
// Note that it holds no embedding model: queries and records arrive
// pre-embedded, so the backend only stores and searches vectors.
struct RedisVectorStore {
    conn: MultiplexedConnection,
    key: String,
}

impl RedisVectorStore {
    async fn new(redis_url: &str, key: &str) -> Result<Self, redis::RedisError> {
        let client = Client::open(redis_url)?;

        Ok(Self {
            conn: client.get_multiplexed_async_connection().await?,
            key: key.to_string(),
        })
    }

    /// Insert precomputed records.
    async fn insert(&mut self, records: Vec<StoreRecord>) -> Result<(), VectorStoreError> {
        for record in records {
            // Index every embedding of the record under the record's id.
            for embedding in record.embeddings.iter() {
                // Convert it to Vec<f32> for Redis
                let vec_f32: Vec<f32> = embedding.vec.iter().map(|&x| x as f32).collect();

                let _: bool = self
                    .conn
                    .vadd_options(
                        &self.key,
                        VectorAddInput::Fp32(&vec_f32),
                        &record.id,
                        &VAddOptions::default().set_attributes(record.payload.clone()),
                    )
                    .await
                    .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;
            }
        }

        Ok(())
    }

    // If we wanted to filter query results, creating a simple
    // 'RedisSearchFilter' would be the easiest way. Alternatively, you can use
    // vector_store::request::Filter as your filter DSL and keep a `filter`
    // field on the request.

    /// Returns the top N most similar documents for a pre-embedded query.
    async fn top_n(&self, req: VectorSearchRequest) -> Result<Vec<SearchHit>, VectorStoreError> {
        let ids = self.top_n_ids(req).await?;

        // For each result, fetch the attributes
        let mut output = Vec::with_capacity(ids.len());
        for (score, id) in ids {
            let attrs: Option<String> = self
                .conn
                .clone()
                .vgetattr(&self.key, &id)
                .await
                .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

            let payload: serde_json::Value = attrs
                .as_deref()
                .map(serde_json::from_str)
                .transpose()
                .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?
                .unwrap_or_default();

            output.push(SearchHit { id, score, payload });
        }

        Ok(output)
    }

    /// Returns the top N most similar document IDs as `(score, id)` tuples.
    async fn top_n_ids(
        &self,
        req: VectorSearchRequest,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        // Search with the first query embedding. (A backend can also merge
        // results across all query embeddings.)
        let embedding = req.query().first();
        let vec_f32: Vec<f32> = embedding.vec.iter().map(|&x| x as f32).collect();

        let opts = VSimOptions::default()
            .set_count(req.samples() as usize)
            .set_with_scores(true);

        let results: Vec<(String, f64)> = self
            .conn
            .clone()
            .vsim_options(&self.key, VectorSimilaritySearchInput::Fp32(&vec_f32), &opts)
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        // Convert from (id, score) to (score, id)
        Ok(results.into_iter().map(|(id, score)| (score, id)).collect())
    }
}

/// Our content
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Document {
    title: String,
    content: String,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize OpenAI client from environment
    let openai_client = openai::Client::from_env()?;
    // Convert it to an EmbeddingModel
    let embedding_model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    // Create the Redis vector store
    let mut store = RedisVectorStore::new("redis://127.0.0.1:6379", "test_vectors").await?;

    // Sample documents to index
    let documents = [
        Document {
            title: "Rust Programming".to_string(),
            content: "Rust is a systems programming language focused on safety and performance."
                .to_string(),
        },
        Document {
            title: "Haskell Programming".to_string(),
            content: "Haskell is a functional programming language known for its category theory informed abstractions"
                .to_string(),
        },
        Document {
            title: "OCaml Programming".into(),
            content: "OCaml is a functional programming language primarily concerned with pragmatism and systems programming.".into()
        },
        Document {
            title: "Machine Learning".to_string(),
            content: "Machine learning is a subset of AI that enables systems to learn from data."
                .to_string(),
        },
    ];

    // Embed the documents and add them to the vector store. Embedding happens
    // *outside* the store: it only ever sees precomputed vectors.
    println!("Adding documents to Redis vector store...");
    let mut records = Vec::new();
    for (i, doc) in documents.iter().enumerate() {
        let embedding: Embedding = embedding_model.embed_text(&doc.content).await?;
        records.push(StoreRecord::new(
            format!("doc_{i}"),
            doc,
            OneOrMany::one(embedding),
        )?);
        println!("  Added: '{}'", doc.title);
    }
    store.insert(records).await?;

    // Query the vector store
    let query = "What programming language is best for systems programming?";
    println!("\nQuery: '{}'", query);

    // Embed the query, then create a pre-embedded request
    let query_embedding = embedding_model.embed_text(query).await?;
    let req = VectorSearchRequest::builder()
        .query(query_embedding)
        .samples(2)
        .build();

    // Execute the query
    let results = store.top_n(req).await?;

    println!("\nResults:");
    for hit in results {
        let doc: Document = hit.payload_as()?;
        println!("  [{:.4}] {} - '{}'", hit.score, hit.id, doc.title);
    }

    Ok(())
}
