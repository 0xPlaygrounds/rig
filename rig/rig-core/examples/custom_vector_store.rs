//! Example: Implementing a custom vector store backend
//!
//! This demonstrates how to implement `VectorStoreIndex` for any
//! vector database. Use this as a template for your own backend.
use redis::{
    AsyncCommands, Client,
    aio::MultiplexedConnection,
    vector_sets::{VAddOptions, VSimOptions, VectorAddInput, VectorSimilaritySearchInput},
};
use rig::{
    client::{EmbeddingsClient, ProviderClient},
    embeddings::EmbeddingModel,
    providers::openai,
    vector_store::{VectorSearchRequest, VectorStoreError, VectorStoreIndex, request::Filter},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

// This is the struct representing our vector store backend
struct RedisVectorStore<E> {
    conn: MultiplexedConnection,
    key: String,
    embedding_model: E,
}

impl<E: EmbeddingModel> RedisVectorStore<E> {
    async fn new(
        redis_url: &str,
        key: &str,
        embedding_model: E,
    ) -> Result<Self, redis::RedisError> {
        let client = Client::open(redis_url)?;

        Ok(Self {
            conn: client.get_multiplexed_async_connection().await?,
            key: key.to_string(),
            embedding_model,
        })
    }

    // Add a single document
    async fn add_document<T: Serialize>(
        &mut self,
        id: &str,
        content: &str,
        metadata: &T,
    ) -> Result<(), VectorStoreError> {
        // Get the embedding vector for your content
        let embedding = self
            .embedding_model
            .embed_text(content)
            .await
            .map_err(VectorStoreError::EmbeddingError)?;

        // Convert it to Vec<f32> for Redis
        let vec_f32: Vec<f32> = embedding.vec.iter().map(|&x| x as f32).collect();

        // Serialize metadata as JSON
        let attrs = serde_json::to_value(metadata)
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let _: bool = self
            .conn
            .vadd_options(
                &self.key,
                VectorAddInput::Fp32(&vec_f32),
                id,
                &VAddOptions::default().set_attributes(attrs),
            )
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        Ok(())
    }
}

impl<E: EmbeddingModel + Send + Sync> VectorStoreIndex for RedisVectorStore<E> {
    // Irrelevant for our program, but if we wanted to filter out query results
    // creating a simple 'RedisSearchFilter' would be the easiest way.
    // Alternatively, you can use vector_store::request::Filter as your filter DSL.
    //
    // For example, to filter out results with a distance >= 0.2 from our query:
    // ```rust
    // let req = VectorSearchRequest::builder()
    //      .query(query)
    //      .samples(2)
    //      .filter(Filter::lt("Distance", 0.2))
    //      .build()?;
    // ```
    type Filter = Filter<serde_json::Value>;

    async fn top_n<T: DeserializeOwned + Send>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        // Get the embedding vector for your content
        let embedding = self
            .embedding_model
            .embed_text(req.query())
            .await
            .map_err(VectorStoreError::EmbeddingError)?;

        // Convert to Vec<f32> for Redis
        let vec_f32: Vec<f32> = embedding.vec.iter().map(|&x| x as f32).collect();

        let results: Vec<(String, f64)> = self
            .conn
            .clone()
            .vsim_options(
                &self.key,
                VectorSimilaritySearchInput::Fp32(&vec_f32),
                &VSimOptions::default()
                    .set_count(req.samples() as usize)
                    .set_with_scores(true),
            )
            .await
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        // For each result, fetch the attributes
        let mut output = Vec::with_capacity(results.len());
        for (id, score) in results {
            let attrs: Option<String> = self
                .conn
                .clone()
                .vgetattr(&self.key, &id)
                .await
                .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

            let metadata: T = attrs
                .as_deref()
                .map(serde_json::from_str)
                .transpose()
                .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?
                .unwrap_or_else(|| serde_json::from_str("{}").unwrap());

            output.push((score, id, metadata));
        }

        Ok(output)
    }

    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let embedding = self
            .embedding_model
            .embed_text(req.query())
            .await
            .map_err(VectorStoreError::EmbeddingError)?;

        let vec_f32: Vec<f32> = embedding.vec.iter().map(|&x| x as f32).collect();

        let opts = VSimOptions::default()
            .set_count(req.samples() as usize)
            .set_with_scores(true);

        let results: Vec<(String, f64)> = self
            .conn
            .clone()
            .vsim_options(
                &self.key,
                VectorSimilaritySearchInput::Fp32(&vec_f32),
                &opts,
            )
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
    let openai_client = openai::Client::from_env();
    // Convert it to an EmbeddingModel
    let embedding_model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    // Create the Redis vector store
    let mut store =
        RedisVectorStore::new("redis://127.0.0.1:6379", "test_vectors", embedding_model).await?;

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

    // Add documents to the vector store
    println!("Adding documents to Redis vector store...");
    for (i, doc) in documents.iter().enumerate() {
        store
            .add_document(&format!("doc_{}", i), &doc.content, doc)
            .await?;
        println!("  Added: '{}'", doc.title);
    }

    // Query the vector store
    let query = "What programming language is best for systems programming?";
    println!("\nQuery: '{}'", query);

    // Create a query
    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(2)
        .build()?;

    // Execute the query
    let results: Vec<(f64, String, Document)> = store.top_n(req).await?;

    println!("\nResults:");
    for (score, id, doc) in results {
        println!("  [{:.4}] {} - '{}'", score, id, doc.title);
    }

    Ok(())
}
