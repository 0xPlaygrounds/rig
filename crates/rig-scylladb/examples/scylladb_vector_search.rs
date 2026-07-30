use rig_core::OneOrMany;
use rig_core::embeddings::{default_concurrency, embed_documents};
use rig_core::http_runtime::HttpRuntime;
use rig_core::{Embed, providers::openai, vector_store::request::VectorSearchRequest};
use rig_scylladb::{ScyllaDbVectorStore, create_session};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Embed, Clone, Debug, Deserialize, Serialize)]
struct Word {
    id: String,
    #[embed]
    definition: String,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize tracing for logging
    tracing_subscriber::fmt::init();

    // Create ScyllaDB session
    // In production, you would use your ScyllaDB cluster endpoints
    let session = create_session("127.0.0.1:9042").await?;

    // Embedding configuration is plain data plus a shared HTTP runtime.
    let embed_cfg = openai::functions::EmbeddingConfig::from_env(openai::TEXT_EMBEDDING_ADA_002)?;
    let rt = HttpRuntime::new();
    let max_documents = openai::functions::DESCRIPTOR
        .max_embedding_documents
        .unwrap_or(usize::MAX);

    // Create ScyllaDB vector store. Queries arrive pre-embedded, so the store
    // itself holds no embedding model.
    let vector_store = ScyllaDbVectorStore::new(
        session,
        "word_definitions", // keyspace
        "words",            // table
        1536,               // dimensions for text-embedding-ada-002
    )
    .await?;

    // Create sample word definitions
    let words = vec![
        Word {
            id: "doc0".to_string(),
            definition: "A large language model trained by OpenAI".to_string(),
        },
        Word {
            id: "doc1".to_string(),
            definition: "A high-performance NoSQL database compatible with Cassandra".to_string(),
        },
        Word {
            id: "doc2".to_string(),
            definition: "A systems programming language focused on safety and performance"
                .to_string(),
        },
        Word {
            id: "doc3".to_string(),
            definition: "A vector database for storing and querying high-dimensional data"
                .to_string(),
        },
        Word {
            id: "doc4".to_string(),
            definition: "An asynchronous runtime for Rust programming language".to_string(),
        },
    ];

    // Generate embeddings for the documents. Embedding happens *outside* the
    // store: it only ever sees precomputed vectors.
    let embeddings = embed_documents(
        words.clone(),
        max_documents,
        default_concurrency(max_documents),
        |texts| openai::functions::embed(&embed_cfg, &rt, texts),
    )
    .await?;

    tracing::info!(
        "Inserting {} word definitions into ScyllaDB...",
        words.len()
    );

    // Insert documents with their embeddings. The store's id column is a UUID,
    // so give each stored record a fresh UUID id.
    vector_store
        .insert_as(
            embeddings
                .into_iter()
                .map(|(document, embedding)| (Uuid::new_v4().to_string(), document, embedding))
                .collect(),
        )
        .await?;

    tracing::info!("Documents inserted successfully!");

    // Test similarity search: embed the query, then send the pre-embedded request
    let query = "What is Rust programming language?";
    let query_embedding = openai::functions::embed(&embed_cfg, &rt, vec![query.to_string()])
        .await?
        .embeddings
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no embedding returned for the query"))?;
    let req = VectorSearchRequest::new(OneOrMany::one(query_embedding), 3);
    tracing::info!("Searching for: '{}'", query);

    let results = vector_store.top_n_as::<Word>(req.clone()).await?;

    tracing::info!("Top 3 similar definitions:");
    for (i, (score, id, word)) in results.iter().enumerate() {
        tracing::info!(
            "{}. Score: {:.4}, ID: {}, Definition: {}",
            i + 1,
            score,
            id,
            word.definition
        );
    }

    // Test ID-only search
    tracing::info!("Searching for IDs only...");
    let id_results = vector_store.top_n_ids(req).await?;

    tracing::info!("Top 2 similar document IDs:");
    for (i, (score, id)) in id_results.iter().enumerate() {
        tracing::info!("{}. Score: {:.4}, ID: {}", i + 1, score, id);
    }

    // Test with different query
    let database_query = "distributed database system";
    tracing::info!("Searching for: '{}'", database_query);
    let query_embedding =
        openai::functions::embed(&embed_cfg, &rt, vec![database_query.to_string()])
            .await?
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("no embedding returned for the query"))?;
    let req = VectorSearchRequest::new(OneOrMany::one(query_embedding), 2);

    let db_results = vector_store.top_n_as::<Word>(req).await?;

    tracing::info!("Top 2 similar definitions:");
    for (i, (score, id, word)) in db_results.iter().enumerate() {
        tracing::info!(
            "{}. Score: {:.4}, ID: {}, Definition: {}",
            i + 1,
            score,
            id,
            word.definition
        );
    }

    tracing::info!("✅ ScyllaDB vector search example completed successfully!");

    Ok(())
}
