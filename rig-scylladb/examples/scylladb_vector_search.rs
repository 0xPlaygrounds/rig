use rig::{
    Embed,
    client::EmbeddingsClient,
    embeddings::EmbeddingsBuilder,
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::VectorStoreIndex,
};
use rig_scylladb::{ScyllaDbVectorStore, create_session};
use serde::{Deserialize, Serialize};
use std::env;

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
    let session = create_session("127.0.0.1:9042")
        .await
        .expect("Failed to create ScyllaDB session");

    // Create OpenAI client and embedding model
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);
    let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    // Create ScyllaDB vector store
    let vector_store = ScyllaDbVectorStore::new(
        model.clone(),
        session,
        "word_definitions", // keyspace
        "words",            // table
        1536,               // dimensions for text-embedding-ada-002
    )
    .await
    .expect("Failed to create ScyllaDB vector store");

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

    // Generate embeddings for the documents
    let embeddings = EmbeddingsBuilder::new(model.clone())
        .documents(words.clone())?
        .build()
        .await?;

    tracing::info!(
        "Inserting {} word definitions into ScyllaDB...",
        words.len()
    );

    // Insert documents with their embeddings
    let documents_with_embeddings = embeddings
        .iter()
        .map(|(document, embedding)| (document.clone(), embedding.clone()))
        .collect::<Vec<_>>();

    vector_store
        .insert_documents(documents_with_embeddings)
        .await
        .expect("Failed to insert documents");

    tracing::info!("Documents inserted successfully!");

    // Test similarity search
    let query = "What is Rust programming language?";
    tracing::info!("Searching for: '{}'", query);

    let results = vector_store
        .top_n::<Word>(query, 3)
        .await
        .expect("Failed to search vectors");

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
    let id_results = vector_store
        .top_n_ids(query, 2)
        .await
        .expect("Failed to search vector IDs");

    tracing::info!("Top 2 similar document IDs:");
    for (i, (score, id)) in id_results.iter().enumerate() {
        tracing::info!("{}. Score: {:.4}, ID: {}", i + 1, score, id);
    }

    // Test with different query
    let database_query = "distributed database system";
    tracing::info!("Searching for: '{}'", database_query);

    let db_results = vector_store
        .top_n::<Word>(database_query, 2)
        .await
        .expect("Failed to search vectors");

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
