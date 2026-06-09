// Redis vector store example demonstrating:
// - Basic vector similarity search
// - Metadata field extraction and filtered search
// - Document round-tripping
//
// Prerequisites:
//   export OPENAI_API_KEY=<your-key>
//   docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
//   redis-cli FT.CREATE products_idx ON HASH PREFIX 1 "prod:" SCHEMA \
//     document TEXT embedded_text TEXT embedding VECTOR FLAT 6 \
//     TYPE FLOAT32 DIM 1536 DISTANCE_METRIC COSINE \
//     category TAG price NUMERIC
//
// Run:
//   cargo run --example vector_search_redis --features rig-core/derive

use rig_core::vector_store::request::SearchFilter;
use rig_core::{
    Embed,
    client::{EmbeddingsClient, ProviderClient},
    embeddings::EmbeddingsBuilder,
    providers::openai,
    vector_store::{InsertDocuments, VectorStoreIndex, request::VectorSearchRequest},
};
use rig_redis::{Filter, RedisVectorStore, filter::RedisValue};
use serde::{Deserialize, Serialize};

#[derive(Embed, Serialize, Deserialize, Clone, Debug)]
struct Product {
    name: String,
    category: String,
    price: f64,
    #[embed]
    description: String,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let openai_client = openai::Client::from_env()?;
    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_3_SMALL);

    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
    let redis_client = redis::Client::open(redis_url)?;

    // Key prefix must match the index PREFIX configuration.
    // Metadata fields must match the index SCHEMA (category TAG, price NUMERIC).
    let vector_store = RedisVectorStore::new(
        model.clone(),
        redis_client,
        "products_idx".to_string(),
        "embedding".to_string(),
    )
    .await?
    .with_key_prefix("prod:".to_string())
    .with_metadata_fields(vec!["category".to_string(), "price".to_string()]);

    // --- Insert documents ---
    let products = vec![
        Product {
            name: "Gaming Laptop".to_string(),
            category: "Electronics".to_string(),
            price: 1500.0,
            description: "High-performance gaming laptop with RTX 4080 GPU and 32GB RAM"
                .to_string(),
        },
        Product {
            name: "Wool Sweater".to_string(),
            category: "Clothing".to_string(),
            price: 75.0,
            description: "Warm merino wool sweater, perfect for cold winter days".to_string(),
        },
        Product {
            name: "Mechanical Keyboard".to_string(),
            category: "Electronics".to_string(),
            price: 45.0,
            description: "Compact 65% mechanical keyboard with Cherry MX switches".to_string(),
        },
        Product {
            name: "Running Shoes".to_string(),
            category: "Clothing".to_string(),
            price: 120.0,
            description: "Lightweight running shoes with responsive cushioning for road running"
                .to_string(),
        },
    ];

    let documents = EmbeddingsBuilder::new(model.clone())
        .documents(products)?
        .build()
        .await?;

    vector_store.insert_documents(documents).await?;
    println!("Inserted 4 products\n");

    // --- Basic vector search (no filter) ---
    let query = "computer peripherals";
    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(2)
        .build();

    let results = vector_store.top_n::<Product>(req).await?;

    println!("=== Basic search: \"{query}\" (top 2) ===");
    for (score, id, doc) in &results {
        println!(
            "  [{score:.4}] {id} -> {} (${}, {})",
            doc.name, doc.price, doc.category
        );
    }
    println!();

    // --- Filtered search: only Electronics ---
    let query = "something for gaming";
    let filter = Filter::eq("category", RedisValue::String("Electronics".to_string()));
    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(4)
        .filter(filter)
        .build();

    let results = vector_store.top_n::<Product>(req).await?;

    println!("=== Filtered search: \"{query}\" (category=Electronics) ===");
    for (score, id, doc) in &results {
        println!(
            "  [{score:.4}] {id} -> {} (${}, {})",
            doc.name, doc.price, doc.category
        );
    }
    println!();

    // --- Filtered search: price under $100 ---
    let query = "affordable gear";
    let filter = Filter::lt("price", RedisValue::Number(100.0));
    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(4)
        .filter(filter)
        .build();

    let results = vector_store.top_n::<Product>(req).await?;

    println!("=== Filtered search: \"{query}\" (price < 100) ===");
    for (score, id, doc) in &results {
        println!(
            "  [{score:.4}] {id} -> {} (${}, {})",
            doc.name, doc.price, doc.category
        );
    }
    println!();

    // --- Combined filter: Electronics AND price > 100 ---
    let query = "premium tech";
    let filter = Filter::eq("category", RedisValue::String("Electronics".to_string()))
        .and(Filter::gt("price", RedisValue::Number(100.0)));
    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(4)
        .filter(filter)
        .build();

    let results = vector_store.top_n::<Product>(req).await?;

    println!("=== Combined filter: \"{query}\" (Electronics AND price > 100) ===");
    for (score, id, doc) in &results {
        println!(
            "  [{score:.4}] {id} -> {} (${}, {})",
            doc.name, doc.price, doc.category
        );
    }

    Ok(())
}
