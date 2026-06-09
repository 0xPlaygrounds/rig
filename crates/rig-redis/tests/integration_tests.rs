//! Integration tests for rig-redis vector store.
//!
//! These tests require a Redis Stack instance (with RediSearch module).
//! They use testcontainers to spin up an isolated Redis Stack container by default.
//!
//! Set `REDIS_URL` environment variable to use an external Redis instance instead.
//!
//! Run with: `cargo test --test integration_tests -- --ignored`
#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used
)]

use rig_core::client::EmbeddingsClient;
use rig_core::vector_store::request::SearchFilter;
use rig_core::{
    Embed,
    embeddings::EmbeddingsBuilder,
    providers::openai,
    vector_store::{InsertDocuments, VectorStoreIndex, request::VectorSearchRequest},
};
use rig_redis::RedisVectorStore;
use rig_redis::filter::Filter;
use rig_redis::filter::RedisValue;
use serde_json::json;
use testcontainers::{
    GenericImage,
    core::{IntoContainerPort, WaitFor},
    runners::AsyncRunner,
};
use tokio::time::{Duration, sleep};

const REDIS_PORT: u16 = 6379;
const VECTOR_FIELD: &str = "embedding";

#[derive(Embed, Clone, serde::Deserialize, serde::Serialize, Debug, PartialEq)]
struct Word {
    id: String,
    #[embed]
    definition: String,
}

/// A document type with metadata fields for filtering tests.
#[derive(Embed, Clone, serde::Deserialize, serde::Serialize, Debug, PartialEq)]
struct Product {
    name: String,
    category: String,
    price: f64,
    in_stock: bool,
    #[embed]
    description: String,
}

/// Get Redis connection info.
///
/// If `REDIS_URL` is set, uses that external instance.
/// Otherwise, starts an isolated testcontainers Redis Stack instance.
async fn get_redis_connection() -> (
    String,
    u16,
    Option<testcontainers::ContainerAsync<GenericImage>>,
) {
    if let Ok(url) = std::env::var("REDIS_URL") {
        // Parse host:port from URL like redis://host:port
        let url = url.strip_prefix("redis://").unwrap_or(&url);
        let parts: Vec<&str> = url.split(':').collect();
        let host = parts.first().unwrap_or(&"127.0.0.1").to_string();
        let port: u16 = parts
            .get(1)
            .and_then(|p| p.parse().ok())
            .unwrap_or(REDIS_PORT);
        println!("Using external Redis instance at {host}:{port}");
        (host, port, None)
    } else {
        println!("Starting new Redis Stack container via testcontainers");
        let container = GenericImage::new("redis/redis-stack", "latest")
            .with_exposed_port(REDIS_PORT.tcp())
            .with_wait_for(WaitFor::Duration {
                length: std::time::Duration::from_secs(3),
            })
            .start()
            .await
            .expect("Failed to start Redis Stack container. Is Docker/Podman running?");

        let port = container.get_host_port_ipv4(REDIS_PORT).await.unwrap();
        let host = container.get_host().await.unwrap().to_string();

        (host, port, Some(container))
    }
}

/// Verifies that the Redis instance has RediSearch module loaded.
async fn verify_redisearch(client: &redis::Client) -> bool {
    let mut con = match client.get_multiplexed_async_connection().await {
        Ok(con) => con,
        Err(_) => return false,
    };

    // FT._LIST returns an array of index names. If the command errors,
    // RediSearch is not available.
    let result: Result<redis::Value, _> = redis::cmd("FT._LIST").query_async(&mut con).await;
    result.is_ok()
}

async fn setup_redis_index(
    client: &redis::Client,
    index_name: &str,
    dimensions: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut con = client.get_multiplexed_async_connection().await?;

    // Drop existing index if it exists (DD flag deletes associated documents)
    let _: Result<String, _> = redis::cmd("FT.DROPINDEX")
        .arg(index_name)
        .arg("DD")
        .query_async(&mut con)
        .await;

    // Create vector index with PREFIX to associate documents with this index
    let prefix = format!("{index_name}:");
    let _: String = redis::cmd("FT.CREATE")
        .arg(index_name)
        .arg("ON")
        .arg("HASH")
        .arg("PREFIX")
        .arg(1)
        .arg(&prefix)
        .arg("SCHEMA")
        .arg("document")
        .arg("TEXT")
        .arg("embedded_text")
        .arg("TEXT")
        .arg(VECTOR_FIELD)
        .arg("VECTOR")
        .arg("FLAT")
        .arg(6)
        .arg("TYPE")
        .arg("FLOAT32")
        .arg("DIM")
        .arg(dimensions)
        .arg("DISTANCE_METRIC")
        .arg("COSINE")
        .query_async(&mut con)
        .await?;

    // Wait for index to be ready
    sleep(Duration::from_millis(1000)).await;

    Ok(())
}

/// Creates a RediSearch index with additional metadata fields for filtering.
async fn setup_redis_index_with_metadata(
    client: &redis::Client,
    index_name: &str,
    dimensions: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut con = client.get_multiplexed_async_connection().await?;

    let _: Result<String, _> = redis::cmd("FT.DROPINDEX")
        .arg(index_name)
        .arg("DD")
        .query_async(&mut con)
        .await;

    let prefix = format!("{index_name}:");
    let _: String = redis::cmd("FT.CREATE")
        .arg(index_name)
        .arg("ON")
        .arg("HASH")
        .arg("PREFIX")
        .arg(1)
        .arg(&prefix)
        .arg("SCHEMA")
        .arg("document")
        .arg("TEXT")
        .arg("embedded_text")
        .arg("TEXT")
        .arg(VECTOR_FIELD)
        .arg("VECTOR")
        .arg("FLAT")
        .arg(6)
        .arg("TYPE")
        .arg("FLOAT32")
        .arg("DIM")
        .arg(dimensions)
        .arg("DISTANCE_METRIC")
        .arg("COSINE")
        // Metadata fields
        .arg("category")
        .arg("TAG")
        .arg("price")
        .arg("NUMERIC")
        .arg("in_stock")
        .arg("TAG")
        .query_async(&mut con)
        .await?;

    sleep(Duration::from_millis(1000)).await;

    Ok(())
}

async fn cleanup_redis_index(
    client: &redis::Client,
    index_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut con = client.get_multiplexed_async_connection().await?;

    // Drop index and associated documents
    let _: Result<String, _> = redis::cmd("FT.DROPINDEX")
        .arg(index_name)
        .arg("DD")
        .query_async(&mut con)
        .await;

    // Wait for cleanup to complete
    sleep(Duration::from_millis(100)).await;

    Ok(())
}

#[tokio::test]
#[ignore = "requires Docker/Podman for testcontainers"]
async fn test_vector_search_basic() {
    let (host, port, _container) = get_redis_connection().await;
    let index_name = "test_vector_search_basic";

    let redis_url = format!("redis://{host}:{port}");
    let redis_client = redis::Client::open(redis_url).unwrap();

    assert!(
        verify_redisearch(&redis_client).await,
        "RediSearch module not available on this Redis instance"
    );

    let server = httpmock::MockServer::start();

    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .header("Authorization", "Bearer TEST")
            .json_body(json!({
                "input": [
                    "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets",
                    "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
                    "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans."
                ],
                "model": "text-embedding-ada-002",
            }));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": (0..1536).map(|i| if i < 512 { 1.0 } else { 0.0 }).collect::<Vec<f64>>(), "index": 0},
                    {"object": "embedding", "embedding": (0..1536).map(|i| if (512..1024).contains(&i) { 1.0 } else { 0.0 }).collect::<Vec<f64>>(), "index": 1},
                    {"object": "embedding", "embedding": (0..1536).map(|i| if i >= 1024 { 1.0 } else { 0.0 }).collect::<Vec<f64>>(), "index": 2}
                ],
                "model": "text-embedding-ada-002",
                "usage": {"prompt_tokens": 8, "total_tokens": 8}
            }));
    });

    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .header("Authorization", "Bearer TEST")
            .json_body(json!({
                "input": ["What is a linglingdong?"],
                "model": "text-embedding-ada-002",
            }));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": (0..1536).map(|i| if i >= 1024 { 1.0 } else { 0.0 }).collect::<Vec<f64>>(), "index": 0}
                ],
                "model": "text-embedding-ada-002",
                "usage": {"prompt_tokens": 8, "total_tokens": 8}
            }));
    });

    let openai_client = openai::Client::builder()
        .api_key("TEST")
        .base_url(server.base_url())
        .build()
        .unwrap();

    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    setup_redis_index(&redis_client, index_name, 1536)
        .await
        .unwrap();

    let vector_store = RedisVectorStore::new(
        model.clone(),
        redis_client.clone(),
        index_name.to_string(),
        VECTOR_FIELD.to_string(),
    )
    .await
    .unwrap()
    .with_key_prefix(format!("{}:", index_name));

    let words = vec![
        Word {
            id: "doc0".to_string(),
            definition: "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets".to_string(),
        },
        Word {
            id: "doc1".to_string(),
            definition: "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
        },
        Word {
            id: "doc2".to_string(),
            definition: "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
        }
    ];

    let documents = EmbeddingsBuilder::new(model.clone())
        .documents(words)
        .unwrap()
        .build()
        .await
        .unwrap();

    vector_store.insert_documents(documents).await.unwrap();

    sleep(Duration::from_millis(500)).await;

    let req = VectorSearchRequest::builder()
        .query("What is a linglingdong?")
        .samples(1)
        .build();

    let results = vector_store.top_n::<Word>(req).await.unwrap();

    assert_eq!(results.len(), 1);
    let (score, _, doc) = &results[0];
    assert!(score.is_finite());
    assert!(doc.definition.contains("linglingdong"));

    cleanup_redis_index(&redis_client, index_name)
        .await
        .unwrap();
}

#[tokio::test]
#[ignore = "requires Docker/Podman for testcontainers"]
async fn test_top_n_ids() {
    let (host, port, _container) = get_redis_connection().await;
    let index_name = "test_top_n_ids";

    let redis_url = format!("redis://{host}:{port}");
    let redis_client = redis::Client::open(redis_url).unwrap();

    assert!(
        verify_redisearch(&redis_client).await,
        "RediSearch module not available on this Redis instance"
    );

    let server = httpmock::MockServer::start();

    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .json_body(json!({
                "input": [
                    "First test document",
                    "Second test document"
                ],
                "model": "text-embedding-ada-002",
            }));
        then.status(200).json_body(json!({
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": vec![0.5; 1536], "index": 0},
                {"object": "embedding", "embedding": vec![0.6; 1536], "index": 1}
            ],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 4, "total_tokens": 4}
        }));
    });

    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .json_body(json!({
                "input": ["test query"],
                "model": "text-embedding-ada-002",
            }));
        then.status(200).json_body(json!({
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": vec![0.55; 1536], "index": 0}
            ],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
        }));
    });

    let openai_client = openai::Client::builder()
        .api_key("TEST")
        .base_url(server.base_url())
        .build()
        .unwrap();

    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    setup_redis_index(&redis_client, index_name, 1536)
        .await
        .unwrap();

    let vector_store = RedisVectorStore::new(
        model.clone(),
        redis_client.clone(),
        index_name.to_string(),
        VECTOR_FIELD.to_string(),
    )
    .await
    .unwrap()
    .with_key_prefix(format!("{}:", index_name));

    let words = vec![
        Word {
            id: "test1".to_string(),
            definition: "First test document".to_string(),
        },
        Word {
            id: "test2".to_string(),
            definition: "Second test document".to_string(),
        },
    ];

    let documents = EmbeddingsBuilder::new(model.clone())
        .documents(words)
        .unwrap()
        .build()
        .await
        .unwrap();

    vector_store.insert_documents(documents).await.unwrap();

    sleep(Duration::from_millis(500)).await;

    let req = VectorSearchRequest::builder()
        .query("test query")
        .samples(2)
        .build();

    let results = vector_store.top_n_ids(req).await.unwrap();

    assert_eq!(results.len(), 2);
    assert!(results[0].0.is_finite());
    assert!(!results[0].1.is_empty());

    cleanup_redis_index(&redis_client, index_name)
        .await
        .unwrap();
}

#[tokio::test]
#[ignore = "requires Docker/Podman for testcontainers"]
async fn test_threshold_filtering() {
    let (host, port, _container) = get_redis_connection().await;
    let index_name = "test_threshold_filtering";

    let redis_url = format!("redis://{host}:{port}");
    let redis_client = redis::Client::open(redis_url).unwrap();

    assert!(
        verify_redisearch(&redis_client).await,
        "RediSearch module not available on this Redis instance"
    );

    let server = httpmock::MockServer::start();

    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .json_body(json!({
                "input": [
                    "Document with low similarity",
                    "Document with high similarity"
                ],
                "model": "text-embedding-ada-002",
            }));
        then.status(200).json_body(json!({
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": vec![0.1; 1536], "index": 0},
                {"object": "embedding", "embedding": vec![0.9; 1536], "index": 1}
            ],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 4, "total_tokens": 4}
        }));
    });

    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .json_body(json!({
                "input": ["test query"],
                "model": "text-embedding-ada-002",
            }));
        then.status(200).json_body(json!({
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": vec![0.85; 1536], "index": 0}
            ],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
        }));
    });

    let openai_client = openai::Client::builder()
        .api_key("TEST")
        .base_url(server.base_url())
        .build()
        .unwrap();

    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    setup_redis_index(&redis_client, index_name, 1536)
        .await
        .unwrap();

    let vector_store = RedisVectorStore::new(
        model.clone(),
        redis_client.clone(),
        index_name.to_string(),
        VECTOR_FIELD.to_string(),
    )
    .await
    .unwrap()
    .with_key_prefix(format!("{}:", index_name));

    let words = vec![
        Word {
            id: "low_score".to_string(),
            definition: "Document with low similarity".to_string(),
        },
        Word {
            id: "high_score".to_string(),
            definition: "Document with high similarity".to_string(),
        },
    ];

    let documents = EmbeddingsBuilder::new(model.clone())
        .documents(words)
        .unwrap()
        .build()
        .await
        .unwrap();

    vector_store.insert_documents(documents).await.unwrap();

    sleep(Duration::from_millis(500)).await;

    let req = VectorSearchRequest::builder()
        .query("test query")
        .samples(10)
        .threshold(0.5)
        .build();

    let results = vector_store.top_n::<Word>(req).await.unwrap();

    for (score, _, _) in &results {
        assert!(score >= &0.5, "All results should meet threshold");
    }

    cleanup_redis_index(&redis_client, index_name)
        .await
        .unwrap();
}

#[tokio::test]
#[ignore = "requires Docker/Podman for testcontainers"]
async fn test_insert_multiple_embeddings() {
    let (host, port, _container) = get_redis_connection().await;
    let index_name = "test_insert_multiple_embeddings";

    let redis_url = format!("redis://{host}:{port}");
    let redis_client = redis::Client::open(redis_url).unwrap();

    assert!(
        verify_redisearch(&redis_client).await,
        "RediSearch module not available on this Redis instance"
    );

    let server = httpmock::MockServer::start();

    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .json_body(json!({
                "input": [
                    "First batch document",
                    "Second batch document",
                    "Third batch document"
                ],
                "model": "text-embedding-ada-002",
            }));
        then.status(200).json_body(json!({
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": vec![0.1; 1536], "index": 0},
                {"object": "embedding", "embedding": vec![0.2; 1536], "index": 1},
                {"object": "embedding", "embedding": vec![0.3; 1536], "index": 2}
            ],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 6, "total_tokens": 6}
        }));
    });

    let openai_client = openai::Client::builder()
        .api_key("TEST")
        .base_url(server.base_url())
        .build()
        .unwrap();

    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    setup_redis_index(&redis_client, index_name, 1536)
        .await
        .unwrap();

    let vector_store = RedisVectorStore::new(
        model.clone(),
        redis_client.clone(),
        index_name.to_string(),
        VECTOR_FIELD.to_string(),
    )
    .await
    .unwrap()
    .with_key_prefix(format!("{}:", index_name));

    let words = vec![
        Word {
            id: "batch1".to_string(),
            definition: "First batch document".to_string(),
        },
        Word {
            id: "batch2".to_string(),
            definition: "Second batch document".to_string(),
        },
        Word {
            id: "batch3".to_string(),
            definition: "Third batch document".to_string(),
        },
    ];

    let documents = EmbeddingsBuilder::new(model.clone())
        .documents(words)
        .unwrap()
        .build()
        .await
        .unwrap();

    vector_store.insert_documents(documents).await.unwrap();

    sleep(Duration::from_millis(500)).await;

    // Verify documents were inserted by searching
    let mut con = redis_client
        .get_multiplexed_async_connection()
        .await
        .unwrap();
    let keys: Vec<String> = redis::cmd("KEYS")
        .arg(format!("{index_name}:*"))
        .query_async(&mut con)
        .await
        .unwrap();

    assert!(keys.len() >= 3, "Should have inserted at least 3 documents");

    cleanup_redis_index(&redis_client, index_name)
        .await
        .unwrap();
}

#[tokio::test]
#[ignore = "requires Docker/Podman for testcontainers"]
async fn test_empty_results() {
    let (host, port, _container) = get_redis_connection().await;
    let index_name = "test_empty_results";

    let redis_url = format!("redis://{host}:{port}");
    let redis_client = redis::Client::open(redis_url).unwrap();

    assert!(
        verify_redisearch(&redis_client).await,
        "RediSearch module not available on this Redis instance"
    );

    let server = httpmock::MockServer::start();

    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .json_body(json!({
                "input": ["query with no results"],
                "model": "text-embedding-ada-002",
            }));
        then.status(200).json_body(json!({
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": vec![0.5; 1536], "index": 0}
            ],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
        }));
    });

    let openai_client = openai::Client::builder()
        .api_key("TEST")
        .base_url(server.base_url())
        .build()
        .unwrap();

    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    setup_redis_index(&redis_client, index_name, 1536)
        .await
        .unwrap();

    let vector_store = RedisVectorStore::new(
        model.clone(),
        redis_client.clone(),
        index_name.to_string(),
        VECTOR_FIELD.to_string(),
    )
    .await
    .unwrap()
    .with_key_prefix(format!("{}:", index_name));

    let req = VectorSearchRequest::builder()
        .query("query with no results")
        .samples(5)
        .build();

    let results = vector_store.top_n::<Word>(req).await.unwrap();

    assert_eq!(results.len(), 0);

    cleanup_redis_index(&redis_client, index_name)
        .await
        .unwrap();
}

// =============================================================================
// Metadata filtering tests
// =============================================================================

/// Helper to create a mock embedding server that returns distinct vectors for products.
fn setup_product_embedding_mocks(server: &httpmock::MockServer) {
    // Mock for inserting 3 products
    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .json_body(json!({
                "input": [
                    "A high-end gaming laptop with RTX graphics",
                    "A cozy wool sweater for winter",
                    "A budget-friendly mechanical keyboard"
                ],
                "model": "text-embedding-ada-002",
            }));
        then.status(200).json_body(json!({
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": (0..1536).map(|i| if i < 512 { 1.0 } else { 0.0 }).collect::<Vec<f64>>(), "index": 0},
                {"object": "embedding", "embedding": (0..1536).map(|i| if (512..1024).contains(&i) { 1.0 } else { 0.0 }).collect::<Vec<f64>>(), "index": 1},
                {"object": "embedding", "embedding": (0..1536).map(|i| if i >= 1024 { 1.0 } else { 0.0 }).collect::<Vec<f64>>(), "index": 2}
            ],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 6, "total_tokens": 6}
        }));
    });

    // Mock for search query "electronics"
    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .json_body(json!({
                "input": ["electronics"],
                "model": "text-embedding-ada-002",
            }));
        then.status(200).json_body(json!({
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": (0..1536).map(|i| if i < 512 { 0.9 } else { 0.1 }).collect::<Vec<f64>>(), "index": 0}
            ],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 1, "total_tokens": 1}
        }));
    });

    // Mock for search query "find products"
    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .json_body(json!({
                "input": ["find products"],
                "model": "text-embedding-ada-002",
            }));
        then.status(200).json_body(json!({
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": vec![0.5; 1536], "index": 0}
            ],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
        }));
    });
}

/// Creates a vector store with metadata fields and inserts test products.
async fn setup_product_store(
    redis_client: &redis::Client,
    index_name: &str,
    server: &httpmock::MockServer,
) -> RedisVectorStore<rig_core::providers::openai::EmbeddingModel> {
    let openai_client = openai::Client::builder()
        .api_key("TEST")
        .base_url(server.base_url())
        .build()
        .unwrap();

    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    setup_redis_index_with_metadata(redis_client, index_name, 1536)
        .await
        .unwrap();

    let vector_store = RedisVectorStore::new(
        model.clone(),
        redis_client.clone(),
        index_name.to_string(),
        VECTOR_FIELD.to_string(),
    )
    .await
    .unwrap()
    .with_key_prefix(format!("{}:", index_name))
    .with_metadata_fields(vec![
        "category".to_string(),
        "price".to_string(),
        "in_stock".to_string(),
    ]);

    let products = vec![
        Product {
            name: "Gaming Laptop".to_string(),
            category: "Electronics".to_string(),
            price: 1500.0,
            in_stock: true,
            description: "A high-end gaming laptop with RTX graphics".to_string(),
        },
        Product {
            name: "Wool Sweater".to_string(),
            category: "Clothing".to_string(),
            price: 75.0,
            in_stock: true,
            description: "A cozy wool sweater for winter".to_string(),
        },
        Product {
            name: "Mechanical Keyboard".to_string(),
            category: "Electronics".to_string(),
            price: 45.0,
            in_stock: false,
            description: "A budget-friendly mechanical keyboard".to_string(),
        },
    ];

    let documents = EmbeddingsBuilder::new(model.clone())
        .documents(products)
        .unwrap()
        .build()
        .await
        .unwrap();

    vector_store.insert_documents(documents).await.unwrap();

    sleep(Duration::from_millis(500)).await;

    vector_store
}

#[tokio::test]
#[ignore = "requires Docker/Podman for testcontainers"]
async fn test_metadata_filter_by_tag() {
    let (host, port, _container) = get_redis_connection().await;
    let index_name = "test_metadata_filter_by_tag";

    let redis_url = format!("redis://{host}:{port}");
    let redis_client = redis::Client::open(redis_url).unwrap();

    assert!(
        verify_redisearch(&redis_client).await,
        "RediSearch module not available on this Redis instance"
    );

    let server = httpmock::MockServer::start();
    setup_product_embedding_mocks(&server);

    let vector_store = setup_product_store(&redis_client, index_name, &server).await;

    // Filter by category = "Electronics" — should return 2 results
    let filter = Filter::eq("category", RedisValue::String("Electronics".to_string()));
    let req = VectorSearchRequest::builder()
        .query("find products")
        .samples(10)
        .filter(filter)
        .build();

    let results = vector_store.top_n::<Product>(req).await.unwrap();

    assert_eq!(results.len(), 2, "Should find 2 Electronics products");
    for (_, _, product) in &results {
        assert_eq!(product.category, "Electronics");
    }

    cleanup_redis_index(&redis_client, index_name)
        .await
        .unwrap();
}

#[tokio::test]
#[ignore = "requires Docker/Podman for testcontainers"]
async fn test_metadata_filter_by_numeric_range() {
    let (host, port, _container) = get_redis_connection().await;
    let index_name = "test_metadata_filter_numeric";

    let redis_url = format!("redis://{host}:{port}");
    let redis_client = redis::Client::open(redis_url).unwrap();

    assert!(
        verify_redisearch(&redis_client).await,
        "RediSearch module not available on this Redis instance"
    );

    let server = httpmock::MockServer::start();
    setup_product_embedding_mocks(&server);

    let vector_store = setup_product_store(&redis_client, index_name, &server).await;

    // Filter by price < 100 — should return Wool Sweater (75) and Keyboard (45)
    let filter = Filter::lt("price", RedisValue::Number(100.0));
    let req = VectorSearchRequest::builder()
        .query("find products")
        .samples(10)
        .filter(filter)
        .build();

    let results = vector_store.top_n::<Product>(req).await.unwrap();

    assert_eq!(results.len(), 2, "Should find 2 products under $100");
    for (_, _, product) in &results {
        assert!(product.price < 100.0, "All products should be under $100");
    }

    cleanup_redis_index(&redis_client, index_name)
        .await
        .unwrap();
}

#[tokio::test]
#[ignore = "requires Docker/Podman for testcontainers"]
async fn test_metadata_filter_by_boolean() {
    let (host, port, _container) = get_redis_connection().await;
    let index_name = "test_metadata_filter_boolean";

    let redis_url = format!("redis://{host}:{port}");
    let redis_client = redis::Client::open(redis_url).unwrap();

    assert!(
        verify_redisearch(&redis_client).await,
        "RediSearch module not available on this Redis instance"
    );

    let server = httpmock::MockServer::start();
    setup_product_embedding_mocks(&server);

    let vector_store = setup_product_store(&redis_client, index_name, &server).await;

    // Filter by in_stock = true — should return Gaming Laptop and Wool Sweater
    let filter = Filter::eq("in_stock", RedisValue::Bool(true));
    let req = VectorSearchRequest::builder()
        .query("find products")
        .samples(10)
        .filter(filter)
        .build();

    let results = vector_store.top_n::<Product>(req).await.unwrap();

    assert_eq!(results.len(), 2, "Should find 2 in-stock products");
    for (_, _, product) in &results {
        assert!(product.in_stock, "All products should be in stock");
    }

    cleanup_redis_index(&redis_client, index_name)
        .await
        .unwrap();
}

#[tokio::test]
#[ignore = "requires Docker/Podman for testcontainers"]
async fn test_metadata_filter_combined_and() {
    let (host, port, _container) = get_redis_connection().await;
    let index_name = "test_metadata_filter_combined";

    let redis_url = format!("redis://{host}:{port}");
    let redis_client = redis::Client::open(redis_url).unwrap();

    assert!(
        verify_redisearch(&redis_client).await,
        "RediSearch module not available on this Redis instance"
    );

    let server = httpmock::MockServer::start();
    setup_product_embedding_mocks(&server);

    let vector_store = setup_product_store(&redis_client, index_name, &server).await;

    // Filter: category = "Electronics" AND in_stock = true
    // Should return only Gaming Laptop (Keyboard is out of stock)
    let filter = Filter::eq("category", RedisValue::String("Electronics".to_string()))
        .and(Filter::eq("in_stock", RedisValue::Bool(true)));

    let req = VectorSearchRequest::builder()
        .query("find products")
        .samples(10)
        .filter(filter)
        .build();

    let results = vector_store.top_n::<Product>(req).await.unwrap();

    assert_eq!(
        results.len(),
        1,
        "Should find 1 in-stock Electronics product"
    );
    assert_eq!(results[0].2.name, "Gaming Laptop");

    cleanup_redis_index(&redis_client, index_name)
        .await
        .unwrap();
}

#[tokio::test]
#[ignore = "requires Docker/Podman for testcontainers"]
async fn test_metadata_filter_combined_or() {
    let (host, port, _container) = get_redis_connection().await;
    let index_name = "test_metadata_filter_or";

    let redis_url = format!("redis://{host}:{port}");
    let redis_client = redis::Client::open(redis_url).unwrap();

    assert!(
        verify_redisearch(&redis_client).await,
        "RediSearch module not available on this Redis instance"
    );

    let server = httpmock::MockServer::start();
    setup_product_embedding_mocks(&server);

    let vector_store = setup_product_store(&redis_client, index_name, &server).await;

    // Filter: category = "Electronics" OR category = "Clothing"
    // Should return all 3 products
    let filter = Filter::eq("category", RedisValue::String("Electronics".to_string())).or(
        Filter::eq("category", RedisValue::String("Clothing".to_string())),
    );

    let req = VectorSearchRequest::builder()
        .query("find products")
        .samples(10)
        .filter(filter)
        .build();

    let results = vector_store.top_n::<Product>(req).await.unwrap();

    assert_eq!(results.len(), 3, "Should find all 3 products");

    cleanup_redis_index(&redis_client, index_name)
        .await
        .unwrap();
}

#[tokio::test]
#[ignore = "requires Docker/Podman for testcontainers"]
async fn test_metadata_filter_no_match() {
    let (host, port, _container) = get_redis_connection().await;
    let index_name = "test_metadata_filter_no_match";

    let redis_url = format!("redis://{host}:{port}");
    let redis_client = redis::Client::open(redis_url).unwrap();

    assert!(
        verify_redisearch(&redis_client).await,
        "RediSearch module not available on this Redis instance"
    );

    let server = httpmock::MockServer::start();
    setup_product_embedding_mocks(&server);

    let vector_store = setup_product_store(&redis_client, index_name, &server).await;

    // Filter by a non-existent category — should return 0 results
    let filter = Filter::eq("category", RedisValue::String("Books".to_string()));
    let req = VectorSearchRequest::builder()
        .query("find products")
        .samples(10)
        .filter(filter)
        .build();

    let results = vector_store.top_n::<Product>(req).await.unwrap();

    assert_eq!(results.len(), 0, "Should find no products");

    cleanup_redis_index(&redis_client, index_name)
        .await
        .unwrap();
}

#[tokio::test]
#[ignore = "requires Docker/Podman for testcontainers"]
async fn test_metadata_filter_numeric_gt() {
    let (host, port, _container) = get_redis_connection().await;
    let index_name = "test_metadata_filter_gt";

    let redis_url = format!("redis://{host}:{port}");
    let redis_client = redis::Client::open(redis_url).unwrap();

    assert!(
        verify_redisearch(&redis_client).await,
        "RediSearch module not available on this Redis instance"
    );

    let server = httpmock::MockServer::start();
    setup_product_embedding_mocks(&server);

    let vector_store = setup_product_store(&redis_client, index_name, &server).await;

    // Filter: price > 100 — should only return Gaming Laptop (1500)
    let filter = Filter::gt("price", RedisValue::Number(100.0));
    let req = VectorSearchRequest::builder()
        .query("find products")
        .samples(10)
        .filter(filter)
        .build();

    let results = vector_store.top_n::<Product>(req).await.unwrap();

    assert_eq!(results.len(), 1, "Should find 1 product over $100");
    assert_eq!(results[0].2.name, "Gaming Laptop");

    cleanup_redis_index(&redis_client, index_name)
        .await
        .unwrap();
}

#[tokio::test]
#[ignore = "requires Docker/Podman for testcontainers"]
async fn test_metadata_fields_missing_from_document() {
    let (host, port, _container) = get_redis_connection().await;
    let index_name = "test_metadata_missing_field";

    let redis_url = format!("redis://{host}:{port}");
    let redis_client = redis::Client::open(redis_url).unwrap();

    assert!(
        verify_redisearch(&redis_client).await,
        "RediSearch module not available on this Redis instance"
    );

    let server = httpmock::MockServer::start();

    // Mock for Word insertion (no category/price fields in Word struct)
    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .json_body(json!({
                "input": ["A simple test document"],
                "model": "text-embedding-ada-002",
            }));
        then.status(200).json_body(json!({
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": vec![0.5; 1536], "index": 0}
            ],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
        }));
    });

    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .json_body(json!({
                "input": ["search query"],
                "model": "text-embedding-ada-002",
            }));
        then.status(200).json_body(json!({
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": vec![0.5; 1536], "index": 0}
            ],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 1, "total_tokens": 1}
        }));
    });

    let openai_client = openai::Client::builder()
        .api_key("TEST")
        .base_url(server.base_url())
        .build()
        .unwrap();

    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    setup_redis_index(&redis_client, index_name, 1536)
        .await
        .unwrap();

    // Configure metadata_fields that don't exist in the Word struct
    let vector_store = RedisVectorStore::new(
        model.clone(),
        redis_client.clone(),
        index_name.to_string(),
        VECTOR_FIELD.to_string(),
    )
    .await
    .unwrap()
    .with_key_prefix(format!("{}:", index_name))
    .with_metadata_fields(vec![
        "category".to_string(),
        "nonexistent_field".to_string(),
    ]);

    let words = vec![Word {
        id: "test1".to_string(),
        definition: "A simple test document".to_string(),
    }];

    let documents = EmbeddingsBuilder::new(model.clone())
        .documents(words)
        .unwrap()
        .build()
        .await
        .unwrap();

    // Should not panic or error — missing fields are silently skipped
    vector_store.insert_documents(documents).await.unwrap();

    sleep(Duration::from_millis(500)).await;

    // The document should still be searchable (just without metadata filters)
    let req = VectorSearchRequest::builder()
        .query("search query")
        .samples(5)
        .build();

    let results = vector_store.top_n::<Word>(req).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].2.id, "test1");

    cleanup_redis_index(&redis_client, index_name)
        .await
        .unwrap();
}

#[tokio::test]
#[ignore = "requires Docker/Podman for testcontainers"]
async fn test_metadata_no_fields_configured() {
    let (host, port, _container) = get_redis_connection().await;
    let index_name = "test_metadata_no_fields";

    let redis_url = format!("redis://{host}:{port}");
    let redis_client = redis::Client::open(redis_url).unwrap();

    assert!(
        verify_redisearch(&redis_client).await,
        "RediSearch module not available on this Redis instance"
    );

    let server = httpmock::MockServer::start();

    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .json_body(json!({
                "input": ["A simple test document"],
                "model": "text-embedding-ada-002",
            }));
        then.status(200).json_body(json!({
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": vec![0.5; 1536], "index": 0}
            ],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
        }));
    });

    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .json_body(json!({
                "input": ["search"],
                "model": "text-embedding-ada-002",
            }));
        then.status(200).json_body(json!({
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": vec![0.5; 1536], "index": 0}
            ],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 1, "total_tokens": 1}
        }));
    });

    let openai_client = openai::Client::builder()
        .api_key("TEST")
        .base_url(server.base_url())
        .build()
        .unwrap();

    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    setup_redis_index(&redis_client, index_name, 1536)
        .await
        .unwrap();

    // No metadata fields configured — backward compatible
    let vector_store = RedisVectorStore::new(
        model.clone(),
        redis_client.clone(),
        index_name.to_string(),
        VECTOR_FIELD.to_string(),
    )
    .await
    .unwrap()
    .with_key_prefix(format!("{}:", index_name));

    let words = vec![Word {
        id: "compat".to_string(),
        definition: "A simple test document".to_string(),
    }];

    let documents = EmbeddingsBuilder::new(model.clone())
        .documents(words)
        .unwrap()
        .build()
        .await
        .unwrap();

    vector_store.insert_documents(documents).await.unwrap();

    sleep(Duration::from_millis(500)).await;

    let req = VectorSearchRequest::builder()
        .query("search")
        .samples(5)
        .build();

    let results = vector_store.top_n::<Word>(req).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].2.id, "compat");

    cleanup_redis_index(&redis_client, index_name)
        .await
        .unwrap();
}

#[tokio::test]
#[ignore = "requires Docker/Podman for testcontainers"]
async fn test_metadata_filter_with_top_n_ids() {
    let (host, port, _container) = get_redis_connection().await;
    let index_name = "test_metadata_top_n_ids";

    let redis_url = format!("redis://{host}:{port}");
    let redis_client = redis::Client::open(redis_url).unwrap();

    assert!(
        verify_redisearch(&redis_client).await,
        "RediSearch module not available on this Redis instance"
    );

    let server = httpmock::MockServer::start();
    setup_product_embedding_mocks(&server);

    let vector_store = setup_product_store(&redis_client, index_name, &server).await;

    // Filter by category = "Clothing" — should return 1 result
    let filter = Filter::eq("category", RedisValue::String("Clothing".to_string()));
    let req = VectorSearchRequest::builder()
        .query("find products")
        .samples(10)
        .filter(filter)
        .build();

    let results = vector_store.top_n_ids(req).await.unwrap();

    assert_eq!(results.len(), 1, "Should find 1 Clothing product");
    assert!(!results[0].1.is_empty());

    cleanup_redis_index(&redis_client, index_name)
        .await
        .unwrap();
}
