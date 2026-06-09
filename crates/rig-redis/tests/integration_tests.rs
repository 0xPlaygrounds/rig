#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used
)]

use rig_core::client::EmbeddingsClient;
use rig_core::{
    Embed,
    embeddings::EmbeddingsBuilder,
    providers::openai,
    vector_store::{InsertDocuments, VectorStoreIndex, request::VectorSearchRequest},
};
use rig_redis::RedisVectorStore;
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

struct TestRedis {
    client: redis::Client,
    _container: Option<testcontainers::ContainerAsync<GenericImage>>,
}

async fn redis_has_search(client: &redis::Client) -> bool {
    let Ok(mut con) = client.get_multiplexed_async_connection().await else {
        return false;
    };

    redis::cmd("FT._LIST")
        .query_async::<Vec<String>>(&mut con)
        .await
        .is_ok()
}

async fn get_redis_connection() -> Option<TestRedis> {
    if let Ok(redis_url) = std::env::var("REDIS_URL") {
        let client = match redis::Client::open(redis_url.clone()) {
            Ok(client) => client,
            Err(err) => {
                eprintln!("Skipping Redis integration tests: invalid REDIS_URL ({err})");
                return None;
            }
        };

        if redis_has_search(&client).await {
            return Some(TestRedis {
                client,
                _container: None,
            });
        }

        eprintln!("Skipping Redis integration tests: REDIS_URL does not expose RediSearch");
        return None;
    }

    let container = match GenericImage::new("redis/redis-stack", "latest")
        .with_exposed_port(REDIS_PORT.tcp())
        .with_wait_for(WaitFor::Duration {
            length: std::time::Duration::from_secs(3),
        })
        .start()
        .await
    {
        Ok(container) => container,
        Err(err) => {
            eprintln!(
                "Skipping Redis integration tests: could not start Redis Stack container ({err})"
            );
            return None;
        }
    };

    let port = match container.get_host_port_ipv4(REDIS_PORT).await {
        Ok(port) => port,
        Err(err) => {
            eprintln!("Skipping Redis integration tests: could not read Redis Stack port ({err})");
            return None;
        }
    };
    let host = match container.get_host().await {
        Ok(host) => host.to_string(),
        Err(err) => {
            eprintln!("Skipping Redis integration tests: could not read Redis Stack host ({err})");
            return None;
        }
    };
    let client = match redis::Client::open(format!("redis://{host}:{port}")) {
        Ok(client) => client,
        Err(err) => {
            eprintln!("Skipping Redis integration tests: invalid container Redis URL ({err})");
            return None;
        }
    };

    if !redis_has_search(&client).await {
        eprintln!(
            "Skipping Redis integration tests: Redis Stack container does not expose RediSearch"
        );
        return None;
    }

    Some(TestRedis {
        client,
        _container: Some(container),
    })
}

fn unique_index_name(base: &str) -> String {
    format!("{}_{}", base, uuid::Uuid::new_v4().simple())
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
async fn test_vector_search_basic() {
    let Some(redis) = get_redis_connection().await else {
        return;
    };
    let index_name = unique_index_name("test_vector_search_basic");

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

    let redis_client = redis.client;

    setup_redis_index(&redis_client, &index_name, 1536)
        .await
        .unwrap();

    let vector_store = RedisVectorStore::new(
        model.clone(),
        redis_client.clone(),
        index_name.clone(),
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
    // Redis returns cosine distance (0 = identical, higher = more different)
    // So we just check it's a valid number
    assert!(score.is_finite());
    assert!(doc.definition.contains("linglingdong"));

    cleanup_redis_index(&redis_client, &index_name)
        .await
        .unwrap();
}

#[tokio::test]
async fn test_top_n_ids() {
    let Some(redis) = get_redis_connection().await else {
        return;
    };
    let index_name = unique_index_name("test_top_n_ids");

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

    let redis_client = redis.client;

    setup_redis_index(&redis_client, &index_name, 1536)
        .await
        .unwrap();

    let vector_store = RedisVectorStore::new(
        model.clone(),
        redis_client.clone(),
        index_name.clone(),
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
    // Redis returns cosine distance, so scores can be 0 or positive
    assert!(results[0].0.is_finite());
    assert!(!results[0].1.is_empty());

    cleanup_redis_index(&redis_client, &index_name)
        .await
        .unwrap();
}

#[tokio::test]
async fn test_threshold_filtering() {
    let Some(redis) = get_redis_connection().await else {
        return;
    };
    let index_name = unique_index_name("test_threshold_filtering");

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

    let redis_client = redis.client;

    setup_redis_index(&redis_client, &index_name, 1536)
        .await
        .unwrap();

    let vector_store = RedisVectorStore::new(
        model.clone(),
        redis_client.clone(),
        index_name.clone(),
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

    cleanup_redis_index(&redis_client, &index_name)
        .await
        .unwrap();
}

#[tokio::test]
async fn test_insert_multiple_embeddings() {
    let Some(redis) = get_redis_connection().await else {
        return;
    };
    let index_name = unique_index_name("test_insert_multiple_embeddings");

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

    let redis_client = redis.client;

    setup_redis_index(&redis_client, &index_name, 1536)
        .await
        .unwrap();

    let vector_store = RedisVectorStore::new(
        model.clone(),
        redis_client.clone(),
        index_name.clone(),
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

    // Verify documents were inserted
    let mut con = redis_client
        .get_multiplexed_async_connection()
        .await
        .unwrap();
    let keys: Vec<String> = redis::cmd("KEYS")
        .arg(format!("{index_name}:*"))
        .query_async(&mut con)
        .await
        .unwrap();

    // Should have at least 3 documents (one per embedding)
    assert!(keys.len() >= 3, "Should have inserted at least 3 documents");

    cleanup_redis_index(&redis_client, &index_name)
        .await
        .unwrap();
}

#[tokio::test]
async fn test_empty_results() {
    let Some(redis) = get_redis_connection().await else {
        return;
    };
    let index_name = unique_index_name("test_empty_results");

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

    let redis_client = redis.client;

    setup_redis_index(&redis_client, &index_name, 1536)
        .await
        .unwrap();

    let vector_store = RedisVectorStore::new(
        model.clone(),
        redis_client.clone(),
        index_name.clone(),
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

    cleanup_redis_index(&redis_client, &index_name)
        .await
        .unwrap();
}
