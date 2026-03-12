use rig::client::EmbeddingsClient;
use rig::{
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

/// Check if Redis is already running on localhost:6379
async fn is_redis_running() -> bool {
    match redis::Client::open("redis://127.0.0.1:6379") {
        Ok(client) => client.get_multiplexed_async_connection().await.is_ok(),
        Err(_) => false,
    }
}

/// Get Redis connection info - either from existing instance or new container
async fn get_redis_connection() -> (
    String,
    u16,
    Option<testcontainers::ContainerAsync<GenericImage>>,
) {
    if is_redis_running().await {
        println!("Using existing Redis instance on localhost:6379");
        ("127.0.0.1".to_string(), REDIS_PORT, None)
    } else {
        println!("Starting new Redis Stack container");
        let container = GenericImage::new("redis/redis-stack", "latest")
            .with_exposed_port(REDIS_PORT.tcp())
            .with_wait_for(WaitFor::Duration {
                length: std::time::Duration::from_secs(3),
            })
            .start()
            .await
            .expect("Failed to start Redis Stack container");

        let port = container.get_host_port_ipv4(REDIS_PORT).await.unwrap();
        let host = container.get_host().await.unwrap().to_string();

        (host, port, Some(container))
    }
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
    let (host, port, _container) = get_redis_connection().await;
    let index_name = "test_vector_search_basic";

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

    let redis_url = format!("redis://{host}:{port}");
    let redis_client = redis::Client::open(redis_url).unwrap();

    setup_redis_index(&redis_client, index_name, 1536)
        .await
        .unwrap();

    let vector_store = RedisVectorStore::new(
        model.clone(),
        redis_client.clone(),
        index_name.to_string(),
        VECTOR_FIELD.to_string(),
    )
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
        .build()
        .unwrap();

    let results = vector_store.top_n::<Word>(req).await.unwrap();

    assert_eq!(results.len(), 1);
    let (score, _, doc) = &results[0];
    // Redis returns cosine distance (0 = identical, higher = more different)
    // So we just check it's a valid number
    assert!(score.is_finite());
    assert!(doc.definition.contains("linglingdong"));

    cleanup_redis_index(&redis_client, index_name)
        .await
        .unwrap();
}

#[tokio::test]
async fn test_top_n_ids() {
    let (host, port, _container) = get_redis_connection().await;
    let index_name = "test_top_n_ids";

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

    let redis_url = format!("redis://{host}:{port}");
    let redis_client = redis::Client::open(redis_url).unwrap();

    setup_redis_index(&redis_client, index_name, 1536)
        .await
        .unwrap();

    let vector_store = RedisVectorStore::new(
        model.clone(),
        redis_client.clone(),
        index_name.to_string(),
        VECTOR_FIELD.to_string(),
    )
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
        .build()
        .unwrap();

    let results = vector_store.top_n_ids(req).await.unwrap();

    assert_eq!(results.len(), 2);
    // Redis returns cosine distance, so scores can be 0 or positive
    assert!(results[0].0.is_finite());
    assert!(!results[0].1.is_empty());

    cleanup_redis_index(&redis_client, index_name)
        .await
        .unwrap();
}

#[tokio::test]
async fn test_threshold_filtering() {
    let (host, port, _container) = get_redis_connection().await;
    let index_name = "test_threshold_filtering";

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

    let redis_url = format!("redis://{host}:{port}");
    let redis_client = redis::Client::open(redis_url).unwrap();

    setup_redis_index(&redis_client, index_name, 1536)
        .await
        .unwrap();

    let vector_store = RedisVectorStore::new(
        model.clone(),
        redis_client.clone(),
        index_name.to_string(),
        VECTOR_FIELD.to_string(),
    )
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
        .build()
        .unwrap();

    let results = vector_store.top_n::<Word>(req).await.unwrap();

    for (score, _, _) in &results {
        assert!(score >= &0.5, "All results should meet threshold");
    }

    cleanup_redis_index(&redis_client, index_name)
        .await
        .unwrap();
}

#[tokio::test]
async fn test_insert_multiple_embeddings() {
    let (host, port, _container) = get_redis_connection().await;
    let index_name = "test_insert_multiple_embeddings";

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

    let redis_url = format!("redis://{host}:{port}");
    let redis_client = redis::Client::open(redis_url).unwrap();

    setup_redis_index(&redis_client, index_name, 1536)
        .await
        .unwrap();

    let vector_store = RedisVectorStore::new(
        model.clone(),
        redis_client.clone(),
        index_name.to_string(),
        VECTOR_FIELD.to_string(),
    )
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
        .arg("*")
        .query_async(&mut con)
        .await
        .unwrap();

    // Should have at least 3 documents (one per embedding)
    assert!(keys.len() >= 3, "Should have inserted at least 3 documents");

    cleanup_redis_index(&redis_client, index_name)
        .await
        .unwrap();
}

#[tokio::test]
async fn test_empty_results() {
    let (host, port, _container) = get_redis_connection().await;
    let index_name = "test_empty_results";

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

    let redis_url = format!("redis://{host}:{port}");
    let redis_client = redis::Client::open(redis_url).unwrap();

    setup_redis_index(&redis_client, index_name, 1536)
        .await
        .unwrap();

    let vector_store = RedisVectorStore::new(
        model.clone(),
        redis_client.clone(),
        index_name.to_string(),
        VECTOR_FIELD.to_string(),
    )
    .with_key_prefix(format!("{}:", index_name));

    let req = VectorSearchRequest::builder()
        .query("query with no results")
        .samples(5)
        .build()
        .unwrap();

    let results = vector_store.top_n::<Word>(req).await.unwrap();

    assert_eq!(results.len(), 0);

    cleanup_redis_index(&redis_client, index_name)
        .await
        .unwrap();
}
