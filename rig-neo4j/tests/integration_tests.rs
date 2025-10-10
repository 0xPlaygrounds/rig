use serde_json::json;
use testcontainers::{
    GenericImage, ImageExt,
    core::{IntoContainerPort, WaitFor},
    runners::AsyncRunner,
};

use futures::{StreamExt, TryStreamExt};
use rig::vector_store::VectorStoreIndex;
use rig::{
    Embed, OneOrMany,
    embeddings::{Embedding, EmbeddingsBuilder},
    providers::openai,
};
use rig::{client::EmbeddingsClient, vector_store::request::VectorSearchRequest};
use rig_neo4j::{Neo4jClient, ToBoltType, vector_index::SearchParams};

const BOLT_PORT: u16 = 7687;
const HTTP_PORT: u16 = 7474;

#[derive(Embed, Clone, serde::Deserialize, Debug)]
struct Word {
    id: String,
    #[embed]
    definition: String,
}

#[tokio::test]
async fn vector_search_test() {
    // Setup a local Neo 4J container for testing. NOTE: docker service must be running.
    let container = GenericImage::new("neo4j", "latest")
        .with_wait_for(WaitFor::Duration {
            length: std::time::Duration::from_secs(5),
        })
        .with_exposed_port(BOLT_PORT.tcp())
        .with_exposed_port(HTTP_PORT.tcp())
        .with_env_var("NEO4J_AUTH", "none")
        .start()
        .await
        .expect("Failed to start Neo 4J container");

    let port = container.get_host_port_ipv4(BOLT_PORT).await.expect("");
    let host = container.get_host().await.expect("").to_string();

    let neo4j_client = Neo4jClient::connect(&format!("neo4j://{host}:{port}"), "", "")
        .await
        .expect("");

    // Setup mock openai API
    let server = httpmock::MockServer::start();

    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .header("Authorization", "Bearer TEST")
            .header("Content-Type", "application/json")
            .json_body(json!({
                "input": [
                    "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets",
                    "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
                    "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans."
                ],
                "model": "text-embedding-ada-002",
                "dimensions": 1536,
            }));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "object": "list",
                "data": [
                  {
                    "object": "embedding",
                    "embedding": vec![-0.001; 1536],
                    "index": 0
                  },
                  {
                    "object": "embedding",
                    "embedding": vec![0.0023064255; 1536],
                    "index": 1
                  },
                  {
                    "object": "embedding",
                    "embedding": vec![-0.001; 1536],
                    "index": 2
                  },
                ],
                "model": "text-embedding-ada-002",
                "usage": {
                  "prompt_tokens": 8,
                  "total_tokens": 8
                }
            }
        ));
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .header("Authorization", "Bearer TEST")
            .header("Content-Type", "application/json")
            .json_body(json!({
                "input": [
                    "What is a glarb?",
                ],
                "model": "text-embedding-ada-002",
                "dimensions": 1536
            }));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                    "object": "list",
                    "data": [
                      {
                        "object": "embedding",
                        "embedding": vec![0.0024064254; 1536],
                        "index": 0
                      }
                    ],
                    "model": "text-embedding-ada-002",
                    "usage": {
                      "prompt_tokens": 8,
                      "total_tokens": 8
                    }
                }
            ));
    });

    // Initialize OpenAI client
    let openai_client = openai::Client::builder("TEST")
        .base_url(&server.base_url())
        .build();

    // Select the embedding model and generate our embeddings
    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    let embeddings = create_embeddings(model.clone()).await;

    futures::stream::iter(embeddings)
        .map(|(doc, embeddings)| {
            neo4j_client.graph.run(
                neo4rs::query(
                    "
                        CREATE
                            (document:DocumentEmbeddings {
                                id: $id,
                                document: $document,
                                embedding: $embedding})
                        RETURN document",
                )
                .param("id", doc.id)
                // Here we use the first embedding but we could use any of them.
                // Neo4j only takes primitive types or arrays as properties.
                .param("embedding", embeddings.first().vec.clone())
                .param("document", doc.definition.to_bolt_type()),
            )
        })
        .buffer_unordered(3)
        .try_collect::<Vec<_>>()
        .await
        .expect("");

    // Create a vector index on our vector store
    println!("Creating vector index...");
    neo4j_client
        .graph
        .run(neo4rs::query(
            "CREATE VECTOR INDEX vector_index IF NOT EXISTS
                FOR (m:DocumentEmbeddings)
                ON m.embedding
                OPTIONS { indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                    }}",
        ))
        .await
        .expect("");

    // ℹ️ The index name must be unique among both indexes and constraints.
    // A newly created index is not immediately available but is created in the background.

    // Check if the index exists with db.awaitIndex(), the call timeouts if the index is not ready
    let index_exists = neo4j_client
        .graph
        .run(neo4rs::query("CALL db.awaitIndex('vector_index')"))
        .await;
    if index_exists.is_err() {
        println!("Index not ready, waiting for index...");
        std::thread::sleep(std::time::Duration::from_secs(5));
    }

    println!("Index exists: {index_exists:?}");

    // Create a vector index on our vector store
    // IMPORTANT: Reuse the same model that was used to generate the embeddings
    let index = neo4j_client
        .get_index(model, "vector_index", SearchParams::default())
        .await
        .expect("");

    let query = "What is a glarb?";
    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(1)
        .build()
        .expect("VectorSearchRequest should not fail to build here");

    // Query the index
    let results = index.top_n::<serde_json::Value>(req).await.expect("");

    let (_, _, value) = &results.first().expect("");

    assert_eq!(
        value,
        &serde_json::json!({
            "id": "doc1",
            "document": "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
            "embedding": serde_json::Value::Null
        })
    )
}

async fn create_embeddings(model: openai::EmbeddingModel) -> Vec<(Word, OneOrMany<Embedding>)> {
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

    EmbeddingsBuilder::new(model)
        .documents(words)
        .expect("")
        .build()
        .await
        .expect("")
}
