use rig::{embeddings::EmbeddingsBuilder, vector_store::VectorStoreIndex, Embed};
use rig_postgres::PostgresVectorStore;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::{postgres::PgPoolOptions, PgPool};
use testcontainers::{
    core::{IntoContainerPort, WaitFor},
    runners::AsyncRunner,
    ContainerAsync, GenericImage, ImageExt,
};

const POSTGRES_PORT: u16 = 5432;

#[derive(Embed, Clone, Serialize, Deserialize, Debug, PartialEq)]
struct Word {
    id: String,
    #[embed]
    definition: String,
}

#[tokio::test]
async fn vector_search_test() {
    let container = start_container().await;

    let host = container.get_host().await.unwrap().to_string();
    let port = container
        .get_host_port_ipv4(POSTGRES_PORT)
        .await
        .expect("Error getting docker port");

    println!("Container started on host:port {}:{}", host, port);

    // connect to Postgres
    let pg_pool = connect_to_postgres(host, port).await;

    // run migrations on Postgres
    sqlx::migrate!("./tests/migrations")
        .run(&pg_pool)
        .await
        .expect("Failed to run migrations");

    println!("Connected to postgres and ran migrations");

    // init fake openai service
    let openai_mock = create_openai_mock_service().await;
    let openai_client = rig::providers::openai::Client::from_url("TEST", &openai_mock.base_url());

    let model = openai_client.embedding_model(rig::providers::openai::TEXT_EMBEDDING_ADA_002);

    // create test documents with mocked embeddings
    let words = vec![
        Word {
            id: "0981d983-a5f8-49eb-89ea-f7d3b2196d2e".to_string(),
            definition: "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets".to_string(),
        },
        Word {
            id: "62a36d43-80b6-4fd6-990c-f75bb02287d1".to_string(),
            definition: "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
        },
        Word {
            id: "f9e17d59-32e5-440c-be02-b2759a654824".to_string(),
            definition: "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
        }
    ];

    let documents = EmbeddingsBuilder::new(model.clone())
        .documents(words)
        .unwrap()
        .build()
        .await
        .expect("Failed to create embeddings");

    // insert documents into vector store
    let vector_store = PostgresVectorStore::default(model, pg_pool.clone());

    vector_store
        .insert_documents(documents)
        .await
        .expect("Failed to insert documents");

    let documents_count: i64 = sqlx::query_scalar("SELECT count(*) FROM documents")
        .fetch_one(&pg_pool)
        .await
        .expect("Failed to fetch documents count");

    assert_eq!(documents_count, 3);

    // search for a document
    let results = vector_store
        .top_n::<Word>("What is a linglingdong?", 1)
        .await
        .expect("Failed to search for document");

    let (score, id, doc) = results[0].clone();
    println!("Score: {}, ID: {}, Document: {:?}", score, id, doc);

    assert_eq!(results.len(), 1);
    assert_eq!(doc, Word {
        id: "f9e17d59-32e5-440c-be02-b2759a654824".to_string(),
        definition: "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
    });
}

async fn start_container() -> ContainerAsync<GenericImage> {
    // Setup a local postgres container for testing. NOTE: docker service must be running.
    GenericImage::new("pgvector/pgvector", "pg17")
        .with_wait_for(WaitFor::message_on_stderr(
            "database system is ready to accept connections",
        ))
        .with_exposed_port(POSTGRES_PORT.tcp())
        .with_env_var("POSTGRES_USER", "postgres")
        .with_env_var("POSTGRES_PASSWORD", "postgres")
        .with_env_var("POSTGRES_DB", "rig")
        .start()
        .await
        .expect("Failed to start postgres with pgvector container")
}

async fn connect_to_postgres(host: String, port: u16) -> PgPool {
    // connect to Postgres
    PgPoolOptions::new()
        .max_connections(50)
        .idle_timeout(std::time::Duration::from_secs(5))
        .connect(&format!(
            "postgres://postgres:postgres@{}:{}/rig",
            host, port
        ))
        .await
        .expect("Failed to create postgres pool")
}

async fn create_openai_mock_service() -> httpmock::MockServer {
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
                  {
                    "object": "embedding",
                    "embedding": vec![0.0043064255; 1536],
                    "index": 0
                  },
                  {
                    "object": "embedding",
                    "embedding": vec![0.0043064255; 1536],
                    "index": 1
                  },
                  {
                    "object": "embedding",
                    "embedding": vec![0.0023064255; 1536],
                    "index": 2
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
    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .header("Authorization", "Bearer TEST")
            .json_body(json!({
                "input": [
                    "What is a linglingdong?"
                ],
                "model": "text-embedding-ada-002",
            }));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                    "object": "list",
                    "data": [
                      {
                        "object": "embedding",
                        "embedding": vec![0.0024; 1536],
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

    server
}
