#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

use rig::OneOrMany;
use serde_json::json;
use testcontainers::{
    GenericImage,
    core::{IntoContainerPort, WaitFor},
    runners::AsyncRunner,
};

use qdrant_client::{
    Payload, Qdrant,
    qdrant::{
        CreateCollectionBuilder, Distance, PointStruct, QueryPointsBuilder, UpsertPointsBuilder,
        VectorParamsBuilder,
    },
};
use rig::http_runtime::HttpRuntime;
use rig::qdrant::QdrantVectorStore;
use rig::vector_store::request::VectorSearchRequest;
use rig::{Embed, embeddings::embed_documents, providers::openai};

const QDRANT_PORT: u16 = 6333;
const QDRANT_PORT_SECONDARY: u16 = 6334;
const COLLECTION_NAME: &str = "rig-collection";

fn skip_if_docker_unavailable(test_name: &str) -> bool {
    let docker_socket = std::path::Path::new("/var/run/docker.sock");
    if std::env::var_os("DOCKER_HOST").is_some() || docker_socket.exists() {
        return false;
    }

    eprintln!("skipping {test_name}: Docker is unavailable");
    true
}

#[derive(Embed, Clone, serde::Deserialize, serde::Serialize, Debug)]
struct Word {
    id: String,
    #[embed]
    definition: String,
}

#[tokio::test]
async fn vector_search_test() {
    if skip_if_docker_unavailable("vector_search_test") {
        return;
    }

    // Setup a local qdrant container for testing. NOTE: docker service must be running.
    let container = GenericImage::new("qdrant/qdrant", "latest")
        .with_wait_for(WaitFor::Duration {
            length: std::time::Duration::from_secs(5),
        })
        .with_exposed_port(QDRANT_PORT.tcp())
        .with_exposed_port(QDRANT_PORT_SECONDARY.tcp())
        .start()
        .await
        .expect("Failed to start qdrant container");

    let port = container
        .get_host_port_ipv4(QDRANT_PORT_SECONDARY)
        .await
        .unwrap();
    let host = container.get_host().await.unwrap().to_string();

    let client = Qdrant::from_url(&format!("http://{host}:{port}"))
        .build()
        .unwrap();

    // Create a collection with 1536 dimensions if it doesn't exist
    // Note: Make sure the dimensions match the size of the embeddings returned by the
    // model you are using
    if !client.collection_exists(COLLECTION_NAME).await.unwrap() {
        client
            .create_collection(
                CreateCollectionBuilder::new(COLLECTION_NAME)
                    .vectors_config(VectorParamsBuilder::new(1536, Distance::Cosine)),
            )
            .await
            .unwrap();
    }

    // Setup mock openai API
    let server = httpmock::MockServer::start();

    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .header("Authorization", "Bearer TEST")
            .json_body(json!({
                "input": [
                    "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets",
                    "Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
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
                        "embedding": vec![0.002; 1536],
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

    // Configure the (mocked) OpenAI embeddings face
    let cfg = openai::functions::EmbeddingConfig::new(openai::TEXT_EMBEDDING_ADA_002)
        .with_api_key("TEST")
        .with_base_url(server.base_url());
    let rt = HttpRuntime::new();

    let points = create_points(&cfg, &rt).await;

    client
        .upsert_points(UpsertPointsBuilder::new(COLLECTION_NAME, points).wait(true))
        .await
        .unwrap();

    let query_params = QueryPointsBuilder::new(COLLECTION_NAME).with_payload(true);
    let vector_store = QdrantVectorStore::new(client, query_params.build());

    // Queries arrive pre-embedded: embed the query text with the (mocked)
    // embedding model, then pass the embedding to the store.
    let query_embedding =
        openai::functions::embed(&cfg, &rt, vec!["What is a linglingdong?".to_string()])
            .await
            .unwrap()
            .embeddings
            .into_iter()
            .next()
            .unwrap();
    let req = VectorSearchRequest::new(OneOrMany::one(query_embedding), 1);

    let results = vector_store
        .top_n_as::<serde_json::Value>(req)
        .await
        .unwrap();

    let (_, _, value) = &results.first().unwrap();

    assert_eq!(
        value,
        &serde_json::json!({
            "definition": "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.",
            "id": "f9e17d59-32e5-440c-be02-b2759a654824"
        })
    )
}

async fn create_points(
    cfg: &openai::functions::EmbeddingConfig,
    rt: &HttpRuntime,
) -> Vec<PointStruct> {
    let words = vec![
        Word {
            id: "0981d983-a5f8-49eb-89ea-f7d3b2196d2e".to_string(),
            definition: "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets".to_string(),
        },
        Word {
            id: "62a36d43-80b6-4fd6-990c-f75bb02287d1".to_string(),
            definition: "Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
        },
        Word {
            id: "f9e17d59-32e5-440c-be02-b2759a654824".to_string(),
            definition: "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
        }
    ];

    let max_documents = openai::functions::DESCRIPTOR
        .max_embedding_documents
        .unwrap_or(usize::MAX);
    let documents = embed_documents(
        words,
        max_documents,
        rig::embeddings::default_concurrency(max_documents),
        |texts| openai::functions::embed(cfg, rt, texts),
    )
    .await
    .unwrap();

    documents
        .into_iter()
        .map(|(d, embeddings)| {
            let vec: Vec<f32> = embeddings.first().vec.iter().map(|&x| x as f32).collect();
            PointStruct::new(
                d.id.clone(),
                vec,
                Payload::try_from(serde_json::to_value(&d).unwrap()).unwrap(),
            )
        })
        .collect()
}
