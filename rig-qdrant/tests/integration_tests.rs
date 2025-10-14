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
use rig::{
    Embed, embeddings::EmbeddingsBuilder, providers::openai, vector_store::VectorStoreIndex,
};
use rig::{client::EmbeddingsClient, vector_store::request::VectorSearchRequest};
use rig_qdrant::QdrantVectorStore;

const QDRANT_PORT: u16 = 6333;
const QDRANT_PORT_SECONDARY: u16 = 6334;
const COLLECTION_NAME: &str = "rig-collection";

#[derive(Embed, Clone, serde::Deserialize, serde::Serialize, Debug)]
struct Word {
    id: String,
    #[embed]
    definition: String,
}

#[tokio::test]
async fn vector_search_test() {
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
                "dimensions": 1536,
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

    // Initialize OpenAI client
    let openai_client = openai::Client::builder("TEST")
        .base_url(&server.base_url())
        .build();
    // let openai_client = openai::Client::from_env();

    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    let points = create_points(model.clone()).await;

    client
        .upsert_points(UpsertPointsBuilder::new(COLLECTION_NAME, points).wait(true))
        .await
        .unwrap();

    let query_params = QueryPointsBuilder::new(COLLECTION_NAME).with_payload(true);
    let vector_store = QdrantVectorStore::new(client, model, query_params.build());

    let query = "What is a linglingdong?";
    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(1)
        .build()
        .expect("VectorSearchRequest should not fail to build here");

    let results = vector_store.top_n::<serde_json::Value>(req).await.unwrap();

    let (_, _, value) = &results.first().unwrap();

    assert_eq!(
        value,
        &serde_json::json!({
            "definition": "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.",
            "id": "f9e17d59-32e5-440c-be02-b2759a654824"
        })
    )
}

async fn create_points(model: openai::EmbeddingModel) -> Vec<PointStruct> {
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

    let documents = EmbeddingsBuilder::new(model)
        .documents(words)
        .unwrap()
        .build()
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
