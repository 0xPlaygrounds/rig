use serde_json::json;
use testcontainers::{
    GenericImage,
    core::{IntoContainerPort, WaitFor},
    runners::AsyncRunner,
};

use crate::common::{WORD_DEFINITIONS, mock_embeddings, openai_client, skip_if_docker_unavailable};
use qdrant_client::{
    Payload, Qdrant,
    qdrant::{
        CreateCollectionBuilder, Distance, PointStruct, QueryPointsBuilder, UpsertPointsBuilder,
        VectorParamsBuilder,
    },
};
use rig::qdrant::QdrantVectorStore;
use rig::vector_store::request::VectorSearchRequest;
use rig::{
    Embed, driver::Model, embeddings::EmbeddingsBuilder, providers::openai,
    vector_store::VectorStoreIndex,
};

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
    if skip_if_docker_unavailable("vector_search_test") {
        return;
    }

    // Setup a local qdrant container for testing. NOTE: docker service must be running.
    // Pinned like `pgvector:pg17` / `scylla:5.4`: a floating `latest` defeats
    // layer caching and lets a rerun silently test a different database version.
    let container = GenericImage::new("qdrant/qdrant", "v1.19.0")
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

    mock_embeddings(
        &server,
        json!({ "input": WORD_DEFINITIONS, "model": "text-embedding-ada-002" }),
        [
            vec![0.0043064255; 1536],
            vec![0.0043064255; 1536],
            vec![0.0023064255; 1536],
        ],
    );
    mock_embeddings(
        &server,
        json!({ "input": ["What is a linglingdong?"], "model": "text-embedding-ada-002" }),
        [vec![0.002; 1536]],
    );

    let openai_client = openai_client(&server);

    let model = openai_client.embedding(openai::TEXT_EMBEDDING_ADA_002, None);

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
        .build();

    let results = vector_store.top_n::<serde_json::Value>(req).await.unwrap();

    let (_, _, value) = &results.first().unwrap();

    assert_eq!(
        value,
        &serde_json::json!({
            "definition": "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.",
            "id": "f9e17d59-32e5-440c-be02-b2759a654824"
        })
    );
}

async fn create_points(model: Model<openai::wire::Embeddings>) -> Vec<PointStruct> {
    let ids = [
        "0981d983-a5f8-49eb-89ea-f7d3b2196d2e",
        "62a36d43-80b6-4fd6-990c-f75bb02287d1",
        "f9e17d59-32e5-440c-be02-b2759a654824",
    ];
    let words = ids
        .into_iter()
        .zip(WORD_DEFINITIONS)
        .map(|(id, definition)| Word {
            id: id.to_string(),
            definition: definition.to_string(),
        });

    let documents = EmbeddingsBuilder::new(model)
        .documents(words)
        .unwrap()
        .build()
        .await
        .unwrap();

    documents
        .into_iter()
        .map(|(d, embeddings)| {
            let vec: Vec<f32> = embeddings
                .first()
                .expect("expected at least one embedding")
                .vec
                .iter()
                .map(|&x| x as f32)
                .collect();
            PointStruct::new(
                d.id.clone(),
                vec,
                Payload::try_from(serde_json::to_value(&d).unwrap()).unwrap(),
            )
        })
        .collect()
}
