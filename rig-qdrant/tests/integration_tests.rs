use testcontainers::{
    core::{IntoContainerPort, WaitFor},
    runners::AsyncRunner,
    GenericImage,
};

use qdrant_client::{
    qdrant::{
        CreateCollectionBuilder, Distance, PointStruct, QueryPointsBuilder, UpsertPointsBuilder,
        VectorParamsBuilder,
    },
    Payload, Qdrant,
};
use rig::{
    embeddings::EmbeddingsBuilder, providers::openai, vector_store::VectorStoreIndex, Embed,
};
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

    // Initialize OpenAI client.
    let openai_client = openai::Client::from_env();

    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    let points = create_points(model.clone()).await;

    client
        .upsert_points(UpsertPointsBuilder::new(COLLECTION_NAME, points))
        .await
        .unwrap();

    let query_params = QueryPointsBuilder::new(COLLECTION_NAME).with_payload(true);
    let vector_store = QdrantVectorStore::new(client, model, query_params.build());

    let results = vector_store
        .top_n::<serde_json::Value>("What is a linglingdong?", 1)
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
