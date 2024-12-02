use mongodb::{
    bson::{self, doc},
    options::ClientOptions,
    Collection, SearchIndexModel,
};
use rig::{
    embeddings::EmbeddingsBuilder, providers::openai, vector_store::VectorStoreIndex, Embed,
};
use rig_mongodb::{MongoDbVectorIndex, SearchParams};
use testcontainers::{
    core::{IntoContainerPort, WaitFor},
    runners::AsyncRunner,
    GenericImage, ImageExt,
};

#[derive(Embed, Clone, serde::Deserialize, serde::Serialize, Debug, PartialEq)]
struct FakeDefinition {
    #[serde(rename = "_id")]
    id: String,
    #[embed]
    definition: String,
}

const VECTOR_SEARCH_INDEX_NAME: &str = "vector_index";
const MONGODB_PORT: u16 = 27017;
const COLLECTION_NAME: &str = "fake_definitions";
const DATABASE_NAME: &str = "rig";

/// Setup a local MongoDB Atlas container for testing. NOTE: docker service must be running.
/// This includes running the container with `testcontainers`, and creating a database and collection
/// that will be used by integration tests.
async fn setup_mongo_server() -> Collection<bson::Document> {
    // Setup local MongoDB Atlas
    let container = GenericImage::new("mongodb/mongodb-atlas-local", "latest")
        .with_exposed_port(MONGODB_PORT.tcp())
        .with_wait_for(WaitFor::Duration {
            length: std::time::Duration::from_secs(10),
        })
        .with_env_var("MONGODB_INITDB_ROOT_USERNAME", "riguser")
        .with_env_var("MONGODB_INITDB_ROOT_PASSWORD", "rigpassword")
        .start()
        .await
        .expect("Failed to start MongoDB Atlas container");

    let port = container.get_host_port_ipv4(MONGODB_PORT).await.unwrap();

    // Initialize MongoDB client
    let options = ClientOptions::parse(format!(
        "mongodb://riguser:rigpassword@localhost:{port}/?directConnection=true"
    ))
    .await
    .expect("MongoDB connection string should be valid");

    let mongodb_client =
        mongodb::Client::with_options(options).expect("MongoDB client options should be valid");

    // Initialize MongoDB database and collection
    mongodb_client
        .database(DATABASE_NAME)
        .create_collection(COLLECTION_NAME)
        .await
        .expect("Collection should be created");

    // Get the created collection
    let collection: Collection<bson::Document> = mongodb_client
        .database(DATABASE_NAME)
        .collection(COLLECTION_NAME);

    // Create a vector search index on the collection
    collection
        .create_search_index(
            SearchIndexModel::builder()
                .name(Some(VECTOR_SEARCH_INDEX_NAME.to_string()))
                .index_type(Some(mongodb::SearchIndexType::VectorSearch))
                .definition(doc! {
                    "fields": [{
                        "numDimensions": 1536,
                        "path": "embeddings.vec",
                        "similarity": "cosine",
                        "type": "vector"
                    }]
                })
                .build(),
        )
        .await
        .expect("Failed to create search index");

    collection
}

#[tokio::test]
async fn vector_search_test() {
    // Initialize OpenAI client
    let openai_client = openai::Client::from_env();

    let collection = setup_mongo_server().await;

    // Select the embedding model and generate our embeddings
    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    let linglingdong = FakeDefinition {
        id: "doc2".to_string(),
        definition: "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
    };

    let fake_definitions = vec![
        FakeDefinition {
            id: "doc0".to_string(),
            definition: "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets".to_string(),
        },
        FakeDefinition {
            id: "doc1".to_string(),
            definition: "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
        },
        linglingdong.clone()
    ];

    let embeddings = EmbeddingsBuilder::new(model.clone())
        .documents(fake_definitions)
        .unwrap()
        .build()
        .await
        .unwrap();

    let mongo_documents = embeddings
        .iter()
        .map(|(FakeDefinition { id, definition, .. }, embedding)| {
            doc! {
                "id": id.clone(),
                "definition": definition.clone(),
                "embedding": embedding.first().vec.clone(),
            }
        })
        .collect::<Vec<_>>();

    collection.insert_many(mongo_documents).await.unwrap();

    // Create a vector index on our vector store.
    // Note: a vector index called "vector_index" must exist on the MongoDB collection you are querying.
    // IMPORTANT: Reuse the same model that was used to generate the embeddings
    let index = MongoDbVectorIndex::new(
        collection,
        model,
        VECTOR_SEARCH_INDEX_NAME,
        SearchParams::new(),
    )
    .await
    .unwrap();

    // Query the index
    let mut results = index
        .top_n::<serde_json::Value>("What is a linglingdong?", 1)
        .await
        .unwrap();

    if results.is_empty() {
        results = index
            .top_n::<serde_json::Value>("What is a linglingdong?", 1)
            .await
            .expect("Failed to query vector index");
    }

    let result_string = &results.first().unwrap();

    assert_eq!(
        result_string.2,
        serde_json::to_value(&linglingdong).unwrap()
    );
}
