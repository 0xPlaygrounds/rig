use mongodb::{
    bson::{self, doc},
    options::ClientOptions,
    Collection, SearchIndexModel,
};
use rig::{
    embeddings::{DocumentEmbeddings, EmbeddingsBuilder},
    providers::openai,
    vector_store::VectorStoreIndex,
};
use rig_mongodb::MongoDbVectorIndex;
use testcontainers::{
    core::{IntoContainerPort, WaitFor},
    runners::AsyncRunner,
    GenericImage, ImageExt,
};

const VECTOR_SEARCH_INDEX_NAME: &str = "vector_index";
const MONGODB_PORT: u16 = 27017;

/// Setup a local MongoDB Atlas container for testing.
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
        .database("rig")
        .create_collection("fake_definitions")
        .await
        .expect("Collection should be created");

    // Get the created collection
    let collection: Collection<bson::Document> = mongodb_client
        .database("rig")
        .collection("fake_definitions");

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

    let embeddings = EmbeddingsBuilder::new(model.clone())
        .simple_document("doc0", "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets")
        .simple_document("doc1", "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.")
        .simple_document("doc2", "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.")
        .build()
        .await
        .expect("Failed to build embeddings");

    // Add embeddings to vector store
    collection
        .clone_with_type::<DocumentEmbeddings>()
        .insert_many(embeddings)
        .await
        .expect("Failed to insert embeddings");

    // Create a vector index on our vector store
    let vector_index = MongoDbVectorIndex::new(
        collection,
        model,
        VECTOR_SEARCH_INDEX_NAME,
        rig_mongodb::SearchParams::new(),
    )
    .await
    .expect("Failed to create Rig vector index");

    // Query the index
    let mut results = vector_index
        .top_n::<serde_json::Value>("What is a linglingdong?", 1)
        .await
        .expect("Failed to query vector index");

    if results.is_empty() {
        results = vector_index
            .top_n::<serde_json::Value>("What is a linglingdong?", 1)
            .await
            .expect("Failed to query vector index");
    }

    let result_string = &results.first().unwrap().1;

    assert_eq!(result_string, "\"doc2\"");
}
