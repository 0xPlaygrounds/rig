use mongodb::{bson::{self, doc}, options::ClientOptions, Collection, SearchIndexModel};
use rig::{embeddings::{DocumentEmbeddings, EmbeddingsBuilder}, providers::openai, vector_store::VectorStoreIndex};
use rig_mongodb::MongoDbVectorIndex;
use testcontainers::{core::{IntoContainerPort, WaitFor}, runners::AsyncRunner, GenericImage, ImageExt};

const VECTOR_SEARCH_INDEX_NAME: &str = "vector_index";

#[tokio::test]
async fn integration_test() {
    // Initialize OpenAI client
    let openai_client = openai::Client::from_env();
    
    // Setup local MongoDB Atlas
   let container = GenericImage::new("mongodb/mongodb-atlas-local", "latest")
        .with_exposed_port(27017.tcp())
        .with_wait_for(WaitFor::Duration { length: std::time::Duration::from_secs(10) })
        .with_env_var("MONGODB_INITDB_ROOT_USERNAME", "riguser")
        .with_env_var("MONGODB_INITDB_ROOT_PASSWORD", "rigpassword")
        .start()
        .await
        .expect("Failed to start MongoDB Atlas container");

    let port = container.get_host_port_ipv4(27017).await.unwrap();

    // Initialize MongoDB client
    let options = ClientOptions::parse(format!("mongodb://riguser:rigpassword@localhost:{port}/?directConnection=true"))
        .await
        .expect("MongoDB connection string should be valid");

    let mongodb_client =
        mongodb::Client::with_options(options).expect("MongoDB client options should be valid");

    // Initialize MongoDB vector store
    mongodb_client
        .database("rig")
        .create_collection("fake_definitions")
        .await
        .expect("Collection should be created");

    // Initialize MongoDB vector store
    let collection: Collection<bson::Document> = mongodb_client
        .database("rig")
        .collection("fake_definitions");

    // Create a vector search index
    collection.create_search_index(
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
            .build()
    ).await
    .expect("Failed to create search index");

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
    ).await.expect("Failed to create Rig vector index");

    // Query the index
    let results = vector_index
        .top_n::<serde_json::Value>("What is a linglingdong?", 1)
        .await
        .expect("Failed to query vector index");

    let result_string = &results.first().unwrap().1;

    assert_eq!(
        result_string,
        "\"doc2\""
    );
}