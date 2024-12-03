use mongodb::{
    bson::{self, doc},
    options::ClientOptions,
    Collection, SearchIndexModel,
};
use rig::{
    embeddings::EmbeddingsBuilder, providers::openai, vector_store::VectorStoreIndex, Embed,
};
use rig_mongodb::{MongoDbVectorIndex, SearchParams};
use serde_json::json;
use testcontainers::{
    core::{IntoContainerPort, WaitFor},
    runners::AsyncRunner,
    GenericImage, ImageExt,
};
use tokio::time::{sleep, Duration};

#[derive(Embed, Clone, serde::Deserialize, serde::Serialize, Debug, PartialEq)]
struct Word {
    #[serde(rename = "_id")]
    id: String,
    #[embed]
    definition: String,
}

const VECTOR_SEARCH_INDEX_NAME: &str = "vector_index";
const MONGODB_PORT: u16 = 27017;
const COLLECTION_NAME: &str = "words";
const DATABASE_NAME: &str = "rig";
const USERNAME: &str = "riguser";
const PASSWORD: &str = "rigpassword";

#[tokio::test]
async fn vector_search_test() {
    // Initialize OpenAI client
    let openai_client = openai::Client::from_env();

    // Select the embedding model and generate our embeddings
    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    // Setup a local MongoDB Atlas container for testing. NOTE: docker service must be running.
    let container = GenericImage::new("mongodb/mongodb-atlas-local", "latest")
        .with_exposed_port(MONGODB_PORT.tcp())
        .with_wait_for(WaitFor::Duration {
            length: std::time::Duration::from_secs(5),
        })
        .with_env_var("MONGODB_INITDB_ROOT_USERNAME", USERNAME)
        .with_env_var("MONGODB_INITDB_ROOT_PASSWORD", PASSWORD)
        .start()
        .await
        .expect("Failed to start MongoDB Atlas container");

    let port = container.get_host_port_ipv4(MONGODB_PORT).await.unwrap();
    let host = container.get_host().await.unwrap().to_string();

    let collection = bootstrap_collection(host, port).await;

    let embeddings = create_embeddings(model.clone()).await;

    collection.insert_many(embeddings).await.unwrap();

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

    sleep(Duration::from_secs(5)).await;

    // Query the index
    let results = index
        .top_n::<serde_json::Value>("What is a linglingdong?", 1)
        .await
        .unwrap();

    let (score, _, value) = &results.first().unwrap();

    assert_eq!(
        *value,
        json!({
            "_id": "doc2".to_string(),
            "definition": "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
            "score": score
        })
    )
}

async fn bootstrap_collection(host: String, port: u16) -> Collection<bson::Document> {
    // Initialize MongoDB client
    let options = ClientOptions::parse(format!(
        "mongodb://{USERNAME}:{PASSWORD}@{host}:{port}/?directConnection=true"
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
                        "path": "embedding",
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

async fn create_embeddings(model: openai::EmbeddingModel) -> Vec<bson::Document> {
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

    let embeddings = EmbeddingsBuilder::new(model)
        .documents(words)
        .unwrap()
        .build()
        .await
        .unwrap();

    embeddings
        .iter()
        .map(|(Word { id, definition, .. }, embedding)| {
            doc! {
                "_id": id.clone(),
                "definition": definition.clone(),
                "embedding": embedding.first().vec.clone(),
            }
        })
        .collect()
}
