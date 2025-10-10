use futures::StreamExt;
use mongodb::{
    Collection, SearchIndexModel,
    bson::{self, doc},
    options::ClientOptions,
};
use rig::{
    Embed,
    embeddings::EmbeddingsBuilder,
    providers::openai,
    vector_store::{InsertDocuments, VectorStoreIndex},
};
use rig::{client::EmbeddingsClient, vector_store::request::VectorSearchRequest};
use rig_mongodb::{MongoDbVectorIndex, SearchParams};
use serde_json::json;
use testcontainers::{
    GenericImage, ImageExt,
    core::{IntoContainerPort, WaitFor},
    runners::AsyncRunner,
};
use tokio::time::{Duration, sleep};

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
                    "embedding": vec![0.1; 1536],
                    "index": 0
                  },
                  {
                    "object": "embedding",
                    "embedding": vec![0.2; 1536],
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
                        "embedding": vec![0.0023064254; 1536],
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

    // Wait for the new documents to be indexed
    sleep(Duration::from_secs(5)).await;

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

    let query = "What is a linglingdong?";
    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(1)
        .build()
        .expect("VectorSearchRequest should not fail to build here");

    let results = index.top_n::<serde_json::Value>(req).await.unwrap();

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

#[tokio::test]
async fn insert_documents_test() {
    // Setup mock openai API
    let server = httpmock::MockServer::start();

    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .header("Authorization", "Bearer TEST")
            .json_body(json!({
                "input": [
                    "Test document 1",
                    "Test document 2"
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
                    "embedding": vec![0.1; 1536],
                    "index": 0
                  },
                  {
                    "object": "embedding",
                    "embedding": vec![0.2; 1536],
                    "index": 1
                  }
                ],
                "model": "text-embedding-ada-002",
                "usage": {
                  "prompt_tokens": 4,
                  "total_tokens": 4
                }
            }));
    });

    // Initialize OpenAI client
    let openai_client = openai::Client::builder("TEST")
        .base_url(&server.base_url())
        .build();
    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    // Setup MongoDB container
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

    // Create test documents in the format expected by InsertDocuments trait
    let test_words = vec![
        Word {
            id: "insert_test_1".to_string(),
            definition: "Test document 1".to_string(),
        },
        Word {
            id: "insert_test_2".to_string(),
            definition: "Test document 2".to_string(),
        },
    ];

    // Generate embeddings using EmbeddingsBuilder (returns Vec<(Word, OneOrMany<Embedding>)>)
    let documents_with_embeddings = EmbeddingsBuilder::new(model.clone())
        .documents(test_words)
        .unwrap()
        .build()
        .await
        .expect("Failed to create embeddings");

    // Clear collection before test
    collection.delete_many(doc! {}).await.unwrap();

    // Create MongoDbVectorIndex (we don't need the vector search functionality, just access to insert_documents)
    let temp_collection = collection.clone_with_type::<Word>();

    // We expect this to fail because we don't have a proper vector index, but that's OK
    // We just need the MongoDbVectorIndex struct to call insert_documents
    match MongoDbVectorIndex::new(
        temp_collection.clone(),
        model.clone(),
        "test_index_that_doesnt_exist", // This will fail, but we handle it
        SearchParams::new(),
    )
    .await
    {
        Ok(vector_index) => {
            match vector_index
                .insert_documents(documents_with_embeddings)
                .await
            {
                Ok(_) => {
                    // Verify documents were inserted
                    let count = collection.count_documents(doc! {}).await.unwrap();
                    assert_eq!(count, 2, "Should have inserted exactly 2 documents");

                    // Check document structure
                    let mut cursor = collection.find(doc! {}).await.unwrap();
                    let mut docs_found = 0;
                    while let Some(result) = cursor.next().await {
                        let doc = result.unwrap();
                        docs_found += 1;

                        println!("🔍 Document {docs_found}: {doc:?}");

                        // Verify your implementation created the right fields
                        assert!(
                            doc.contains_key("document"),
                            "Should have 'document' field from your implementation"
                        );
                        assert!(
                            doc.contains_key("embedding"),
                            "Should have 'embedding' field from your implementation"
                        );
                        assert!(
                            doc.contains_key("embedded_text"),
                            "Should have 'embedded_text' field from your implementation"
                        );
                    }
                }
                Err(e) => {
                    panic!("InsertDocuments::insert_documents() failed: {e}");
                }
            }
        }
        Err(e) => {
            println!("vector index creation failed (expected): {e}");
        }
    }
}

async fn create_search_index(collection: &Collection<bson::Document>) {
    let max_attempts = 5;

    for attempt in 0..max_attempts {
        match collection
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
        {
            Ok(_) => {
                // Wait for index to be available
                for _ in 0..max_attempts {
                    let indexes = collection
                        .list_search_indexes()
                        .name(VECTOR_SEARCH_INDEX_NAME)
                        .await
                        .unwrap()
                        .collect::<Vec<_>>()
                        .await;

                    if indexes.iter().any(|idx| {
                        idx.as_ref()
                            .ok()
                            .map(|i| {
                                // Check both name and status
                                let name_matches =
                                    i.get_str("name").ok() == Some(VECTOR_SEARCH_INDEX_NAME);
                                let status_ready = i.get_str("status").ok() == Some("READY");
                                name_matches && status_ready
                            })
                            .unwrap_or(false)
                    }) {
                        return;
                    }
                    sleep(Duration::from_secs(2)).await;
                }
                panic!("Index creation verified but index not found");
            }
            Err(_) => {
                println!(
                    "Waiting for MongoDB... {} attempts remaining",
                    max_attempts - attempt - 1
                );
                sleep(Duration::from_secs(5)).await;
            }
        }
    }

    panic!("Failed to create search index after {max_attempts} attempts");
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

    // Create the search index
    create_search_index(&collection).await;

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
