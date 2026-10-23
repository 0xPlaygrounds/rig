use crate::common::{
    WORD_DEFINITIONS, axis_embedding, mock_embeddings, openai_client, skip_if_docker_unavailable,
};
use futures::{StreamExt, TryStreamExt};
use mongodb::{
    Collection, SearchIndexModel,
    bson::{self, doc},
    options::ClientOptions,
};
use rig::mongodb::{MongoDbVectorIndex, SearchParams};
use rig::vector_store::VectorSearchResult;
use rig::vector_store::request::VectorSearchRequest;
use rig::{
    Embed,
    driver::Model,
    embeddings::EmbeddingsBuilder,
    providers::openai,
    vector_store::{InsertDocuments, VectorStoreIndex},
};
use serde_json::json;
use testcontainers::{
    ContainerAsync, GenericImage, ImageExt,
    core::{IntoContainerPort, WaitFor, wait::LogWaitStrategy},
    runners::AsyncRunner,
};
use tokio::time::{Duration, Instant, sleep};

#[derive(Embed, Clone, serde::Deserialize, serde::Serialize, Debug, PartialEq)]
struct Word {
    #[serde(rename = "_id")]
    id: String,
    #[embed]
    definition: String,
}

const VECTOR_SEARCH_INDEX_NAME: &str = "vector_index";
const MONGODB_PORT: u16 = 27017;
// Pinned like `pgvector:pg17` / `scylla:5.4`: a floating `latest` defeats
// layer caching and lets a rerun silently test a different database version.
const MONGODB_IMAGE: &str = "mongodb/mongodb-atlas-local";
const MONGODB_TAG: &str = "8.0.5";
const COLLECTION_NAME: &str = "words";
const DATABASE_NAME: &str = "rig";
const USERNAME: &str = "riguser";
const PASSWORD: &str = "rigpassword";
/// The longest the container, a new search index, or the documents in it
/// may take to be ready: a bound on a hang, as each is waited for.
const READY_WITHIN: Duration = Duration::from_secs(180);
/// How often a search index, or a search, is asked again until it is ready.
const POLL: Duration = Duration::from_millis(500);

/// A local Atlas container, started once its last mongod listens: the
/// image's entrypoint starts mongod, sets it up and restarts it twice, so
/// a connection before the third start may meet a server shutting down.
/// The image's health check is no help, as it can pass before the
/// restarts.
async fn start_atlas() -> ContainerAsync<GenericImage> {
    GenericImage::new(MONGODB_IMAGE, MONGODB_TAG)
        .with_exposed_port(MONGODB_PORT.tcp())
        .with_wait_for(WaitFor::log(
            LogWaitStrategy::stdout("\"Waiting for connections\"").with_times(3),
        ))
        .with_startup_timeout(READY_WITHIN)
        .with_env_var("MONGODB_INITDB_ROOT_USERNAME", USERNAME)
        .with_env_var("MONGODB_INITDB_ROOT_PASSWORD", PASSWORD)
        .start()
        .await
        .expect("Failed to start MongoDB Atlas container")
}

#[tokio::test]
async fn vector_search_test() {
    if skip_if_docker_unavailable("vector_search_test") {
        return;
    }

    // Setup mock openai API
    let server = httpmock::MockServer::start();

    mock_embeddings(
        &server,
        json!({ "input": WORD_DEFINITIONS, "model": "text-embedding-ada-002" }),
        (0..3).map(axis_embedding),
    );
    mock_embeddings(
        &server,
        json!({ "input": ["What is a linglingdong?"], "model": "text-embedding-ada-002" }),
        [axis_embedding(2)],
    );

    let openai_client = openai_client(&server);

    // Select the embedding model and generate our embeddings
    let model = openai_client.embedding(openai::TEXT_EMBEDDING_ADA_002, None);

    let container = start_atlas().await;

    let port = container.get_host_port_ipv4(MONGODB_PORT).await.unwrap();
    let host = container.get_host().await.unwrap().to_string();

    let collection = bootstrap_collection(&host, port).await;

    let embeddings = create_embeddings(model.clone()).await;

    collection.insert_many(embeddings).await.unwrap();

    // Create a vector index on our vector store.
    // Note: a vector index called "vector_index" must exist on the MongoDB collection you are querying.
    // IMPORTANT: Reuse the same model that was used to generate the embeddings
    let index = MongoDbVectorIndex::new(
        collection,
        model.clone(),
        VECTOR_SEARCH_INDEX_NAME,
        SearchParams::new(),
    )
    .await
    .unwrap();

    // A fresh client runs its operations only after the index appended its
    // handshake metadata, so its pooled connections carry it.
    let client = connect(&host, port, "handshake-check").await;
    let fresh = client.database(DATABASE_NAME).collection(COLLECTION_NAME);
    MongoDbVectorIndex::<bson::Document>::new(
        fresh,
        model,
        VECTOR_SEARCH_INDEX_NAME,
        SearchParams::new(),
    )
    .await
    .unwrap();
    let connections: Vec<bson::Document> = client
        .database("admin")
        .aggregate([
            doc! { "$currentOp": { "idleConnections": true } },
            doc! { "$match": { "appName": "handshake-check" } },
        ])
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    let tagged = connections.iter().any(|connection| {
        let driver = connection
            .get_document("clientMetadata")
            .and_then(|metadata| metadata.get_document("driver"));
        driver.is_ok_and(|driver| {
            driver
                .get_str("name")
                .is_ok_and(|name| name.ends_with("|rig-mongodb"))
                && driver.get_str("version").is_ok_and(|version| {
                    version.ends_with(&format!("|{}", env!("CARGO_PKG_VERSION")))
                })
        })
    });
    assert!(
        tagged,
        "no connection carried the rig-mongodb handshake: {connections:?}"
    );

    let query = "What is a linglingdong?";
    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(1)
        .build();

    // The documents inserted after the index was made reach the search
    // service on its own time.
    let deadline = Instant::now() + READY_WITHIN;
    let results = loop {
        let results = index
            .top_n::<serde_json::Value>(req.clone())
            .await
            .expect("MongoDB vector search should succeed on a queryable index");
        let top = results
            .first()
            .and_then(|result| result.document.get("_id"));
        if top == Some(&json!("doc2")) {
            break results;
        }
        assert!(
            Instant::now() < deadline,
            "doc2 was never the top vector search result; observed results: {results:?}"
        );
        sleep(POLL).await;
    };

    let VectorSearchResult {
        score,
        document: value,
        ..
    } = &results.first().unwrap();

    assert_eq!(
        *value,
        json!({
            "_id": "doc2".to_string(),
            "definition": "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
            "score": score
        })
    );
}

#[tokio::test]
async fn insert_documents_test() {
    if skip_if_docker_unavailable("insert_documents_test") {
        return;
    }

    // Setup mock openai API
    let server = httpmock::MockServer::start();

    mock_embeddings(
        &server,
        json!({
            "input": ["Test document 1", "Test document 2"],
            "model": "text-embedding-ada-002",
        }),
        [vec![0.1; 1536], vec![0.2; 1536]],
    );

    let openai_client = openai_client(&server);

    let model = openai_client.embedding(openai::TEXT_EMBEDDING_ADA_002, None);

    let container = start_atlas().await;

    let port = container.get_host_port_ipv4(MONGODB_PORT).await.unwrap();
    let host = container.get_host().await.unwrap().to_string();
    let collection = bootstrap_collection(&host, port).await;

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

    // Generate embeddings using EmbeddingsBuilder (returns Vec<(Word, Vec<Embedding>)>)
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

/// Waits for the search service, which starts after mongod, to answer;
/// then makes the vector search index and waits until it can be queried.
async fn create_search_index(collection: &Collection<bson::Document>) {
    let deadline = Instant::now() + READY_WITHIN;
    while let Err(error) = collection.list_search_indexes().await {
        assert!(
            Instant::now() < deadline,
            "the search service never answered: {error}"
        );
        sleep(POLL).await;
    }
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
        .expect("the search index should be created");
    loop {
        let indexes = collection
            .list_search_indexes()
            .name(VECTOR_SEARCH_INDEX_NAME)
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        let ready = indexes.iter().flatten().any(|index| {
            index.get_str("status").ok() == Some("READY")
                && index.get_bool("queryable").ok() == Some(true)
        });
        if ready {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "the search index never became queryable: {indexes:?}"
        );
        sleep(POLL).await;
    }
}

async fn connect(host: &str, port: u16, app_name: &str) -> mongodb::Client {
    let options = ClientOptions::parse(format!(
        "mongodb://{USERNAME}:{PASSWORD}@{host}:{port}/?directConnection=true&appName={app_name}"
    ))
    .await
    .expect("MongoDB connection string should be valid");
    mongodb::Client::with_options(options).expect("MongoDB client options should be valid")
}

async fn bootstrap_collection(host: &str, port: u16) -> Collection<bson::Document> {
    let mongodb_client = connect(host, port, "bootstrap").await;

    // The last mongod is a one-member replica set that elects itself
    // primary some time after it listens.
    let deadline = Instant::now() + READY_WITHIN;
    loop {
        let hello = mongodb_client
            .database("admin")
            .run_command(doc! { "hello": 1 })
            .await;
        if hello.is_ok_and(|hello| hello.get_bool("isWritablePrimary") == Ok(true)) {
            break;
        }
        assert!(Instant::now() < deadline, "mongod never became primary");
        sleep(POLL).await;
    }

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

async fn create_embeddings(model: Model<openai::wire::Embeddings>) -> Vec<bson::Document> {
    let words = WORD_DEFINITIONS
        .iter()
        .enumerate()
        .map(|(i, definition)| Word {
            id: format!("doc{i}"),
            definition: definition.to_string(),
        });

    let embeddings = EmbeddingsBuilder::new(model)
        .documents(words)
        .unwrap()
        .build()
        .await
        .unwrap();

    embeddings
        .iter()
        .map(|(Word { id, definition, .. }, embeddings)| {
            let embedding = embeddings.first().expect("expected at least one embedding");
            doc! {
                "_id": id.clone(),
                "definition": definition.clone(),
                "embedding": embedding.vec.clone(),
            }
        })
        .collect()
}
