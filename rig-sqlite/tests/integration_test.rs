use rig::vector_store::request::VectorSearchRequest;
use serde_json::json;

use rig::client::EmbeddingsClient;
use rig::vector_store::VectorStoreIndex;
use rig::{
    Embed, OneOrMany,
    embeddings::{Embedding, EmbeddingsBuilder},
    providers::openai,
};
use rig_sqlite::{Column, ColumnValue, SqliteVectorStore, SqliteVectorStoreTable};
use rusqlite::ffi::{sqlite3, sqlite3_api_routines, sqlite3_auto_extension};
use sqlite_vec::sqlite3_vec_init;
use tokio_rusqlite::Connection;

#[derive(Embed, Clone, serde::Deserialize, Debug)]
struct Word {
    id: String,
    #[embed]
    definition: String,
}

impl SqliteVectorStoreTable for Word {
    fn name() -> &'static str {
        "documents"
    }

    fn schema() -> Vec<Column> {
        vec![
            Column::new("id", "TEXT PRIMARY KEY"),
            Column::new("definition", "TEXT"),
        ]
    }

    fn id(&self) -> String {
        self.id.clone()
    }

    fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)> {
        vec![
            ("id", Box::new(self.id.clone())),
            ("definition", Box::new(self.definition.clone())),
        ]
    }
}

type SqliteExtensionFn =
    unsafe extern "C" fn(*mut sqlite3, *mut *mut i8, *const sqlite3_api_routines) -> i32;

#[tokio::test]
async fn vector_search_test() {
    // Initialize the `sqlite-vec`extension
    // See: https://alexgarcia.xyz/sqlite-vec/rust.html
    unsafe {
        sqlite3_auto_extension(Some(std::mem::transmute::<*const (), SqliteExtensionFn>(
            sqlite3_vec_init as *const (),
        )));
    }

    // Initialize SQLite connection
    let conn = Connection::open("vector_store.db")
        .await
        .expect("Could not initialize SQLite connection");

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
                    "embedding": vec![-0.001; 1536],
                    "index": 0
                  },
                  {
                    "object": "embedding",
                    "embedding": vec![0.0023064255; 1536],
                    "index": 1
                  },
                  {
                    "object": "embedding",
                    "embedding": vec![-0.001; 1536],
                    "index": 2
                  },
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
                    "What is a glarb?",
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
                        "embedding": vec![0.0024064254; 1536],
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

    let embeddings = create_embeddings(model.clone()).await;

    // Initialize SQLite vector store
    let vector_store = SqliteVectorStore::new(conn, &model)
        .await
        .expect("Could not initialize SQLite vector store");

    // Add embeddings to vector store
    vector_store
        .add_rows(embeddings)
        .await
        .expect("Could not add embeddings to vector store");

    // Create a vector index on our vector store
    let index = vector_store.index(model);
    let query = "What is a glarb?";
    let samples = 1;
    let req = VectorSearchRequest::builder()
        .samples(samples)
        .query(query)
        .build()
        .expect("VectorSearchRequest should not fail to build here");

    // Query the index
    let results = index.top_n::<serde_json::Value>(req).await.expect("");

    let (_, _, value) = &results.first().expect("");

    assert_eq!(
        value,
        &serde_json::json!({
            "id": "doc1",
            "definition": "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
        })
    )
}

async fn create_embeddings(model: openai::EmbeddingModel) -> Vec<(Word, OneOrMany<Embedding>)> {
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

    EmbeddingsBuilder::new(model)
        .documents(words)
        .expect("")
        .build()
        .await
        .expect("")
}
