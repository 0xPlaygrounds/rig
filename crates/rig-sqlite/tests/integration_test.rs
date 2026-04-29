#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

use rig::vector_store::request::{SearchFilter, VectorSearchRequest};
use serde_json::json;

use rig::client::EmbeddingsClient;
use rig::vector_store::{InsertDocuments, VectorStoreIndex};
use rig::{
    Embed, OneOrMany,
    embeddings::{Embedding, EmbeddingsBuilder},
    providers::openai,
};
use rig_sqlite::{
    Column, ColumnValue, SqliteSearchFilter, SqliteVectorStore, SqliteVectorStoreTable,
};
use rusqlite::ffi::{sqlite3, sqlite3_api_routines, sqlite3_auto_extension};
use sqlite_vec::sqlite3_vec_init;
use tokio_rusqlite::Connection;

#[derive(Embed, Clone, serde::Deserialize, serde::Serialize, Debug)]
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

fn register_sqlite_vec_extension() {
    // Initialize the `sqlite-vec`extension
    // See: https://alexgarcia.xyz/sqlite-vec/rust.html

    unsafe {
        sqlite3_auto_extension(Some(std::mem::transmute::<*const (), SqliteExtensionFn>(
            sqlite3_vec_init as *const (),
        )));
    }
}

async fn open_test_connection(name: &str) -> Connection {
    Connection::open(format!("file:{name}?mode=memory"))
        .await
        .expect("Could not initialize SQLite connection")
}

#[tokio::test]
async fn vector_search_test() {
    register_sqlite_vec_extension();

    let conn = open_test_connection("vector_search_test").await;
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

    let openai_client = openai::Client::builder()
        .api_key("TEST")
        .base_url(server.base_url())
        .build()
        .unwrap();
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
        .filter(SqliteSearchFilter::eq("id", "doc1".into()).not())
        .build();

    // Query the index
    let results = index.top_n::<serde_json::Value>(req).await.expect("");
    assert!(results.is_empty());
}

#[tokio::test]
async fn insert_documents_test() {
    register_sqlite_vec_extension();

    let conn = open_test_connection("insert_documents_test").await;
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

    let openai_client = openai::Client::builder()
        .api_key("TEST")
        .base_url(server.base_url())
        .build()
        .unwrap();
    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);
    let embeddings = create_embeddings(model.clone()).await;

    let vector_store: SqliteVectorStore<_, Word> = SqliteVectorStore::new(conn.clone(), &model)
        .await
        .expect("Could not initialize SQLite vector store");

    vector_store
        .insert_documents(embeddings)
        .await
        .expect("Could not add embeddings to vector store");

    let (document_count, embedding_count) = conn
        .call(|conn| {
            let document_count: i64 =
                conn.query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))?;
            let embedding_count: i64 =
                conn.query_row("SELECT COUNT(*) FROM documents_embeddings", [], |row| {
                    row.get(0)
                })?;

            Ok((document_count, embedding_count))
        })
        .await
        .expect("Could not verify inserted rows");

    assert_eq!(document_count, 3);
    assert_eq!(embedding_count, 3);
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
