//! SQLite (`sqlite-vec`) vector search over precomputed embeddings.
//!
//! Embeddings are produced by `openai::functions::embed` — a free function
//! over a plain `EmbeddingConfig` plus a shared `HttpRuntime` — and batched
//! with `embed_documents`, the replacement for the retired
//! `EmbeddingsBuilder`. Because there is no embedding-model object to ask
//! for its dimensionality, the store is told the vector width directly.

use rig_core::Embed;
use rig_core::OneOrMany;
use rig_core::embeddings::{default_concurrency, embed_documents};
use rig_core::http_runtime::HttpRuntime;
use rig_core::providers::openai;
use rig_core::vector_store::request::VectorSearchRequest;
use rig_sqlite::{
    Column, ColumnValue, SqliteDistanceMetric, SqliteVectorStore, SqliteVectorStoreTable,
};
use rusqlite::ffi::{sqlite3, sqlite3_api_routines, sqlite3_auto_extension};
use serde::{Deserialize, Serialize};
use sqlite_vec::sqlite3_vec_init;
use std::os::raw::c_char;
use tokio_rusqlite::Connection;

#[derive(Embed, Clone, Debug, Deserialize, Serialize)]
struct Document {
    id: String,
    #[embed]
    content: String,
}

impl SqliteVectorStoreTable for Document {
    fn name() -> &'static str {
        "documents"
    }

    fn schema() -> Vec<Column> {
        vec![
            Column::new("id", "TEXT PRIMARY KEY"),
            Column::new("content", "TEXT"),
        ]
    }

    fn id(&self) -> String {
        self.id.clone()
    }

    fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)> {
        vec![
            ("id", Box::new(self.id.clone())),
            ("content", Box::new(self.content.clone())),
        ]
    }
}

type SqliteExtensionFn =
    unsafe extern "C" fn(*mut sqlite3, *mut *mut c_char, *const sqlite3_api_routines) -> i32;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::DEBUG.into()),
        )
        .init();

    // Initialize the `sqlite-vec`extension
    // See: https://alexgarcia.xyz/sqlite-vec/rust.html
    unsafe {
        sqlite3_auto_extension(Some(std::mem::transmute::<*const (), SqliteExtensionFn>(
            sqlite3_vec_init as *const (),
        )));
    }

    // Initialize SQLite connection
    let conn = Connection::open("vector_store.db").await?;

    // Embedding configuration is plain data plus a shared HTTP runtime.
    let embed_cfg = openai::functions::EmbeddingConfig::from_env(openai::TEXT_EMBEDDING_ADA_002)?;
    // The config knows its model's native width; no need to restate it.
    let embedding_dims = embed_cfg
        .ndims()
        .ok_or_else(|| anyhow::anyhow!("text-embedding-ada-002 has a known vector width"))?;
    let rt = HttpRuntime::new();
    let max_documents = openai::functions::DESCRIPTOR
        .max_embedding_documents
        .unwrap_or(usize::MAX);

    let documents = vec![
        Document {
            id: "doc0".to_string(),
            content: "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets".to_string(),
        },
        Document {
            id: "doc1".to_string(),
            content: "Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
        },
        Document {
            id: "doc2".to_string(),
            content: "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
        },
    ];

    let embeddings = embed_documents(
        documents,
        max_documents,
        default_concurrency(max_documents),
        |texts| openai::functions::embed(&embed_cfg, &rt, texts),
    )
    .await?;

    // Initialize SQLite vector store. Queries arrive pre-embedded, so the
    // store only needs the embedding dimensionality.
    let vector_store: SqliteVectorStore<Document> =
        SqliteVectorStore::with_distance_metric(conn, embedding_dims, SqliteDistanceMetric::Cosine)
            .await?;

    // Add precomputed embeddings to the vector store. The row's own `id`
    // column identifies each stored document.
    vector_store
        .insert_as(
            embeddings
                .into_iter()
                .map(|(doc, embeddings)| (doc.id.clone(), doc, embeddings))
                .collect(),
        )
        .await?;

    // Embed the query, then send the pre-embedded request
    let query = "What is a linglingdong?";
    let query_embedding = openai::functions::embed(&embed_cfg, &rt, vec![query.to_string()])
        .await?
        .embeddings
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no embedding returned for the query"))?;
    let samples = 1;
    let req = VectorSearchRequest::new(OneOrMany::one(query_embedding), samples);

    // Query the store
    let results = vector_store
        .top_n_as::<Document>(req.clone())
        .await?
        .into_iter()
        .collect::<Vec<_>>();

    println!("Results: {results:?}");

    let id_results = vector_store
        .top_n_ids(req)
        .await?
        .into_iter()
        .collect::<Vec<_>>();

    println!("ID results: {id_results:?}");

    Ok(())
}
