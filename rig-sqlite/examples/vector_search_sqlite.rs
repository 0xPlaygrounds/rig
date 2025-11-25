use rig::client::{EmbeddingsClient, ProviderClient};
use rig::providers::openai;
use rig::vector_store::request::VectorSearchRequest;
use rig::{
    Embed, embeddings::EmbeddingsBuilder, providers::openai::Client, vector_store::VectorStoreIndex,
};
use rig_sqlite::{Column, ColumnValue, SqliteVectorStore, SqliteVectorStoreTable};
use rusqlite::ffi::{sqlite3, sqlite3_api_routines, sqlite3_auto_extension};
use serde::Deserialize;
use sqlite_vec::sqlite3_vec_init;
use tokio_rusqlite::Connection;

#[derive(Embed, Clone, Debug, Deserialize)]
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
    unsafe extern "C" fn(*mut sqlite3, *mut *mut i8, *const sqlite3_api_routines) -> i32;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::DEBUG.into()),
        )
        .init();

    // Initialize OpenAI client
    let openai_client = Client::from_env();

    // Initialize the `sqlite-vec`extension
    // See: https://alexgarcia.xyz/sqlite-vec/rust.html
    unsafe {
        sqlite3_auto_extension(Some(std::mem::transmute::<*const (), SqliteExtensionFn>(
            sqlite3_vec_init as *const (),
        )));
    }

    // Initialize SQLite connection
    let conn = Connection::open("vector_store.db").await?;

    // Select the embedding model and generate our embeddings
    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    let documents = vec![
        Document {
            id: "doc0".to_string(),
            content: "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets".to_string(),
        },
        Document {
            id: "doc1".to_string(),
            content: "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
        },
        Document {
            id: "doc2".to_string(),
            content: "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
        },
    ];

    let embeddings = EmbeddingsBuilder::new(model.clone())
        .documents(documents)?
        .build()
        .await?;

    // Initialize SQLite vector store
    let vector_store = SqliteVectorStore::new(conn, &model).await?;

    // Add embeddings to vector store
    vector_store.add_rows(embeddings).await?;

    // Create a vector index on our vector store
    let index = vector_store.index(model);

    let query = "What is a linglingdong?";
    let samples = 1;
    let req = VectorSearchRequest::builder()
        .samples(samples)
        .query(query)
        .build()?;

    // Query the index
    let results = index
        .top_n::<Document>(req.clone())
        .await?
        .into_iter()
        .collect::<Vec<_>>();

    println!("Results: {results:?}");

    let id_results = index.top_n_ids(req).await?.into_iter().collect::<Vec<_>>();

    println!("ID results: {id_results:?}");

    Ok(())
}
