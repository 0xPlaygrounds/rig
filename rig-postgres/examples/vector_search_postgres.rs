use rig::{
    embeddings::EmbeddingsBuilder,
    providers::openai::{Client, TEXT_EMBEDDING_3_SMALL},
    vector_store::VectorStoreIndex,
    Embed,
};
use rig_postgres::{Column, PostgresVectorIndex, PostgresVectorStore, PostgresVectorStoreTable};
use serde::Deserialize;
use tokio_postgres::types::ToSql;

#[derive(Clone, Debug, Deserialize, Embed)]
pub struct Document {
    id: String,
    #[embed]
    content: String,
}

impl PostgresVectorStoreTable for Document {
    fn name() -> &'static str {
        "documents"
    }

    fn schema() -> Vec<Column> {
        vec![
            Column::new("id", "TEXT PRIMARY KEY"),
            Column::new("content", "TEXT"),
        ]
    }

    fn column_values(&self) -> Vec<(&'static str, Box<dyn ToSql + Sync>)> {
        vec![
            ("id", Box::new(self.id.clone())),
            ("content", Box::new(self.content.clone())),
        ]
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // tracing_subscriber::fmt().with_env_filter(
    //     tracing_subscriber::EnvFilter::from_default_env()
    //         .add_directive(tracing::Level::DEBUG.into())
    //         .add_directive("hyper=off".parse().unwrap())
    // ).init();

    // set up postgres connection
    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL not set");
    let db_config: tokio_postgres::Config = database_url.parse()?;
    let (psql, connection) = db_config.connect(tokio_postgres::NoTls).await?;

    tokio::spawn(async move {
        if let Err(e) = connection.await {
            tracing::error!("Connection error: {}", e);
        }
    });

    // set up embedding model
    let openai_api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai = Client::new(&openai_api_key);
    let model = openai.embedding_model(TEXT_EMBEDDING_3_SMALL);

    // generate embeddings
    let documents: Vec<Document> = vec![
        "The Mediterranean diet emphasizes fish, olive oil, and vegetables, believed to reduce chronic diseases.",
        "Photosynthesis in plants converts light energy into glucose and produces essential oxygen.",
        "20th-century innovations, from radios to smartphones, centered on electronic advancements.",
    ].into_iter().map(|content| Document {
        id: uuid::Uuid::new_v4().to_string(),
        content: content.to_string(),
    }).collect();
    let embeddings = EmbeddingsBuilder::new(model.clone())
        .documents(documents)?
        .build()
        .await?;

    // add embeddings to store
    let store = PostgresVectorStore::new(psql, &model).await?;
    store.add_rows(embeddings).await?;

    // query the index
    let index = PostgresVectorIndex::new(model, store);
    let results = index.top_n::<Document>("What is photosynthesis", 1).await?;
    println!("top_n results: \n{:?}", results);

    let ids = index.top_n_ids("What is photosynthesis?", 1).await?;
    println!("top_n_ids results:\n{:?}", ids);

    Ok(())
}
