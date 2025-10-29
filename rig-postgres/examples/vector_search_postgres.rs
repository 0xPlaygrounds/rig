use rig::client::EmbeddingsClient;
use rig::vector_store::request::VectorSearchRequest;
use rig::{
    Embed,
    embeddings::EmbeddingsBuilder,
    vector_store::{InsertDocuments, VectorStoreIndex},
};
use rig_postgres::PostgresVectorStore;
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPoolOptions;

// A vector search needs to be performed on the `definitions` field, so we derive the `Embed` trait for `WordDefinition`
// and tag that field with `#[embed]`.
// We are not going to store the definitions on our database so we skip the `Serialize` trait
#[derive(Embed, Serialize, Deserialize, Clone, Debug, Eq, PartialEq, Default)]
struct WordDefinition {
    id: String,
    word: String,
    #[serde(skip)] // we don't want to serialize this field, we use only to create embeddings
    #[embed]
    definitions: Vec<String>,
}

impl std::fmt::Display for WordDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.word)
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // load environment variables from .env file
    dotenvy::dotenv().ok();

    // Create OpenAI client
    let openai_client = rig::providers::openai::Client::from_env();
    let model = openai_client.embedding_model(rig::providers::openai::TEXT_EMBEDDING_3_SMALL);

    // setup Postgres
    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL not set");
    let pool = PgPoolOptions::new()
        .max_connections(50)
        .idle_timeout(std::time::Duration::from_secs(5))
        .connect(&database_url)
        .await
        .expect("Failed to create postgres pool");

    // make sure database is setup
    sqlx::migrate!("./examples/migrations").run(&pool).await?;

    // create test documents with mocked embeddings
    let words = vec![
        WordDefinition {
            id: "doc0".to_string(),
            word: "flurbo".to_string(),
            definitions: vec![
                "1. *flurbo* (name): A flurbo is a green alien that lives on cold planets.".to_string(),
                "2. *flurbo* (name): A fictional digital currency that originated in the animated series Rick and Morty.".to_string()
            ]
        },
        WordDefinition {
            id: "doc1".to_string(),
            word: "glarb-glarb".to_string(),
            definitions: vec![
                "1. *glarb-glarb* (noun): A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
                "2. *glarb-glarb* (noun): A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy.".to_string()
            ]
        },
        WordDefinition {
            id: "doc2".to_string(),
            word: "linglingdong".to_string(),
            definitions: vec![
                "1. *linglingdong* (noun): A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
                "2. *linglingdong* (noun): A rare, mystical instrument crafted by the ancient monks of the Nebulon Mountain Ranges on the planet Quarm.".to_string()
            ]
        }];

    let documents = EmbeddingsBuilder::new(model.clone())
        .documents(words)
        .unwrap()
        .build()
        .await
        .expect("Failed to create embeddings");

    // delete documents from table to have a clean start (optional, not recommended for production)
    sqlx::query("TRUNCATE documents").execute(&pool).await?;

    // init vector store
    let vector_store = PostgresVectorStore::with_defaults(model, pool);
    vector_store.insert_documents(documents).await?;

    // query vector
    let query = "What does \"glarb-glarb\" mean?";
    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(2)
        .build()
        .expect("VectorSearchRequest should not fail to build here");

    let results = vector_store.top_n::<WordDefinition>(req).await?;

    println!("#{} results for query: {}", results.len(), query);
    for (distance, _id, doc) in results.iter() {
        println!("Result distance {distance} for word: {doc}");

        // expected output (even if we have 2 entries on glarb-glarb the index only gives closest match)
        // Result distance 0.2988549857990437 for word: glarb-glarb
        //Result distance 0.7072261746390949 for word: linglingdong
    }

    Ok(())
}
