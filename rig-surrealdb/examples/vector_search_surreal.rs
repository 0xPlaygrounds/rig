use rig::{embeddings::EmbeddingsBuilder, vector_store::VectorStoreIndex, Embed};
use rig_surrealdb::{SurrealVectorStore, Ws};
use serde::{Deserialize, Serialize};
use surrealdb::{opt::auth::Root, Surreal};

// A vector search needs to be performed on the `definitions` field, so we derive the `Embed` trait for `WordDefinition`
// and tag that field with `#[embed]`.
// We are not going to store the definitions on our database so we skip the `Serialize` trait
#[derive(Embed, Serialize, Deserialize, Clone, Debug, Eq, PartialEq, Default)]
struct WordDefinition {
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

    let surreal = Surreal::new::<Ws>("localhost:9999").await?;

    surreal
        .signin(Root {
            username: "root",
            password: "root",
        })
        .await?;

    surreal.use_ns("example").use_db("example").await?;

    surreal
        .query(include_str!("./migrations.surql"))
        .await
        .unwrap();

    // create test documents with mocked embeddings
    let words = vec![
        WordDefinition {
            word: "flurbo".to_string(),
            definitions: vec![
                "1. *flurbo* (name): A flurbo is a green alien that lives on cold planets.".to_string(),
                "2. *flurbo* (name): A fictional digital currency that originated in the animated series Rick and Morty.".to_string()
            ]
        },
        WordDefinition {

            word: "glarb-glarb".to_string(),
            definitions: vec![
                "1. *glarb-glarb* (noun): A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
                "2. *glarb-glarb* (noun): A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy.".to_string()
            ]
        },
        WordDefinition {

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

    // init vector store
    let vector_store = SurrealVectorStore::with_defaults(model, surreal);
    vector_store.insert_documents(documents).await?;

    // query vector
    let query = "What does \"glarb-glarb\" mean?";

    let results = vector_store.top_n::<WordDefinition>(query, 2).await?;

    println!("#{} results for query: {}", results.len(), query);
    for (distance, _id, doc) in results.iter() {
        println!("Result distance {} for word: {}", distance, doc);

        // expected output (even if we have 2 entries on glarb-glarb the index only gives closest match)
        // Result distance 0.2988549857990437 for word: glarb-glarb
        //Result distance 0.7072261746390949 for word: linglingdong
    }

    Ok(())
}
