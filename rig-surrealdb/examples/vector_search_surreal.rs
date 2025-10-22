use rig::client::EmbeddingsClient;
use rig::vector_store::request::VectorSearchRequest;
use rig::{
    Embed,
    embeddings::EmbeddingsBuilder,
    vector_store::{InsertDocuments, VectorStoreIndex},
};
use rig_surrealdb::{Mem, SurrealVectorStore};
use serde::{Deserialize, Serialize};
use surrealdb::Surreal;

// A vector search needs to be performed on the `definitions` field, so we derive the `Embed` trait for `WordDefinition`
// and tag that field with `#[embed]`.
// We are not going to store the definitions on our database so we skip the `Serialize` trait
#[derive(Embed, Serialize, Deserialize, Clone, Debug, Eq, PartialEq, Default)]
struct WordDefinition {
    word: String,
    #[serde(skip)] // we don't want to serialize this field, we use only to create embeddings
    #[embed]
    definition: String,
}

impl std::fmt::Display for WordDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.word)
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_client = rig::providers::openai::Client::from_env();
    let model = openai_client.embedding_model(rig::providers::openai::TEXT_EMBEDDING_3_SMALL);

    let surreal = Surreal::new::<Mem>(()).await?;

    surreal.use_ns("example").use_db("example").await?;

    // create test documents with mocked embeddings
    let words = vec![
        WordDefinition {
            word: "flurbo".to_string(),
            definition: "1. *flurbo* (name): A fictional digital currency that originated in the animated series Rick and Morty.".to_string()
        },
        WordDefinition {
            word: "glarb-glarb".to_string(),
            definition: "1. *glarb-glarb* (noun): A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy.".to_string()
        },
        WordDefinition {
            word: "linglingdong".to_string(),
            definition: "1. *linglingdong* (noun): A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
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
    println!("Attempting vector search with query: {query}");

    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(2)
        .build()?;

    let results = vector_store.top_n::<WordDefinition>(req).await?;

    println!("{} results for query: {}", results.len(), query);
    for (distance, _id, doc) in results.iter() {
        println!("Result distance {distance} for word: {doc}");

        // expected output
        // Result distance 0.693218142100547 for word: glarb-glarb
        // Result distance 0.2529120980283861 for word: linglingdong
    }

    println!("Attempting vector search with cosine similarity threshold of 0.5 and query: {query}");
    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(2)
        .threshold(0.5)
        .build()?;

    let results = vector_store.top_n::<WordDefinition>(req).await?;

    println!("{} results for query: {}", results.len(), query);
    assert_eq!(results.len(), 1);

    for (distance, _id, doc) in results.iter() {
        println!("Result distance {distance} for word: {doc}");
    }

    Ok(())
}
