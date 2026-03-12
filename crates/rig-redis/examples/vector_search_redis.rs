use rig::client::ProviderClient;
use rig::vector_store::InsertDocuments;
use rig::vector_store::request::VectorSearchRequest;
use rig::{
    Embed, client::EmbeddingsClient, embeddings::EmbeddingsBuilder, vector_store::VectorStoreIndex,
};
use serde::{Deserialize, Serialize};

#[derive(Embed, Serialize, Deserialize, Clone, Debug, Eq, PartialEq, Default)]
struct WordDefinition {
    word: String,
    #[serde(skip)]
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

    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
    let redis_client = redis::Client::open(redis_url)?;

    let vector_store = rig_redis::RedisVectorStore::new(
        model.clone(),
        redis_client,
        "word_idx".to_string(),
        "embedding".to_string(),
    );

    // Create test documents with embeddings
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
        }
    ];

    let documents = EmbeddingsBuilder::new(model.clone())
        .documents(words)
        .unwrap()
        .build()
        .await
        .expect("Failed to create embeddings");

    vector_store.insert_documents(documents).await?;

    // Query vector store
    let query = "What does \"glarb-glarb\" mean?";

    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(2)
        .build()
        .expect("VectorSearchRequest should not fail to build here");

    let results = vector_store.top_n::<WordDefinition>(req).await?;

    println!("#{} results for query: {}", results.len(), query);
    for (score, _id, doc) in results.iter() {
        println!("Result score {score} for word: {doc}");
    }

    Ok(())
}
