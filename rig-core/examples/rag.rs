use std::{env, vec};

use rig::{
    completion::Prompt,
    embeddings::EmbeddingsBuilder,
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::in_memory_store::InMemoryVectorStore,
    Embed,
};
use serde::Serialize;

// Data to be RAGged.
// A vector search needs to be performed on the `definitions` field, so we derive the `Embed` trait for `WordDefinition`
// and tag that field with `#[embed]`.
#[derive(Embed, Serialize, Clone, Debug, Eq, PartialEq, Default)]
struct WordDefinition {
    id: String,
    word: String,
    #[embed]
    definitions: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    // Create OpenAI client
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);

    let embedding_model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    // Generate embeddings for the definitions of all the documents using the specified embedding model.
    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .documents(vec![
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
            },
        ])?
        .build()
        .await?;

    // Create vector store with the embeddings
    let vector_store = InMemoryVectorStore::from_documents(embeddings);

    // Create vector store index
    let index = vector_store.index(embedding_model);

    let rag_agent = openai_client.agent("gpt-4")
        .preamble("
            You are a dictionary assistant here to assist the user in understanding the meaning of words.
            You will find additional non-standard word definitions that could be useful below.
        ")
        .dynamic_context(1, index)
        .build();

    // Prompt the agent and print the response
    let response = rag_agent.prompt("What does \"glarb-glarb\" mean?").await?;

    println!("{}", response);

    Ok(())
}
