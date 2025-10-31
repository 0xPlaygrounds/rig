use std::vec;

use rig::client::{CompletionClient, EmbeddingsClient, ProviderClient};
use rig::{
    Embed, completion::Prompt, embeddings::EmbeddingsBuilder,
    vector_store::in_memory_store::InMemoryVectorStore,
};
use rig_bedrock::client::Client;
use rig_bedrock::{completion::AMAZON_NOVA_LITE, embedding::AMAZON_TITAN_EMBED_TEXT_V2_0};
use serde::Serialize;
use tracing::info;

// Data to be RAG-ed.
// A vector search needs to be performed on the `definitions` field, so we derive the `Embed` trait for `WordDefinition`
// and tag that field with `#[embed]`.
#[derive(rig_derive::Embed, Serialize, Clone, Debug, Eq, PartialEq, Default)]
struct WordDefinition {
    id: String,
    word: String,
    #[embed]
    definitions: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    let client = Client::from_env();
    let embedding_model = client.embedding_model_with_ndims(AMAZON_TITAN_EMBED_TEXT_V2_0, 256);

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

    let rag_agent = client.agent(AMAZON_NOVA_LITE)
        .preamble("
            You are a dictionary assistant here to assist the user in understanding the meaning of words.
            You will find additional non-standard word definitions that could be useful below.
        ")
        .dynamic_context(1, index)
        .build();

    // Prompt the agent and print the response
    let response = rag_agent.prompt("What does \"glarb-glarb\" mean?").await?;

    info!("{}", response);

    Ok(())
}
