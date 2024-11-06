use std::{env, vec};

use rig::{
    completion::Prompt,
    embeddings::EmbeddingsBuilder,
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::in_memory_store::InMemoryVectorStore,
    Embed,
};
use serde::Serialize;

// Shape of data that needs to be RAG'ed.
// A vector search needs to be performed on the definitions, so we derive the `Embed` trait for `FakeDefinition`
// and tag that field with `#[embed]`.
#[derive(Embed, Serialize, Clone, Debug, Eq, PartialEq, Default)]
struct FakeDefinition {
    id: String,
    #[embed]
    definitions: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);

    let embedding_model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    // Generate embeddings for the definitions of all the documents using the specified embedding model.
    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .documents(vec![
            FakeDefinition {
                id: "doc0".to_string(),
                definitions: vec![
                    "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets.".to_string(),
                    "Definition of a *flurbo*: A fictional digital currency that originated in the animated series Rick and Morty.".to_string()
                ]
            },
            FakeDefinition {
                id: "doc1".to_string(),
                definitions: vec![
                    "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
                    "Definition of a *glarb-glarb*: A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy.".to_string()
                ]
            },
            FakeDefinition {
                id: "doc2".to_string(),
                definitions: vec![
                    "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
                    "Definition of a *linglingdong*: A rare, mystical instrument crafted by the ancient monks of the Nebulon Mountain Ranges on the planet Quarm.".to_string()
                ]
            },
        ])?
        .build()
        .await?;

    let index = InMemoryVectorStore::default()
        .add_documents_with_id(embeddings, |definition| definition.id.clone())?
        .index(embedding_model);

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
