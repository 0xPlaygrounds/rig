use std::env;

use rig::{
    completion::Prompt,
    embeddings::EmbeddingsBuilder,
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::{in_memory_store::InMemoryVectorStore, VectorStore},
};
use rig_derive::Embed;
use serde::Serialize;

#[derive(Embed, Clone, Serialize, Eq, PartialEq, Default)]
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

    // Create vector store, compute embeddings and load them in the store
    let mut vector_store = InMemoryVectorStore::default();

    let fake_definitions = vec![
        FakeDefinition {
            id: "doc0".to_string(),
            definitions: vec![
                "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets".to_string(),
                "Definition of a *flurbo*: A unit of currency used in a bizarre or fantastical world, often associated with eccentric societies or sci-fi settings.".to_string()
            ]
        },
        FakeDefinition {
            id: "doc1".to_string(),
            definitions: vec![
                "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
                "Definition of a *glarb-glarb*: A mysterious, bubbling substance often found in swamps, alien planets, or under mysterious circumstances.".to_string()
            ]
        },
        FakeDefinition {
            id: "doc2".to_string(),
            definitions: vec![
                "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string()
            ]
        }
    ];

    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .documents(fake_definitions)
        .build()
        .await?;

    vector_store
        .add_documents(
            embeddings
                .into_iter()
                .enumerate()
                .map(|(i, (fake_definition, embeddings))| {
                    (format!("doc{i}"), fake_definition, embeddings)
                })
                .collect(),
        )
        .await?;

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
