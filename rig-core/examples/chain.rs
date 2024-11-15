use std::env;

use rig::{
    chain::{self, Chain},
    embeddings::EmbeddingsBuilder,
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::{in_memory_store::InMemoryVectorStore, VectorStore},
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);

    let embedding_model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    // Create vector store, compute embeddings and load them in the store
    let mut vector_store = InMemoryVectorStore::default();

    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .simple_document("doc0", "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets")
        .simple_document("doc1", "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.")
        .simple_document("doc2", "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.")
        .build()
        .await?;

    vector_store.add_documents(embeddings).await?;

    // Create vector store index
    let index = vector_store.index(embedding_model);

    let agent = openai_client.agent("gpt-4")
        .preamble("
            You are a dictionary assistant here to assist the user in understanding the meaning of words.
        ")
        .build();

    let chain = chain::new()
        // Retrieve top document from the index and return it with the prompt
        .lookup(index, 2)
        // Format the prompt with the context documents
        .map(|(query, docs): (_, Vec<String>)| {
            format!(
                "User question: {}\n\nAdditional word definitions:\n{}",
                query,
                docs.join("\n")
            )
        })
        // Prompt the agent
        .prompt(&agent);

    // Prompt the agent and print the response
    let response = chain
        .call("What does \"glarb-glarb\" mean?".to_string())
        .await;

    println!("{}", response);

    Ok(())
}
