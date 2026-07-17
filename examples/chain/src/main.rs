//! Demonstrates retrieval-augmented prompting: look up context from a vector
//! store, fold it into the prompt, then prompt the agent.
//! Requires `OPENAI_API_KEY`.

use rig::completion::Prompt;
use rig::prelude::*;
use rig::providers::openai;
use rig::vector_store::VectorStoreIndex;
use rig::vector_store::request::VectorSearchRequest;
use rig::{
    embeddings::EmbeddingsBuilder, providers::openai::Client,
    vector_store::in_memory_store::InMemoryVectorStore,
};

const QUERY: &str = "What does \"glarb-glarb\" mean?";

fn sample_definitions() -> [&'static str; 3] {
    [
        "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets",
        "Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
        "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.",
    ]
}

fn build_dictionary_agent(
    client: &Client,
) -> rig::agent::Agent<openai::responses_api::ResponsesCompletionModel> {
    client
        .agent(openai::GPT_4)
        .preamble(
            "
            You are a dictionary assistant here to help the user understand non-standard words.
        ",
        )
        .build()
}

fn lookup_context(docs: Vec<(f64, String, String)>, prompt: &str) -> String {
    format!(
        "Non standard word definitions:\n{}\n\n{}",
        docs.into_iter()
            .map(|(_, _, doc)| doc)
            .collect::<Vec<_>>()
            .join("\n"),
        prompt,
    )
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt().init();
    let client = Client::from_env()?;
    let embedding_model = client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    let mut builder = EmbeddingsBuilder::new(embedding_model.clone());
    for definition in sample_definitions() {
        builder = builder.document(definition)?;
    }
    let vector_store = InMemoryVectorStore::from_documents(builder.build().await?);
    let index = vector_store.index(embedding_model);
    let agent = build_dictionary_agent(&client);

    // Retrieve the most relevant definition, fold it into the prompt, then
    // prompt the agent. (The old pipeline ran the lookup "in parallel" with a
    // passthrough of the query; since the passthrough is instant, a plain
    // sequential lookup is equivalent and clearer.)
    let req = VectorSearchRequest::builder()
        .query(QUERY)
        .samples(1)
        .build();
    let prompt = match index.top_n::<String>(req).await {
        Ok(docs) => lookup_context(docs, QUERY),
        Err(err) => {
            println!("Lookup failed: {err}. Prompting without retrieved context.");
            QUERY.to_string()
        }
    };

    let response = agent.prompt(prompt).await?;
    println!("{response}");

    Ok(())
}
