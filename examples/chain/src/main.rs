//! Demonstrates retrieval-augmented prompting: look up context from a vector
//! store, fold it into the prompt, then prompt the agent.
//! Requires `OPENAI_API_KEY`.

use rig::driver::Bound;
use rig::prelude::*;
use rig::providers::openai::{self, OpenAI};
use rig::vector_store::VectorStoreIndex;
use rig::vector_store::request::VectorSearchRequest;
use rig::{embeddings::EmbeddingsBuilder, vector_store::in_memory_store::InMemoryVectorStore};

const QUERY: &str = "What does \"glarb-glarb\" mean?";

fn sample_definitions() -> [&'static str; 3] {
    [
        "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets",
        "Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
        "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.",
    ]
}

fn build_dictionary_agent(client: &Bound<OpenAI>) -> rig::agent::Agent {
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
    let client = OpenAI::from_env()?.bound()?;
    let embedding_model = client.embedding(openai::TEXT_EMBEDDING_ADA_002, None);

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

    let response = agent.prompt(prompt).await?.output;
    println!("{response}");

    Ok(())
}
