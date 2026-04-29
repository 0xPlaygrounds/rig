//! Demonstrates a retrieval-augmented pipeline with `parallel!` and `lookup`.
//! Requires `OPENAI_API_KEY`.
//! Run it to see the pipeline retrieve context and fold it into the final prompt.

use rig::prelude::*;
use rig::providers::openai;
use rig::{
    embeddings::EmbeddingsBuilder,
    parallel,
    pipeline::{self, Op, agent_ops::lookup, passthrough},
    providers::openai::Client,
    vector_store::in_memory_store::InMemoryVectorStore,
};

const QUERY: &str = "What does \"glarb-glarb\" mean?";

fn sample_definitions() -> [&'static str; 3] {
    [
        "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets",
        "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
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

    let chain = pipeline::new()
        .chain(parallel!(
            passthrough::<&str>(),
            lookup::<_, _, String>(index, 1), // Required to specify document type
        ))
        .map(|(prompt, maybe_docs)| match maybe_docs {
            Ok(docs) => lookup_context(docs, prompt),
            Err(err) => {
                println!("Lookup failed: {err}. Prompting without retrieved context.");
                prompt.to_string()
            }
        })
        .prompt(agent);

    let response = chain.call(QUERY).await?;
    println!("{response}");

    Ok(())
}
