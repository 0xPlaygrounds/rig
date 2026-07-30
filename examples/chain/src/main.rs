//! Demonstrates retrieval-augmented prompting: look up context from a vector
//! store, fold it into the prompt, then prompt the agent.
//!
//! Embedding is a free function over plain config data
//! (`openai::functions::embed` + an `EmbeddingConfig` that names the model),
//! and `rig::embeddings::embed_documents` is the document-level entry point
//! that replaced `EmbeddingsBuilder`.
//!
//! Requires `OPENAI_API_KEY`.

use rig::OneOrMany;
use rig::embeddings::{default_concurrency, embed_documents};
use rig::http_runtime::HttpRuntime;
use rig::prelude::*;
use rig::providers::openai;
use rig::vector_store::in_memory_store::InMemoryVectorStore;
use rig::vector_store::request::VectorSearchRequest;

const QUERY: &str = "What does \"glarb-glarb\" mean?";

fn sample_definitions() -> [&'static str; 3] {
    [
        "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets",
        "Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
        "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.",
    ]
}

fn build_dictionary_agent(provider: ProviderConfig) -> rig::agent::Agent {
    AgentBuilder::new(provider)
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
    let rt = HttpRuntime::new();
    let ecfg = openai::functions::EmbeddingConfig::from_env(openai::TEXT_EMBEDDING_ADA_002)?;

    let max_documents = openai::functions::DESCRIPTOR
        .max_embedding_documents
        .unwrap_or(usize::MAX);
    let documents: Vec<String> = sample_definitions().iter().map(|s| s.to_string()).collect();
    let embeddings = embed_documents(
        documents,
        max_documents,
        default_concurrency(max_documents),
        |texts| openai::functions::embed(&ecfg, &rt, texts),
    )
    .await?;
    let vector_store = InMemoryVectorStore::from_documents(embeddings)?;

    let agent = build_dictionary_agent(ProviderConfig::OpenAi(
        openai::functions::Config::from_env(openai::GPT_4)?,
    ));

    // Retrieve the most relevant definition, fold it into the prompt, then
    // prompt the agent. (The old pipeline ran the lookup "in parallel" with a
    // passthrough of the query; since the passthrough is instant, a plain
    // sequential lookup is equivalent and clearer.)
    let query_embedding = openai::functions::embed(&ecfg, &rt, vec![QUERY.to_string()])
        .await?
        .embeddings
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("the embedding provider returned no embedding"))?;
    let req = VectorSearchRequest::new(OneOrMany::one(query_embedding), 1);
    let prompt = match vector_store.top_n_as::<String>(req).await {
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
