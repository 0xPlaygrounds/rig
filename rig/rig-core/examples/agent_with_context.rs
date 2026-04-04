//! Demonstrates adding small context documents directly to an agent.
//! Requires `COHERE_API_KEY`.
//! Run it to see the model answer from the supplied in-memory facts.

use anyhow::Result;
use rig::agent::AgentBuilder;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::cohere::{self, COMMAND_R};

const CONTEXT_DOCS: [&str; 3] = [
    "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets.",
    "Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
    "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.",
];

const CONTEXT_PROMPT: &str = "What does \"glarb-glarb\" mean?";

#[tokio::main]
async fn main() -> Result<()> {
    let client = cohere::Client::from_env();
    let model = client.completion_model(COMMAND_R);
    let agent = CONTEXT_DOCS
        .iter()
        .copied()
        .fold(AgentBuilder::new(model), |builder, doc| {
            builder.context(doc)
        })
        .build();

    let response = agent.prompt(CONTEXT_PROMPT).await?;
    println!("{response}");

    Ok(())
}
