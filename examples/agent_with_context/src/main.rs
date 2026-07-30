//! Demonstrates adding small context documents directly to an agent.
//! Requires `COHERE_API_KEY`.
//! Run it to see the model answer from the supplied in-memory facts.
//!
//! The provider is plain data: `cohere::functions::Config` names the model and
//! is wrapped in [`ProviderConfig`] for [`AgentBuilder`].

use anyhow::Result;
use rig::prelude::*;
use rig::providers::cohere::{self, COMMAND_R};

const CONTEXT_DOCS: [&str; 3] = [
    "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets.",
    "Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
    "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.",
];

const CONTEXT_PROMPT: &str = "What does \"glarb-glarb\" mean?";

#[tokio::main]
async fn main() -> Result<()> {
    let provider = ProviderConfig::Cohere(cohere::functions::Config::from_env(COMMAND_R)?);
    let agent = CONTEXT_DOCS
        .iter()
        .copied()
        .fold(AgentBuilder::new(provider), |builder, doc| {
            builder.context(doc)
        })
        .build();

    let response = agent.prompt(CONTEXT_PROMPT).await?;
    println!("{response}");

    Ok(())
}
