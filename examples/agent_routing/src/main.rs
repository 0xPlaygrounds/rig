//! Demonstrates routing one prompt into different follow-up prompts.
//! Requires `OPENAI_API_KEY`.
//! Run it to see a classifier agent choose which second prompt should run.
//!
//! Both agents come from the same plain-data provider config
//! (`openai::functions::Config`, which names the model) wrapped in
//! [`ProviderConfig`].

use anyhow::{Result, bail};
use rig::prelude::*;
use rig::providers::openai;

const INPUT_PROMPT: &str = "Sheep can self-medicate";
const ROUTER_PREAMBLE: &str = "
    Categorize the user's statement as exactly one of: sheep, cow, dog.
    Return only the category.
";

fn build_router_agent(cfg: &openai::functions::Config) -> rig::agent::Agent {
    AgentBuilder::new(ProviderConfig::OpenAi(cfg.clone()))
        .preamble(ROUTER_PREAMBLE)
        .build()
}

fn build_response_agent(cfg: &openai::functions::Config) -> rig::agent::Agent {
    AgentBuilder::new(ProviderConfig::OpenAi(cfg.clone())).build()
}

fn follow_up_prompt(category: &str) -> Result<&'static str> {
    match category {
        "cow" => Ok("Tell me a fact about the United States of America."),
        "sheep" => Ok("Calculate 5+5 for me. Return only the number."),
        "dog" => Ok("Write me a poem about cashews."),
        other => bail!("could not process category: {other}"),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = openai::functions::Config::from_env(openai::GPT_4)?;
    let category = build_router_agent(&cfg).prompt(INPUT_PROMPT).await?;
    let follow_up = follow_up_prompt(category.trim())?;
    let response = build_response_agent(&cfg).prompt(follow_up).await?;

    println!("Classifier chose: {}", category.trim());
    println!("Follow-up prompt: {follow_up}");
    println!("Response: {}", response.trim());

    Ok(())
}
