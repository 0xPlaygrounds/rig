//! Demonstrates prompt chaining with two agents in sequence.
//! Requires `OPENAI_API_KEY`.
//! Run it to see one agent produce a value that the next agent transforms.
//!
//! Both agents are built from the same plain-data provider config
//! (`openai::functions::Config`, which names the model) wrapped in
//! [`ProviderConfig`] — the config is just cloned per agent.

use anyhow::Result;
use rig::prelude::*;
use rig::providers::openai;

const INPUT_PROMPT: &str = "Please generate a single whole integer that is 0 or 1";
const RNG_PREAMBLE: &str =
    "You are a random number generator. Return only a single whole integer that is either 0 or 1.";
const ADDER_PREAMBLE: &str =
    "Add 1000 to the number you receive, unless it is 0. Return only the final number.";

fn build_rng_agent(cfg: &openai::functions::Config) -> rig::agent::Agent {
    AgentBuilder::new(ProviderConfig::OpenAi(cfg.clone()))
        .preamble(RNG_PREAMBLE)
        .build()
}

fn build_adder_agent(cfg: &openai::functions::Config) -> rig::agent::Agent {
    AgentBuilder::new(ProviderConfig::OpenAi(cfg.clone()))
        .preamble(ADDER_PREAMBLE)
        .build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = openai::functions::Config::from_env(openai::GPT_4)?;
    let seed = build_rng_agent(&cfg).prompt(INPUT_PROMPT).await?;
    let response = build_adder_agent(&cfg).prompt(seed.trim()).await?;

    println!("First agent returned: {}", seed.trim());
    println!("Second agent returned: {}", response.trim());

    Ok(())
}
