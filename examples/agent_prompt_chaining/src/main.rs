//! Demonstrates prompt chaining with two agents in sequence.
//! Requires `OPENAI_API_KEY`.
//! Run it to see one agent produce a value that the next agent transforms.

use anyhow::Result;
use rig::operation::Completion;
use rig::prelude::*;
use rig::providers::openai::{self, OpenAI};

const INPUT_PROMPT: &str = "Please generate a single whole integer that is 0 or 1";
const RNG_PREAMBLE: &str =
    "You are a random number generator. Return only a single whole integer that is either 0 or 1.";
const ADDER_PREAMBLE: &str =
    "Add 1000 to the number you receive, unless it is 0. Return only the final number.";

fn build_rng_agent(model: impl Into<DynModel<Completion>>) -> rig::agent::Agent {
    AgentBuilder::new(model).preamble(RNG_PREAMBLE).build()
}

fn build_adder_agent(model: impl Into<DynModel<Completion>>) -> rig::agent::Agent {
    AgentBuilder::new(model).preamble(ADDER_PREAMBLE).build()
}

#[tokio::main]
async fn main() -> Result<()> {
    // One model serves both agents: erase it once, clone the handle.
    let gpt4 = rig::model(OpenAI::from_env()?.completion(openai::GPT_4)).erase();
    let seed = build_rng_agent(gpt4.clone())
        .prompt(INPUT_PROMPT)
        .await?
        .output;
    let response = build_adder_agent(gpt4).prompt(seed.trim()).await?.output;

    println!("First agent returned: {}", seed.trim());
    println!("Second agent returned: {}", response.trim());

    Ok(())
}
