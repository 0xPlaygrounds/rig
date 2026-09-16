//! Demonstrates prompt chaining with two agents in sequence.
//! Requires `OPENAI_API_KEY`.
//! Run it to see one agent produce a value that the next agent transforms.

use anyhow::Result;
use rig::driver::Bound;
use rig::http_client::BoxedHttpClient;
use rig::prelude::*;
use rig::providers::openai::{self, OpenAI};

const INPUT_PROMPT: &str = "Please generate a single whole integer that is 0 or 1";
const RNG_PREAMBLE: &str =
    "You are a random number generator. Return only a single whole integer that is either 0 or 1.";
const ADDER_PREAMBLE: &str =
    "Add 1000 to the number you receive, unless it is 0. Return only the final number.";

fn build_rng_agent(openai: &Bound<OpenAI, BoxedHttpClient>) -> rig::agent::Agent {
    openai.agent(openai::GPT_4).preamble(RNG_PREAMBLE).build()
}

fn build_adder_agent(openai: &Bound<OpenAI, BoxedHttpClient>) -> rig::agent::Agent {
    openai.agent(openai::GPT_4).preamble(ADDER_PREAMBLE).build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let openai = OpenAI::from_env()?.bound()?;
    let seed = build_rng_agent(&openai).prompt(INPUT_PROMPT).await?.output;
    let response = build_adder_agent(&openai).prompt(seed.trim()).await?.output;

    println!("First agent returned: {}", seed.trim());
    println!("Second agent returned: {}", response.trim());

    Ok(())
}
