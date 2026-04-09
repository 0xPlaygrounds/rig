//! Demonstrates prompt chaining with two agents in sequence.
//! Requires `OPENAI_API_KEY`.
//! Run it to see one agent produce a value that the next agent transforms.

use anyhow::Result;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::openai;
use rig::providers::openai::client::Client;

const INPUT_PROMPT: &str = "Please generate a single whole integer that is 0 or 1";
const RNG_PREAMBLE: &str =
    "You are a random number generator. Return only a single whole integer that is either 0 or 1.";
const ADDER_PREAMBLE: &str =
    "Add 1000 to the number you receive, unless it is 0. Return only the final number.";

fn build_rng_agent(
    client: &Client,
) -> rig::agent::Agent<openai::responses_api::ResponsesCompletionModel> {
    client.agent(openai::GPT_4).preamble(RNG_PREAMBLE).build()
}

fn build_adder_agent(
    client: &Client,
) -> rig::agent::Agent<openai::responses_api::ResponsesCompletionModel> {
    client.agent(openai::GPT_4).preamble(ADDER_PREAMBLE).build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::from_env();
    let seed = build_rng_agent(&client).prompt(INPUT_PROMPT).await?;
    let response = build_adder_agent(&client).prompt(seed.trim()).await?;

    println!("First agent returned: {}", seed.trim());
    println!("Second agent returned: {}", response.trim());

    Ok(())
}
