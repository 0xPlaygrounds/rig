//! Demonstrates routing one prompt into different follow-up prompts.
//! Requires `OPENAI_API_KEY`.
//! Run it to see a classifier agent choose which second prompt should run.

use anyhow::{Result, bail};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::openai;
use rig::providers::openai::client::Client;

const INPUT_PROMPT: &str = "Sheep can self-medicate";
const ROUTER_PREAMBLE: &str = "
    Categorize the user's statement as exactly one of: sheep, cow, dog.
    Return only the category.
";

fn build_router_agent(
    client: &Client,
) -> rig::agent::Agent<openai::responses_api::ResponsesCompletionModel> {
    client
        .agent(openai::GPT_4)
        .preamble(ROUTER_PREAMBLE)
        .build()
}

fn build_response_agent(
    client: &Client,
) -> rig::agent::Agent<openai::responses_api::ResponsesCompletionModel> {
    client.agent(openai::GPT_4).build()
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
    let client = Client::from_env()?;
    let category = build_router_agent(&client).prompt(INPUT_PROMPT).await?;
    let follow_up = follow_up_prompt(category.trim())?;
    let response = build_response_agent(&client).prompt(follow_up).await?;

    println!("Classifier chose: {}", category.trim());
    println!("Follow-up prompt: {follow_up}");
    println!("Response: {}", response.trim());

    Ok(())
}
