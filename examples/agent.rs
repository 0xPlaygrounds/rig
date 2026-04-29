//! Demonstrates the smallest useful agent setup with OpenAI.
//! Requires `OPENAI_API_KEY`.
//! Run it to see the provider/client/agent/prompt flow end to end.

use anyhow::Result;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::openai;

const PREAMBLE: &str = "You are a comedian here to entertain the user using humour and jokes.";
const PROMPT: &str = "Entertain me!";

#[tokio::main]
async fn main() -> Result<()> {
    let agent = openai::Client::from_env()?
        .agent(openai::GPT_4O)
        .preamble(PREAMBLE)
        .build();

    let response = agent.prompt(PROMPT).await?;
    println!("{response}");

    Ok(())
}
