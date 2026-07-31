//! Demonstrates the smallest useful agent setup with OpenAI.
//! Requires `OPENAI_API_KEY`.
//! Run it to see the config/agent/prompt flow end to end: a provider is plain
//! data (`openai::functions::Config`, which names the model), wrapped in
//! `ProviderConfig` for the agent runtime.

use anyhow::Result;
use rig::prelude::*;
use rig::providers::openai;

const PREAMBLE: &str = "You are a comedian here to entertain the user using humour and jokes.";
const PROMPT: &str = "Entertain me!";

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = openai::functions::Config::from_env(openai::GPT_4O)?;
    let agent = AgentBuilder::new(cfg).preamble(PREAMBLE).build();

    let response = agent.prompt(PROMPT).await?;
    println!("{response}");

    Ok(())
}
