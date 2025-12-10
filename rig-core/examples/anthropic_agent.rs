use rig::prelude::*;
use rig::{
    completion::Prompt,
    providers::anthropic::{self},
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create Anthropic client
    let client = anthropic::Client::from_env();

    // Create agent with a single context prompt
    let agent = client
        .agent(anthropic::completion::CLAUDE_3_5_SONNET)
        .preamble("Be precise and concise.")
        .temperature(0.5)
        .build();

    // Prompt the agent and print the response
    let response = agent
        .prompt("When and where and what type is the next solar eclipse?")
        .await?;

    println!("{response}");

    Ok(())
}
