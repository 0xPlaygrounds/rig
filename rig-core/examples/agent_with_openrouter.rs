use rig::completion::Prompt;
use rig::prelude::*;
use rig::providers::openrouter;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let client = openrouter::Client::from_env();

    // Create agent with a single context prompt
    let comedian_agent = client
        .agent("google/gemini-2.5-pro-exp-03-25:free")
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    // Prompt the agent and print the response
    let response = comedian_agent.prompt("Entertain me!").await?;

    println!("{response}");

    Ok(())
}
