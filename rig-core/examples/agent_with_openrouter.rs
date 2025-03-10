use std::env;

use rig::{completion::Prompt, providers};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let client = providers::openrouter::Client::new(
        &env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY not set"),
    );

    // Create agent with a single context prompt
    let comedian_agent = client
        .agent("cognitivecomputations/dolphin3.0-mistral-24b:free")
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    // Prompt the agent and print the response
    let response = comedian_agent.prompt("Entertain me!").await?;
    println!("{}", response);

    Ok(())
}
