use std::env;

use rig::{completion::Prompt, providers};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    // Create OpenAI client
    let client = providers::openai::Client::new(
        &env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set"),
    );

    // Create agent with a single context prompt
    let comedian_agent = client
        .agent("gpt-4o")
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    // Prompt the agent and print the response
    let response = comedian_agent.prompt("Entertain me!").await?;
    println!("{}", response);

    Ok(())
}
