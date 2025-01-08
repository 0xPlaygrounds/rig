use rig::{completion::Prompt, providers};
use std::env;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create Galadriel client
    let client = providers::galadriel::Client::new(
        &env::var("GALADRIEL_API_KEY").expect("GALADRIEL_API_KEY not set"),
        env::var("GALADRIEL_FINE_TUNE_API_KEY").ok().as_deref(),
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
