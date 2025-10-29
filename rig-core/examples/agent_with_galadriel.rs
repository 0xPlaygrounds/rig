use rig::prelude::*;
use rig::{completion::Prompt, providers};

use std::env;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create Galadriel client
    let api_key = &env::var("GALADRIEL_API_KEY").expect("GALADRIEL_API_KEY not set");
    let fine_tune_api_key = env::var("GALADRIEL_FINE_TUNE_API_KEY").ok();
    let mut builder = providers::galadriel::Client::builder(api_key);
    if let Some(fine_tune_api_key) = fine_tune_api_key.as_deref() {
        builder = builder.fine_tune_api_key(fine_tune_api_key);
    }
    let client = builder.build();

    // Create agent with a single context prompt
    let comedian_agent = client
        .agent("gpt-4o")
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    // Prompt the agent and print the response
    let response = comedian_agent.prompt("Entertain me!").await?;
    println!("{response}");
    Ok(())
}
