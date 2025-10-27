use rig::client::Nothing;
use rig::completion::Prompt;
/// This example requires that you have the [`ollama`](https://ollama.com) server running locally.
use rig::prelude::*;
use rig::providers::ollama;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create ollama client
    //
    // In the case of ollama, no API key is necessary, so we can use the `Nothing` struct in its
    // place
    let client: ollama::Client = ollama::Client::new(Nothing).unwrap();

    // Create agent with a single context prompt
    let comedian_agent = client
        .agent("qwen2.5:14b")
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    // Prompt the agent and print the response
    let response = comedian_agent.prompt("Entertain me!").await?;

    println!("{response}");

    Ok(())
}
