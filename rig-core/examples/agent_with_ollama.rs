/// This example requires that you have the [`ollama`](https://ollama.com) server running locally.
use rig::{completion::Prompt, providers::ollama};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create an Ollama client with default base url, a local ollama endpoint
    let client = ollama::Client::new();

    // Create agent with a single context prompt
    let comedian_agent = client
        .agent(ollama::LLAMA_3_2)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    // Prompt the agent and print the response
    let response = comedian_agent.prompt("Entertain me!").await?;
    println!("{}", response);

    Ok(())
}
