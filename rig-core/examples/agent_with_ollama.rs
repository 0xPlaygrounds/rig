/// This example requires that you have the [`ollama`](https://ollama.com) server running locally.
use rig::{completion::Prompt, providers};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create an OpenAI client with a custom base url, a local ollama endpoint
    // The API Key is unnecessary for most local endpoints
    let client = providers::openai::Client::from_url("ollama", "http://localhost:11434/v1");

    // Create agent with a single context prompt
    let comedian_agent = client
        .agent("llama3.2:latest")
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    // Prompt the agent and print the response
    let response = comedian_agent.prompt("Entertain me!").await?;
    println!("{}", response);

    Ok(())
}
