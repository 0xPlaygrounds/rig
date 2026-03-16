/// This example requires that you have a [llamafile](https://github.com/Mozilla-Ocho/llamafile)
/// server running locally at http://localhost:8080 (the default).
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::llamafile;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create llamafile client pointing at the default local server.
    // You can also use `Client::from_url("http://localhost:8080")` or
    // set the LLAMAFILE_API_BASE_URL env var and call `Client::from_env()`.
    let client = llamafile::Client::from_url("http://localhost:8080");

    // Create agent with a preamble
    let agent = client
        .agent(llamafile::LLAMA_CPP)
        .preamble("You are a helpful assistant.")
        .build();

    // Prompt the agent and print the response
    let response = agent
        .prompt("Explain what llamafile is in two sentences.")
        .await?;

    println!("{response}");

    Ok(())
}
