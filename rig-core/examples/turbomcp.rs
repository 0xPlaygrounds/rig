//! An example of how you can use `turbomcp` with Rig to create an MCP friendly agent.
//! This example mirrors the rmcp.rs example to show identical usage patterns.

#![cfg(feature = "turbomcp")]

use rig::{
    client::{CompletionClient, ProviderClient},
    completion::Prompt,
    providers::openai,
};
use turbomcp_client::{Client, SharedClient};
use turbomcp_transport::stdio::StdioTransport;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    // Create TurboMCP client - equivalent to RMCP setup
    let transport = StdioTransport::new();
    let mut client = Client::new(transport);

    // Initialize the client
    client.initialize().await?;

    // Create SharedClient for clean async access (addresses Rig maintainer feedback)
    let shared_client = SharedClient::new(client);

    // For this demo, create a sample tool (in real usage, you'd get tools from the server)
    // This demonstrates the same pattern as the RMCP example
    let tool =
        turbomcp_protocol::types::Tool::with_description("sum", "Calculate the sum of two numbers");

    // Create OpenAI agent - identical pattern to RMCP
    let openai_client = openai::Client::from_env();
    let agent = openai_client
        .agent("gpt-4o")
        .preamble("You are a helpful assistant who has access to a number of tools from an MCP server designed to be used for incrementing and decrementing a counter.")
        .turbomcp_tool(tool, shared_client.clone()) // Same pattern as rmcp_tool()
        .build();

    // Use the agent - identical to RMCP
    let res = agent.prompt("What is 2+5?").multi_turn(2).await.unwrap();

    println!("GPT-4o: {res}");

    Ok(())
}
