//! An example of how you can use TurboMCP with Rig to create an MCP friendly agent.
//!
//! This example demonstrates the same functionality as the rmcp example, but using TurboMCP
//! as the MCP implementation. TurboMCP is a high-performance Rust SDK for the Model Context
//! Protocol with zero-boilerplate development and automatic schema generation.
//!
//! ## Running the Example
//!
//! This example requires:
//! 1. An MCP server running on http://localhost:8080
//! 2. OPENAI_API_KEY environment variable set
//!
//! ```bash
//! # Terminal 1: Start any TurboMCP server (or use your own)
//! cargo run --example http_server -p turbomcp
//!
//! # Terminal 2: Run this example
//! OPENAI_API_KEY=your_api_key cargo run --example turbomcp --features turbomcp
//! ```

use rig::{
    client::{CompletionClient, ProviderClient},
    completion::Prompt,
    providers::openai,
};
use turbomcp_client::{ClientBuilder, StreamableHttpClientConfig, StreamableHttpClientTransport};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    // Create a TurboMCP client that connects to an MCP server
    let config = StreamableHttpClientConfig {
        base_url: "http://localhost:8080".to_string(),
        endpoint_path: "/mcp".to_string(),
        ..Default::default()
    };

    let transport = StreamableHttpClientTransport::new(config);

    let client = ClientBuilder::new()
        .with_tools(true)
        .build_sync(transport);

    // Initialize the client
    tracing::info!("Connecting to TurboMCP server...");
    client.initialize().await?;
    tracing::info!("Connected to TurboMCP server");

    // List tools from the server
    let tools = client.list_tools().await?;
    tracing::info!(
        "Available tools: {:?}",
        tools.iter().map(|t| &t.name).collect::<Vec<_>>()
    );

    if tools.is_empty() {
        tracing::warn!("No tools available from server. Make sure the server has tools registered.");
        return Ok(());
    }

    // Create an OpenAI agent with TurboMCP tools
    let openai_client = openai::Client::from_env();
    let agent = openai_client
        .agent(openai::GPT_4O)
        .preamble(
            "You are a helpful assistant who has access to a number of tools from an MCP server. \
             Use the available tools to answer questions.",
        )
        .turbomcp_tools(tools, client)
        .build();

    // Use the agent to answer a question that may require tool usage
    let res = agent
        .prompt("What tools do you have available?")
        .max_turns(2)
        .await
        .unwrap();

    println!("GPT-4o: {res}");

    Ok(())
}
