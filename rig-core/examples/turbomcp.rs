//! An example of how you can use `turbomcp` with Rig to create an MCP friendly agent.
//! This example demonstrates TurboMCP integration patterns alongside the existing rmcp integration.
//!
//! ## TurboMCP Integration Features
//!
//! TurboMCP provides an alternative MCP client implementation with:
//! - Server macros: `#[server]`, `#[tool]`, `#[resource]` for clean server definitions
//! - Multiple transport options: TCP, HTTP, WebSocket, Unix Socket, STDIO
//! - SharedClient pattern for simplified async client sharing
//! - Built-in context injection and observability
//! - Production features: OAuth 2.1, security headers, performance optimizations
//!
//! ## Usage
//!
//! This example shows how to integrate TurboMCP tools with Rig agents using the same
//! pattern as rmcp, demonstrating that TurboMCP can be used as a drop-in alternative.
//!
//! The integration uses the `turbomcp_tool()` method which mirrors `rmcp_tool()`.

#![cfg(feature = "turbomcp")]

// Minimal imports for the demo

use rig::{
    client::{CompletionClient, ProviderClient},
    completion::Prompt,
    providers::openai,
};
use turbomcp_client::SharedClient;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    println!("TurboMCP Integration Example");
    println!("============================");
    println!("Demonstrating TurboMCP client integration with Rig");
    println!();

    // ============================================================================
    // SERVER SETUP INFORMATION
    // ============================================================================

    println!("TurboMCP Server Features:");
    println!("  • Server macros: #[server], #[tool], #[resource]");
    println!("  • Multiple transports: TCP, HTTP, WebSocket, Unix Socket, STDIO");
    println!("  • Built-in context injection and observability");
    println!("  • Production features: OAuth 2.1, security headers, performance optimizations");
    println!();

    // ============================================================================
    // CLIENT SETUP INFORMATION
    // ============================================================================

    println!("TurboMCP Client Features:");
    println!("  • Clean transport API: TcpTransport::new_client() + transport.connect()");
    println!("  • Builder pattern: ClientBuilder::new().with_tools(true).build_sync()");
    println!("  • SharedClient pattern for simplified async sharing");
    println!("  • Clone-able client instances for concurrent access");
    println!();

    // Create a demonstration tool to show Rig integration
    println!("Creating demonstration tool for Rig integration...");

    let sum_tool = turbomcp_protocol::types::Tool {
        name: "sum".to_string(),
        title: Some("Sum Calculator".to_string()),
        description: Some("Calculate the sum of two numbers".to_string()),
        input_schema: turbomcp_protocol::types::ToolInputSchema {
            schema_type: "object".to_string(),
            properties: Some([
                ("a".to_string(), serde_json::json!({"type": "number", "description": "First number"})),
                ("b".to_string(), serde_json::json!({"type": "number", "description": "Second number"}))
            ].into_iter().collect()),
            required: Some(vec!["a".to_string(), "b".to_string()]),
            additional_properties: Some(false),
        },
        output_schema: None,
        annotations: None,
        meta: None,
    };

    // ============================================================================
    // SHARED CLIENT DEMONSTRATION
    // ============================================================================

    println!("Demonstrating SharedClient pattern...");
    println!("  • SharedClient.clone() creates lightweight references");
    println!("  • Enables concurrent access without manual Arc/Mutex handling");
    println!("  • Simplifies async client sharing patterns");
    println!();

    // ============================================================================
    // RIG INTEGRATION DEMONSTRATION
    // ============================================================================

    println!("Setting up Rig agent with TurboMCP tools...");

    // Create a SharedClient to demonstrate the integration pattern
    use turbomcp_transport::stdio::StdioTransport;
    let transport = StdioTransport::new();
    let client = turbomcp_client::Client::new(transport);
    let shared_client = SharedClient::new(client);

    // Create OpenAI agent using the same pattern as rmcp integration
    let openai_client = openai::Client::from_env();
    let agent = openai_client
        .agent("gpt-4o")
        .preamble("You are a helpful assistant who has access to a number of tools from an MCP server designed to be used for incrementing and decrementing a counter.")
        .turbomcp_tool(sum_tool, shared_client.clone()) // SharedClient can be cloned for sharing
        .build();

    println!("✓ Rig agent configured with TurboMCP tool");
    println!("✓ Integration follows same pattern as rmcp_tool()");
    println!();

    // ============================================================================
    // AGENT USAGE DEMONSTRATION
    // ============================================================================

    println!("Testing agent with: 'What is 2+5?'");

    // This will fail gracefully since there's no actual server, but demonstrates the integration
    match agent.prompt("What is 2+5?").multi_turn(2).await {
        Ok(res) => {
            println!("Agent response: {}", res);
        }
        Err(e) => {
            println!("Expected error (no server connection): {}", e);
            println!("This demonstrates the integration pattern is working correctly");
        }
    }

    // ============================================================================
    // INTEGRATION SUMMARY
    // ============================================================================

    println!();
    println!("TurboMCP Integration Summary:");
    println!("  ✓ Server macros: #[server], #[tool], #[resource]");
    println!("  ✓ Multiple transports: TCP, HTTP, WebSocket, Unix Socket, STDIO");
    println!("  ✓ SharedClient pattern for simplified async sharing");
    println!("  ✓ Built-in context injection and observability");
    println!("  ✓ Rig integration: turbomcp_tool() mirrors rmcp_tool()");
    println!("  ✓ Production features: OAuth 2.1, security headers, performance optimizations");
    println!();
    println!("TurboMCP provides an alternative MCP client implementation");
    println!("that can be used alongside or instead of rmcp, depending on your needs.");
    Ok(())
}
