//! An example demonstrating TurboMCP integration with Rig for LLM agents.
//!
//! This example shows how to:
//! - Create a TurboMCP server with tools and resources using macros
//! - Connect a TurboMCP client to the server via TCP transport
//! - Integrate TurboMCP tools with a Rig agent
//! - Make actual tool calls through the agent
//!
//! The server provides a `sum` tool that adds two numbers, along with
//! sample resources demonstrating the MCP protocol capabilities.

#![cfg(feature = "turbomcp")]

use std::{sync::Arc, net::SocketAddr};
use tokio::sync::Mutex;
use turbomcp::prelude::*;
use turbomcp_client::{SharedClient, Client};
use turbomcp_transport::TcpTransport;

use rig::{
    client::{CompletionClient, ProviderClient},
    completion::Prompt,
    providers::openai,
};

/// Request structure for the sum tool
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct StructRequest {
    pub a: i32,
    pub b: i32,
}

/// Simple counter server that provides math tools and sample resources
#[derive(Clone)]
pub struct Counter {
    pub counter: Arc<Mutex<i32>>,
}

#[server(
    name = "TurboMCP-Rig-Demo",
    version = "1.0.0",
    description = "A simple MCP server with math tools and sample resources"
)]
impl Counter {
    pub fn new() -> Self {
        Self {
            counter: Arc::new(Mutex::new(0)),
        }
    }

    /// Adds two numbers together
    #[tool("Calculate the sum of two numbers")]
    async fn sum(&self, request: StructRequest) -> McpResult<String> {
        Ok((request.a + request.b).to_string())
    }

    /// Sample file system resource
    #[resource("str:////Users/to/some/path/")]
    async fn resource_cwd(&self) -> McpResult<String> {
        Ok("/Users/to/some/path/".to_string())
    }

    /// Sample memo resource
    #[resource("memo://insights")]
    async fn resource_memo(&self) -> McpResult<String> {
        Ok("Business Intelligence Memo\n\nAnalysis has revealed 5 key insights ...".to_string())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    println!("🚀 TurboMCP + Rig Integration Example");
    println!("=====================================");

    let server = Counter::new();

    // Start TurboMCP TCP server
    println!("🚀 Starting TurboMCP TCP server on localhost:8080...");
    tokio::spawn({
        let server = server.clone();
        async move {
            loop {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {
                        println!("Received Ctrl+C, shutting down");
                        break;
                    }
                    result = server.run_tcp("127.0.0.1:8080".parse::<SocketAddr>().unwrap()) => {
                        match result {
                            Ok(_) => break,
                            Err(e) => {
                                eprintln!("Server error: {e:?}");
                                break;
                            }
                        }
                    }
                }
            }
        }
    });

    // Brief delay for server startup
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    println!("🔌 Connecting TurboMCP client to localhost:8080...");

    // Create TurboMCP TCP client transport
    let local_addr = "127.0.0.1:0".parse().unwrap(); // Use any available local port
    let remote_addr = "127.0.0.1:8080".parse().unwrap();
    let transport = TcpTransport::new_client(local_addr, remote_addr);

    // Create shared client
    let client = Client::new(transport);
    let client = SharedClient::new(client);

    // Discover available tools from the server
    println!("📋 Discovering tools from TurboMCP server...");
    let tools = client.list_tools().await.map_err(|e| {
        anyhow::anyhow!("Failed to list tools: {}", e)
    })?;

    println!("✅ TurboMCP client connected!");
    println!("   • Server: localhost:8080");
    println!("   • Tools available: {}", tools.len());
    for tool in &tools {
        println!("     - {}: {}", tool.name, tool.description.as_deref().unwrap_or("No description"));
    }

    // Create Rig agent with OpenAI GPT-4o
    println!("\n🤖 Setting up Rig agent with TurboMCP tools...");
    let openai_client = openai::Client::from_env();

    // Build agent with TurboMCP tools
    let mut agent = openai_client
        .agent("gpt-4o")
        .preamble("You are a helpful assistant who has access to tools from an MCP server for mathematical operations.");

    // Add each discovered tool to the agent
    for tool in &tools {
        agent = agent.turbomcp_tool(tool.clone(), client.clone());
        println!("   ✅ Added tool: {}", tool.name);
    }

    let agent = agent.build();
    println!("   ✅ Agent configured with {} TurboMCP tools", tools.len());

    // Test the agent with a math query
    println!("\n🧪 Testing agent: 'What is 2+5?'");
    let response = agent.prompt("What is 2+5?").await.map_err(|e| {
        anyhow::anyhow!("Agent query failed: {}", e)
    })?;

    println!("🤖 Agent response: {}", response);

    println!("\n✅ Example completed successfully!");
    println!("   • TurboMCP server running with tools and resources");
    println!("   • Client connected and discovered tools");
    println!("   • Rig agent integrated with TurboMCP tools");
    println!("   • Agent successfully executed tool calls");

    println!("\n📝 This example demonstrates:");
    println!("   • Simple server setup with #[server], #[tool], #[resource] macros");
    println!("   • Automatic transport management with SharedClient");
    println!("   • Seamless Rig integration with .turbomcp_tool()");
    println!("   • End-to-end LLM agent functionality with MCP tools");

    Ok(())
}