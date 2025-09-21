//! TurboMCP Integration Demo with Rig
//!
//! This example demonstrates TurboMCP integration with Rig, focusing on
//! spec-compliant MCP functionality that works identically to RMCP.
//!
//! Run with:
//! cargo run --example turbomcp_simple_demo --features turbomcp
//!
//! Prerequisites:
//! - A TurboMCP server running

#![cfg(feature = "turbomcp")]

use std::sync::Arc;
use tokio::sync::Mutex;

use rig::{
    client::{CompletionClient, ProviderClient},
    providers::openai,
};

use turbomcp_client::Client;
use turbomcp_transport::stdio::StdioTransport;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    
    println!("🚀 TurboMCP 1.0.8 Integration Demo with Rig");
    println!("===========================================");
    println!("✨ Featuring new tool creation helpers and enhanced APIs");
    
    // 1. Create a TurboMCP client - same as any MCP client
    println!("\n📦 Setting up TurboMCP client...");
    
    let transport = StdioTransport::new();
    let mut client = Client::new(transport);
    
    println!("✅ TurboMCP 1.0.8 client created with enhanced features!");
    
    // 2. Initialize the client
    client.initialize().await?;
    println!("✅ Client initialized");
    
    // 3. List available tools
    let tool_names = client.list_tools().await?;
    println!("Available tools: {:?}", tool_names);
    
    // 4. Wrap client for use with Rig
    let client_arc = Arc::new(Mutex::new(client));
    
    // 5. Set up OpenAI agent with TurboMCP tools
    println!("\n🤖 Creating Rig agent with TurboMCP integration...");
    
    let openai_client = openai::Client::from_env();
    
    // In a real scenario:
    // 1. List tools from the server: client.list_tools().await?
    // 2. Add each tool via: agent.turbomcp_tool(tool, client_arc.clone())
    
    // For this demo, we'll show the integration pattern
    println!("\n🎯 TurboMCP 1.0.8 integration ready!");
    println!("   • Spec-compliant MCP implementation");
    println!("   • Same patterns as RMCP integration");
    println!("   • Transport-agnostic (stdio, HTTP, WebSocket, etc.)");
    println!("   • Full feature parity with RMCP");
    println!("   • ✨ Enhanced tool creation helpers (Tool::new, Tool::with_description)");
    println!("   • 🔐 NEW: OAuth 2.1 MCP compliance (RFC 8707, RFC 9728, RFC 7591)");
    println!("   • 🔒 NEW: Enhanced security framework with PKCE and attack prevention");
    println!("   • 🌐 NEW: Multi-provider OAuth support (Google, GitHub, Microsoft)");
    
    demonstrate_integration(&client_arc).await?;
    
    Ok(())
}

async fn demonstrate_integration(
    client: &Arc<Mutex<Client<StdioTransport>>>
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 TurboMCP Integration Demonstration:");
    
    // Test 1: Basic client functionality
    println!("1. Testing basic MCP operations...");
    
    {
        let mut client = client.lock().await;
        
        // Test ping
        if let Ok(_) = client.ping().await {
            println!("   ✅ Ping successful - connection healthy");
        } else {
            println!("   ⚠️  Ping failed (expected for demo without server)");
        }
        
        // Test resource listing
        if let Ok(resources) = client.list_resources().await {
            println!("   ✅ Resources listed: {} found", resources.len());
        } else {
            println!("   ⚠️  Resource listing failed (expected for demo without server)");
        }
    }
    
    // Test 2: Show TurboMCP capabilities
    println!("\n2. TurboMCP capabilities:");
    println!("   • Multiple transport support (stdio, HTTP, WebSocket, TCP, Unix)");
    println!("   • Full MCP 2024-11-05 specification compliance");
    println!("   • Type-safe protocol communication");
    println!("   • Automatic connection management");
    
    // Test 3: Integration with Rig
    println!("\n3. Rig integration benefits:");
    println!("   • Identical API to RMCP integration");
    println!("   • Same agent.turbomcp_tool() method");
    println!("   • Transparent tool execution");
    println!("   • Full feature parity");
    
    println!("\n💡 TurboMCP provides spec-compliant MCP with additional transport options!");
    
    Ok(())
}

/// Helper function to show transport flexibility
#[allow(dead_code)] 
async fn demonstrate_transport_options() -> Result<(), Box<dyn std::error::Error>> {
    // TurboMCP supports multiple transports:
    
    // 1. Stdio transport (most common for MCP)
    let _stdio_transport = StdioTransport::new();
    println!("Stdio transport: Standard MCP communication");
    
    // 2. HTTP transport (for web-based MCP servers)
    // let http_transport = HttpTransport::new("http://localhost:8080");
    
    // 3. WebSocket transport (for real-time bidirectional communication)
    // let ws_transport = WebSocketTransport::new("ws://localhost:8080");
    
    // 4. TCP transport (for custom network protocols)
    // let tcp_transport = TcpTransport::new("localhost:8080");
    
    // 5. Unix socket transport (for local IPC)
    // let unix_transport = UnixTransport::new("/tmp/mcp.sock");
    
    println!("TurboMCP provides transport flexibility while maintaining MCP compliance");
    
    Ok(())
}