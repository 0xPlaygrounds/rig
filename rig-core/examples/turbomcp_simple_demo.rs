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

// Note: In actual usage, you would import these for the agent:
// use rig::{
//     client::ProviderClient,
//     providers::openai,
// };

use turbomcp_client::{Client, SharedClient};
use turbomcp_transport::stdio::StdioTransport;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    
    println!("🚀 TurboMCP 1.0.9 Integration Demo with Rig (Enhanced)");
    println!("===========================================");
    println!("✨ Featuring SharedClient from 1.0.9 - comprehensive shared wrapper system!");
    
    // 1. Create a TurboMCP client - same as any MCP client
    println!("\n📦 Setting up TurboMCP client...");
    
    let transport = StdioTransport::new();
    let client = Client::new(transport);
    
    println!("✅ TurboMCP 1.0.9 client created with comprehensive shared features!");

    // 2. Create SharedClient - TurboMCP 1.0.9 native shared wrapper!
    let shared_client = SharedClient::new(client);

    // 3. Initialize the client (gracefully handle no server for demo)
    match shared_client.initialize().await {
        Ok(_) => {
            println!("✅ Client initialized");

            // 4. List available tools
            match shared_client.list_tools().await {
                Ok(tool_names) => println!("Available tools: {:?}", tool_names),
                Err(e) => println!("⚠️  Could not list tools (expected without server): {}", e),
            }
        }
        Err(e) => {
            println!("⚠️  Client initialization failed (expected without server): {}", e);
            println!("✨ This is normal for a demo without an actual MCP server running");
        }
    }
    
    // 5. Set up OpenAI agent with TurboMCP tools
    println!("\n🤖 Creating Rig agent with TurboMCP integration...");
    
    // Note: In real usage, you would set OPENAI_API_KEY environment variable
    println!("   → Would create OpenAI client here (requires OPENAI_API_KEY)");
    
    // In a real scenario:
    // 1. List tools from the server: client.list_tools().await?
    // 2. Add each tool via: agent.turbomcp_tool(tool, client_arc.clone())
    
    // For this demo, we'll show the integration pattern
    println!("\n🎯 TurboMCP 1.0.9 integration ready with comprehensive shared wrapper system!");
    println!("   • Spec-compliant MCP implementation");
    println!("   • Same patterns as RMCP integration");
    println!("   • Transport-agnostic (stdio, HTTP, WebSocket, etc.)");
    println!("   • Full feature parity with RMCP");
    println!("   • ✨ Enhanced tool creation helpers (Tool::new, Tool::with_description)");
    println!("   • 🤝 NATIVE: SharedClient from TurboMCP 1.0.9");
    println!("   • 🔄 NATIVE: SharedTransport for multi-protocol support");
    println!("   • 🌐 NATIVE: SharedServer for bidirectional MCP communication");
    println!("   • ♾️ NEW: Generic TurboMcpTool instead of Arc<dyn>");
    println!("   • 🔐 OAuth 2.1 MCP compliance (RFC 8707, RFC 9728, RFC 7591)");
    println!("   • 🔒 Enhanced security framework with PKCE and attack prevention");
    println!("   • 🧩 Plugin middleware architecture (retry, cache, metrics)");
    
    demonstrate_integration(&shared_client).await;
    
    Ok(())
}

async fn demonstrate_integration(
    client: &SharedClient<StdioTransport>
) {
    println!("\n📊 TurboMCP Integration Demonstration:");
    
    // Test 1: Basic client functionality
    println!("1. Testing basic MCP operations...");
    
    // SharedClient provides comprehensive shared wrapper system:

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
}

/// Helper function to show transport flexibility
#[allow(dead_code)]
async fn demonstrate_transport_options() {
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
}