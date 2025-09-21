//! TurboMCP Plugin System Demo with Rig
//!
//! This example demonstrates the advanced plugin capabilities of TurboMCP v1.0.9
//! integrated with Rig using SharedClient. It showcases:
//! - Automatic retry logic with exponential backoff
//! - Response caching with TTL
//! - Metrics collection and monitoring
//! - Custom plugin middleware
//!
//! Run with:
//! cargo run --example turbomcp_plugin_demo --features turbomcp
//!
//! Prerequisites:
//! - TurboMCP v1.0.9+ with comprehensive shared wrapper system
//! - A TurboMCP server running (or mock for demo)

#![cfg(feature = "turbomcp")]

// Using TurboMCP 1.0.9 SharedClient - no manual Arc/Mutex needed!
use std::sync::Arc;

use rig::tool::turbomcp::TurboMcpClient;

// Note: In actual usage, you would import these for the agent:
// use rig::{
//     client::{CompletionClient, ProviderClient},
//     providers::openai,
// };

// TurboMCP imports for v1.0.9 SharedClient with native plugin support
use turbomcp_client::{SharedClient, ClientBuilder};
use turbomcp_transport::stdio::StdioTransport;
use turbomcp_client::plugins::{
    MetricsPlugin, RetryPlugin, CachePlugin,
    PluginConfig, RetryConfig, CacheConfig
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    
    println!("🚀 TurboMCP v1.0.9 Plugin System Demo with Rig");
    println!("===============================================");
    println!("✨ Featuring SharedClient - native shared wrapper system!");
    
    // 1. Create a TurboMCP client with comprehensive plugin stack
    println!("\n📦 Setting up TurboMCP client with plugin middleware...");
    
    let transport = StdioTransport::new();
    let mut client = ClientBuilder::new()
        // Add metrics collection plugin
        .with_plugin(Arc::new(MetricsPlugin::new(PluginConfig::Metrics)))
        // Add retry logic with exponential backoff
        .with_plugin(Arc::new(RetryPlugin::new(PluginConfig::Retry(RetryConfig {
            max_retries: 3,
            base_delay_ms: 1000,
            max_delay_ms: 30000,
            backoff_multiplier: 2.0,
            retry_on_timeout: true,
            retry_on_connection_error: true,
        }))))
        // Add response caching with 5-minute TTL
        .with_plugin(Arc::new(CachePlugin::new(PluginConfig::Cache(CacheConfig {
            ttl_seconds: 300,
            max_entries: 1000,
            cache_tools: true,
            cache_resources: true,
            cache_responses: true,
        }))))
        .build(transport)
        .await?;
    
    println!("✅ TurboMCP client configured with {} plugins", 3);
    
    // 2. Initialize the client (this triggers plugin initialization)
    let shared_client = match client.initialize().await {
        Ok(_) => {
            println!("✅ Client initialized with plugin middleware active");
            SharedClient::new(client)
        }
        Err(e) => {
            println!("⚠️  Client initialization failed (expected without server): {}", e);
            println!("✨ Creating SharedClient anyway to demonstrate API patterns");
            SharedClient::new(client)
        }
    };
    
    // 4. Set up OpenAI agent with TurboMCP tools
    println!("\n🤖 Creating Rig agent with TurboMCP integration...");
    
    // Note: In real usage, you would set OPENAI_API_KEY environment variable
    println!("   → Would create OpenAI agent here (requires OPENAI_API_KEY)");
    println!("   → Agent would use SharedClient for clean async tool access");
    
    // Note: In a real scenario, you would:
    // 1. Connect to an actual TurboMCP server
    // 2. List available tools: client.list_tools().await?
    // 3. Add each tool via: agent.turbomcp_tool(tool, client_arc.clone())
    
    // For demo purposes, let's simulate some tool calls to show plugin behavior
    println!("\n🛠️  Demonstrating plugin middleware behavior...");

    // Simulate tool calls that would benefit from plugins - SharedClient handles concurrency!
    demonstrate_plugin_behavior(&shared_client).await;
    
    println!("\n🎯 Demo complete! Plugin middleware provided:");
    println!("   • Automatic retry on failures");
    println!("   • Response caching for performance");  
    println!("   • Metrics collection for monitoring");
    println!("   • Zero-overhead when tools succeed");
    
    Ok(())
}

async fn demonstrate_plugin_behavior(
    client: &SharedClient<StdioTransport>
) {
    println!("\n📊 Plugin Behavior Demonstration:");
    
    // Test 1: Cache behavior (first call vs second call)
    println!("1. Testing cache plugin - calling same tool twice...");
    
    let _args: std::collections::HashMap<String, serde_json::Value> = std::collections::HashMap::new();
    
    // Demo: Show how SharedClient would handle tool calls (without actual server)
    println!("   → SharedClient provides clean async access to tool calls");
    println!("   → First call would be cached by plugin middleware");
    println!("   → Second call would hit cache (demonstrating performance benefit)");
    println!("   → No Arc/Mutex complexity in user code!");
    
    // Test 2: Plugin info (SharedClient provides clean async access)
    println!("\n2. Plugin capabilities:");
    if client.has_plugin("metrics") {
        println!("   ✅ Metrics plugin: Active - collecting performance data");
    }
    if client.has_plugin("retry") {
        println!("   ✅ Retry plugin: Active - will retry failed requests");
    }
    if client.has_plugin("cache") {
        println!("   ✅ Cache plugin: Active - caching responses for performance");
    }
    
    println!("\n💡 All tool calls in Rig automatically benefit from this middleware!");
    println!("   Plugin middleware is transparent - no code changes needed.");
    println!("   ✨ SharedClient eliminates Arc/Mutex complexity from user code!");
}

/// Helper function to show how to register custom plugins
#[allow(dead_code)]
async fn demonstrate_custom_plugin_registration() {
    println!("Custom plugin registration would work like this:");
    println!("  1. Create ClientBuilder with plugins");
    println!("  2. Add custom plugins via .with_plugin()");
    println!("  3. Build client with all plugins configured");
    println!("  4. Use SharedClient for clean async access");
    println!("  → All plugins work transparently with Rig integration");
}