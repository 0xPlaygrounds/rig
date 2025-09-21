//! TurboMCP Plugin System Demo with Rig
//!
//! This example demonstrates the advanced plugin capabilities of TurboMCP v1.0.8
//! integrated with Rig. It showcases:
//! - Automatic retry logic with exponential backoff
//! - Response caching with TTL
//! - Metrics collection and monitoring
//! - Custom plugin middleware
//!
//! Run with:
//! cargo run --example turbomcp_plugin_demo --features turbomcp
//!
//! Prerequisites:
//! - TurboMCP v1.0.8+ with plugin system
//! - A TurboMCP server running (or mock for demo)

#![cfg(feature = "turbomcp")]

use std::sync::Arc;
use tokio::sync::Mutex;

use rig::{
    client::{CompletionClient, ProviderClient},
    providers::openai,
};

// TurboMCP imports for v1.0.8 plugin system
use turbomcp_client::{Client, ClientBuilder};
use turbomcp_transport::stdio::StdioTransport;
use turbomcp_client::plugins::{
    MetricsPlugin, RetryPlugin, CachePlugin,
    PluginConfig, RetryConfig, CacheConfig
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    
    println!("🚀 TurboMCP v1.0.8 Plugin System Demo with Rig");
    println!("===============================================");
    
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
    client.initialize().await?;
    println!("✅ Client initialized with plugin middleware active");
    
    // 3. Wrap client for use with Rig
    let client_arc = Arc::new(Mutex::new(client));
    
    // 4. Set up OpenAI agent with TurboMCP tools
    println!("\n🤖 Creating Rig agent with TurboMCP integration...");
    
    let openai_client = openai::Client::from_env();
    let _agent = openai_client
        .agent("gpt-4o-mini")
        .preamble("You are a helpful assistant with access to MCP tools that have advanced middleware capabilities including retry logic, caching, and metrics collection.");
    
    // Note: In a real scenario, you would:
    // 1. Connect to an actual TurboMCP server
    // 2. List available tools: client.list_tools().await?
    // 3. Add each tool via: agent.turbomcp_tool(tool, client_arc.clone())
    
    // For demo purposes, let's simulate some tool calls to show plugin behavior
    println!("\n🛠️  Demonstrating plugin middleware behavior...");
    
    // Simulate tool calls that would benefit from plugins
    demonstrate_plugin_behavior(&client_arc).await?;
    
    println!("\n🎯 Demo complete! Plugin middleware provided:");
    println!("   • Automatic retry on failures");
    println!("   • Response caching for performance");  
    println!("   • Metrics collection for monitoring");
    println!("   • Zero-overhead when tools succeed");
    
    Ok(())
}

async fn demonstrate_plugin_behavior(
    client: &Arc<Mutex<Client<StdioTransport>>>
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Plugin Behavior Demonstration:");
    
    // Test 1: Cache behavior (first call vs second call)
    println!("1. Testing cache plugin - calling same tool twice...");
    
    let args = std::collections::HashMap::new();
    
    // First call - will be cached
    let start = std::time::Instant::now();
    let _result1 = client.lock().await.call_tool("test_tool", Some(args.clone())).await;
    let duration1 = start.elapsed();
    println!("   First call: {:?} (full execution + caching)", duration1);
    
    // Second call - should hit cache
    let start = std::time::Instant::now(); 
    let _result2 = client.lock().await.call_tool("test_tool", Some(args)).await;
    let duration2 = start.elapsed();
    println!("   Second call: {:?} (cache hit - should be faster)", duration2);
    
    // Test 2: Plugin info
    println!("\n2. Plugin capabilities:");
    if client.lock().await.has_plugin("metrics") {
        println!("   ✅ Metrics plugin: Active - collecting performance data");
    }
    if client.lock().await.has_plugin("retry") {
        println!("   ✅ Retry plugin: Active - will retry failed requests");
    }
    if client.lock().await.has_plugin("cache") {
        println!("   ✅ Cache plugin: Active - caching responses for performance");
    }
    
    println!("\n💡 All tool calls in Rig automatically benefit from this middleware!");
    println!("   Plugin middleware is transparent - no code changes needed.");
    
    Ok(())
}

/// Helper function to show how to register custom plugins
#[allow(dead_code)]
async fn demonstrate_custom_plugin_registration() -> Result<(), Box<dyn std::error::Error>> {
    let transport = StdioTransport::new();
    let client = ClientBuilder::new()
        .build(transport)
        .await?;
    
    // You can add custom plugins after client creation
    // let custom_plugin = Arc::new(MyCustomPlugin::new());
    // client.register_plugin(custom_plugin).await?;
    
    // Or check what plugins are registered
    let has_metrics = client.has_plugin("metrics");
    println!("Has metrics plugin: {}", has_metrics);
    
    Ok(())
}