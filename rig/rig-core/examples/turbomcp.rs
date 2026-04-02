//! TurboMCP + Rig integration example.
//!
//! Demonstrates two approaches:
//! - **Basic**: Fetch tools once and build an agent (tools are static).
//! - **Auto-updating**: Use [`TurboMcpClientHandler`] with [`ToolServer`] so the
//!   agent automatically picks up tool changes when the MCP server sends
//!   `notifications/tools/list_changed`.
//!
//! ## Running the Example
//!
//! ```bash
//! OPENAI_API_KEY=your_api_key cargo run --example turbomcp --features turbomcp
//! ```

use std::future::Future;

use rig::{
    client::{CompletionClient, ProviderClient},
    completion::Prompt,
    providers::openai,
    tool::{server::ToolServer, turbomcp::TurboMcpClientHandler},
};
use turbomcp::{
    McpError, McpHandler, McpResult, Prompt as McpPrompt, PromptResult, RequestContext,
    Resource, ResourceResult, ServerInfo, Tool, ToolResult,
};
use turbomcp_client::ClientBuilder;

// ── In-process MCP server ─────────────────────────────────────────────────

/// A simple MCP server that exposes math tools.
///
/// In a real deployment, this would run in a separate process.
/// Here it runs in-process via the channel transport for demonstration.
#[derive(Clone)]
struct MathServer;

impl McpHandler for MathServer {
    fn server_info(&self) -> ServerInfo {
        ServerInfo::new("math-server", "1.0.0")
    }

    fn list_tools(&self) -> Vec<Tool> {
        vec![
            Tool::new("add", "Add two numbers"),
            Tool::new("multiply", "Multiply two numbers"),
        ]
    }

    fn list_resources(&self) -> Vec<Resource> {
        vec![]
    }

    fn list_prompts(&self) -> Vec<McpPrompt> {
        vec![]
    }

    fn call_tool<'a>(
        &'a self,
        name: &'a str,
        args: serde_json::Value,
        _ctx: &'a RequestContext,
    ) -> impl Future<Output = McpResult<ToolResult>> + Send + 'a {
        let name = name.to_string();
        async move {
            match name.as_str() {
                "add" => {
                    let a = args.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
                    let b = args.get("b").and_then(|v| v.as_i64()).unwrap_or(0);
                    Ok(ToolResult::text(format!("{}", a + b)))
                }
                "multiply" => {
                    let a = args.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
                    let b = args.get("b").and_then(|v| v.as_i64()).unwrap_or(0);
                    Ok(ToolResult::text(format!("{}", a * b)))
                }
                _ => Err(McpError::tool_not_found(&name)),
            }
        }
    }

    fn read_resource<'a>(
        &'a self,
        uri: &'a str,
        _ctx: &'a RequestContext,
    ) -> impl Future<Output = McpResult<ResourceResult>> + Send + 'a {
        let uri = uri.to_string();
        async move { Err(McpError::resource_not_found(&uri)) }
    }

    fn get_prompt<'a>(
        &'a self,
        name: &'a str,
        _args: Option<serde_json::Value>,
        _ctx: &'a RequestContext,
    ) -> impl Future<Output = McpResult<PromptResult>> + Send + 'a {
        let name = name.to_string();
        async move { Err(McpError::prompt_not_found(&name)) }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    // ── Approach 1: In-process server with auto-updating tools ────────────

    // Start an in-process MCP server via the zero-copy channel transport.
    // In production, you'd connect to an external server over HTTP/WebSocket.
    let (client_transport, _server_handle) =
        turbomcp_server::transport::channel::run_in_process(&MathServer)
            .await
            .expect("failed to start in-process server");

    let client = ClientBuilder::new().with_tools(true).build_sync(client_transport);

    // Create a shared ToolServer so the handler can update tools at runtime.
    let tool_server_handle = ToolServer::new().run();

    // TurboMcpClientHandler connects to the MCP server and auto-refreshes
    // tools whenever the server sends `notifications/tools/list_changed`.
    let handler = TurboMcpClientHandler::new(client, tool_server_handle.clone());
    let _client = handler.connect().await.inspect_err(|e| {
        tracing::error!("TurboMCP client error: {:?}", e);
    })?;

    tracing::info!("Connected to TurboMCP server");

    // Show registered tools
    let tool_defs = tool_server_handle.get_tool_defs(None).await?;
    tracing::info!(
        "Registered tools: {:?}",
        tool_defs.iter().map(|t| &t.name).collect::<Vec<_>>()
    );

    // Build an agent using the shared tool server handle.
    // When the server updates its tools, the agent sees the changes automatically.
    let openai_client = openai::Client::from_env();
    let agent = openai_client
        .agent(openai::GPT_4O)
        .preamble("You are a helpful math assistant with access to calculator tools.")
        .tool_server_handle(tool_server_handle)
        .build();

    let res = agent.prompt("What is 7 + 15?").max_turns(2).await?;
    println!("GPT-4o: {res}");

    // ── Approach 2: Static tools (simpler, no auto-update) ────────────────

    // For simpler use cases where you don't need auto-updating tools,
    // you can fetch tools once and pass them directly to the agent builder.
    //
    // let tools = client.list_tools().await?;
    // let agent = openai_client
    //     .agent(openai::GPT_4O)
    //     .preamble("You are a helpful assistant.")
    //     .turbomcp_tools(tools, client)
    //     .build();

    Ok(())
}
