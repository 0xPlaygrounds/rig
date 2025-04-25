use anyhow::Result;
use rig::{
    completion::Prompt,
    providers::{self},
};
use rmcp::{
    model::{CallToolResult, Content},
    tool,
    transport::{SseServer, SseTransport},
    Error as McpError, ServerHandler, ServiceExt,
};

#[derive(Clone)]
struct McpController;

#[tool(tool_box)]
impl McpController {
    #[tool(name = "Add", description = "Adds two numbers together.")]
    async fn add_tool(
        #[tool(param)]
        #[schemars(description = "The first number to add")]
        a: f64,
        #[tool(param)]
        #[schemars(description = "The second number to add")]
        b: f64,
    ) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult::success(vec![Content::text(
            (a + b).to_string(),
        )]))
    }
}

#[tool(tool_box)]
impl ServerHandler for McpController {}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt().init();

    // Create the MCP server
    let serve_ct = SseServer::serve("127.0.0.1:3000".parse()?)
        .await?
        .with_service(|| McpController);

    // Create the MCP client
    let transport = SseTransport::start("http://127.0.0.1:3000/sse").await?;
    let mcp_client = None.serve(transport).await?;

    let tools_list_res = mcp_client.list_all_tools().await?;
    println!("Tools: {:?}", tools_list_res);

    tracing::info!("Building RIG agent");
    let completion_model = providers::openai::Client::from_env();
    let mut agent_builder = completion_model.agent("gpt-4o");

    // Add MCP tools to the agent
    agent_builder = tools_list_res
        .into_iter()
        .fold(agent_builder, |builder, tool| {
            builder.mcp_tool(tool, mcp_client.clone())
        });
    let agent = agent_builder.build();

    tracing::info!("Prompting RIG agent");
    let response = agent.prompt("Add 10 + 10").await?;
    tracing::info!("Agent response: {:?}", response);

    serve_ct.cancel();

    Ok(())
}
