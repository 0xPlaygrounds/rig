use anyhow::Result;
use rig::prelude::*;

use mcp_core::{
    client::ClientBuilder,
    server::Server,
    tool_text_content,
    transport::{ClientSseTransportBuilder, ServerSseTransport},
    types::{ClientCapabilities, Implementation, ServerCapabilities, ToolResponseContent},
};

use mcp_core_macros::tool;

use rig::{
    completion::Prompt,
    providers::{self},
};

use serde_json::json;

#[tool(
    name = "Add",
    description = "Adds two numbers together.",
    params(a = "The first number to add", b = "The second number to add")
)]

async fn add_tool(a: f64, b: f64) -> Result<ToolResponseContent> {
    Ok(tool_text_content!((a + b).to_string()))
}

#[tokio::main]

async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt().init();

    // Create the MCP server

    let mcp_server_protocol = Server::builder("add".to_string(), "1.0".to_string())
        .capabilities(ServerCapabilities {
            tools: Some(json!({

                "listChanged": false,

            })),

            ..Default::default()
        })
        .register_tool(AddTool::tool(), AddTool::call())
        .build();

    let mcp_server_transport =
        ServerSseTransport::new("127.0.0.1".to_string(), 3000, mcp_server_protocol);

    tokio::spawn(async move { Server::start(mcp_server_transport).await });

    // Create the MCP client

    let mcp_client = ClientBuilder::new(
        ClientSseTransportBuilder::new("http://127.0.0.1:3000/sse".to_string()).build(),
    )
    .build();

    // Start the MCP client

    mcp_client.open().await?;

    let init_res = mcp_client
        .initialize(
            Implementation {
                name: "mcp-client".to_string(),

                version: "0.1.0".to_string(),
            },
            ClientCapabilities::default(),
        )
        .await?;

    println!("Initialized: {:?}", init_res);

    let tools_list_res = mcp_client.list_tools(None, None).await?;

    println!("Tools: {:?}", tools_list_res);

    tracing::info!("Building RIG agent");

    let completion_model = providers::openai::Client::from_env();

    let mut agent_builder = completion_model.agent("gpt-4o");

    // Add MCP tools to the agent

    agent_builder = tools_list_res
        .tools
        .into_iter()
        .fold(agent_builder, |builder, tool| {
            builder.mcp_tool(tool, mcp_client.clone().into())
        });

    let agent = agent_builder.build();

    tracing::info!("Prompting RIG agent");

    let response = agent.prompt("Add 10 + 10").await?;

    tracing::info!("Agent response: {:?}", response);

    Ok(())
}
