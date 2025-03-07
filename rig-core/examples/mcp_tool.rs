use serde_json::json;

use rig::{
    completion::Prompt,
    providers::{self},
};

use mcp_core::{
    client::ClientBuilder,
    server::Server,
    tool_error_response, tool_text_response,
    tools::ToolHandlerFn,
    transport::{ClientSseTransportBuilder, ServerSseTransport},
    types::{
        CallToolRequest, CallToolResponse, ClientCapabilities, Implementation, ServerCapabilities,
        Tool, ToolResponseContent,
    },
};

pub struct AddTool;

impl AddTool {
    pub fn tool() -> Tool {
        Tool {
            name: "Add".to_string(),
            description: Some("Adds two numbers together.".to_string()),
            input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "number",
                            "description": "The first number to add"
                        },
                        "b": {
                            "type": "number",
                            "description": "The second number to add"
                        }
                    },
                    "required": [
                        "a",
                        "b"
                    ]
            }),
        }
    }

    pub async fn call() -> ToolHandlerFn {
        move |req: CallToolRequest| {
            Box::pin(async move {
                let args = req.arguments.unwrap_or_default();

                let a = match args["a"].as_f64() {
                    Some(val) => val,
                    None => {
                        return tool_error_response!(anyhow::anyhow!(
                            "Missing or invalid 'a' parameter"
                        ))
                    }
                };
                let b = match args["b"].as_f64() {
                    Some(val) => val,
                    None => {
                        return tool_error_response!(anyhow::anyhow!(
                            "Missing or invalid 'b' parameter"
                        ))
                    }
                };

                tool_text_response!((a + b).to_string())
            })
        }
    }
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
        .register_tool(AddTool::tool(), AddTool::call().await)
        .build();
    let mcp_server_transport =
        ServerSseTransport::new("127.0.0.1".to_string(), 3000, mcp_server_protocol);

    // Start the MCP server in the background
    tokio::spawn(async move { Server::start(mcp_server_transport).await });

    // Create the MCP client
    let mcp_client = ClientBuilder::new(
        ClientSseTransportBuilder::new("http://localhost:3000".to_string()).build(),
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
