use mcp_core::{
    client::Client,
    run_http_server,
    server::Server,
    sse::http_server::Host,
    tool_error_response, tool_text_response,
    transport::{ClientSseTransport, Transport},
    types::{
        CallToolRequest, CallToolResponse, Implementation, ServerCapabilities, Tool,
        ToolResponseContent,
    },
};
use serde_json::json;
use std::sync::Arc;

use rig::{
    completion::Prompt,
    providers::{self},
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt().init();

    tokio::spawn(async move {
        run_http_server(
            Host {
                host: "127.0.0.1".to_string(),
                port: 3000,
                public_url: None,
            },
            None,
            |transport| async move {
                let mut server_builder =
                    Server::builder(transport).capabilities(ServerCapabilities {
                        tools: Some(json!({})),
                        ..Default::default()
                    });

                server_builder.register_tool(
                    Tool {
                        name: "Add".to_string(),
                        description: Some("Adds two numbers together.".to_string()),
                        input_schema: json!({
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
                    },
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
                    },
                );

                Ok(server_builder.build())
            },
        )
        .await
    });

    let transport = ClientSseTransport::builder("http://127.0.0.1:3000".to_string()).build();
    transport.open().await?;

    let mcp_client = Arc::new(Client::builder(transport).use_strict().build());
    let mcp_client_clone = mcp_client.clone();
    tokio::spawn(async move { mcp_client_clone.start().await });

    let init_res = mcp_client
        .initialize(Implementation {
            name: "mcp-client".to_string(),
            version: "0.1.0".to_string(),
        })
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
            builder.mcp_tool(tool, mcp_client.clone())
        });
    let agent = agent_builder.build();

    tracing::info!("Prompting RIG agent");
    let response = agent.prompt("Add 10 + 10").await?;
    tracing::info!("Agent response: {:?}", response);

    Ok(())
}
