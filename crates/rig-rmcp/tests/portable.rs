//! The rig-core-only path: with `default-features = false`, `rig-rmcp` depends
//! on rig-core alone and exposes MCP tools as `DynamicTool`s. This
//! test drives one against an in-process rmcp server without touching
//! rig-agent (it compiles and passes under both feature configurations).
#![cfg(not(target_family = "wasm"))]
#![allow(clippy::expect_used)]

use std::sync::Arc;

use rig_core::tool::DynamicTool;
use rig_rmcp::{McpConnection, McpTool};
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ClientConfig, ContentBlock, ErrorData,
    Implementation, ProtocolVersion, ServerCapabilities, ServerConfig, Tool,
};
use rmcp::service::RequestContext;
use rmcp::{RoleServer, ServerHandler, ServiceExt};

#[derive(Clone)]
struct EchoServer;

impl ServerHandler for EchoServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_protocol_version(ProtocolVersion::LATEST)
            .with_server_info(Implementation::new("rig-rmcp-portable-test", "0.1.0"))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let who = request
            .arguments
            .as_ref()
            .and_then(|args| args.get("who"))
            .and_then(|value| value.as_str())
            .unwrap_or("nobody")
            .to_owned();
        let result = match request.name.as_ref() {
            "greet" => CallToolResult::success(vec![ContentBlock::text(format!("hello {who}"))]),
            _ => CallToolResult::error(vec![ContentBlock::text("unknown tool")]),
        };
        Ok(result.into())
    }
}

fn tool(name: &str) -> Tool {
    Tool::new(
        name.to_string(),
        format!("{name} tool"),
        Arc::new(serde_json::Map::new()),
    )
}

#[tokio::test]
async fn mcp_tool_runs_as_a_dynamic_tool_without_rig_agent() {
    let (c2s, sfc) = tokio::io::duplex(8192);
    let (s2c, cfs) = tokio::io::duplex(8192);
    let server_task = tokio::spawn(async move {
        let running = EchoServer.serve((sfc, s2c)).await.expect("server start");
        running.waiting().await.expect("server error");
    });
    let client = McpConnection::new(
        ClientConfig::default()
            .serve((cfs, c2s))
            .await
            .expect("client connect"),
    );

    // One tool, converted through `TryFrom`.
    let greet: DynamicTool = McpTool::new(tool("greet"), &client)
        .try_into()
        .expect("tool name");
    assert_eq!(greet.name(), "greet");
    assert_eq!(greet.definition().description, "greet tool");
    let output = rig_agent::tool::RegisteredTool::from_dynamic(greet)
        .execute(
            serde_json::json!({ "who": "rig" }).to_string(),
            &mut rig_agent::tool::ToolContext::new(),
        )
        .await
        .into_result()
        .expect("greet succeeds");
    assert!(output.render().contains("hello rig"), "{output:?}");

    // A tool the server reports as an error becomes a failed call that still
    // carries the tool's output.
    let mut many = [tool("greet"), tool("missing")]
        .into_iter()
        .map(|tool| DynamicTool::try_from(McpTool::new(tool, &client)).expect("tool name"));
    let _greet_again = many.next().expect("two tools");
    let missing = many.next().expect("two tools");
    let error = rig_agent::tool::RegisteredTool::from_dynamic(missing)
        .execute("{}".to_owned(), &mut rig_agent::tool::ToolContext::new())
        .await
        .into_result()
        .expect_err("server-reported error surfaces as Err");
    assert!(
        error.to_string().contains("reported an execution error"),
        "{error}"
    );

    // A server tool without a name cannot be advertised to a model.
    assert!(DynamicTool::try_from(McpTool::new(tool(""), &client)).is_err());

    drop(client);
    server_task.abort();
}
