//! The rig-core-only path: with `default-features = false`, `rig-rmcp` depends
//! on rig-core alone and exposes MCP tools as `PortableDynamicTool`s. This
//! test drives one against an in-process rmcp server without touching
//! rig-agent (it compiles and passes under both feature configurations).
#![cfg(not(target_family = "wasm"))]
#![allow(clippy::expect_used)]

use std::sync::Arc;

use rig_core::tool::PortableDynamicTool;
use rig_rmcp::{McpTool, tools_from_server};
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ClientInfo, ContentBlock, ErrorData, Implementation,
    ProtocolVersion, ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::RequestContext;
use rmcp::{RoleServer, ServerHandler, ServiceExt};

#[derive(Clone)]
struct EchoServer;

impl ServerHandler for EchoServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_protocol_version(ProtocolVersion::LATEST)
            .with_server_info(Implementation::new("rig-rmcp-portable-test", "0.1.0"))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let who = request
            .arguments
            .as_ref()
            .and_then(|args| args.get("who"))
            .and_then(|value| value.as_str())
            .unwrap_or("nobody")
            .to_owned();
        match request.name.as_ref() {
            "greet" => Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "hello {who}"
            ))])),
            _ => Ok(CallToolResult::error(vec![ContentBlock::text(
                "unknown tool",
            )])),
        }
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
async fn mcp_tool_runs_as_a_portable_dynamic_tool_without_rig_agent() {
    let (c2s, sfc) = tokio::io::duplex(8192);
    let (s2c, cfs) = tokio::io::duplex(8192);
    let server_task = tokio::spawn(async move {
        let running = EchoServer.serve((sfc, s2c)).await.expect("server start");
        running.waiting().await.expect("server error");
    });
    let client = ClientInfo::default()
        .serve((cfs, c2s))
        .await
        .expect("client connect");
    let peer = client.peer().clone();

    // One tool, converted through `From`.
    let greet: PortableDynamicTool = McpTool::from_mcp_server(tool("greet"), peer.clone()).into();
    assert_eq!(greet.name(), "greet");
    assert_eq!(greet.definition().description, "greet tool");
    let output = greet
        .execute(serde_json::json!({ "who": "rig" }))
        .await
        .expect("greet succeeds");
    assert!(output.render().contains("hello rig"), "{output:?}");

    // The model-visible alias must not replace the server's wire name.
    let aliased: PortableDynamicTool =
        McpTool::from_mcp_server_with_name("safe_greet", tool("greet"), peer.clone()).into();
    assert_eq!(aliased.name(), "safe_greet");
    let aliased_output = aliased
        .execute(serde_json::json!({ "who": "alias" }))
        .await
        .expect("aliased greet succeeds using the original MCP name");
    assert!(aliased_output.render().contains("hello alias"));

    // A tool the server reports as an error becomes a failed call that still
    // carries the tool's output.
    let mut many = tools_from_server([tool("greet"), tool("missing")], &peer, None)
        .into_iter()
        .map(PortableDynamicTool::from);
    let _greet_again = many.next().expect("two tools");
    let missing = many.next().expect("two tools");
    let error = missing
        .execute(serde_json::json!({}))
        .await
        .expect_err("server-reported error surfaces as Err");
    assert!(
        error.to_string().contains("reported an execution error"),
        "{error}"
    );

    drop(client);
    server_task.abort();
}
