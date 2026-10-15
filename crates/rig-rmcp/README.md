# rig-rmcp

MCP (Model Context Protocol) tool support for [Rig](https://crates.io/crates/rig), built on the [`rmcp`](https://docs.rs/rmcp) SDK. Depends on **rig-core only**.

## Connection ownership

Retain the `RunningService` for as long as its tools are used. Cloning
`service.peer()` gives the tool a request handle but does not retain the
connection owner. Dropping the service cancels the connection asynchronously;
registries retire its tools after the transport closes.

For manual registration, move the connected service into the application state
alongside the agent. This function takes an `AgentBuilder` and the service
returned by `ClientInfo::serve`:

```rust,no_run
use rig_agent::{Agent, AgentBuilder, tool::server::ToolServer};
use rig_rmcp::{
    McpTool,
    rmcp::{RoleClient, model::ClientInfo, service::RunningService},
};

struct AppState {
    agent: Agent,
    _mcp_services: Vec<RunningService<RoleClient, ClientInfo>>,
}

async fn register_mcp(
    builder: AgentBuilder,
    service: RunningService<RoleClient, ClientInfo>,
) -> Result<AppState, Box<dyn std::error::Error + Send + Sync>> {
    let mut tools = ToolServer::new();
    for definition in service.list_all_tools().await? {
        let tool = McpTool::from_mcp_server(definition, service.peer().clone());
        tools = tools.dynamic_tool(tool.try_into()?);
    }
    let agent = builder.tool_server_handle(tools.run()).build();
    Ok(AppState {
        agent,
        _mcp_services: vec![service],
    })
}
```

Store the whole `AppState` and keep it alive for every prompt. Moving only its
`agent` into another state value leaves the service behind to be dropped.

For automatic tool-list synchronization, also retain the service returned by
`McpClientHandler::connect`. With a `rig_agent::tool::server::ToolServerHandle`
sink, that service has type
`RunningService<RoleClient, McpClientHandler<ToolServerHandle>>`.

## Adapters

- `McpTool` — one MCP server tool; convert it into a `rig_core::tool::DynamicTool` (`TryFrom`, failing on an empty tool name) and register it wherever dynamic tools go (`rig-agent`'s `AgentBuilder::dynamic_tool`, `ToolServer::dynamic_tool`, or any other runtime). The dynamic tool carries a liveness probe bound to the MCP transport, so registries can retire it on disconnect. `tools_from_server` converts a whole tool list.
- `McpClientHandler<S: rig_core::tool::ManagedToolSink>` — an rmcp client handler that keeps a tool registry in sync with `notifications/tools/list_changed`. rig-agent's `ToolServerHandle` implements `ManagedToolSink`, so `McpClientHandler::new(client_info, tool_server_handle.clone()).connect(transport)` is the agent usage; other runtimes implement the sink for their own registry.
- Per call, an `rmcp::model::Meta` placed in the runtime's `rig_core::tool::ToolContext` is forwarded as the request's `_meta`, and the response's `structuredContent`, `Meta`, and raw `CallToolResult` are published to the context's result map for hooks (`preserve_mcp_result`).

The `rig` facade re-exports this crate as `rig::tool::rmcp` behind its `rmcp` feature.

Native-only: rmcp's `ClientHandler` requires `Send + Sync` unconditionally, which rig's wasm tool registry cannot satisfy; building for `wasm32` fails with a single explanatory `compile_error!`.
