# rig-rmcp

MCP (Model Context Protocol) tool support for [Rig](https://crates.io/crates/rig), built on the [`rmcp`](https://docs.rs/rmcp) SDK. Depends on **rig-core only**.

- `McpTool` — one MCP server tool; convert it into a `rig_core::tool::DynamicTool` (`TryFrom`, failing on an empty tool name) and register it wherever dynamic tools go (`rig-agent`'s `AgentBuilder::dynamic_tool`, `ToolServer::dynamic_tool`, or any other runtime). The dynamic tool carries a liveness probe bound to the MCP transport, so registries can retire it on disconnect. `tools_from_server` converts a whole tool list.
- `McpClientHandler<S: rig_core::tool::ManagedToolSink>` — an rmcp client handler that keeps a tool registry in sync with `notifications/tools/list_changed`. rig-agent's `ToolServerHandle` implements `ManagedToolSink`, so `McpClientHandler::new(client_info, tool_server_handle.clone()).connect(transport)` is the agent usage; other runtimes implement the sink for their own registry.
- Per call, an `rmcp::model::Meta` placed in the runtime's `rig_core::tool::ToolContext` is forwarded as the request's `_meta`, and the response's `structuredContent`, `Meta`, and raw `CallToolResult` are published to the context's result map for hooks (`preserve_mcp_result`).

The `rig` facade re-exports this crate as `rig::tool::rmcp` behind its `rmcp` feature.

Native-only: rmcp's `ClientHandler` requires `Send + Sync` unconditionally, which rig's wasm tool registry cannot satisfy; building for `wasm32` fails with a single explanatory `compile_error!`.

## Model-visible tool aliases

Use `McpTool::with_model_name` before converting to a `DynamicTool` to choose the name exposed to the model and used for registry lookup. `McpTool::definition()` retains the original server definition, and MCP `tools/call` requests use its original name. Without an alias, both names are the server's name. Timeouts, cancellation, context metadata, and result handling keep their existing behavior.

Choose unique aliases that satisfy your model provider's naming rules. `ToolName::new` rejects an empty alias; conversion also rejects an empty raw MCP name. Rig leaves sanitization and collision handling to the caller.

For two servers that both advertise `fetch`, convert their adapters with distinct names:

```rust
use rig_core::message::{EmptyToolName, ToolName};
use rig_core::tool::DynamicTool;
use rig_rmcp::McpTool;

fn separate_fetch_tools(
    first: McpTool,
    second: McpTool,
) -> Result<[DynamicTool; 2], EmptyToolName> {
    Ok([
        first
            .with_model_name(ToolName::new("first_fetch")?)
            .try_into()?,
        second
            .with_model_name(ToolName::new("second_fetch")?)
            .try_into()?,
    ])
}
```

Register both returned tools in the same `rig-agent` `ToolServer` with `dynamic_tool`. The model can call `first_fetch` or `second_fetch`; the selected server receives `fetch`.
