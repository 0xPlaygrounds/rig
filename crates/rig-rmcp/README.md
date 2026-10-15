# rig-rmcp

MCP (Model Context Protocol) tool support for [Rig](https://crates.io/crates/rig), built on the [`rmcp`](https://docs.rs/rmcp) SDK. Depends on **rig-core only**.

- `McpTool` — one MCP server tool; convert it into a `rig_core::tool::DynamicTool` (`TryFrom`, failing on an empty tool name) and register it wherever dynamic tools go (`rig-agent`'s `AgentBuilder::dynamic_tool`, `ToolServer::dynamic_tool`, or any other runtime). The dynamic tool carries a liveness probe bound to the MCP transport, so registries can retire it on disconnect. `tools_from_server` converts a whole tool list.
- `McpClientHandler<S: rig_core::tool::ManagedToolSink>` — an rmcp client handler that keeps a tool registry in sync with `notifications/tools/list_changed`. rig-agent's `ToolServerHandle` implements `ManagedToolSink`, so `McpClientHandler::new(client_info, tool_server_handle.clone()).connect(transport)` is the agent usage; other runtimes implement the sink for their own registry.
- Per call, `McpMeta(rmcp::model::Meta)` placed in the runtime's `rig_core::tool::ToolContext` is forwarded as the request's `_meta`, and the response's `structuredContent`, `Meta`, and raw `CallToolResult` are published to the context's result map for hooks (`preserve_mcp_result`).

The `rig` facade re-exports this crate as `rig::tool::rmcp` behind its `rmcp` feature.

## Metadata derived for each call

Attach an `McpMetaProvider` to an `McpTool` with `with_meta_provider` before converting it to a `DynamicTool`. The provider runs synchronously for each dynamic tool execution and reads the tool name, final parsed MCP arguments, and an immutable per-call `ToolContext`. It does not mutate the run context. The provider and its metadata stay outside the tool schema and model-visible arguments.

Returning `Some(meta)` replaces the context's `McpMeta` for that request; returning `None` uses the context metadata. Cloned tools share the provider. Direct `McpTool::execute_mcp(args, meta)` calls use their explicit metadata and do not invoke the provider.

This example assumes the server recognizes `com.example/idempotency-key` and uses a stable `order_id` to identify one `create_order` operation:

```rust
use rig_core::message::EmptyToolName;
use rig_core::tool::{DynamicTool, ToolContext};
use rig_rmcp::rmcp::model::JsonObject;
use rig_rmcp::{McpMetaProvider, McpTool, Meta};
use serde_json::Value;

struct OrderMetadata;

impl McpMetaProvider for OrderMetadata {
    fn meta_for(
        &self,
        tool: &str,
        arguments: Option<&JsonObject>,
        _context: &ToolContext,
    ) -> Option<Meta> {
        if tool != "create_order" {
            return None;
        }
        let order_id = arguments?.get("order_id")?.as_str()?;
        let mut meta = Meta::new();
        meta.0.insert(
            "com.example/idempotency-key".into(),
            Value::String(format!("create_order:{order_id}")),
        );
        Some(meta)
    }
}

fn order_tool(tool: McpTool) -> Result<DynamicTool, EmptyToolName> {
    tool.with_meta_provider(OrderMetadata).try_into()
}
```

Native-only: rmcp's `ClientHandler` requires `Send + Sync` unconditionally, which rig's wasm tool registry cannot satisfy; building for `wasm32` fails with a single explanatory `compile_error!`.
