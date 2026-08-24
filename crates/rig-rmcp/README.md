# rig-rmcp

MCP (Model Context Protocol) tool support for [Rig](https://crates.io/crates/rig), built on the [`rmcp`](https://docs.rs/rmcp) SDK. Depends on **rig-core only**.

- `McpTool` — one MCP server tool; convert it into a `rig_core::tool::PortableDynamicTool` (`From`) and register it wherever portable tools go (`rig-agent`'s `AgentBuilder::portable_dynamic_tool`, `ToolServer::portable_dynamic_tool`, or any other runtime). The portable tool carries a liveness probe bound to the MCP transport, so registries can retire it on disconnect. `tools_from_server` converts a whole tool list.
- `McpClientHandler<S: rig_core::tool::ManagedToolSink>` — an rmcp client handler that keeps a tool registry in sync with `notifications/tools/list_changed`. rig-agent's `ToolServerHandle` implements `ManagedToolSink`, so `McpClientHandler::new(client_info, tool_server_handle.clone()).connect(transport)` is the agent usage; other runtimes implement the sink for their own registry.
- Per call, an `rmcp::model::Meta` placed in the runtime's `rig_core::tool::ToolContext` is forwarded as the request's `_meta`, and the response's `structuredContent`, `Meta`, and raw `CallToolResult` are published to the context's result map for hooks (`preserve_mcp_result`).

The `rig` facade re-exports this crate as `rig::tool::rmcp` behind its `rmcp` feature.

Native-only: rmcp's `ClientHandler` requires `Send + Sync` unconditionally, which rig's wasm tool registry cannot satisfy; building for `wasm32` fails with a single explanatory `compile_error!`.
