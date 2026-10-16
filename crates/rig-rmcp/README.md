# rig-rmcp

MCP (Model Context Protocol) client support for [Rig](https://crates.io/crates/rig), built on the [`rmcp`](https://docs.rs/rmcp) SDK. Depends on **rig-core only**.

- `McpConnection` owns a connected rmcp client (`McpConnection::new(client_config.serve(transport).await?)`). Clones share it, and it closes when the last clone, or every tool holding one, is dropped, or on `close()`.
- `McpTool` is one server tool, from `McpConnection::tools` or `McpTool::new(definition, &connection)`. It keeps its connection open, so an agent that holds its tools keeps the server connected. Convert it into a `rig_core::tool::DynamicTool` (`TryFrom`, failing on an empty tool name) and register it wherever dynamic tools go. The dynamic tool carries a liveness probe bound to the transport, so registries retire it on disconnect.
- `McpTool::with_name` offers the tool to the model under another name, while calls still send the server the tool's MCP name: namespace one server's tools, or tell apart two servers' tools of the same name.
- Per call, an `McpMeta` in the runtime's `rig_core::tool::ToolContext` is forwarded as the request's `_meta`, and `McpTool::with_call_meta` adds entries derived from the call itself (the MCP name and the exact arguments sent), for signatures or idempotency keys. The raw `CallToolResult` (with the response's `structuredContent` and `_meta`) is published to the context's result map for hooks (`McpCallToolResult`). A tool that declares an output schema answers the model with its structured content alone.
- `McpConnection::resources` and `read_resource` list and read the server's resources. `resource_context(uris)` is a retrieval handler that reads them again for every model call: pass it to rig-agent's `AgentBuilder::dynamic_context_handler` to give an agent live resources as context.
- `McpClientHandler<S: rig_core::tool::ManagedToolSink>` is an rmcp client handler that keeps a tool registry in sync with `notifications/tools/list_changed`. rig-agent's `ToolServerHandle` implements `ManagedToolSink`, so `McpClientHandler::new(client_config, tool_server_handle.clone()).connect(transport)` is the agent usage. The registry owns those tools, so hold the `McpConnection` that `connect` returns for as long as they should work.

The `rig` facade re-exports this crate as `rig::tool::rmcp` behind its `rmcp` feature. The `rmcp_example` package shows a local server's tools and resources and a hosted server over streamable HTTP.

Native-only: rmcp's `ClientHandler` requires `Send + Sync` unconditionally, which rig's wasm tool registry cannot satisfy; building for `wasm32` fails with a single explanatory `compile_error!`.
