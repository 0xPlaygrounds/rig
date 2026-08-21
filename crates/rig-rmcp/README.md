# rig-rmcp

MCP (Model Context Protocol) tool support for [Rig](https://crates.io/crates/rig), built on the [`rmcp`](https://docs.rs/rmcp) SDK.

- With default features (`agent`): `McpTool` implements rig-agent's contextual `ErasedTool` (MCP `_meta` passthrough from the `ToolContext`, raw results preserved on it), `McpClientHandler` keeps a `ToolServer` in sync with `notifications/tools/list_changed`, and `RmcpAgentBuilderExt` / `RmcpToolServerExt` add `rmcp_tool(s)` to the agent and tool-server builders. The `rig` facade re-exports all of it as `rig::tool::rmcp` behind its `rmcp` feature.
- With `default-features = false`: depends on rig-core only. MCP tools are exposed as `rig_core::tool::PortableDynamicTool`s (no `_meta` passthrough, no preserved raw result — there is no context to put them on), usable from any runtime that consumes portable tools.

Native-only: rmcp's `ClientHandler` requires `Send + Sync` unconditionally, which rig's wasm tool registry cannot satisfy; building for `wasm32` fails with a single explanatory `compile_error!`.
