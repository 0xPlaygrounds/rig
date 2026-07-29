# rig-mcp

Host-owned MCP (Model Context Protocol) toolset for Rig's session runtime.

`McpToolset` wraps an already-connected [`rmcp`](https://crates.io/crates/rmcp)
client and exposes its tools as typed, model-facing definitions plus a
call-by-name entry point:

```rust,ignore
use rig_mcp::McpToolset;
use rmcp::ServiceExt;

let service = client_info.serve(transport).await?;
let mut toolset = McpToolset::from_sink(service.peer().clone()).await?;

// Hand these to the model.
let definitions = toolset.definitions();

// Dispatch a model tool call.
let outcome = toolset
    .call("get_weather", &serde_json::json!({"city": "Lisbon"}), None)
    .await?;
let model_visible = outcome.result; // rig_core::tool::ToolResult
let wire = outcome.raw;             // rmcp::model::CallToolResult (structuredContent, _meta, ...)

// Host-initiated tool-list refresh (replaces the classic push reconciliation).
toolset.refresh().await?;
```

## Scope

This crate serves the session runtime, where the host owns tool routing:

- the tool list is a snapshot the host refreshes explicitly (`refresh`), not a
  notification-driven registry;
- per-call `_meta` (SEP-1319) is passed explicitly to `call`, not smuggled
  through a type-map context;
- each call returns a typed `McpCallOutcome` with both the model-visible
  `ToolResult` and the raw wire result.

The classic agent runtime (`rig-agent`) keeps its own rmcp integration —
`McpClientHandler` with `tools/list_changed` reconciliation — which this crate
does not replace or modify.

## Semantics shared with the classic runtime

Ported from `rig-agent`'s rmcp module:

- **Argument validation** — only JSON objects (or `null` for no-argument calls)
  are forwarded; arrays and scalars are rejected instead of being coerced into
  a different request.
- **Timeouts** — a 300s default per-call timeout with a two-phase deadline
  (send + response wait) and a detached, bounded best-effort cancellation
  notification on elapse, so a lost response cannot hang the host and a stalled
  transport cannot accumulate cancellation tasks. Tool-list fetches are bounded
  by a 30s default deadline across all pages.
- **Content mapping** — MCP `ContentBlock`s map to Rig `ToolResultContent`
  without stringifying: text and images stay typed, everything else is
  preserved as structured JSON; `structuredContent` replaces rmcp's canonical
  text fallback rather than duplicating it; `is_error: true` becomes a failed
  `ToolResult` whose model output is the server-provided error content.

MCP support is native-only (matching `rig-agent`); on wasm targets the crate
compiles to an empty library.
