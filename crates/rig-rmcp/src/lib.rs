//! MCP (Model Context Protocol) tool support for Rig via the `rmcp` crate.
//!
//! With default features (`agent`) this crate provides [`McpClientHandler`], a client handler that reacts to
//! `notifications/tools/list_changed` by re-fetching the tool list and updating
//! rig-agent's `ToolServer`. Individual MCP tools are registered through the
//! [`RmcpAgentBuilderExt`] / [`RmcpToolServerExt`] `rmcp_tool` builder methods
//! (in [`prelude`]).
//!
//! With `default-features = false` the crate depends on rig-core only: MCP
//! tools are exposed as [`PortableDynamicTool`](rig_core::tool::PortableDynamicTool)s
//! (`From<McpTool>`), minus the
//! two context-dependent conveniences (`_meta` passthrough, preserved raw result).
//!
//! # Example
//!
//! ```rust,ignore
//! use rig_rmcp::McpClientHandler;
//! use rig_agent::tool::server::ToolServer;
//! use rmcp::ServiceExt;
//!
//! // 1. Create a ToolServer and get a handle
//! let tool_server_handle = ToolServer::new().run();
//!
//! // 2. Create a handler that auto-updates tools on list changes
//! let handler = McpClientHandler::new(client_info, tool_server_handle.clone());
//!
//! // 3. Connect to the MCP server and register initial tools
//! let mcp_service = handler.connect(transport).await?;
//!
//! // 4. Build an agent using the shared tool server handle
//! let agent = openai_client
//!     .agent(openai::GPT_5_2)
//!     .preamble("You are a helpful assistant.")
//!     .tool_server_handle(tool_server_handle)
//!     .build();
//! ```
//!
//! # Per-call metadata
//!
//! Rig's MCP adapter forwards an [`rmcp::model::Meta`] (re-exported here as
//! [`Meta`]) placed in a rig-agent `ToolContext` as the MCP request's `_meta`
//! (SEP-1319) — the idiomatic channel for per-call values such as auth tokens,
//! session ids, or A2A `context_id`/`task_id`, which the model never sees:
//!
//! ```rust,ignore
//! use rig_rmcp::Meta;
//! use rig_agent::tool::ToolContext;
//!
//! let mut meta = Meta::new();
//! meta.0.insert("authorization".into(), serde_json::json!("Bearer …"));
//! let mut context = ToolContext::new();
//! context.insert(meta);
//! let answer = agent.prompt("…").tool_context(context).await?;
//! ```
//!
//! # Response metadata
//!
//! MCP responses retain their protocol data in the per-dispatch rig-agent
//! `ToolContext` (`agent` feature). Result hooks can inspect the untouched
//! [`rmcp::model::CallToolResult`], its `structuredContent` as a
//! [`serde_json::Value`], and response [`Meta`] with
//! `event.tool_context.result::<T>()`. These values are host-only; only the
//! response's ordered presentation content is sent to the model.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used,
        clippy::unreachable
    )
)]

// MCP is native-only. rmcp's `ClientHandler` is declared
// `Sized + Send + Sync + 'static` unconditionally — its `local` feature relaxes
// the future bounds but not the handler itself — and rig's tool registry holds
// `Arc<dyn ErasedTool>` values that are deliberately neither `Send` nor `Sync`
// on wasm (`WasmCompatSend`/`WasmCompatSync` are no-op markers there). The two
// maybe-`Send` abstractions cannot be reconciled from this side, so raise one
// sentence instead of a page of trait errors.
#[cfg(target_family = "wasm")]
compile_error!(
    "the `rmcp` feature is native-only: rmcp's `ClientHandler` requires \
     `Send + Sync` unconditionally (its `local` feature relaxes only futures), \
     which rig's wasm tool registry cannot satisfy. Disable `rmcp` for wasm targets."
);

#[cfg(not(target_family = "wasm"))]
mod native;
#[cfg(not(target_family = "wasm"))]
pub use native::*;

#[cfg(all(not(target_family = "wasm"), feature = "agent"))]
#[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
pub mod agent;
#[cfg(all(not(target_family = "wasm"), feature = "agent"))]
pub use agent::{McpClientHandler, RmcpAgentBuilderExt, RmcpToolServerExt};

/// Bring the `rmcp_tool*` builder extension traits into scope.
#[cfg(all(not(target_family = "wasm"), feature = "agent"))]
#[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
pub mod prelude {
    pub use crate::agent::{RmcpAgentBuilderExt, RmcpToolServerExt};
}

/// The rmcp SDK this crate is built against, so callers and rig agree on one version.
#[cfg(not(target_family = "wasm"))]
pub use rmcp;
