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
//! Host-owned MCP tools for Rig's session runtime.
//!
//! This crate provides [`McpToolset`], a typed wrapper around an
//! already-connected [`rmcp`] client that:
//!
//! - fetches the server's tool list (paginated) and exposes it as
//!   [`rig_core::completion::ToolDefinition`]s,
//! - dispatches calls by name with the same argument validation, two-phase
//!   deadline, and best-effort cancellation semantics as Rig's classic agent
//!   runtime (`rig-agent`'s `rmcp` integration, which remains unchanged), and
//! - returns a typed [`McpCallOutcome`] carrying both the model-visible
//!   [`rig_core::tool::ToolResult`] and the raw wire
//!   [`rmcp::model::CallToolResult`].
//!
//! Unlike the classic runtime's push-based `tools/list_changed` reconciliation,
//! the toolset is host-owned: the host decides when to [`McpToolset::refresh`]
//! the tool list, and per-call metadata (`_meta`, SEP-1319) is passed
//! explicitly to [`McpToolset::call`] instead of being smuggled through a
//! type-map context.
//!
//! # Example
//!
//! ```rust,ignore
//! use rig_mcp::McpToolset;
//! use rmcp::ServiceExt;
//!
//! let service = client_info.serve(transport).await?;
//! let toolset = McpToolset::from_sink(service.peer().clone()).await?;
//!
//! let definitions = toolset.definitions(); // hand these to the model
//! let outcome = toolset.call("get_weather", &serde_json::json!({"city": "Lisbon"}), None).await?;
//! let model_visible = outcome.result; // rig_core::tool::ToolResult
//! let wire = outcome.raw;             // rmcp::model::CallToolResult
//! ```
//!
//! This crate is native-only; on `wasm` targets it compiles to an empty
//! library so workspace-wide builds are unaffected.

#[cfg(not(target_family = "wasm"))]
mod toolset;

#[cfg(not(target_family = "wasm"))]
pub use toolset::{
    DEFAULT_MCP_REFRESH_TIMEOUT, DEFAULT_MCP_TOOL_TIMEOUT, McpCallOutcome, McpError, McpToolset,
};
