//! MCP (Model Context Protocol) tool support for Rig via the `rmcp` crate.
//!
//! This crate depends on rig-core only. It provides:
//!
//! - `McpTool`, one MCP server tool, usable as a rig-core
//!   [`PortableDynamicTool`](rig_core::tool::PortableDynamicTool) via `From`
//!   (with a liveness probe bound to the MCP transport), and
//!   `tools_from_server` for a whole tool list;
//! - `McpClientHandler`, an rmcp client handler that keeps any
//!   [`ManagedToolSink`](rig_core::tool::ManagedToolSink) — rig-agent's
//!   `ToolServerHandle`, for example — in sync with the server's tool list,
//!   reacting to `notifications/tools/list_changed`.
//!
//! Per call, an [`rmcp::model::RequestMetaObject`] placed in the runtime's
//! [`ToolContext`](rig_core::tool::ToolContext) is forwarded as the request's
//! `_meta`, and the response's `structuredContent`, [`MetaObject`], and raw
//! result are published to the context's result map (`preserve_mcp_result`).
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
//! Rig's MCP adapter forwards an [`rmcp::model::RequestMetaObject`]
//! (re-exported here as [`RequestMetaObject`]) placed in a rig-agent
//! `ToolContext` as the MCP request's `_meta` (SEP-1319) — the idiomatic
//! channel for per-call values such as auth tokens, session ids, or A2A
//! `context_id`/`task_id`, which the model never sees:
//!
//! ```rust,ignore
//! use rig_rmcp::RequestMetaObject;
//! use rig_agent::tool::ToolContext;
//!
//! let mut meta = RequestMetaObject::new();
//! meta.insert("authorization".into(), serde_json::json!("Bearer …"));
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
//! [`serde_json::Value`], and response [`MetaObject`] with
//! `event.tool_context.result::<T>()`. These values are host-only; only the
//! response's ordered presentation content is sent to the model.
//!
//! Request and response `_meta` are distinct types in MCP: a request carries a
//! [`RequestMetaObject`] (which additionally reserves keys such as
//! `progressToken`), a result carries a plain [`MetaObject`]. Both deref
//! through to the underlying JSON map ([`RequestMetaObject`] via
//! [`MetaObject`]), so the key-level API is the same.

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

// MCP is native-only, but no longer for the reason this comment carried through
// rmcp 2.x. That one — rmcp's `ClientHandler` demanding `Send + Sync` that
// rig's wasm tool registry cannot supply, its `Arc<dyn ErasedTool>` being
// deliberately neither (`rig-core`'s `WasmCompatSend`/`WasmCompatSync` are
// no-op markers there) — was fixed upstream in rmcp 3.0: `local` now gates the
// handler bound (`Sized + 'static`) as it already gated the future bound
// (`MaybeSendFuture`). rmcp core with `client` + `local` (and default features
// off — the default `server` feature drags `uuid/v4` → `getrandom` in) compiles
// for wasm32-unknown-unknown.
//
// The blocker moved downstream, to the transport:
//
//   - `transport-streamable-http-client-reqwest` — the only browser-plausible
//     one — does not build for wasm. `default_http_client()` calls
//     `reqwest::redirect::Policy::none()` and `pool_max_idle_per_host(0)`
//     ungated; reqwest's wasm backend has neither (no `redirect` module, no
//     such `ClientBuilder` method).
//   - No other transport is usable there either. The generic ones
//     (`SinkStreamTransport`, `AsyncRwTransport`, `WorkerTransport`) compile,
//     but `Transport<R>: Send` with `Send` futures is unconditional — `local`
//     relaxes handlers, never transports — and browser futures (`JsFuture`)
//     are `!Send`. The rest are native by construction: child process, unix
//     socket, stdio.
//   - rmcp's `local` swaps `tokio::spawn` for `tokio::task::spawn_local`, which
//     wants a tokio `LocalSet` — single-threaded *native* tokio, not a browser
//     event loop. Rig has no spawn abstraction to bridge that; it never spawns
//     on wasm outside this crate's cancellation delivery.
//
// So a wasm MCP client needs upstream `Send`-bound relaxations on rmcp's
// transport layer, a hand-written `Transport` over fetch/EventSource, and a
// spawn shim — a feature project, not a `cfg` fix. Until then `rmcp` is
// declared only under `[target.'cfg(not(target_family = "wasm"))'.dependencies]`
// and this gate raises one sentence instead of a page of trait errors.
#[cfg(target_family = "wasm")]
compile_error!(
    "the `rmcp` feature is native-only: rmcp ships no wasm-capable client \
     transport (its streamable-HTTP client calls reqwest APIs that do not \
     exist on wasm; every other transport is native by construction or \
     blocked by rmcp's unconditional `Send` bounds). Disable `rmcp` for \
     wasm targets."
);

#[cfg(not(target_family = "wasm"))]
mod native;
#[cfg(not(target_family = "wasm"))]
pub use native::*;

#[cfg(not(target_family = "wasm"))]
mod handler;
#[cfg(not(target_family = "wasm"))]
pub use handler::McpClientHandler;

#[cfg(all(test, not(target_family = "wasm")))]
mod tests;

/// The rmcp SDK this crate is built against, so callers and rig agree on one version.
#[cfg(not(target_family = "wasm"))]
pub use rmcp;
