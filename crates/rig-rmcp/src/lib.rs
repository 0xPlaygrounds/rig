//! Native MCP tool adapters and managed tool-list synchronization for Rig.
//! [`McpTool`] converts to a context-aware
//! [`DynamicTool`](rig_core::tool::DynamicTool); [`McpClientHandler`] refreshes
//! registrations when the server reports tool-list changes.
//!
//! [`McpMeta`] supplies request metadata outside model-visible arguments.
//! Responses publish raw results, structured content, and response metadata
//! to the context result map; model output uses ordered presentation content.
//!
//! ```
//! use rig_core::tool::ToolContext;
//! use rig_rmcp::{McpMeta, RequestMetaObject};
//!
//! let mut context = ToolContext::new();
//! context.insert(McpMeta(RequestMetaObject::default()))?;
//! # Ok::<(), rig_core::tool::ToolContextError>(())
//! ```

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

// rmcp has no client transport that builds for WASM, and its transport traits
// require `Send` futures that browser futures cannot provide.
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
