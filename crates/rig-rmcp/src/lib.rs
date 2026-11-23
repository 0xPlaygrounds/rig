//! MCP clients for Rig on the rmcp SDK.
//!
//! [`McpConnection`] owns a connected client: its [`McpTool`]s convert to
//! context-aware [`DynamicTool`](rig_core::tool::DynamicTool)s and keep it
//! open, and its resources can be read directly or as agent context.
//! [`McpClientHandler`] instead keeps a tool registry in sync with the
//! server's tool list.
//!
//! [`McpMeta`] in the call's context supplies request `_meta` outside
//! model-visible arguments, and [`McpTool::with_call_meta`] derives it from
//! each call. Responses publish raw results, structured content, and response
//! metadata to the context result map; model output uses ordered presentation
//! content.
//!
//! ```
//! use rig_core::tool::ToolContext;
//! use rig_rmcp::McpMeta;
//! use rig_rmcp::rmcp::model::RequestMetaObject;
//!
//! let mut context = ToolContext::new();
//! context.insert(McpMeta(RequestMetaObject::new()))?;
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

// rmcp requires Send + Sync handlers even with local futures, while Rig's
// WASM tool registry does not provide those bounds.
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

#[cfg(not(target_family = "wasm"))]
mod connection;
#[cfg(not(target_family = "wasm"))]
pub use connection::{McpConnection, McpResourceContext};

#[cfg(not(target_family = "wasm"))]
mod handler;
#[cfg(not(target_family = "wasm"))]
pub use handler::McpClientHandler;

#[cfg(all(test, not(target_family = "wasm")))]
mod tests;

/// The rmcp SDK this crate is built against, so callers and rig agree on one version.
#[cfg(not(target_family = "wasm"))]
pub use rmcp;

/// A client config naming the client `name` at `version`, with default
/// capabilities, that offers the newest MCP protocol version with an
/// `initialize` handshake.
///
/// rmcp's default config offers its newest known version, which can be an
/// unreleased draft; some servers accept that offer and then refuse the
/// session. A server that asks for an older version still gets it.
///
/// ```
/// let config = rig_rmcp::client_config("my-app", "1.0");
/// assert!(config.protocol_version.has_initialize());
/// ```
#[cfg(not(target_family = "wasm"))]
pub fn client_config(
    name: impl Into<String>,
    version: impl Into<String>,
) -> rmcp::model::ClientConfig {
    use rmcp::model::{ClientCapabilities, ClientConfig, Implementation, ProtocolVersion};

    ClientConfig::new(
        ClientCapabilities::default(),
        Implementation::new(name, version),
    )
    .with_protocol_version(ProtocolVersion::LATEST_WITH_INITIALIZE)
}
