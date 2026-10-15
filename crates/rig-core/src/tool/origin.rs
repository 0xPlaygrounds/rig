//! Where a registered tool's implementation lives, for hosts that apply
//! different policy to native, MCP and other tools. Dispatch is the same for
//! every origin; the origin only describes the tool.
//!
//! ```
//! use rig_core::tool::{McpToolOrigin, ToolOrigin};
//!
//! let origin = ToolOrigin::Mcp(McpToolOrigin::new("search").with_server("docs"));
//! assert!(matches!(&origin, ToolOrigin::Mcp(mcp) if mcp.server() == Some("docs")));
//! ```

use std::any::Any;
use std::sync::Arc;

use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// Where a registered tool's implementation lives.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum ToolOrigin {
    /// A typed [`Tool`](super::Tool) that runs in process.
    Native {
        /// The tool's Rust type name, for display and logs.
        type_name: &'static str,
    },
    /// A runtime-defined [`DynamicTool`](super::DynamicTool) callback that
    /// runs in process.
    Dynamic,
    /// A tool an MCP server runs.
    Mcp(McpToolOrigin),
    /// A handler registered under its own descriptor, such as a replayer or
    /// a host's own bus handler.
    Handler,
}

#[cfg(not(target_family = "wasm"))]
type Source = Arc<dyn Any + Send + Sync>;
#[cfg(target_family = "wasm")]
type Source = Arc<dyn Any>;

/// The MCP server and server-side tool behind a registration, plus the
/// adapter that calls it when the integration attaches one.
#[derive(Clone)]
pub struct McpToolOrigin {
    server: Option<String>,
    tool: String,
    source: Option<Source>,
}

impl McpToolOrigin {
    /// The tool named `tool` on its server.
    pub fn new(tool: impl Into<String>) -> Self {
        Self {
            server: None,
            tool: tool.into(),
            source: None,
        }
    }

    /// The same origin, naming the server that serves the tool.
    pub fn with_server(mut self, server: impl Into<String>) -> Self {
        self.server = Some(server.into());
        self
    }

    /// The same origin, carrying the integration's adapter for the tool so a
    /// host can reach it with [`Self::source`].
    pub fn with_source<S>(mut self, source: S) -> Self
    where
        S: Any + WasmCompatSend + WasmCompatSync,
    {
        self.source = Some(Arc::new(source));
        self
    }

    /// The server's name, when the server gave one.
    pub fn server(&self) -> Option<&str> {
        self.server.as_deref()
    }

    /// The tool's name on its server.
    pub fn tool(&self) -> &str {
        &self.tool
    }

    /// The attached adapter, when it is an `S`. `rig-rmcp` attaches its
    /// `McpTool`, which holds the server's tool definition and client.
    pub fn source<S: Any>(&self) -> Option<&S> {
        self.source.as_deref()?.downcast_ref()
    }
}

impl std::fmt::Debug for McpToolOrigin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpToolOrigin")
            .field("server", &self.server)
            .field("tool", &self.tool)
            .field("has_source", &self.source.is_some())
            .finish()
    }
}
