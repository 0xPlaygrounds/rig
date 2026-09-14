//! Externally managed tool sources.
//!
//! A tool registry that wants to accept tools from a source it does not own —
//! a remote protocol whose tool list can change or disconnect (MCP is the
//! first such source, via `rig-rmcp`) — implements [`ManagedToolSink`]. The
//! source hands over [`PortableDynamicTool`]s and keeps the returned
//! [`ManagedToolToken`]s; on refresh it reconciles against them so a newer
//! local or peer-source registration under the same name is never clobbered
//! by a stale refresh, and names the source no longer offers are removed.
//! Liveness comes from [`PortableDynamicTool::is_live`], so a sink can retire
//! disconnected tools without probing by execution.

use std::collections::HashMap;
use std::sync::Arc;

use super::PortableDynamicTool;

/// Opaque identity for one managed registry generation.
///
/// Minted by a [`ManagedToolSink`] when it installs a tool; two tokens are
/// equal only if they are the same generation.
#[derive(Clone, Debug)]
pub struct ManagedToolToken(Arc<()>);

impl ManagedToolToken {
    /// Mint a fresh generation. Only sinks should call this.
    pub fn new() -> Self {
        Self(Arc::new(()))
    }
}

impl Default for ManagedToolToken {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for ManagedToolToken {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ManagedToolToken {}

/// A tool registry that accepts tools from an external, refreshable source.
pub trait ManagedToolSink {
    /// Atomically install the initial tools owned by one source.
    /// Last-registration-wins: an existing name is replaced. Tools that
    /// report `!is_live()` are skipped. Returns one generation token per
    /// installed name, to hand back to [`Self::reconcile_managed_tools`].
    fn add_managed_tools(
        &self,
        tools: Vec<PortableDynamicTool>,
    ) -> HashMap<String, ManagedToolToken>;

    /// Atomically reconcile one source's registrations with a refreshed tool
    /// list. Existing names change only while their `expected` generation is
    /// still current (newer local or peer-source registrations win); names
    /// missing from `tools` and still owned by this source are removed.
    /// Returns the new generation tokens.
    fn reconcile_managed_tools(
        &self,
        expected: HashMap<String, ManagedToolToken>,
        tools: Vec<PortableDynamicTool>,
    ) -> HashMap<String, ManagedToolToken>;
}
