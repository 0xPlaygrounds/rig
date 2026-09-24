//! Registry contracts for refreshable external tool sources.
//! Generation tokens protect newer registrations from stale refreshes, while
//! [`DynamicTool::is_live`] reports disconnection without tool execution.
//!
//! ```
//! use rig_core::tool::ManagedToolToken;
//!
//! let token = ManagedToolToken::new();
//! assert_eq!(token, token.clone());
//! assert_ne!(token, ManagedToolToken::new());
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use super::DynamicTool;

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
    fn add_managed_tools(&self, tools: Vec<DynamicTool>) -> HashMap<String, ManagedToolToken>;

    /// Atomically reconcile one source's registrations with a refreshed tool
    /// list. Existing names change only while their `expected` generation is
    /// still current (newer local or peer-source registrations win); names
    /// missing from `tools` and still owned by this source are removed.
    /// Returns the new generation tokens.
    fn reconcile_managed_tools(
        &self,
        expected: HashMap<String, ManagedToolToken>,
        tools: Vec<DynamicTool>,
    ) -> HashMap<String, ManagedToolToken>;
}
