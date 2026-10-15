//! Request-scoped tool advertisements: the pinned registrations rig
//! dispatches and the provider-hosted tools the provider runs.
//!
//! ```
//! use rig_agent::tool::{CatalogEntry, ToolCatalog, ToolOrigin};
//! fn mcp_tools(catalog: &ToolCatalog) -> Vec<&str> {
//!     catalog
//!         .entries()
//!         .filter_map(|entry| match entry {
//!             CatalogEntry::Registered(tool) if matches!(tool.origin(), ToolOrigin::Mcp(_)) => {
//!                 Some(tool.definition().name.as_str())
//!             }
//!             _ => None,
//!         })
//!         .collect()
//! }
//! ```

use std::collections::BTreeSet;

use indexmap::IndexMap;

use rig_core::{
    completion::{ProviderToolDefinition, ToolDefinition},
    effect::{Key, family},
    tool::{ToolContext, ToolResult},
};

use super::registry::{RegisteredTool, ToolDispatch, execute_tool};

/// Opaque generation lease retained by catalogs. Registries may retire replaced
/// bindings once no catalog holds their lease.
#[cfg(not(target_family = "wasm"))]
pub type ToolLease = std::sync::Arc<dyn std::any::Any + Send + Sync>;
/// Opaque generation lease retained by catalogs, without thread-safety bounds on WASM.
#[cfg(target_family = "wasm")]
pub type ToolLease = std::sync::Arc<dyn std::any::Any>;

/// The tools one request advertises, with the exact registrations that
/// serve them. Inline dispatch retains the original handler after replacement.
/// Registry-generated bus keys also pin their generation. An explicitly reused
/// key instead reaches its latest bus binding. An in-place replacement under
/// the same name and key shares its existing lease. After removal or a key
/// change, the replacement has a new lease; an older generation's catalog
/// cannot extend or shorten that new binding's lifetime.
#[derive(Clone)]
pub struct ToolCatalog {
    definitions: Vec<ToolDefinition>,
    tools: IndexMap<String, RegisteredTool>,
    provider_tools: Vec<ProviderToolDefinition>,
    /// The tokens the catalog keeps alive for as long as it exists (see
    /// [`ToolLease`]).
    leases: Vec<ToolLease>,
}

/// One entry of a [`ToolCatalog`].
#[derive(Clone, Copy, Debug)]
pub enum CatalogEntry<'a> {
    /// A tool rig dispatches; its [`origin`](RegisteredTool::origin) says
    /// where it runs.
    Registered(&'a RegisteredTool),
    /// A tool the provider runs on its side.
    ProviderHosted(&'a ProviderToolDefinition),
}

impl ToolCatalog {
    /// A catalog over `tools`, advertised under their own definitions, and
    /// `provider_tools`.
    pub(crate) fn from_registered(
        tools: IndexMap<String, RegisteredTool>,
        provider_tools: Vec<ProviderToolDefinition>,
    ) -> Self {
        let definitions = tools
            .values()
            .map(|tool| tool.definition().clone())
            .collect();
        Self {
            definitions,
            tools,
            provider_tools,
            leases: Vec::new(),
        }
    }

    /// Attach the registry's generation leases (see the field).
    pub(crate) fn with_leases(mut self, leases: Vec<ToolLease>) -> Self {
        self.leases = leases;
        self
    }

    /// The advertised definitions of the registered tools.
    pub fn definitions(&self) -> &[ToolDefinition] {
        &self.definitions
    }

    /// The provider-hosted tools every request declares.
    pub fn provider_tools(&self) -> &[ProviderToolDefinition] {
        &self.provider_tools
    }

    /// Every entry: the registered tools in advertisement order, then the
    /// provider-hosted tools.
    pub fn entries(&self) -> impl Iterator<Item = CatalogEntry<'_>> {
        self.tools
            .values()
            .map(CatalogEntry::Registered)
            .chain(self.provider_tools.iter().map(CatalogEntry::ProviderHosted))
    }

    /// The advertised names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tools.keys().map(String::as_str)
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the catalog has no registered tools.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Whether `name` is in the catalog.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// The registration behind `name`.
    pub fn get(&self, name: &str) -> Option<&RegisteredTool> {
        self.tools.get(name)
    }

    /// The bus key behind `name`, when the catalog has it.
    pub fn key(&self, name: &str) -> Option<&Key<family::Tool>> {
        self.tools.get(name).map(RegisteredTool::key)
    }

    /// Every registration, in advertisement order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &RegisteredTool)> {
        self.tools.iter().map(|(name, tool)| (name.as_str(), tool))
    }

    /// Run `tool_name` inline, publishing its result metadata into
    /// `context`.
    pub async fn execute(
        &self,
        tool_name: &str,
        args: &str,
        context: &mut ToolContext,
    ) -> ToolResult {
        context.clear_dispatch_result();
        self.execute_scoped(tool_name, args, context)
            .await
            .publish_to(context)
    }

    /// Take the definitions out, leaving the registrations.
    pub fn take_definitions(&mut self) -> Vec<ToolDefinition> {
        std::mem::take(&mut self.definitions)
    }

    /// Keep only `names`.
    pub(crate) fn retain_names(&mut self, names: &BTreeSet<String>) {
        self.definitions
            .retain(|definition| names.contains(definition.name.as_str()));
        self.tools.retain(|name, _| names.contains(name));
    }

    /// Run `tool_name` on a dispatch-scoped copy of `context`.
    pub async fn execute_scoped(
        &self,
        tool_name: &str,
        args: &str,
        context: &ToolContext,
    ) -> ToolDispatch {
        let tool = self.tools.get(tool_name).cloned();
        execute_tool(tool_name, args.to_string(), tool, context).await
    }
}

#[cfg(test)]
mod tests;
