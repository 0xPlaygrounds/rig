//! The advertisement snapshot of a tool set: the definitions a request
//! carries plus the registrations behind them, pinned at snapshot time.

use std::collections::BTreeSet;

use indexmap::IndexMap;

use rig_core::{
    completion::ToolDefinition,
    effect::{Key, family},
    tool::{ToolContext, ToolResult},
};

use super::registry::{RegisteredTool, ToolDispatch, dispatch_tool};

/// An opaque token a catalog keeps alive for as long as it exists: a
/// registry that retires replaced generations lazily hands one per pinned
/// registration and sweeps a generation once no catalog holds its token —
/// the token's own drop is the registry's cue.
#[cfg(not(target_family = "wasm"))]
pub type ToolLease = std::sync::Arc<dyn std::any::Any + Send + Sync>;
/// An opaque token a catalog keeps alive for as long as it exists (browser
/// wasm: no `Send + Sync`, no threads).
#[cfg(target_family = "wasm")]
pub type ToolLease = std::sync::Arc<dyn std::any::Any>;

/// The tools one request advertises, with the exact registrations that
/// serve them. A catalog is a snapshot: replacing a tool in the registry
/// after the snapshot does not change what the snapshot dispatches to.
#[derive(Clone)]
pub struct ToolCatalog {
    definitions: Vec<ToolDefinition>,
    tools: IndexMap<String, RegisteredTool>,
    /// The tokens the catalog keeps alive for as long as it exists (see
    /// [`ToolLease`]).
    leases: Vec<ToolLease>,
}

impl ToolCatalog {
    /// A catalog over `tools`, advertised under their map names.
    pub fn from_registered(tools: IndexMap<String, RegisteredTool>) -> Self {
        let definitions = tools
            .iter()
            .map(|(name, tool)| tool.definition_with_name(name.clone()))
            .collect();
        Self {
            definitions,
            tools,
            leases: Vec::new(),
        }
    }

    /// Attach the registry's generation leases (see the field).
    pub fn with_leases(mut self, leases: Vec<ToolLease>) -> Self {
        self.leases = leases;
        self
    }

    /// The advertised definitions.
    pub fn definitions(&self) -> &[ToolDefinition] {
        &self.definitions
    }

    /// The advertised names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tools.keys().map(String::as_str)
    }

    /// Number of tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the catalog is empty.
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
        self.dispatch(tool_name, args, context)
            .await
            .publish_to(context)
    }

    /// [`Self::execute`] by value.
    pub async fn execute_owned(
        self,
        tool_name: String,
        args: String,
        mut context: ToolContext,
    ) -> (ToolResult, ToolContext) {
        let result = self.execute(&tool_name, &args, &mut context).await;
        (result, context)
    }

    /// Take the definitions out, leaving the registrations.
    pub fn take_definitions(&mut self) -> Vec<ToolDefinition> {
        std::mem::take(&mut self.definitions)
    }

    /// Keep only `names`.
    pub fn retain_names(&mut self, names: &BTreeSet<String>) {
        self.definitions
            .retain(|definition| names.contains(&definition.name));
        self.tools.retain(|name, _| names.contains(name));
    }

    /// Run `tool_name` on a dispatch-scoped copy of `context`.
    pub async fn dispatch(
        &self,
        tool_name: &str,
        args: &str,
        context: &ToolContext,
    ) -> ToolDispatch {
        let tool = self.tools.get(tool_name).cloned();
        dispatch_tool(tool_name, args.to_string(), tool, context).await
    }
}

#[cfg(test)]
mod tests;
