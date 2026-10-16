//! Request-scoped tool advertisements and their pinned registrations.
//!
//! ```
//! use rig_agent::tool::ToolCatalog;
//! fn advertised_names(catalog: &ToolCatalog) -> Vec<&str> {
//!     catalog.names().collect()
//! }
//! ```

use std::collections::BTreeSet;

use indexmap::IndexMap;

use rig_core::{
    completion::ToolDefinition,
    effect::{Key, family},
    tool::{ToolContext, ToolResult},
};

use super::registry::{RegisteredTool, ToolDispatch, execute_tool};

/// The tools one request advertises, with the exact registrations that
/// serve them: a call runs the registration the catalog pinned, even after
/// the registry replaced or removed it.
#[derive(Clone)]
pub struct ToolCatalog {
    definitions: Vec<ToolDefinition>,
    tools: IndexMap<String, RegisteredTool>,
}

impl ToolCatalog {
    /// A catalog over `tools`, advertised under their own definitions.
    pub(crate) fn from_registered(tools: IndexMap<String, RegisteredTool>) -> Self {
        let definitions = tools.values().map(RegisteredTool::definition).collect();
        Self { definitions, tools }
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

    /// The key behind `name`, when the catalog has it.
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
