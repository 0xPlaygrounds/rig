//! A pinned, retrieval-free view of a set of tools: definitions plus dispatch.

use std::collections::BTreeSet;

use indexmap::IndexMap;

use crate::completion::ToolDefinition;

use super::{
    ToolContext, ToolResult,
    contextual::{RegisteredTool, ToolDispatch, dispatch_tool},
};

/// A pinned view of a tool registry: provider definitions plus the exact
/// implementations behind them.
///
/// A driver takes one per model turn, so registration changes after a
/// catalog is built take effect on the next turn and calls from the current
/// turn dispatch through these pinned handles — the implementation cannot
/// drift from the schema the provider received. Build one from a
/// [`ToolSet`](super::ToolSet) with [`ToolSet::catalog`](super::ToolSet::catalog)
/// (always-exposed tools, registration order) or from an explicit ordered
/// map with [`from_registered`](Self::from_registered) (how `rig-agent`'s
/// registry prepends retrieved tools). Read
/// [`definitions`](Self::definitions) / [`names`](Self::names) or
/// [`execute`](Self::execute) against it without touching any live registry.
///
/// Cloning shares the pinned tool handles (they are `Arc`s) and copies the
/// definitions.
#[derive(Clone)]
pub struct ToolCatalog {
    definitions: Vec<ToolDefinition>,
    tools: IndexMap<String, RegisteredTool>,
}

impl ToolCatalog {
    /// Pin an ordered map of registered tools; each definition is advertised
    /// under its map key.
    pub fn from_registered(tools: IndexMap<String, RegisteredTool>) -> Self {
        let definitions = tools
            .iter()
            .map(|(name, tool)| tool.definition_with_name(name.clone()))
            .collect();
        Self { definitions, tools }
    }

    /// Provider-facing definitions in the same order as their pinned handles.
    pub fn definitions(&self) -> &[ToolDefinition] {
        &self.definitions
    }

    /// Registered names in exposure order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tools.keys().map(String::as_str)
    }

    /// Number of pinned tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the catalog pins no tools.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Whether `name` is pinned.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Execute a pinned tool by name through the canonical structured path,
    /// publishing its result metadata back to `context`. Later registry
    /// changes do not affect which implementation runs.
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

    /// Moves the definitions out of the catalog. Per-turn request assembly is
    /// the usual consumer and never reads them again, so it takes them
    /// instead of deep-cloning every tool's JSON schema each turn.
    pub fn take_definitions(&mut self) -> Vec<ToolDefinition> {
        std::mem::take(&mut self.definitions)
    }

    /// Narrow both provider exposure and dispatch to one allow-list.
    pub fn retain_names(&mut self, names: &BTreeSet<String>) {
        self.definitions
            .retain(|definition| names.contains(&definition.name));
        self.tools.retain(|name, _| names.contains(name));
    }

    /// Dispatch through the exact implementation advertised for this turn,
    /// keeping the per-dispatch context (for hooks) instead of publishing it.
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
mod tests {
    use super::*;
    use crate::tool::{PortableDynamicTool, ToolOutput, ToolSet};

    fn portable(name: &str) -> PortableDynamicTool {
        let reply = format!("{name}!");
        PortableDynamicTool::new(
            name,
            format!("the {name} tool"),
            serde_json::json!({"type": "object"}),
            move |_| {
                let reply = reply.clone();
                Box::pin(async move { Ok(ToolOutput::text(reply)) })
            },
        )
    }

    /// `ToolSet::catalog()` advertises exactly the always-exposed tools, in
    /// registration order, and dispatches by the advertised name.
    #[tokio::test]
    async fn catalog_matches_definitions_and_dispatches_by_name() {
        let mut set = ToolSet::default();
        set.add_portable_dynamic_tool(portable("alpha"));
        set.add_portable_dynamic_tool(portable("beta"));
        let mut retrieval_only = ToolSet::default();
        retrieval_only.add_portable_dynamic_tool(portable("gamma"));
        set.add_retrievable_tools(retrieval_only);

        let catalog = set.catalog();
        assert_eq!(catalog.names().collect::<Vec<_>>(), ["alpha", "beta"]);
        assert_eq!(catalog.definitions().len(), 2);
        assert_eq!(
            catalog.definitions(),
            &set.tool_definitions()[..2],
            "always-exposed definitions are the set's, in order"
        );
        assert!(set.contains("gamma") && !catalog.contains("gamma"));

        let mut context = ToolContext::new();
        let result = catalog.execute("beta", "{}", &mut context).await;
        assert_eq!(result.output().as_text(), Some("beta!"));

        let missing = catalog.execute("gamma", "{}", &mut context).await;
        assert!(!missing.is_success(), "retrieval-only tools are not pinned");
    }

    #[tokio::test]
    async fn retain_names_narrows_definitions_and_dispatch() {
        let mut set = ToolSet::default();
        set.add_portable_dynamic_tool(portable("alpha"));
        set.add_portable_dynamic_tool(portable("beta"));
        let mut catalog = set.catalog();
        catalog.retain_names(&BTreeSet::from(["beta".to_string()]));
        assert_eq!(catalog.names().collect::<Vec<_>>(), ["beta"]);
        assert_eq!(catalog.take_definitions().len(), 1);
        assert!(catalog.definitions().is_empty());
        assert!(
            !catalog
                .execute("alpha", "{}", &mut ToolContext::new())
                .await
                .is_success()
        );
    }
}
