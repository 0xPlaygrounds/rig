use std::sync::Arc;

use std::collections::HashMap;

use indexmap::IndexMap;
use std::sync::{PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::tool::ErasedTool;

use crate::{
    completion::{CompletionError, ToolDefinition},
    tool::{
        DynamicTool, PortableDynamicTool, RegisteredTool, Tool, ToolCatalog, ToolContext,
        ToolDispatch, ToolResult, ToolSet, dispatch_tool,
    },
};
use rig_core::vector_store::{
    VectorSearchRequest, VectorStoreError, VectorStoreIndexDyn, request::Filter,
};

/// A pinned view of the tool registry: provider definitions plus the exact
/// implementations behind them — [`rig_core::tool::ToolCatalog`] under the
/// name the agent runtime has always used.
///
/// The agent loop takes one per turn ([`ToolServerHandle::snapshot`] for the
/// registry as it stands, the retrieval-aware `snapshot_tool_defs` for a
/// prompt), so registration changes after a snapshot is built take effect on
/// the next turn and calls from the current turn dispatch through these
/// pinned handles.
pub type ToolRegistrySnapshot = ToolCatalog;

/// Shared state behind a `ToolServerHandle`.
struct ToolServerState {
    /// Vector indexes used to select retrieval-only tools for each prompt.
    retrieval_indexes: Vec<(usize, Arc<dyn VectorStoreIndexDyn>)>,
    /// The authoritative ordered registry for execution and exposure.
    toolset: ToolSet,
    /// Generation tokens for registrations owned by external tool sources.
    /// A normal registration clears the token, preventing a stale handler
    /// refresh from replacing or removing the newer tool.
    managed_generations: HashMap<String, ManagedToolToken>,
}

impl ToolServerState {
    /// Remove remote registrations whose transport can no longer accept calls.
    /// In-process tools use the default live state, while both handler-managed
    /// and directly registered MCP tools report their transport state.
    fn retire_disconnected_tools(&mut self) {
        let disconnected = self
            .toolset
            .names()
            .filter(|name| self.toolset.get(name).is_none_or(|tool| !tool.is_live()))
            .map(str::to_owned)
            .collect::<Vec<_>>();

        for name in disconnected {
            self.toolset.delete_tool(&name);
            self.managed_generations.remove(&name);
            tracing::debug!(tool_name = %name, "retired disconnected tool registration");
        }
    }
}

pub use rig_core::tool::{ManagedToolSink, ManagedToolToken};

/// Builder for constructing a [`ToolServerHandle`].
///
/// Accumulates tools and configuration, then produces a shared handle via
/// [`run()`](ToolServer::run).
pub struct ToolServer {
    retrieval_indexes: Vec<(usize, Arc<dyn VectorStoreIndexDyn>)>,
    toolset: ToolSet,
}

impl Default for ToolServer {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolServer {
    pub fn new() -> Self {
        Self {
            retrieval_indexes: Vec::new(),
            toolset: ToolSet::default(),
        }
    }

    /// Add a static tool to the agent. Re-registering an existing name
    /// replaces the implementation (last wins) and keeps its position.
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        self.toolset.add_tool(tool);
        self
    }

    /// Add a runtime-defined tool. Re-registering an existing name replaces
    /// the implementation and keeps its position.
    pub fn dynamic_tool(mut self, tool: DynamicTool) -> Self {
        self.toolset.add_dynamic_tool(tool);
        self
    }

    /// Add several runtime-defined tools in order.
    pub fn dynamic_tools(self, tools: Vec<DynamicTool>) -> Self {
        tools.into_iter().fold(self, Self::dynamic_tool)
    }

    /// Add a context-free dynamic tool through the classic registry adapter.
    pub fn portable_dynamic_tool(mut self, tool: PortableDynamicTool) -> Self {
        self.toolset.add_portable_dynamic_tool(tool);
        self
    }

    /// Register a pre-erased tool — the extension point for adapters that
    /// implement [`ErasedTool`] directly (remote tool protocols such as MCP,
    /// provided by companion crates).
    pub fn erased_tool(mut self, tool: Arc<dyn ErasedTool>) -> Self {
        self.toolset.add_erased(tool);
        self
    }

    /// Configure tools retrieved from a vector index for each prompt.
    pub fn retrieved_tools(
        mut self,
        sample: usize,
        index: impl VectorStoreIndexDyn + 'static,
        toolset: ToolSet,
    ) -> Self {
        self.retrieval_indexes.push((sample, Arc::new(index)));
        self.toolset.add_retrievable_tools(toolset);
        self
    }

    /// Consume the builder and return a shared [`ToolServerHandle`].
    pub fn run(self) -> ToolServerHandle {
        ToolServerHandle(Arc::new(RwLock::new(ToolServerState {
            retrieval_indexes: self.retrieval_indexes,
            toolset: self.toolset,
            managed_generations: HashMap::new(),
        })))
    }
}

/// A cheaply-cloneable handle to the shared tool server state.
///
/// All operations acquire locks directly on the underlying state.
/// Multiple handles (e.g. across agents) can share the same state
/// without channel-based message routing.
#[derive(Clone)]
pub struct ToolServerHandle(Arc<RwLock<ToolServerState>>);

impl ToolServerHandle {
    /// Shared registry state under the single poisoning policy: a panic
    /// inside one of the short sync critical sections cannot leave the
    /// registry logically torn, so a poisoned lock is recovered rather than
    /// propagated.
    fn state(&self) -> RwLockReadGuard<'_, ToolServerState> {
        self.0.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Exclusive registry state; same poisoning policy as [`Self::state`].
    fn state_mut(&self) -> RwLockWriteGuard<'_, ToolServerState> {
        self.0.write().unwrap_or_else(PoisonError::into_inner)
    }

    /// Register through `add`, then drop any stale managed-generation
    /// entry so the (re)registered name follows last-registration-wins.
    fn register(&self, add: impl FnOnce(&mut ToolSet) -> String) {
        let mut state = self.state_mut();
        let _name = add(&mut state.toolset);
        state.managed_generations.remove(&_name);
    }

    /// Register a new static tool. Re-registering an existing name replaces
    /// the implementation (last wins) and keeps its position.
    pub fn add_tool<T>(&self, tool: T)
    where
        T: Tool + 'static,
    {
        self.register(|toolset| toolset.add_tool(tool));
    }

    /// Register a runtime-defined static tool.
    pub fn add_dynamic_tool(&self, tool: DynamicTool) {
        self.register(|toolset| toolset.add_dynamic_tool(tool));
    }

    /// Register a context-free dynamic tool through the classic adapter.
    pub fn add_portable_dynamic_tool(&self, tool: PortableDynamicTool) {
        self.register(|toolset| toolset.add_portable_dynamic_tool(tool));
    }

    /// Atomically install the initial tools owned by one external tool source
    /// (an MCP client handler, for example). Last-registration-wins: an existing
    /// name is replaced. Tools that report `!is_live()` are skipped. Returns the
    /// generation token per installed name, to hand back to
    /// [`Self::reconcile_managed_erased_tools`] on refresh.
    pub fn add_managed_erased_tools(
        &self,
        tools: Vec<Arc<dyn ErasedTool>>,
    ) -> HashMap<String, ManagedToolToken> {
        let mut state = self.state_mut();
        let mut managed = HashMap::with_capacity(tools.len());

        for tool in tools {
            // The initial list fetch can complete just before the transport
            // closes. Avoid installing a registration that can never execute.
            if !tool.is_live() {
                tracing::debug!(
                    tool_name = %tool.name(),
                    "ignored initial registration from disconnected MCP owner"
                );
                continue;
            }

            let name = state.toolset.add_erased(tool);
            let token = ManagedToolToken::new();
            state
                .managed_generations
                .insert(name.clone(), token.clone());
            managed.insert(name, token);
        }

        managed
    }

    /// Atomically reconcile one external source's registrations with a
    /// refreshed tool list. Existing names are changed only when their expected
    /// generation remains current; newer local or peer-source registrations
    /// win. Names missing from the new list (and owned by this source) are
    /// removed. Returns the new generation tokens.
    pub fn reconcile_managed_erased_tools(
        &self,
        mut expected: HashMap<String, ManagedToolToken>,
        tools: Vec<Arc<dyn ErasedTool>>,
    ) -> HashMap<String, ManagedToolToken> {
        let mut state = self.state_mut();
        let mut refreshed = HashMap::with_capacity(tools.len());
        let mut managed_order = Vec::with_capacity(tools.len());
        let mut seen = std::collections::HashSet::with_capacity(tools.len());

        // A generation only protects a live owner. MCP service shutdown closes
        // the sink held by its registered tools, so retire those generations
        // before deciding whether another handler may reclaim a name. Local
        // in-process registrations stay live; directly registered MCP tools are
        // also retired when their sink closes.
        state.retire_disconnected_tools();

        for tool in tools {
            // A refresh that raced with service shutdown may already have
            // fetched definitions before the transport closed. Do not let
            // that stale refresh recreate an owner we just retired.
            if !tool.is_live() {
                tracing::debug!(
                    tool_name = %tool.name(),
                    "ignored registration from disconnected MCP owner"
                );
                continue;
            }

            let name = tool.name();
            if !seen.insert(name.clone()) {
                tracing::warn!(tool_name = %name, "ignoring duplicate MCP tool definition");
                continue;
            }
            let present = state.toolset.contains(&name);
            let may_register = match expected.remove(&name) {
                Some(token) if present => state.managed_generations.get(&name) == Some(&token),
                // A stale expected token protects a live newer registration,
                // not an empty slot. Once the competitor disappears, this full
                // server snapshot must converge in one reconciliation.
                Some(_) => true,
                None => !present,
            };

            if may_register {
                state.toolset.add_erased(tool);
                let token = ManagedToolToken::new();
                state
                    .managed_generations
                    .insert(name.clone(), token.clone());
                refreshed.insert(name.clone(), token);
                managed_order.push(name);
            } else {
                tracing::debug!(
                    tool_name = name,
                    "MCP refresh left a newer same-name registration intact"
                );
            }
        }

        for (name, token) in expected {
            if state.managed_generations.get(&name) == Some(&token) {
                state.toolset.delete_tool(&name);
                state.managed_generations.remove(&name);
            }
        }

        // A full MCP list is ordered. Move only entries this handler actually
        // owns to the end in that order, matching remove/re-register semantics;
        // live local or peer-handler competitors retain their relative slots.
        for name in managed_order {
            state.toolset.move_to_end(&name);
        }

        refreshed
    }

    /// Merge an entire toolset into the server in registration order.
    /// Existing names are replaced (last wins) and keep their position.
    pub fn append_toolset(&self, toolset: ToolSet) {
        let mut state = self.state_mut();
        let names = toolset.names().map(str::to_owned).collect::<Vec<_>>();
        state.toolset.add_tools(toolset);
        for name in names {
            state.managed_generations.remove(&name);
        }
    }

    /// Remove a tool by name.
    pub fn remove_tool(&self, tool_name: &str) {
        let mut state = self.state_mut();
        state.toolset.delete_tool(tool_name);
        state.managed_generations.remove(tool_name);
    }

    /// Look up and execute a tool through the canonical structured path.
    ///
    /// The implementation handle is cloned under a brief read lock, so a long
    /// execution never blocks registration changes. The tool receives one
    /// snapshot of the supplied inbound values. Its result metadata is
    /// published back to `context`, while mutations to its inbound snapshot are
    /// discarded.
    pub async fn execute(
        &self,
        tool_name: &str,
        args: &str,
        context: &mut ToolContext,
    ) -> ToolResult {
        context.clear_dispatch_result();
        let dispatch = self.dispatch(tool_name, args, context).await;
        dispatch.publish_to(context)
    }

    /// Run `f` against the registry state, first retiring disconnected MCP
    /// tools (which needs a write lock) when that feature is compiled in.
    fn with_registry<R>(&self, f: impl FnOnce(&ToolServerState) -> R) -> R {
        let mut state = self.state_mut();
        state.retire_disconnected_tools();
        f(&state)
    }

    /// Run one isolated dispatch and retain its full context for agent hooks.
    pub(crate) async fn dispatch(
        &self,
        tool_name: &str,
        args: &str,
        context: &ToolContext,
    ) -> ToolDispatch {
        let tool = self.with_registry(|state| state.toolset.get(tool_name).cloned());
        dispatch_tool(tool_name, args.to_string(), tool, context).await
    }

    /// The registry as it stands, synchronously: every always-exposed
    /// registration in registration order, after retiring tools whose remote
    /// backing disconnected — the same path [`execute`](Self::execute) and
    /// the agent loop resolve through. No retrieval, no executor, no `.await`,
    /// so a tick-driven host can call it every frame.
    ///
    /// For the retrieval-aware view that also selects dynamic tools for a
    /// prompt, use the async [`tool_defs`](Self::tool_defs).
    pub fn snapshot(&self) -> ToolRegistrySnapshot {
        let tools = self.with_registry(|state| snapshot_registered_tools(state, &[]));
        ToolCatalog::from_registered(tools)
    }

    /// Provider definitions of the registry as it stands — the definitions of
    /// [`snapshot`](Self::snapshot), synchronously. Equivalent to
    /// `tool_defs(None)` without the future.
    pub fn static_tool_defs(&self) -> Vec<ToolDefinition> {
        let mut snapshot = self.snapshot();
        snapshot.take_definitions()
    }

    /// A clone of the current registry as a [`ToolSet`]: shares the tool
    /// implementations (they are `Arc`s) and copies names, ordering, and
    /// exposure flags. Use it to fork the registry — build a second server
    /// with the same tools — or inspect it outside the lock. Disconnected
    /// tools are retired first.
    pub fn toolset(&self) -> ToolSet {
        self.with_registry(|state| state.toolset.clone())
    }

    /// Retrieve tool definitions, optionally using a prompt to select
    /// dynamic tools from configured vector stores.
    ///
    /// This is the retrieval-aware, async read: with a prompt it runs the
    /// configured vector-store lookups to pick dynamic tools. If you only need
    /// the registry as it stands, [`static_tool_defs`](Self::static_tool_defs)
    /// / [`snapshot`](Self::snapshot) give the same always-exposed definitions
    /// synchronously.
    pub async fn tool_defs(
        &self,
        prompt: Option<String>,
    ) -> Result<Vec<ToolDefinition>, ToolServerError> {
        Ok(self.snapshot_tool_defs(prompt).await?.take_definitions())
    }

    /// Resolve one ordered provider/dispatch snapshot for an agent turn.
    ///
    /// Retrieval runs without holding the registry lock. Once the selected IDs
    /// are known, one read lock resolves every dynamic and always-exposed name
    /// to an exact implementation. That single instant is the turn boundary:
    /// later replacements are visible only to the next snapshot.
    pub(crate) async fn snapshot_tool_defs(
        &self,
        prompt: Option<String>,
    ) -> Result<ToolRegistrySnapshot, ToolServerError> {
        let retrieval_indexes = {
            let state = self.state();
            state.retrieval_indexes.clone()
        };

        let dynamic_tool_ids = if let Some(ref text) = prompt {
            // Create a future for each dynamic tool index
            let search_futures = retrieval_indexes.iter().map(|(num_sample, index)| {
                let text = text.clone();
                let num_sample = *num_sample;
                let index = index.clone();

                async move {
                    let req = VectorSearchRequest::builder()
                        .query(text)
                        .samples(num_sample as u64)
                        .build();

                    let ids = index
                        .as_ref()
                        .top_n_ids(req.map_filter(Filter::interpret))
                        .await?
                        .into_iter()
                        .map(|(_, id)| id)
                        .collect::<Vec<String>>();

                    Ok::<_, VectorStoreError>(ids)
                }
            });

            // Execute searches concurrently and collect/flatten the IDs
            futures::future::try_join_all(search_futures)
                .await
                .map_err(|e| {
                    ToolServerError::DefinitionError(CompletionError::RequestError(Box::new(e)))
                })?
                .into_iter()
                .flatten()
                .collect::<Vec<String>>()
        } else {
            Vec::new()
        };

        let tools = self.with_registry(|state| snapshot_registered_tools(state, &dynamic_tool_ids));

        Ok(ToolCatalog::from_registered(tools))
    }
}

fn snapshot_registered_tools(
    state: &ToolServerState,
    dynamic_tool_ids: &[String],
) -> IndexMap<String, RegisteredTool> {
    let mut tools = IndexMap::new();
    let insert = |tools: &mut IndexMap<String, RegisteredTool>, name: &str, warn_missing| {
        if tools.contains_key(name) {
            tracing::debug!(
                tool_name = %name,
                "dropping duplicate tool definition from the request"
            );
            return;
        }
        match state.toolset.get(name).cloned() {
            Some(tool) => {
                tools.insert(name.to_string(), tool);
            }
            // A dynamic ID the model asked for but the toolset lacks is worth
            // an operator warning; a retired always-exposed tool is not.
            None if warn_missing => {
                tracing::warn!("Tool implementation not found in toolset: {name}");
            }
            None => {}
        }
    };

    // Retrieved tools remain first, in index/result order. Duplicate IDs and
    // dynamic/static overlap retain the first provider declaration.
    for name in dynamic_tool_ids {
        insert(&mut tools, name, true);
    }
    for name in state.toolset.always_exposed_names() {
        insert(&mut tools, name, false);
    }
    tools
}

// Compile-time thread-safety contract: a registry snapshot or a forked
// `ToolSet` is held in shared host state on native targets.
#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<ToolSet>();
    assert_send_sync_static::<ToolRegistrySnapshot>();
};

#[derive(Debug, thiserror::Error)]
pub enum ToolServerError {
    #[error("Failed to retrieve tool definitions: {0}")]
    DefinitionError(CompletionError),
}
/// The registry contract external tool sources (e.g. `rig-rmcp`'s MCP client
/// handler) program against: portable tools in, generation tokens out.
impl ManagedToolSink for ToolServerHandle {
    fn add_managed_tools(
        &self,
        tools: Vec<PortableDynamicTool>,
    ) -> HashMap<String, ManagedToolToken> {
        self.add_managed_erased_tools(
            tools
                .into_iter()
                .map(|tool| Arc::new(DynamicTool::from(tool)) as Arc<dyn ErasedTool>)
                .collect(),
        )
    }

    fn reconcile_managed_tools(
        &self,
        expected: HashMap<String, ManagedToolToken>,
        tools: Vec<PortableDynamicTool>,
    ) -> HashMap<String, ManagedToolToken> {
        self.reconcile_managed_erased_tools(
            expected,
            tools
                .into_iter()
                .map(|tool| Arc::new(DynamicTool::from(tool)) as Arc<dyn ErasedTool>)
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests;
