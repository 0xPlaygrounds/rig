//! The runtime tool registry an agent advertises from and dispatches
//! through.
//!
//! A [`ToolServerHandle`] is the definition/advertisement surface: it owns
//! the [`ToolSet`] (descriptors plus staged handlers) and publishes every
//! registration onto the buses attached to it — each agent built with the
//! handle attaches its bus at build time, and later additions, removals and
//! MCP reconciles are pushed to every attached bus as they happen. A
//! request's snapshot ([`ToolCatalog`]) pins the *generation* of each tool:
//! registrations are served under generation-qualified keys
//! (`tool:<name>#<n>`), a replacement registers a new generation, and a
//! generation is retired from the buses only once no snapshot references
//! it. Execution during a run goes through the bus; the inline `execute`
//! here serves the standalone use (no agent, no bus).

use std::collections::HashMap;
use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard, Weak};

use indexmap::IndexMap;
use rig_core::{
    bus::{Dispatcher, adapters::RetrieveAdapter},
    effect::HandlerKey,
    vector_store::{VectorStoreIndex, request::DynamicSearchFilter},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

use crate::{
    completion::{CompletionError, ToolDefinition},
    tool::{
        DynamicTool, PortableDynamicTool, RegisteredTool, Tool, ToolCatalog, ToolContext,
        ToolDispatch, ToolResult, ToolSet, dispatch_tool,
    },
};

/// The per-request snapshot of the registry: a [`ToolCatalog`].
pub type ToolRegistrySnapshot = ToolCatalog;

/// A retrieval index registered for tool retrieval: its bus key and how
/// many tools to sample.
#[derive(Clone)]
struct RetrievalIndex {
    samples: usize,
    key: HandlerKey,
    handler: rig_core::bus::ErasedHandler,
}

/// A retired generation waiting for its last snapshot to drop.
struct RetiredGeneration {
    key: HandlerKey,
    lease: Weak<()>,
}

struct ToolServerState {
    retrieval_indexes: Vec<RetrievalIndex>,
    toolset: ToolSet,
    /// The lease each live registration hands to snapshots.
    leases: HashMap<String, Arc<()>>,
    retired: Vec<RetiredGeneration>,
    managed_generations: HashMap<String, ManagedToolToken>,
    /// The buses this registry publishes onto.
    buses: Vec<Dispatcher>,
    next_generation: u64,
}

impl ToolServerState {
    fn generation_key(&mut self, name: &str) -> HandlerKey {
        let n = self.next_generation;
        self.next_generation += 1;
        HandlerKey::from(format!("tool:{name}#{n}"))
    }

    fn publish(&self, tool: &RegisteredTool) {
        for bus in &self.buses {
            bus.register_erased(tool.key().clone(), tool.handler().clone());
        }
    }

    fn unpublish(&self, key: &HandlerKey) {
        for bus in &self.buses {
            bus.deregister(key);
        }
    }

    /// Retire the current generation of `name`: the key stays served until
    /// every snapshot holding it has dropped.
    fn retire(&mut self, name: &str) {
        if let (Some(tool), Some(lease)) = (self.toolset.get(name), self.leases.remove(name)) {
            self.retired.push(RetiredGeneration {
                key: tool.key().clone(),
                lease: Arc::downgrade(&lease),
            });
        }
    }

    fn sweep_retired(&mut self) {
        let retired = std::mem::take(&mut self.retired);
        let mut kept = Vec::with_capacity(retired.len());
        for retired in retired {
            if retired.lease.strong_count() == 0 {
                self.unpublish(&retired.key);
            } else {
                kept.push(retired);
            }
        }
        self.retired = kept;
        self.buses.retain(|bus| !bus.is_closed());
    }

    /// Insert a registration under a fresh generation, publishing it.
    fn register(&mut self, tool: RegisteredTool, always_exposed: bool) -> String {
        let name = tool.name();
        // A registration under the default `tool:<name>` key gets a fresh
        // generation; an explicit key (a replayer's recorded key, a host's
        // own) is served as given.
        let tool = if tool.has_default_key() {
            let key = self.generation_key(&name);
            tool.with_key(key)
        } else {
            tool
        };
        self.retire(&name);
        self.publish(&tool);
        self.leases.insert(name.clone(), Arc::new(()));
        if always_exposed {
            self.toolset.add_registered(tool);
        } else {
            let mut set = ToolSet::default();
            set.add_registered(tool);
            self.toolset.add_retrievable_tools(set);
        }
        name
    }

    fn remove(&mut self, name: &str) {
        self.retire(name);
        self.toolset.delete_tool(name);
        self.managed_generations.remove(name);
    }

    fn retire_disconnected_tools(&mut self) {
        let disconnected = self
            .toolset
            .names()
            .filter(|name| self.toolset.get(name).is_none_or(|tool| !tool.is_live()))
            .map(str::to_owned)
            .collect::<Vec<_>>();

        for name in disconnected {
            self.remove(&name);
            tracing::debug!(tool_name = %name, "retired disconnected tool registration");
        }
        self.sweep_retired();
    }

    fn attach(&mut self, bus: &Dispatcher) {
        for (_, tool) in self.toolset.iter() {
            bus.register_erased(tool.key().clone(), tool.handler().clone());
        }
        for index in &self.retrieval_indexes {
            bus.register_erased(index.key.clone(), index.handler.clone());
        }
        self.buses.push(bus.clone());
    }
}

pub use rig_core::tool::{ManagedToolSink, ManagedToolToken};

/// A tool registry under construction; [`ToolServer::run`] turns it into
/// the shareable [`ToolServerHandle`].
pub struct ToolServer {
    retrieval_indexes: Vec<RetrievalIndex>,
    toolset: ToolSet,
    next_index: usize,
}

impl Default for ToolServer {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolServer {
    /// An empty registry.
    pub fn new() -> Self {
        Self {
            retrieval_indexes: Vec::new(),
            toolset: ToolSet::default(),
            next_index: 0,
        }
    }

    /// Add a typed tool.
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        self.toolset.add_tool(tool);
        self
    }

    /// Add a runtime-defined tool.
    pub fn dynamic_tool(mut self, tool: DynamicTool) -> Self {
        self.toolset.add_dynamic_tool(tool);
        self
    }

    /// Add runtime-defined tools.
    pub fn dynamic_tools(self, tools: Vec<DynamicTool>) -> Self {
        tools.into_iter().fold(self, Self::dynamic_tool)
    }

    /// Add a portable tool.
    pub fn portable_dynamic_tool(mut self, tool: PortableDynamicTool) -> Self {
        self.toolset.add_portable_dynamic_tool(tool);
        self
    }

    /// Add a registration built elsewhere.
    pub fn registered_tool(mut self, tool: RegisteredTool) -> Self {
        self.toolset.add_registered(tool);
        self
    }

    /// Add retrievable tools: `sample` of them are advertised per request,
    /// chosen by `index`.
    pub fn retrieved_tools<I, F>(mut self, sample: usize, index: I, toolset: ToolSet) -> Self
    where
        I: VectorStoreIndex<Filter = F> + 'static,
        F: DynamicSearchFilter + WasmCompatSend + WasmCompatSync + 'static,
    {
        let n = self.next_index;
        self.next_index += 1;
        self.retrieval_indexes.push(RetrievalIndex {
            samples: sample,
            key: HandlerKey::from(format!("retrieve:tools#{n}")),
            handler: rig_core::bus::ErasedHandler::new(RetrieveAdapter::new(index)),
        });
        self.toolset.add_retrievable_tools(toolset);
        self
    }

    /// Start the registry.
    pub fn run(self) -> ToolServerHandle {
        let mut state = ToolServerState {
            retrieval_indexes: self.retrieval_indexes,
            toolset: ToolSet::default(),
            leases: HashMap::new(),
            retired: Vec::new(),
            managed_generations: HashMap::new(),
            buses: Vec::new(),
            next_generation: 0,
        };
        let exposed: Vec<String> = self
            .toolset
            .always_exposed_names()
            .map(str::to_owned)
            .collect();
        for (name, tool) in self.toolset.iter() {
            state.register(tool.clone(), exposed.iter().any(|exposed| exposed == name));
        }
        ToolServerHandle(Arc::new(RwLock::new(state)))
    }
}

/// The shareable, runtime-mutable tool registry.
#[derive(Clone)]
pub struct ToolServerHandle(Arc<RwLock<ToolServerState>>);

impl ToolServerHandle {
    fn state(&self) -> RwLockReadGuard<'_, ToolServerState> {
        self.0.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn state_mut(&self) -> RwLockWriteGuard<'_, ToolServerState> {
        self.0.write().unwrap_or_else(PoisonError::into_inner)
    }

    /// Publish every registration onto `bus`, now and as they change.
    pub fn attach(&self, bus: &Dispatcher) {
        self.state_mut().attach(bus);
    }

    fn register(&self, tool: RegisteredTool) {
        let mut state = self.state_mut();
        let name = state.register(tool, true);
        state.managed_generations.remove(&name);
    }

    /// Add a typed tool.
    pub fn add_tool<T>(&self, tool: T)
    where
        T: Tool + 'static,
    {
        self.register(RegisteredTool::from_tool(tool));
    }

    /// Add a runtime-defined tool.
    pub fn add_dynamic_tool(&self, tool: DynamicTool) {
        self.register(RegisteredTool::from_dynamic(tool));
    }

    /// Add a portable tool.
    pub fn add_portable_dynamic_tool(&self, tool: PortableDynamicTool) {
        self.register(RegisteredTool::from_dynamic(DynamicTool::from_portable(
            tool,
        )));
    }

    /// Add a registration built elsewhere.
    pub fn add_registered_tool(&self, tool: RegisteredTool) {
        self.register(tool);
    }

    fn add_managed(&self, tools: Vec<RegisteredTool>) -> HashMap<String, ManagedToolToken> {
        let mut state = self.state_mut();
        let mut managed = HashMap::with_capacity(tools.len());

        for tool in tools {
            if !tool.is_live() {
                tracing::debug!(
                    tool_name = %tool.name(),
                    "ignored initial registration from disconnected MCP owner"
                );
                continue;
            }

            let name = state.register(tool, true);
            let token = ManagedToolToken::new();
            state
                .managed_generations
                .insert(name.clone(), token.clone());
            managed.insert(name, token);
        }

        managed
    }

    fn reconcile_managed(
        &self,
        mut expected: HashMap<String, ManagedToolToken>,
        tools: Vec<RegisteredTool>,
    ) -> HashMap<String, ManagedToolToken> {
        let mut state = self.state_mut();
        let mut refreshed = HashMap::with_capacity(tools.len());
        let mut managed_order = Vec::with_capacity(tools.len());
        let mut seen = std::collections::HashSet::with_capacity(tools.len());

        state.retire_disconnected_tools();

        for tool in tools {
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
                Some(_) => true,
                None => !present,
            };

            if may_register {
                state.register(tool, true);
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
                state.remove(&name);
            }
        }

        for name in managed_order {
            state.toolset.move_to_end(&name);
        }

        refreshed
    }

    /// Merge a tool set's registrations.
    pub fn append_toolset(&self, toolset: ToolSet) {
        let mut state = self.state_mut();
        let exposed: Vec<String> = toolset.always_exposed_names().map(str::to_owned).collect();
        for (name, tool) in toolset.iter() {
            let always_exposed = exposed.iter().any(|exposed| exposed == name);
            let name = state.register(tool.clone(), always_exposed);
            state.managed_generations.remove(&name);
        }
    }

    /// Remove the tool named `tool_name`.
    pub fn remove_tool(&self, tool_name: &str) {
        let mut state = self.state_mut();
        state.remove(tool_name);
        state.sweep_retired();
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
        let dispatch = self.dispatch(tool_name, args, context).await;
        dispatch.publish_to(context)
    }

    fn with_registry<R>(&self, f: impl FnOnce(&ToolServerState) -> R) -> R {
        let mut state = self.state_mut();
        state.retire_disconnected_tools();
        f(&state)
    }

    pub(crate) async fn dispatch(
        &self,
        tool_name: &str,
        args: &str,
        context: &ToolContext,
    ) -> ToolDispatch {
        let tool = self.with_registry(|state| state.toolset.get(tool_name).cloned());
        dispatch_tool(tool_name, args.to_string(), tool, context).await
    }

    /// The always-exposed registrations, pinned.
    pub fn snapshot(&self) -> ToolRegistrySnapshot {
        let (tools, leases) = self.with_registry(|state| snapshot_registered_tools(state, &[]));
        ToolCatalog::from_registered(tools).with_leases(leases)
    }

    /// The always-exposed definitions.
    pub fn static_tool_defs(&self) -> Vec<ToolDefinition> {
        let mut snapshot = self.snapshot();
        snapshot.take_definitions()
    }

    /// A fork of the registry's tool set.
    pub fn toolset(&self) -> ToolSet {
        self.with_registry(|state| state.toolset.clone())
    }

    /// The definitions a request with `prompt` advertises.
    pub async fn tool_defs(
        &self,
        prompt: Option<String>,
    ) -> Result<Vec<ToolDefinition>, ToolServerError> {
        Ok(self.snapshot_tool_defs(prompt).await?.take_definitions())
    }

    pub(crate) async fn snapshot_tool_defs(
        &self,
        prompt: Option<String>,
    ) -> Result<ToolRegistrySnapshot, ToolServerError> {
        let retrieval_indexes = {
            let state = self.state();
            state.retrieval_indexes.clone()
        };

        let dynamic_tool_ids = if let Some(ref text) = prompt {
            let search_futures = retrieval_indexes.iter().map(|index| {
                let text = text.clone();
                let samples = index.samples;
                let handler = index.handler.clone();

                async move {
                    let req = rig_core::vector_store::request::VectorSearchRequest::builder()
                        .query(text)
                        .samples(samples as u64)
                        .build();
                    // Retrieval for advertisement runs inline on the registry's
                    // own handler: it is a registry read, not a run effect.
                    let outcome = rig_core::bus::serve_inline(
                        &handler,
                        rig_core::effect::EffectKind::Retrieve {
                            query: rig_core::effect::RetrieveQuery::TopNIds {
                                req: req
                                    .map_filter(rig_core::vector_store::request::Filter::interpret),
                            },
                        },
                    )
                    .await
                    .map_err(|report| {
                        rig_core::vector_store::VectorStoreError::DatastoreError(Box::new(report))
                    })?;
                    let ids = match outcome {
                        rig_core::effect::Outcome::Documents(
                            rig_core::effect::RetrievedDocuments::Ids(ids),
                        ) => ids.into_iter().map(|(_, id)| id).collect::<Vec<String>>(),
                        other => {
                            return Err(rig_core::vector_store::VectorStoreError::DatastoreError(
                                Box::new(rig_core::error::ErrorReport::new(
                                    rig_core::error::ErrorKind::Internal,
                                    format!(
                                        "tool retrieval answered with a {} outcome",
                                        other.family()
                                    ),
                                )),
                            ));
                        }
                    };

                    Ok::<_, rig_core::vector_store::VectorStoreError>(ids)
                }
            });

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

        let (tools, leases) =
            self.with_registry(|state| snapshot_registered_tools(state, &dynamic_tool_ids));

        Ok(ToolCatalog::from_registered(tools).with_leases(leases))
    }
}

fn snapshot_registered_tools(
    state: &ToolServerState,
    dynamic_tool_ids: &[String],
) -> (IndexMap<String, RegisteredTool>, Vec<Arc<()>>) {
    let mut tools = IndexMap::new();
    let mut leases = Vec::new();
    let mut insert = |tools: &mut IndexMap<String, RegisteredTool>, name: &str, warn_missing| {
        if tools.contains_key(name) {
            tracing::debug!(
                tool_name = %name,
                "dropping duplicate tool definition from the request"
            );
            return;
        }
        match state.toolset.get(name).cloned() {
            Some(tool) => {
                if let Some(lease) = state.leases.get(name) {
                    leases.push(lease.clone());
                }
                tools.insert(name.to_string(), tool);
            }
            None if warn_missing => {
                tracing::warn!("Tool implementation not found in toolset: {name}");
            }
            None => {}
        }
    };

    for name in dynamic_tool_ids {
        insert(&mut tools, name, true);
    }
    for name in state.toolset.always_exposed_names() {
        insert(&mut tools, name, false);
    }
    (tools, leases)
}

#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<ToolSet>();
    assert_send_sync_static::<ToolRegistrySnapshot>();
    assert_send_sync_static::<ToolServerHandle>();
};

/// Errors from reading the registry.
#[derive(Debug, thiserror::Error)]
pub enum ToolServerError {
    /// The advertised definitions could not be computed.
    #[error("Failed to retrieve tool definitions: {0}")]
    DefinitionError(CompletionError),
}

impl ManagedToolSink for ToolServerHandle {
    fn add_managed_tools(
        &self,
        tools: Vec<PortableDynamicTool>,
    ) -> HashMap<String, ManagedToolToken> {
        self.add_managed(
            tools
                .into_iter()
                .map(|tool| RegisteredTool::from_dynamic(DynamicTool::from_portable(tool)))
                .collect(),
        )
    }

    fn reconcile_managed_tools(
        &self,
        expected: HashMap<String, ManagedToolToken>,
        tools: Vec<PortableDynamicTool>,
    ) -> HashMap<String, ManagedToolToken> {
        self.reconcile_managed(
            expected,
            tools
                .into_iter()
                .map(|tool| RegisteredTool::from_dynamic(DynamicTool::from_portable(tool)))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests;
