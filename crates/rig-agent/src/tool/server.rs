//! The runtime tool registry an agent advertises from and dispatches
//! through.
//!
//! A [`ToolServerHandle`] is the definition/advertisement surface: it owns
//! the [`ToolSet`] (descriptors plus staged handlers) and publishes every
//! registration onto the buses attached to it — each agent built with the
//! handle attaches its bus at build time, and later additions, removals and
//! MCP reconciles are pushed to every attached bus as they happen. A
//! request's snapshot ([`ToolCatalog`]) pins the *generation* of each tool:
//! registrations are served under owner- and generation-qualified keys
//! (`<owner>/tool:<name>#<n>`, the owner being the registry's label —
//! `tools#<m>` by default, [`ToolServer::owner`] to name it), a replacement
//! registers a new generation, and a generation is deregistered from the
//! buses when the last snapshot referencing it drops (or, failing that, on
//! the next registry read). A registration that carries its own key (a
//! replayer's recorded key, a host's own) is served under that key as
//! given. Execution during a run goes through the bus; the inline
//! `execute` here serves the standalone use (no agent, no bus).

use std::collections::HashMap;
use std::sync::{
    Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard, Weak,
    atomic::{AtomicU64, Ordering},
};

use indexmap::IndexMap;
use rig_bus::Registrar;
use rig_core::serve::adapters::RetrieveAdapter;
use rig_core::{
    effect::Key,
    effect::{HandlerKey, family},
    vector_store::{VectorStoreIndex, request::DynamicSearchFilter},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

use crate::{
    completion::{CompletionError, ToolDefinition},
    tool::{
        DynamicTool, PortableDynamicTool, RegisteredTool, Tool, ToolCatalog, ToolContext,
        ToolDispatch, ToolLease, ToolResult, ToolSet, dispatch_tool,
    },
};

/// The per-process counter behind a registry's default owner label.
static NEXT_REGISTRY: AtomicU64 = AtomicU64::new(0);

/// The per-request snapshot of the registry: a [`ToolCatalog`].
pub type ToolRegistrySnapshot = ToolCatalog;

/// A retrieval index registered for tool retrieval: its bus key and how
/// many tools to sample.
#[derive(Clone)]
struct RetrievalIndex {
    samples: usize,
    key: HandlerKey,
    handler: rig_core::serve::ErasedHandler,
}

/// A retrieval index added to a [`ToolServer`], keyed once the registry
/// knows its owner label.
struct PendingRetrievalIndex {
    samples: usize,
    index: usize,
    handler: rig_core::serve::ErasedHandler,
}

/// The lease a live registration hands to every snapshot that pins it.
/// Dropping the last clone after the generation was retired sweeps the
/// registry, so the retired key leaves the buses when the last request
/// that could dispatch to it is gone — not on the next registry read.
struct LeaseToken {
    registry: Weak<RwLock<ToolServerState>>,
}

impl Drop for LeaseToken {
    fn drop(&mut self) {
        let Some(registry) = self.registry.upgrade() else {
            return;
        };
        // A writer holding the lock (a `register` retiring this very
        // generation) sweeps before it releases; a reader holding it makes
        // the next read sweep. Either way the key leaves the buses.
        if let Ok(mut state) = registry.try_write() {
            state.sweep_retired();
        }
    }
}

/// A retired generation waiting for its last snapshot to drop.
struct RetiredGeneration {
    key: HandlerKey,
    lease: Weak<LeaseToken>,
}

struct ToolServerState {
    /// The registry's label, the owner segment of every key it mints.
    owner: String,
    /// The registry itself, for the leases it mints.
    registry: Weak<RwLock<ToolServerState>>,
    retrieval_indexes: Vec<RetrievalIndex>,
    toolset: ToolSet,
    /// The lease each live registration hands to snapshots.
    leases: HashMap<String, Arc<LeaseToken>>,
    retired: Vec<RetiredGeneration>,
    managed_generations: HashMap<String, ManagedToolToken>,
    /// The buses this registry publishes onto, by their registrars.
    buses: Vec<Registrar>,
    next_generation: u64,
}

impl ToolServerState {
    fn generation_key(&mut self, name: &str) -> Key<family::Tool> {
        let n = self.next_generation;
        self.next_generation += 1;
        Key::new_unchecked(HandlerKey::from(format!("{}/tool:{name}#{n}", self.owner)))
    }

    fn lease(&self) -> Arc<LeaseToken> {
        Arc::new(LeaseToken {
            registry: self.registry.clone(),
        })
    }

    fn publish(&self, tool: &RegisteredTool) {
        for bus in &self.buses {
            crate::agent::bus::register_generated(
                bus.register_erased(tool.key().raw().clone(), tool.handler().clone()),
            );
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
                key: tool.key().raw().clone(),
                lease: Arc::downgrade(&lease),
            });
            // The lease's own drop cannot take the lock this thread holds;
            // `sweep_retired` below is what serves it.
            drop(lease);
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
        // An explicit key that is the live key replaces the handler in
        // place: the live registration is not retired, so the key is never
        // deregistered under the new handler.
        let same_key = self
            .toolset
            .get(&name)
            .is_some_and(|live| live.key() == tool.key());
        if !same_key {
            self.retire(&name);
        }
        self.publish(&tool);
        if !same_key {
            self.leases.insert(name.clone(), self.lease());
        }
        if always_exposed {
            self.toolset.add_registered(tool);
        } else {
            let mut set = ToolSet::default();
            set.add_registered(tool);
            self.toolset.add_retrievable_tools(set);
        }
        self.sweep_retired();
        name
    }

    fn remove(&mut self, name: &str) {
        self.retire(name);
        self.toolset.delete_tool(name);
        self.managed_generations.remove(name);
        self.sweep_retired();
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

    fn attach(&mut self, bus: &Registrar) {
        for (_, tool) in self.toolset.iter() {
            crate::agent::bus::register_generated(
                bus.register_erased(tool.key().raw().clone(), tool.handler().clone()),
            );
        }
        for index in &self.retrieval_indexes {
            crate::agent::bus::register_generated(
                bus.register_erased(index.key.clone(), index.handler.clone()),
            );
        }
        self.buses.push(bus.clone());
    }
}

pub use rig_core::tool::{ManagedToolSink, ManagedToolToken};

/// A tool registry under construction; [`ToolServer::run`] turns it into
/// the shareable [`ToolServerHandle`].
pub struct ToolServer {
    owner: Option<String>,
    retrieval_indexes: Vec<PendingRetrievalIndex>,
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
            owner: None,
            retrieval_indexes: Vec::new(),
            toolset: ToolSet::default(),
            next_index: 0,
        }
    }

    /// Name the registry: the owner segment of every key it mints
    /// (`<owner>/tool:<name>#<n>`, `<owner>/retrieve:tools#<n>`). The
    /// default is `tools#<m>` from a per-process counter, distinct per
    /// registry; name it when a host shares one bus between registries and
    /// wants to read the keys.
    pub fn owner(mut self, label: impl Into<String>) -> Self {
        self.owner = Some(label.into());
        self
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
        self.retrieval_indexes.push(PendingRetrievalIndex {
            samples: sample,
            index: n,
            handler: rig_core::serve::ErasedHandler::new(RetrieveAdapter::new(index)),
        });
        self.toolset.add_retrievable_tools(toolset);
        self
    }

    /// Add retrievable tools chosen per request by `handler` — any
    /// retrieval-family handler answering `TopNIds`, such as a replayer
    /// answering a recorded index from an effect log — under the same key
    /// [`retrieved_tools`](Self::retrieved_tools) would give an index.
    pub fn retrieved_tools_handler(
        mut self,
        sample: usize,
        handler: impl rig_core::serve::Serve + 'static,
        toolset: ToolSet,
    ) -> Self {
        let n = self.next_index;
        self.next_index += 1;
        self.retrieval_indexes.push(PendingRetrievalIndex {
            samples: sample,
            index: n,
            handler: rig_core::serve::ErasedHandler::new(handler),
        });
        self.toolset.add_retrievable_tools(toolset);
        self
    }

    /// Start the registry.
    pub fn run(self) -> ToolServerHandle {
        let owner = self
            .owner
            .unwrap_or_else(|| format!("tools#{}", NEXT_REGISTRY.fetch_add(1, Ordering::Relaxed)));
        let retrieval_indexes = self
            .retrieval_indexes
            .into_iter()
            .map(|pending| RetrievalIndex {
                samples: pending.samples,
                key: HandlerKey::from(format!("{owner}/retrieve:tools#{}", pending.index)),
                handler: pending.handler,
            })
            .collect();
        let registry = Arc::new_cyclic(|registry| {
            RwLock::new(ToolServerState {
                owner,
                registry: registry.clone(),
                retrieval_indexes,
                toolset: ToolSet::default(),
                leases: HashMap::new(),
                retired: Vec::new(),
                managed_generations: HashMap::new(),
                buses: Vec::new(),
                next_generation: 0,
            })
        });
        let handle = ToolServerHandle(registry);
        let exposed: Vec<String> = self
            .toolset
            .always_exposed_names()
            .map(str::to_owned)
            .collect();
        {
            let mut state = handle.state_mut();
            for (name, tool) in self.toolset.iter() {
                state.register(tool.clone(), exposed.iter().any(|exposed| exposed == name));
            }
        }
        handle
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
    pub fn attach(&self, bus: &Registrar) {
        self.state_mut().attach(bus);
    }

    /// The registry's label: the owner segment of the keys it mints.
    pub fn owner(&self) -> String {
        self.state().owner.clone()
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

    /// The `Retrieve` effects a request with `prompt` needs answered before
    /// its dynamic tools can be advertised: one `TopNIds` query per
    /// retrieval index, under the key the index is registered on every
    /// attached bus. The engine dispatches them at the boundary and hands
    /// the ids back to [`ToolServerHandle::snapshot_with_dynamic`].
    /// The keys of every retrieval index this registry serves.
    pub fn retrieval_keys(&self) -> Vec<HandlerKey> {
        self.state()
            .retrieval_indexes
            .iter()
            .map(|index| index.key.clone())
            .collect()
    }

    pub fn retrieval_effects(
        &self,
        prompt: Option<String>,
    ) -> Vec<(HandlerKey, rig_core::effect::EffectKind)> {
        let Some(text) = prompt else {
            return Vec::new();
        };
        let retrieval_indexes = {
            let state = self.state();
            state.retrieval_indexes.clone()
        };
        retrieval_indexes
            .into_iter()
            .map(|index| {
                let req = rig_core::vector_store::request::VectorSearchRequest::builder()
                    .query(text.clone())
                    .samples(index.samples as u64)
                    .build();
                (
                    index.key,
                    rig_core::effect::EffectKind::Retrieve {
                        query: rig_core::effect::RetrieveQuery::TopNIds {
                            req: req.map_filter(rig_core::vector_store::request::Filter::interpret),
                        },
                    },
                )
            })
            .collect()
    }

    /// The registrations a request advertises: the always-exposed ones plus
    /// the retrieved tools named by `dynamic_tool_ids`.
    pub fn snapshot_with_dynamic(&self, dynamic_tool_ids: &[String]) -> ToolRegistrySnapshot {
        let (tools, leases) =
            self.with_registry(|state| snapshot_registered_tools(state, dynamic_tool_ids));
        ToolCatalog::from_registered(tools).with_leases(leases)
    }

    /// The registrations a request with `prompt` advertises, answering the
    /// retrieval queries inline on the registry's own handlers. This is the
    /// standalone registry read ([`ToolServerHandle::tool_defs`]); the
    /// engine dispatches the same queries on the bus instead.
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
                    let outcome = rig_core::serve::serve_inline(
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
) -> (IndexMap<String, RegisteredTool>, Vec<ToolLease>) {
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
                    leases.push(lease.clone() as ToolLease);
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
const _: () = {
    const fn assert_send_sync_static<T: Send + Sync + 'static>() {}
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
