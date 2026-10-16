//! Mutable tool registries that publish their registrations to attached
//! handler tables.
//!
//! A registration under the default `tool:<name>` key is served under a fresh
//! generation key (`<owner>/tool:<name>#<n>`); an explicit key is used
//! verbatim. A replaced generation leaves the tables at once: a request
//! catalog keeps the registration it pinned and runs it directly.
//!
//! ```
//! use rig_agent::tool::server::ToolServer;
//! let registry = ToolServer::new().owner("assistant").run();
//! assert!(registry.snapshot().is_empty());
//! ```

use std::collections::HashMap;
use std::sync::{
    Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard,
    atomic::{AtomicU64, Ordering},
};

use crate::handlers::{Handlers, WeakHandlers};
use indexmap::IndexMap;
use rig_core::error::ProviderError;
use rig_core::serve::adapters::RetrieveAdapter;
use rig_core::{
    effect::Key,
    effect::{HandlerKey, family},
    vector_store::{VectorStoreIndex, request::DynamicSearchFilter},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

use crate::{
    agent::register_generated,
    completion::ToolDefinition,
    tool::{DynamicTool, RegisteredTool, Tool, ToolCatalog, ToolSet},
};

/// The per-process counter behind a registry's default owner label.
static NEXT_REGISTRY: AtomicU64 = AtomicU64::new(0);

/// A retrieval index registered for tool retrieval: its key and how many
/// tools to sample.
#[derive(Clone)]
struct RetrievalIndex {
    samples: usize,
    key: HandlerKey,
    handler: rig_core::serve::ErasedHandler,
}

struct ToolServerState {
    /// The registry's label, the owner segment of every key it mints.
    owner: String,
    retrieval_indexes: Vec<RetrievalIndex>,
    toolset: ToolSet,
    managed_generations: HashMap<String, ManagedToolToken>,
    /// The name whose registration was published last under each key: an
    /// explicit key can be reused across names.
    bindings: HashMap<HandlerKey, String>,
    /// The tables this registry publishes onto, while they live.
    tables: Vec<WeakHandlers>,
    next_generation: u64,
}

impl ToolServerState {
    fn generation_key(&mut self, name: &str) -> Key<family::Tool> {
        let n = self.next_generation;
        self.next_generation += 1;
        Key::new_unchecked(HandlerKey::from(format!("{}/tool:{name}#{n}", self.owner)))
    }

    /// The tables still alive, forgetting the dropped ones.
    fn tables(&mut self) -> Vec<Handlers> {
        let mut live = Vec::with_capacity(self.tables.len());
        self.tables.retain(|table| match table.upgrade() {
            Some(table) => {
                live.push(table);
                true
            }
            None => false,
        });
        live
    }

    fn publish(&mut self, tool: &RegisteredTool) {
        for table in self.tables() {
            register_generated(
                table.register_erased(tool.key().raw().clone(), tool.handler().clone()),
            );
        }
    }

    /// Take the key of `name`'s registration off the tables, unless a later
    /// registration was published under it.
    fn unpublish(&mut self, name: &str) {
        let Some(key) = self.toolset.get(name).map(|tool| tool.key().raw().clone()) else {
            return;
        };
        if self.bindings.get(&key).is_some_and(|owner| owner == name) {
            self.bindings.remove(&key);
            for table in self.tables() {
                table.deregister(&key);
            }
        }
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
        // An explicit key that is the live key replaces the handler in place.
        let same_key = self
            .toolset
            .get(&name)
            .is_some_and(|live| live.key() == tool.key());
        if !same_key {
            self.unpublish(&name);
        }
        self.publish(&tool);
        self.bindings.insert(tool.key().raw().clone(), name.clone());
        if always_exposed {
            self.toolset.add_registered_tool(tool);
        } else {
            let mut set = ToolSet::default();
            set.add_registered_tool(tool);
            self.toolset.add_retrievable_tools(set);
        }
        name
    }

    fn remove(&mut self, name: &str) {
        self.unpublish(name);
        self.toolset.remove_tool(name);
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
    }

    fn attach(&mut self, table: &Handlers) {
        for (name, tool) in self.toolset.iter() {
            if self
                .bindings
                .get(tool.key().raw())
                .is_some_and(|owner| owner == name)
            {
                register_generated(
                    table.register_erased(tool.key().raw().clone(), tool.handler().clone()),
                );
            }
        }
        for index in &self.retrieval_indexes {
            register_generated(table.register_erased(index.key.clone(), index.handler.clone()));
        }
        self.tables.push(table.downgrade());
    }
}

pub use rig_core::tool::{ManagedToolSink, ManagedToolToken};

/// A tool registry under construction; [`ToolServer::run`] turns it into
/// the shareable [`ToolServerHandle`].
pub struct ToolServer {
    owner: Option<String>,
    /// Retrieval indexes by sample count, keyed once the registry knows its
    /// owner label.
    retrieval_indexes: Vec<(usize, rig_core::serve::ErasedHandler)>,
    toolset: ToolSet,
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
        }
    }

    /// Name the registry: the owner segment of every key it mints
    /// (`<owner>/tool:<name>#<n>`, `<owner>/retrieve:tools#<n>`). The
    /// default is `tools#<m>` from a per-process counter, distinct per
    /// registry; name it when a host shares one table between registries and
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

    /// Add a registration built elsewhere.
    pub fn registered_tool(mut self, tool: RegisteredTool) -> Self {
        self.toolset.add_registered_tool(tool);
        self
    }

    /// Add retrievable tools: `sample` of them are advertised per request,
    /// chosen by `index`.
    pub fn retrieved_tools<I, F>(self, sample: usize, index: I, toolset: ToolSet) -> Self
    where
        I: VectorStoreIndex<Filter = F> + 'static,
        F: DynamicSearchFilter + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.retrieved_tools_handler(sample, RetrieveAdapter::new(index), toolset)
    }

    /// Add tools selected per request by a retrieval-family handler answering
    /// `TopNIds`, registered under a generated retrieval index key.
    pub fn retrieved_tools_handler(
        mut self,
        sample: usize,
        handler: impl rig_core::serve::Serve + 'static,
        toolset: ToolSet,
    ) -> Self {
        self.retrieval_indexes
            .push((sample, rig_core::serve::ErasedHandler::new(handler)));
        self.toolset.add_retrievable_tools(toolset);
        self
    }

    /// Start the registry.
    pub fn run(mut self) -> ToolServerHandle {
        let owner = self
            .owner
            .take()
            .unwrap_or_else(|| format!("tools#{}", NEXT_REGISTRY.fetch_add(1, Ordering::Relaxed)));
        let handle = ToolServerHandle(Arc::new(RwLock::new(ToolServerState {
            owner,
            retrieval_indexes: Vec::new(),
            toolset: ToolSet::default(),
            managed_generations: HashMap::new(),
            bindings: HashMap::new(),
            tables: Vec::new(),
            next_generation: 0,
        })));
        handle.extend(self);
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

    /// Publish every registration onto `table`, now and as they change.
    pub fn attach(&self, table: &Handlers) {
        self.state_mut().attach(table);
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

    /// Add `server`'s tools and retrieval indexes, keyed under this
    /// registry's owner (`server`'s own owner label is not used).
    pub(crate) fn extend(&self, server: ToolServer) {
        {
            let mut state = self.state_mut();
            for (samples, handler) in server.retrieval_indexes {
                let n = state.retrieval_indexes.len();
                let key = HandlerKey::from(format!("{}/retrieve:tools#{n}", state.owner));
                for table in state.tables() {
                    register_generated(table.register_erased(key.clone(), handler.clone()));
                }
                state.retrieval_indexes.push(RetrievalIndex {
                    samples,
                    key,
                    handler,
                });
            }
        }
        self.add_tools(server.toolset);
    }

    /// Merge a tool set's registrations.
    pub fn add_tools(&self, toolset: ToolSet) {
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
        self.state_mut().remove(tool_name);
    }

    /// Run `f` over the registry, retiring disconnected tools first: under
    /// the read lock while every tool is live, the write lock otherwise.
    fn with_registry<R>(&self, f: impl FnOnce(&ToolServerState) -> R) -> R {
        {
            let state = self.state();
            if state.toolset.iter().all(|(_, tool)| tool.is_live()) {
                return f(&state);
            }
        }
        let mut state = self.state_mut();
        state.retire_disconnected_tools();
        f(&state)
    }

    /// The always-exposed registrations, pinned.
    pub fn snapshot(&self) -> ToolCatalog {
        ToolCatalog::from_registered(
            self.with_registry(|state| snapshot_registered_tools(state, &[])),
        )
    }

    /// A fork of the registry's tool set.
    pub fn toolset(&self) -> ToolSet {
        self.with_registry(|state| state.toolset.clone())
    }

    /// Retrieve advertised definitions for `prompt`, returning retrieval failures
    /// or incompatible result shapes as definition errors.
    pub async fn tool_defs(
        &self,
        prompt: Option<String>,
    ) -> Result<Vec<Arc<ToolDefinition>>, ToolServerError> {
        Ok(self.snapshot_tool_defs(prompt).await?.take_definitions())
    }

    /// Return registered retrieval index keys in registration order.
    pub fn retrieval_keys(&self) -> Vec<HandlerKey> {
        self.state()
            .retrieval_indexes
            .iter()
            .map(|index| index.key.clone())
            .collect()
    }

    /// Build one `TopNIds` effect per retrieval index for the supplied prompt.
    /// Returns no effects without a prompt; pass returned IDs to
    /// [`Self::snapshot_with_dynamic`] after dispatch.
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

    /// Pin retrieved tools in ID order, then always-exposed tools in registration
    /// order. Deduplicates names and skips missing registrations.
    pub fn snapshot_with_dynamic(&self, dynamic_tool_ids: &[String]) -> ToolCatalog {
        ToolCatalog::from_registered(
            self.with_registry(|state| snapshot_registered_tools(state, dynamic_tool_ids)),
        )
    }

    /// The registrations a request with `prompt` advertises, answering the
    /// retrieval queries inline on the registry's own handlers. This is the
    /// standalone registry read ([`ToolServerHandle::tool_defs`]); the
    /// engine dispatches the same queries through its handlers instead.
    pub(crate) async fn snapshot_tool_defs(
        &self,
        prompt: Option<String>,
    ) -> Result<ToolCatalog, ToolServerError> {
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
                        rig_core::effect::Answered::Documents(
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
                .map_err(|e| ToolServerError::DefinitionError(ProviderError::request(e)))?
                .into_iter()
                .flatten()
                .collect::<Vec<String>>()
        } else {
            Vec::new()
        };

        Ok(ToolCatalog::from_registered(self.with_registry(|state| {
            snapshot_registered_tools(state, &dynamic_tool_ids)
        })))
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
    tools
}

#[cfg(not(target_family = "wasm"))]
const _: () = {
    const fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<ToolSet>();
    assert_send_sync_static::<ToolCatalog>();
    assert_send_sync_static::<ToolServerHandle>();
};

/// Errors from reading the registry.
#[derive(Debug, thiserror::Error)]
pub enum ToolServerError {
    /// The advertised definitions could not be computed.
    #[error("Failed to retrieve tool definitions: {0}")]
    DefinitionError(ProviderError),
}

impl ManagedToolSink for ToolServerHandle {
    fn add_managed_tools(&self, tools: Vec<DynamicTool>) -> HashMap<String, ManagedToolToken> {
        self.add_managed(
            tools
                .into_iter()
                .map(RegisteredTool::from_dynamic)
                .collect(),
        )
    }

    fn reconcile_managed_tools(
        &self,
        expected: HashMap<String, ManagedToolToken>,
        tools: Vec<DynamicTool>,
    ) -> HashMap<String, ManagedToolToken> {
        self.reconcile_managed(
            expected,
            tools
                .into_iter()
                .map(RegisteredTool::from_dynamic)
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests;
