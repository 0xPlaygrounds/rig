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
    retrieval_indexes: Vec<(usize, Arc<dyn VectorStoreIndexDyn + Send + Sync>)>,
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
    retrieval_indexes: Vec<(usize, Arc<dyn VectorStoreIndexDyn + Send + Sync>)>,
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
        index: impl VectorStoreIndexDyn + Send + Sync + 'static,
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
mod tests {
    use std::{
        future::{Future, pending, poll_fn},
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
        task::Poll,
        time::Duration,
    };

    use crate::{
        test_utils::{
            BarrierMockToolIndex, MockAddTool, MockBarrierTool, MockControlledTool,
            MockSubtractTool, MockToolIndex,
        },
        tool::{
            Tool, ToolContext, ToolEmbedding, ToolExecutionError, ToolSet,
            server::{ToolServer, ToolServerHandle},
        },
    };

    async fn execute_tool(
        handle: &ToolServerHandle,
        name: &str,
        args: &str,
    ) -> Result<String, ToolExecutionError> {
        execute_tool_with_context(handle, name, args, &mut ToolContext::new()).await
    }

    /// A portable tool whose liveness follows `live`, standing in for a remote
    /// tool whose transport can disconnect.
    fn liveness_gated_tool(name: &str, live: Arc<AtomicBool>) -> crate::tool::PortableDynamicTool {
        crate::tool::PortableDynamicTool::new(
            name,
            "gated",
            serde_json::json!({"type": "object"}),
            |_| Box::pin(async { Ok(crate::tool::ToolOutput::text("ok")) }),
        )
        .with_liveness(move || live.load(Ordering::SeqCst))
    }

    /// The sync snapshot and the async, prompt-less `tool_defs` read the
    /// same always-exposed registry in the same order.
    #[tokio::test]
    async fn sync_snapshot_matches_async_prompt_less_read() {
        let handle = ToolServer::new()
            .tool(MockAddTool)
            .tool(MockSubtractTool)
            .run();

        let sync_defs = handle.static_tool_defs();
        let async_defs = handle.tool_defs(None).await.unwrap();
        assert_eq!(sync_defs.len(), 2);
        assert_eq!(sync_defs, async_defs);

        let snapshot = handle.snapshot();
        assert_eq!(snapshot.definitions(), sync_defs.as_slice());
        assert_eq!(
            snapshot.names().collect::<Vec<_>>(),
            vec!["add", "subtract"]
        );
        assert_eq!(snapshot.len(), 2);
        assert!(!snapshot.is_empty());
    }

    /// A tool whose remote backing disconnected is retired by every read path —
    /// sync snapshot, sync definitions, forked toolset, and the async read.
    #[tokio::test]
    async fn retired_tools_are_absent_from_every_read_path() {
        let live = Arc::new(AtomicBool::new(true));
        let handle = ToolServer::new()
            .tool(MockAddTool)
            .portable_dynamic_tool(liveness_gated_tool("remote", live.clone()))
            .run();
        assert_eq!(
            handle.snapshot().names().collect::<Vec<_>>(),
            vec!["add", "remote"]
        );

        live.store(false, Ordering::SeqCst);

        assert_eq!(handle.snapshot().names().collect::<Vec<_>>(), vec!["add"]);
        assert_eq!(handle.static_tool_defs().len(), 1);
        assert!(!handle.toolset().contains("remote"));
        assert_eq!(handle.tool_defs(None).await.unwrap().len(), 1);
    }

    /// A snapshot pins implementations: it keeps executing the tool it was
    /// taken with after the registry replaces or removes that name.
    #[tokio::test]
    async fn snapshot_executes_pinned_implementation() {
        let handle = ToolServer::new().tool(MockAddTool).run();
        let snapshot = handle.snapshot();
        handle.remove_tool("add");
        assert!(handle.snapshot().is_empty());

        let mut context = ToolContext::new();
        let result = snapshot
            .execute("add", r#"{"x": 2, "y": 3}"#, &mut context)
            .await;
        assert_eq!(result.output().render(), "5");
    }

    /// `toolset()` forks the registry: the fork shares implementations but
    /// later changes on either side stay local.
    #[tokio::test]
    async fn toolset_forks_the_registry() {
        let handle = ToolServer::new().tool(MockAddTool).run();
        let mut fork = handle.toolset();
        assert!(fork.contains("add"));

        fork.add_tool(MockSubtractTool);
        assert!(!handle.snapshot().names().any(|name| name == "subtract"));

        handle.remove_tool("add");
        assert!(fork.contains("add"));

        // The fork builds a second, independent server with the same tools.
        let second = ToolServer::new().run();
        second.append_toolset(fork);
        assert_eq!(
            execute_tool(&second, "add", r#"{"x": 1, "y": 1}"#)
                .await
                .unwrap(),
            "2"
        );
    }

    /// The sync read needs no executor: a plain test reads definitions that
    /// another thread registered, with no runtime in sight.
    #[test]
    fn static_tool_defs_reads_without_a_runtime() {
        let handle = ToolServer::new().run();
        assert!(handle.static_tool_defs().is_empty());

        let writer = handle.clone();
        std::thread::spawn(move || {
            writer.add_tool(MockAddTool);
            writer.add_tool(MockSubtractTool);
        })
        .join()
        .expect("registering thread");

        let names = handle
            .static_tool_defs()
            .into_iter()
            .map(|definition| definition.name)
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["add", "subtract"]);
    }

    async fn execute_tool_with_context(
        handle: &ToolServerHandle,
        name: &str,
        args: &str,
        context: &mut ToolContext,
    ) -> Result<String, ToolExecutionError> {
        let result = handle.execute(name, args, context).await;
        match result.error() {
            Some(error) => Err(error.clone()),
            None => Ok(result.output().render()),
        }
    }

    struct NamedTool;

    impl NamedTool {
        fn new() -> Self {
            Self
        }
    }

    impl Tool for NamedTool {
        const NAME: &'static str = "registered_named";
        type Error = rig::tool::ToolExecutionError;
        type Args = serde_json::Value;
        type Output = String;

        fn description(&self) -> String {
            "uses its canonical name".to_string()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }

        async fn call(
            &self,
            _context: &mut crate::tool::ToolContext,
            _args: Self::Args,
        ) -> Result<Self::Output, crate::tool::ToolExecutionError> {
            Ok("ok".to_string())
        }
    }

    struct ReplacementTool {
        description: &'static str,
        output: &'static str,
    }

    impl Tool for ReplacementTool {
        const NAME: &'static str = "replacement";
        type Error = rig::tool::ToolExecutionError;
        type Args = serde_json::Value;
        type Output = String;

        fn description(&self) -> String {
            self.description.to_string()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }

        async fn call(
            &self,
            _context: &mut ToolContext,
            _args: Self::Args,
        ) -> Result<Self::Output, ToolExecutionError> {
            Ok(self.output.to_string())
        }
    }

    #[derive(Debug, thiserror::Error)]
    #[error("init error")]
    struct InitError;

    impl ToolEmbedding for NamedTool {
        type InitError = InitError;
        type Context = ();
        type State = ();

        fn embedding_docs(&self) -> Vec<String> {
            vec!["named retrieved tool".to_string()]
        }

        fn context(&self) -> Self::Context {}

        fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
            Ok(Self::new())
        }
    }

    #[tokio::test]
    pub async fn test_toolserver() {
        let server = ToolServer::new();

        let handle = server.run();

        handle.add_tool(MockAddTool);
        let res = handle.tool_defs(None).await.unwrap();

        assert_eq!(res.len(), 1);

        let json_args_as_string =
            serde_json::to_string(&serde_json::json!({"x": 2, "y": 5})).unwrap();
        let res = execute_tool(&handle, "add", &json_args_as_string)
            .await
            .unwrap();
        assert_eq!(res, "7");

        handle.remove_tool("add");
        let res = handle.tool_defs(None).await.unwrap();

        assert_eq!(res.len(), 0);
    }

    #[tokio::test]
    async fn definition_snapshot_pins_the_exact_tool_registration() {
        let handle = ToolServer::new()
            .tool(ReplacementTool {
                description: "first schema",
                output: "first implementation",
            })
            .run();
        let snapshot = handle.snapshot_tool_defs(None).await.unwrap();

        handle.add_tool(ReplacementTool {
            description: "second schema",
            output: "second implementation",
        });

        assert_eq!(snapshot.definitions()[0].description, "first schema");
        let dispatch = snapshot
            .dispatch(ReplacementTool::NAME, "{}", &ToolContext::new())
            .await;
        assert_eq!(dispatch.result.output().render(), "first implementation");

        let live = handle
            .dispatch(ReplacementTool::NAME, "{}", &ToolContext::new())
            .await;
        assert_eq!(live.result.output().render(), "second implementation");

        let next_snapshot = handle.snapshot_tool_defs(None).await.unwrap();
        assert_eq!(next_snapshot.definitions()[0].description, "second schema");
        let dispatch = next_snapshot
            .dispatch(ReplacementTool::NAME, "{}", &ToolContext::new())
            .await;
        assert_eq!(dispatch.result.output().render(), "second implementation");
    }

    #[tokio::test]
    pub async fn test_toolserver_append_toolset_matches_add_tool() {
        let mut via_add_tool = {
            let handle = ToolServer::new().run();
            handle.add_tool(MockAddTool);
            handle.add_tool(MockSubtractTool);
            handle.tool_defs(None).await.unwrap()
        };
        via_add_tool.sort_by(|a, b| a.name.cmp(&b.name));

        let mut via_append_toolset = {
            let handle = ToolServer::new().run();
            let mut toolset = ToolSet::default();
            toolset.add_tool(MockAddTool);
            toolset.add_tool(MockSubtractTool);
            handle.append_toolset(toolset);
            handle.tool_defs(None).await.unwrap()
        };
        via_append_toolset.sort_by(|a, b| a.name.cmp(&b.name));

        assert_eq!(via_add_tool.len(), via_append_toolset.len());
        assert!(
            via_add_tool
                .iter()
                .zip(via_append_toolset.iter())
                .all(|(a, b)| a.name == b.name),
            "append_toolset must surface the same LLM-visible tools as add_tool",
        );
    }

    #[tokio::test]
    pub async fn builder_tool_uses_canonical_static_name() {
        let handle = ToolServer::new().tool(NamedTool::new()).run();

        let defs = handle.tool_defs(None).await.unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, NamedTool::NAME);
    }

    #[tokio::test]
    pub async fn handle_add_tool_uses_canonical_static_name() {
        let handle = ToolServer::new().run();
        handle.add_tool(NamedTool::new());

        let defs = handle.tool_defs(None).await.unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, NamedTool::NAME);
    }

    #[tokio::test]
    pub async fn retrieval_resolves_canonical_key() {
        let mut toolset = ToolSet::default();
        toolset.add_retrieved_tool(NamedTool::new());
        let handle = ToolServer::new()
            .retrieved_tools(1, MockToolIndex::new([NamedTool::NAME]), toolset)
            .run();

        let defs = handle
            .tool_defs(Some("use the changing tool".to_string()))
            .await
            .unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, NamedTool::NAME);
    }

    #[tokio::test]
    pub async fn get_tool_defs_preserves_static_registration_order() {
        let handle = ToolServer::new().run();
        handle.add_tool(MockSubtractTool);
        handle.add_tool(MockAddTool);

        let defs = handle.tool_defs(None).await.unwrap();
        assert_eq!(
            defs.iter().map(|def| def.name.as_str()).collect::<Vec<_>>(),
            vec!["subtract", "add"]
        );
    }

    #[tokio::test]
    pub async fn get_tool_defs_dedupes_dynamic_and_static_overlap() {
        // One shared toolset backs both lists, so a dynamically retrieved
        // name that is also static must yield a single definition.
        let handle = ToolServer::new()
            .tool(MockAddTool)
            .retrieved_tools(1, MockToolIndex::new(["add"]), ToolSet::default())
            .run();

        let defs = handle
            .tool_defs(Some("add two numbers".to_string()))
            .await
            .unwrap();
        assert_eq!(
            defs.len(),
            1,
            "dynamic/static name overlap must not produce duplicate declarations: {:?}",
            defs.iter().map(|def| def.name.as_str()).collect::<Vec<_>>()
        );
        assert_eq!(defs[0].name, "add");
    }

    #[tokio::test]
    async fn retrieval_registration_preserves_existing_always_exposure() {
        let handle = ToolServer::new()
            .tool(MockAddTool)
            .retrieved_tools(
                1,
                MockToolIndex::new(["add"]),
                ToolSet::from_tools(vec![MockAddTool]),
            )
            .run();

        let defs = handle.tool_defs(None).await.unwrap();
        assert_eq!(
            defs.iter()
                .map(|definition| definition.name.as_str())
                .collect::<Vec<_>>(),
            vec!["add"],
            "merging a retrieval implementation must not demote an always-exposed registration"
        );
    }

    #[tokio::test]
    pub async fn duplicate_registration_advertises_one_definition() {
        let handle = ToolServer::new().tool(MockAddTool).run();
        handle.add_tool(MockAddTool);

        let mut toolset = ToolSet::default();
        toolset.add_tool(MockAddTool);
        handle.append_toolset(toolset);

        let defs = handle.tool_defs(None).await.unwrap();
        assert_eq!(
            defs.len(),
            1,
            "re-registering a name must not advertise duplicate declarations"
        );
        assert_eq!(defs[0].name, "add");
    }

    #[tokio::test]
    pub async fn test_toolserver_retrieved_tools() {
        // Create a toolset with both tools
        let mut toolset = ToolSet::default();
        toolset.add_tool(MockAddTool);
        toolset.add_tool(MockSubtractTool);

        // Create a mock index that will return "subtract" as the dynamic tool
        let mock_index = MockToolIndex::new(["subtract"]);

        // Build server with static tool "add" and dynamic tools from the mock index
        let server = ToolServer::new().tool(MockAddTool).retrieved_tools(
            1,
            mock_index,
            ToolSet::from_tools(vec![MockSubtractTool]),
        );

        let handle = server.run();

        // Test with None prompt - should only return static tools
        let res = handle.tool_defs(None).await.unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].name, "add");

        // Test with Some prompt - should return both static and dynamic tools
        let res = handle
            .tool_defs(Some("calculate difference".to_string()))
            .await
            .unwrap();
        assert_eq!(res.len(), 2);

        // Check that both tools are present (order may vary)
        let tool_names: Vec<&str> = res.iter().map(|t| t.name.as_str()).collect();
        assert!(tool_names.contains(&"add"));
        assert!(tool_names.contains(&"subtract"));
    }

    #[tokio::test]
    pub async fn test_toolserver_retrieved_tools_missing_implementation() {
        // Create a mock index that returns a tool ID that doesn't exist in the toolset
        let mock_index = MockToolIndex::new(["nonexistent_tool"]);

        // Build server with only static tool, but dynamic index references missing tool
        let server =
            ToolServer::new()
                .tool(MockAddTool)
                .retrieved_tools(1, mock_index, ToolSet::default());

        let handle = server.run();

        // Test with Some prompt - should only return static tool since dynamic tool is missing
        let res = handle
            .tool_defs(Some("some query".to_string()))
            .await
            .unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].name, "add");
    }

    #[tokio::test]
    pub async fn test_toolserver_concurrent_tool_execution() {
        let num_calls = 3;
        let barrier = Arc::new(tokio::sync::Barrier::new(num_calls));

        let server = ToolServer::new().tool(MockBarrierTool::new(barrier.clone()));
        let handle = server.run();

        // Make concurrent calls
        let futures: Vec<_> = (0..num_calls)
            .map(|_| execute_tool(&handle, "barrier_tool", "{}"))
            .collect();

        // If execution is sequential, the first call will block at the barrier forever.
        // We use a 1-second timeout to fail fast instead of hanging the test runner.
        let result =
            tokio::time::timeout(Duration::from_secs(1), futures::future::join_all(futures)).await;

        assert!(
            result.is_ok(),
            "Tool execution deadlocked! Tools are executing sequentially instead of concurrently."
        );

        // All calls should succeed
        for res in result.unwrap() {
            assert!(res.is_ok(), "Tool call failed: {res:?}");
            assert_eq!(res.unwrap(), "done");
        }
    }

    #[tokio::test]
    pub async fn test_toolserver_write_while_tool_running() {
        let started = Arc::new(tokio::sync::Notify::new());
        let allow_finish = Arc::new(tokio::sync::Notify::new());

        // Build server with the controlled tool that waits at a barrier during execution
        let tool = MockControlledTool::new(started.clone(), allow_finish.clone());

        let server = ToolServer::new().tool(tool);
        let handle = server.run();

        // Start tool call in background
        let handle_clone = handle.clone();
        let call_task =
            tokio::spawn(async move { execute_tool(&handle_clone, "controlled", "{}").await });

        // Wait until we are strictly inside `call()`
        started.notified().await;

        // Write to the state (add a tool) while the tool call is mid-execution.
        // If the read lock were incorrectly held across tool execution, this
        // sync call would block forever and the test harness would time out.
        handle.add_tool(MockAddTool);

        // Allow the background tool to finish and clean up
        allow_finish.notify_one();
        let call_result = call_task.await.unwrap();
        assert_eq!(call_result.unwrap(), "42");
    }

    #[tokio::test]
    pub async fn test_toolserver_parallel_retrieval() {
        // We expect exactly 2 parallel searches to hit the barrier at the same time
        let barrier = Arc::new(tokio::sync::Barrier::new(2));

        let index1 = BarrierMockToolIndex::new(barrier.clone(), "add");
        let index2 = BarrierMockToolIndex::new(barrier.clone(), "subtract");

        // Put both tools in the toolset so they resolve correctly
        let mut toolset = ToolSet::default();
        toolset.add_tool(MockAddTool);
        toolset.add_tool(MockSubtractTool);

        let server = ToolServer::new()
            .retrieved_tools(1, index1, ToolSet::default())
            .retrieved_tools(1, index2, toolset);

        let handle = server.run();

        // This will trigger a search across both indices.
        // If fetched sequentially, the first index will wait at the barrier forever.
        let get_defs = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            handle.tool_defs(Some("do math".to_string())),
        )
        .await;

        assert!(
            get_defs.is_ok(),
            "Dynamic tools were fetched sequentially! The first query deadlocked waiting for the second query to start."
        );

        let defs = get_defs.unwrap().unwrap();
        assert_eq!(defs.len(), 2);

        let tool_names: Vec<&str> = defs.iter().map(|t| t.name.as_str()).collect();
        assert!(tool_names.contains(&"add"));
        assert!(tool_names.contains(&"subtract"));
    }

    #[derive(Clone)]
    struct SessionId(String);

    struct CloneTrackedContext {
        clones: Arc<AtomicUsize>,
        value: usize,
    }

    impl Clone for CloneTrackedContext {
        fn clone(&self) -> Self {
            self.clones.fetch_add(1, Ordering::SeqCst);
            Self {
                clones: self.clones.clone(),
                value: self.value,
            }
        }
    }

    #[derive(serde::Deserialize, serde::Serialize)]
    struct ContextReader;

    impl crate::tool::Tool for ContextReader {
        const NAME: &'static str = "context_reader";
        type Error = rig::tool::ToolExecutionError;
        type Args = serde_json::Value;
        type Output = String;

        fn description(&self) -> String {
            "Reads SessionId from context".to_string()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }

        async fn call(
            &self,
            context: &mut ToolContext,
            _args: Self::Args,
        ) -> Result<Self::Output, ToolExecutionError> {
            if let Some(value) = context.get_mut::<CloneTrackedContext>() {
                value.value += 1;
                let result_value = value.value;
                context.insert_result(result_value);
            }
            Ok(context.get::<SessionId>().map_or_else(
                || "no session".to_string(),
                |session| format!("session:{}", session.0),
            ))
        }
    }

    #[tokio::test]
    async fn context_reaches_the_single_execute_path() {
        let handle = ToolServer::new().tool(ContextReader).run();
        let mut context = ToolContext::new();
        context.insert(SessionId("abc-123".to_string()));
        let result = execute_tool_with_context(&handle, "context_reader", "{}", &mut context)
            .await
            .unwrap();
        assert_eq!(result, "session:abc-123");
    }

    #[tokio::test]
    async fn server_dispatch_snapshot_clones_once_and_only_publishes_result_metadata() {
        let handle = ToolServer::new().tool(ContextReader).run();
        let clones = Arc::new(AtomicUsize::new(0));
        let mut context = ToolContext::new();
        context.insert(CloneTrackedContext {
            clones: clones.clone(),
            value: 0,
        });

        let result = execute_tool_with_context(&handle, "context_reader", "{}", &mut context)
            .await
            .unwrap();

        assert_eq!(result, "no session");
        assert_eq!(clones.load(Ordering::SeqCst), 1);
        assert_eq!(
            context
                .get::<CloneTrackedContext>()
                .map(|value| value.value),
            Some(0),
            "tool-local inbound mutations must not change the caller's context"
        );
        assert_eq!(context.result::<usize>(), Some(&1));
    }

    struct PendingTool(Arc<AtomicBool>);

    impl Tool for PendingTool {
        const NAME: &'static str = "pending";
        type Error = rig::tool::ToolExecutionError;
        type Args = ();
        type Output = ();

        fn description(&self) -> String {
            "never completes".into()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        async fn call(
            &self,
            context: &mut ToolContext,
            _args: Self::Args,
        ) -> Result<Self::Output, ToolExecutionError> {
            context.insert_result("unpublished".to_string());
            self.0.store(true, Ordering::SeqCst);
            pending().await
        }
    }

    #[tokio::test]
    async fn cancelled_server_dispatch_does_not_retain_stale_result_metadata() {
        let started = Arc::new(AtomicBool::new(false));
        let handle = ToolServer::new().tool(PendingTool(started.clone())).run();
        let mut context = ToolContext::new();
        context.insert_result("stale".to_string());

        let mut execution = Box::pin(handle.execute(PendingTool::NAME, "null", &mut context));
        tokio::time::timeout(
            Duration::from_secs(1),
            poll_fn(|cx| {
                assert!(execution.as_mut().poll(cx).is_pending());
                started.load(Ordering::SeqCst).then_some(()).map_or_else(
                    || {
                        cx.waker().wake_by_ref();
                        Poll::Pending
                    },
                    Poll::Ready,
                )
            }),
        )
        .await
        .expect("pending tool did not start");
        drop(execution);

        assert!(context.result::<String>().is_none());
    }

    #[tokio::test]
    async fn empty_tool_context_uses_default() {
        let handle = ToolServer::new().tool(ContextReader).run();
        let result = execute_tool(&handle, "context_reader", "{}").await.unwrap();

        assert_eq!(result, "no session");
    }

    #[tokio::test]
    async fn tool_ignoring_context_still_works() {
        let handle = ToolServer::new().tool(MockAddTool).run();
        let mut context = ToolContext::new();
        context.insert(SessionId("ignored".to_string()));
        let args = serde_json::to_string(&serde_json::json!({"x": 3, "y": 7})).unwrap();
        let result = execute_tool_with_context(&handle, "add", &args, &mut context)
            .await
            .unwrap();

        assert_eq!(result, "10");
    }

    #[tokio::test]
    async fn execute_classifies_a_missing_tool_as_not_found() {
        let handle = ToolServer::new().tool(MockAddTool).run();
        let error = execute_tool(&handle, "does_not_exist", "{}")
            .await
            .unwrap_err();
        assert_eq!(error.kind(), crate::tool::ToolErrorKind::NotFound);
        assert!(
            error
                .model_feedback()
                .is_some_and(|feedback| feedback.contains("does_not_exist"))
        );
    }
}
