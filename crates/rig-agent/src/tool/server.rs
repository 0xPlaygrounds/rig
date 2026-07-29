use std::{collections::BTreeSet, sync::Arc};

#[cfg(all(feature = "rmcp", not(target_family = "wasm")))]
use std::collections::HashMap;

use indexmap::IndexMap;
use tokio::sync::RwLock;

#[cfg(all(feature = "rmcp", not(target_family = "wasm")))]
use crate::tool::ErasedTool;

use crate::{
    completion::ToolDefinition,
    tool::{
        DynamicTool, PortableDynamicTool, RegisteredTool, Tool, ToolContext, ToolDispatch,
        ToolResult, ToolSet, dispatch_tool,
    },
};

/// One turn's provider definitions and the exact registry entries behind them.
///
/// Registration changes after this snapshot is built take effect on the next
/// turn. Calls from the current turn dispatch through these pinned handles, so
/// the implementation cannot drift from the schema the provider received.
#[derive(Clone)]
pub(crate) struct ToolRegistrySnapshot {
    definitions: Vec<ToolDefinition>,
    tools: IndexMap<String, RegisteredTool>,
}

impl ToolRegistrySnapshot {
    fn new(tools: IndexMap<String, RegisteredTool>) -> Self {
        let definitions = tools
            .iter()
            .map(|(name, tool)| tool.definition_with_name(name.clone()))
            .collect();
        Self { definitions, tools }
    }

    /// Provider-facing definitions in the same order as their pinned handles.
    pub(crate) fn definitions(&self) -> &[ToolDefinition] {
        &self.definitions
    }

    /// Narrow both provider exposure and dispatch to one per-turn allow-list.
    pub(crate) fn retain_names(&mut self, names: &BTreeSet<String>) {
        self.definitions
            .retain(|definition| names.contains(&definition.name));
        self.tools.retain(|name, _| names.contains(name));
    }

    /// Dispatch through the exact implementation advertised for this turn.
    pub(crate) async fn dispatch(
        &self,
        tool_name: &str,
        args: &str,
        context: &ToolContext,
    ) -> ToolDispatch {
        let tool = self.tools.get(tool_name).cloned();
        dispatch_tool(tool_name, args.to_string(), tool, context).await
    }
}

/// Shared state behind a `ToolServerHandle`.
struct ToolServerState {
    /// The authoritative ordered registry for execution and exposure.
    toolset: ToolSet,
    /// Generation tokens for registrations managed by MCP client handlers.
    /// A normal registration clears the token, preventing a stale handler
    /// refresh from replacing or removing the newer tool.
    #[cfg(all(feature = "rmcp", not(target_family = "wasm")))]
    managed_generations: HashMap<String, ManagedToolToken>,
}

#[cfg(all(feature = "rmcp", not(target_family = "wasm")))]
impl ToolServerState {
    /// Remove remote registrations whose transport can no longer accept calls.
    /// In-process tools use the default live state, while both handler-managed
    /// and directly registered MCP tools report their transport state.
    fn retire_disconnected_tools(&mut self) {
        let disconnected = self
            .toolset
            .tools
            .keys()
            .filter(|name| self.toolset.get(name).is_none_or(|tool| !tool.is_live()))
            .cloned()
            .collect::<Vec<_>>();

        for name in disconnected {
            self.toolset.delete_tool(&name);
            self.managed_generations.remove(&name);
            tracing::debug!(tool_name = %name, "retired disconnected MCP tool registration");
        }
    }
}

/// Opaque identity for one MCP-managed registry generation.
#[cfg(all(feature = "rmcp", not(target_family = "wasm")))]
#[derive(Clone, Debug)]
pub(crate) struct ManagedToolToken(Arc<()>);

#[cfg(all(feature = "rmcp", not(target_family = "wasm")))]
impl ManagedToolToken {
    fn new() -> Self {
        Self(Arc::new(()))
    }
}

#[cfg(all(feature = "rmcp", not(target_family = "wasm")))]
impl PartialEq for ManagedToolToken {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

#[cfg(all(feature = "rmcp", not(target_family = "wasm")))]
impl Eq for ManagedToolToken {}

/// Builder for constructing a [`ToolServerHandle`].
///
/// Accumulates tools and configuration, then produces a shared handle via
/// [`run()`](ToolServer::run).
///
/// Every registered tool is advertised to the provider. To narrow the
/// advertised set per turn (the successor of index-backed dynamic tool
/// retrieval), patch the request from a completion-call hook with
/// [`RequestPatch::active_tools`](crate::agent::hook::RequestPatch::active_tools).
pub struct ToolServer {
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
            toolset: ToolSet::default(),
        }
    }

    pub(crate) fn add_tools(mut self, tools: ToolSet) -> Self {
        self.toolset = tools;
        self
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

    /// Add a context-free dynamic tool through the classic registry adapter.
    pub fn portable_dynamic_tool(mut self, tool: PortableDynamicTool) -> Self {
        self.toolset.add_portable_dynamic_tool(tool);
        self
    }

    /// Add an MCP tool (from `rmcp`) to the agent, bounded by
    /// [`DEFAULT_MCP_TOOL_TIMEOUT`](crate::tool::rmcp::DEFAULT_MCP_TOOL_TIMEOUT)
    /// (see issue #1914). Use [`rmcp_tool_with_timeout`](Self::rmcp_tool_with_timeout)
    /// to change or disable it.
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    #[cfg(all(feature = "rmcp", not(target_family = "wasm")))]
    pub fn rmcp_tool(self, tool: rmcp::model::Tool, client: rmcp::service::ServerSink) -> Self {
        self.rmcp_tool_with_timeout(tool, client, crate::tool::rmcp::DEFAULT_MCP_TOOL_TIMEOUT)
    }

    /// Add an MCP tool (from `rmcp`) with a per-call timeout (see issue #1914).
    ///
    /// Pass a [`Duration`](std::time::Duration) to bound the call, or `None` to
    /// disable the timeout (unbounded).
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    #[cfg(all(feature = "rmcp", not(target_family = "wasm")))]
    pub fn rmcp_tool_with_timeout(
        mut self,
        tool: rmcp::model::Tool,
        client: rmcp::service::ServerSink,
        timeout: impl Into<Option<std::time::Duration>>,
    ) -> Self {
        use crate::tool::rmcp::McpTool;
        self.toolset.add_erased(Arc::new(
            McpTool::from_mcp_server(tool, client).with_timeout(timeout),
        ));
        self
    }

    /// Consume the builder and return a shared [`ToolServerHandle`].
    pub fn run(self) -> ToolServerHandle {
        ToolServerHandle(Arc::new(RwLock::new(ToolServerState {
            toolset: self.toolset,
            #[cfg(all(feature = "rmcp", not(target_family = "wasm")))]
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
    /// Register a new static tool. Re-registering an existing name replaces
    /// the implementation (last wins) and keeps its position.
    pub async fn add_tool<T>(&self, tool: T)
    where
        T: Tool + 'static,
    {
        let mut state = self.0.write().await;
        let _name = state.toolset.add_tool(tool);
        #[cfg(all(feature = "rmcp", not(target_family = "wasm")))]
        state.managed_generations.remove(&_name);
    }

    /// Register a runtime-defined static tool.
    pub async fn add_dynamic_tool(&self, tool: DynamicTool) {
        let mut state = self.0.write().await;
        let _name = state.toolset.add_dynamic_tool(tool);
        #[cfg(all(feature = "rmcp", not(target_family = "wasm")))]
        state.managed_generations.remove(&_name);
    }

    /// Register a context-free dynamic tool through the classic adapter.
    pub async fn add_portable_dynamic_tool(&self, tool: PortableDynamicTool) {
        let mut state = self.0.write().await;
        let _name = state.toolset.add_portable_dynamic_tool(tool);
        #[cfg(all(feature = "rmcp", not(target_family = "wasm")))]
        state.managed_generations.remove(&_name);
    }

    #[cfg(all(feature = "rmcp", test))]
    pub(crate) async fn add_erased_tool(&self, tool: Arc<dyn ErasedTool>) {
        let mut state = self.0.write().await;
        let name = state.toolset.add_erased(tool);
        state.managed_generations.remove(&name);
    }

    /// Atomically install the initial tools owned by one MCP handler.
    /// Initial connection retains the registry's last-registration-wins policy.
    #[cfg(all(feature = "rmcp", not(target_family = "wasm")))]
    pub(crate) async fn add_managed_erased_tools(
        &self,
        tools: Vec<Arc<dyn ErasedTool>>,
    ) -> HashMap<String, ManagedToolToken> {
        let mut state = self.0.write().await;
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

    /// Atomically reconcile one handler's MCP registrations with a refreshed
    /// tool list. Existing names are changed only when their expected generation
    /// remains current; newer local or peer-handler registrations win.
    #[cfg(all(feature = "rmcp", not(target_family = "wasm")))]
    pub(crate) async fn reconcile_managed_erased_tools(
        &self,
        mut expected: HashMap<String, ManagedToolToken>,
        tools: Vec<Arc<dyn ErasedTool>>,
    ) -> HashMap<String, ManagedToolToken> {
        let mut state = self.0.write().await;
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
        let mut ordered_entries = Vec::with_capacity(managed_order.len());
        for name in managed_order {
            if let Some(entry) = state.toolset.tools.shift_remove_entry(&name) {
                ordered_entries.push(entry);
            }
        }
        for (name, registration) in ordered_entries {
            state.toolset.tools.insert(name, registration);
        }

        refreshed
    }

    /// Merge an entire toolset into the server in registration order.
    /// Existing names are replaced (last wins) and keep their position.
    pub async fn append_toolset(&self, toolset: ToolSet) {
        let mut state = self.0.write().await;
        #[cfg(all(feature = "rmcp", not(target_family = "wasm")))]
        let names = toolset.tools.keys().cloned().collect::<Vec<_>>();
        state.toolset.add_tools(toolset);
        #[cfg(all(feature = "rmcp", not(target_family = "wasm")))]
        for name in names {
            state.managed_generations.remove(&name);
        }
    }

    /// Remove a tool by name.
    pub async fn remove_tool(&self, tool_name: &str) {
        let mut state = self.0.write().await;
        state.toolset.delete_tool(tool_name);
        #[cfg(all(feature = "rmcp", not(target_family = "wasm")))]
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
        let ToolDispatch {
            result,
            context: dispatch_context,
        } = self.dispatch(tool_name, args, context).await;
        context.accept_dispatch_result(dispatch_context);
        result
    }

    /// Run one isolated dispatch and retain its full context for agent hooks.
    pub(crate) async fn dispatch(
        &self,
        tool_name: &str,
        args: &str,
        context: &ToolContext,
    ) -> ToolDispatch {
        #[cfg(all(feature = "rmcp", not(target_family = "wasm")))]
        let tool = {
            let mut state = self.0.write().await;
            state.retire_disconnected_tools();
            state.toolset.get(tool_name).cloned()
        };
        #[cfg(not(all(feature = "rmcp", not(target_family = "wasm"))))]
        let tool = {
            let state = self.0.read().await;
            state.toolset.get(tool_name).cloned()
        };
        dispatch_tool(tool_name, args.to_string(), tool, context).await
    }

    /// Retrieve the provider-facing tool definitions in registration order.
    pub async fn get_tool_defs(&self) -> Vec<ToolDefinition> {
        self.snapshot_tool_defs().await.definitions.clone()
    }

    /// Resolve one ordered provider/dispatch snapshot for an agent turn.
    ///
    /// One read lock resolves every registered name to an exact
    /// implementation. That single instant is the turn boundary: later
    /// replacements are visible only to the next snapshot. Per-turn narrowing
    /// happens afterwards through
    /// [`RequestPatch::active_tools`](crate::agent::hook::RequestPatch::active_tools).
    pub(crate) async fn snapshot_tool_defs(&self) -> ToolRegistrySnapshot {
        #[cfg(all(feature = "rmcp", not(target_family = "wasm")))]
        let tools = {
            let mut state = self.0.write().await;
            state.retire_disconnected_tools();
            state.toolset.tools.clone()
        };
        #[cfg(not(all(feature = "rmcp", not(target_family = "wasm"))))]
        let tools = {
            let state = self.0.read().await;
            state.toolset.tools.clone()
        };

        ToolRegistrySnapshot::new(tools)
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
            MockAddTool, MockBarrierTool, MockControlledTool, MockSubtractTool,
        },
        tool::{
            Tool, ToolContext, ToolExecutionError, ToolSet,
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

    #[tokio::test]
    pub async fn test_toolserver() {
        let server = ToolServer::new();

        let handle = server.run();

        handle.add_tool(MockAddTool).await;
        let res = handle.get_tool_defs().await;

        assert_eq!(res.len(), 1);

        let json_args_as_string =
            serde_json::to_string(&serde_json::json!({"x": 2, "y": 5})).unwrap();
        let res = execute_tool(&handle, "add", &json_args_as_string)
            .await
            .unwrap();
        assert_eq!(res, "7");

        handle.remove_tool("add").await;
        let res = handle.get_tool_defs().await;

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
        let snapshot = handle.snapshot_tool_defs().await;

        handle
            .add_tool(ReplacementTool {
                description: "second schema",
                output: "second implementation",
            })
            .await;

        assert_eq!(snapshot.definitions()[0].description, "first schema");
        let dispatch = snapshot
            .dispatch(ReplacementTool::NAME, "{}", &ToolContext::new())
            .await;
        assert_eq!(dispatch.result.output().render(), "first implementation");

        let live = handle
            .dispatch(ReplacementTool::NAME, "{}", &ToolContext::new())
            .await;
        assert_eq!(live.result.output().render(), "second implementation");

        let next_snapshot = handle.snapshot_tool_defs().await;
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
            handle.add_tool(MockAddTool).await;
            handle.add_tool(MockSubtractTool).await;
            handle.get_tool_defs().await
        };
        via_add_tool.sort_by(|a, b| a.name.cmp(&b.name));

        let mut via_append_toolset = {
            let handle = ToolServer::new().run();
            let mut toolset = ToolSet::default();
            toolset.add_tool(MockAddTool);
            toolset.add_tool(MockSubtractTool);
            handle.append_toolset(toolset).await;
            handle.get_tool_defs().await
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

        let defs = handle.get_tool_defs().await;
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, NamedTool::NAME);
    }

    #[tokio::test]
    pub async fn handle_add_tool_uses_canonical_static_name() {
        let handle = ToolServer::new().run();
        handle.add_tool(NamedTool::new()).await;

        let defs = handle.get_tool_defs().await;
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, NamedTool::NAME);
    }

    #[tokio::test]
    pub async fn get_tool_defs_preserves_static_registration_order() {
        let handle = ToolServer::new().run();
        handle.add_tool(MockSubtractTool).await;
        handle.add_tool(MockAddTool).await;

        let defs = handle.get_tool_defs().await;
        assert_eq!(
            defs.iter().map(|def| def.name.as_str()).collect::<Vec<_>>(),
            vec!["subtract", "add"]
        );
    }

    #[tokio::test]
    pub async fn duplicate_registration_advertises_one_definition() {
        let handle = ToolServer::new().tool(MockAddTool).run();
        handle.add_tool(MockAddTool).await;

        let mut toolset = ToolSet::default();
        toolset.add_tool(MockAddTool);
        handle.append_toolset(toolset).await;

        let defs = handle.get_tool_defs().await;
        assert_eq!(
            defs.len(),
            1,
            "re-registering a name must not advertise duplicate declarations"
        );
        assert_eq!(defs[0].name, "add");
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
            assert!(res.is_ok(), "Tool call failed: {:?}", res);
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

        // Try to write to the state (add a tool) while the tool call is mid-execution.
        // If the read lock is incorrectly held across tool execution, this will deadlock.
        let add_result =
            tokio::time::timeout(Duration::from_secs(1), handle.add_tool(MockAddTool)).await;

        assert!(
            add_result.is_ok(),
            "Writing to ToolServer deadlocked! The read lock is being held across tool execution."
        );

        // Allow the background tool to finish and clean up
        allow_finish.notify_one();
        let call_result = call_task.await.unwrap();
        assert_eq!(call_result.unwrap(), "42");
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
            Ok(context
                .get::<SessionId>()
                .map(|session| format!("session:{}", session.0))
                .unwrap_or_else(|| "no session".to_string()))
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
