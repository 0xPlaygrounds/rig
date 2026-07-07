use std::sync::Arc;

use tokio::sync::RwLock;

use crate::{
    completion::ToolDefinition,
    tool::{
        Tool, ToolCallExtensions, ToolDyn, ToolExecutionResult, ToolFailure, ToolSet, ToolSetError,
    },
};

/// Shared state behind a `ToolServerHandle`.
struct ToolServerState {
    /// The toolset where tools are registered and executed. Every registered
    /// tool is advertised, in registration order.
    toolset: ToolSet,
}

/// Builder for constructing a [`ToolServerHandle`].
///
/// Accumulates tools and configuration, then produces a shared handle via
/// [`run()`](ToolServer::run).
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

    /// Add an MCP tool (from `rmcp`) to the agent, bounded by
    /// [`DEFAULT_MCP_TOOL_TIMEOUT`](crate::tool::rmcp::DEFAULT_MCP_TOOL_TIMEOUT)
    /// (see issue #1914). Use [`rmcp_tool_with_timeout`](Self::rmcp_tool_with_timeout)
    /// to change or disable it.
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    #[cfg(feature = "rmcp")]
    pub fn rmcp_tool(self, tool: rmcp::model::Tool, client: rmcp::service::ServerSink) -> Self {
        self.rmcp_tool_with_timeout(tool, client, crate::tool::rmcp::DEFAULT_MCP_TOOL_TIMEOUT)
    }

    /// Add an MCP tool (from `rmcp`) with a per-call timeout (see issue #1914).
    ///
    /// Pass a [`Duration`](std::time::Duration) to bound the call, or `None` to
    /// disable the timeout (unbounded).
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    #[cfg(feature = "rmcp")]
    pub fn rmcp_tool_with_timeout(
        mut self,
        tool: rmcp::model::Tool,
        client: rmcp::service::ServerSink,
        timeout: impl Into<Option<std::time::Duration>>,
    ) -> Self {
        use crate::tool::rmcp::McpTool;
        self.toolset
            .add_tool(McpTool::from_mcp_server(tool, client).with_timeout(timeout));
        self
    }

    /// Consume the builder and return a shared [`ToolServerHandle`].
    pub fn run(self) -> ToolServerHandle {
        ToolServerHandle(Arc::new(RwLock::new(ToolServerState {
            toolset: self.toolset,
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
    pub async fn add_tool(&self, tool: impl ToolDyn + 'static) -> Result<(), ToolServerError> {
        let mut state = self.0.write().await;
        state.toolset.add_tool_boxed(Box::new(tool));
        Ok(())
    }

    /// Merge an entire toolset into the server. Tools from `toolset` become
    /// visible to the LLM via [`Self::get_tool_defs`] in `toolset`'s
    /// registration order. Existing names are replaced (last wins) and keep
    /// their position.
    pub async fn append_toolset(&self, toolset: ToolSet) -> Result<(), ToolServerError> {
        let mut state = self.0.write().await;
        state.toolset.add_tools(toolset);
        Ok(())
    }

    /// Remove a tool by name from the toolset.
    pub async fn remove_tool(&self, tool_name: &str) -> Result<(), ToolServerError> {
        let mut state = self.0.write().await;
        state.toolset.delete_tool(tool_name);
        Ok(())
    }

    /// Look up and execute a tool by name.
    ///
    /// The tool handle is cloned under a brief read lock so that
    /// long-running tool executions never block writers.
    pub async fn call_tool(&self, tool_name: &str, args: &str) -> Result<String, ToolServerError> {
        self.call_tool_with_extensions(tool_name, args, &ToolCallExtensions::EMPTY)
            .await
    }

    /// Look up and execute a tool by name with per-call runtime extensions.
    ///
    /// The extensions are threaded through to [`Tool::call_with_extensions`],
    /// allowing tools to access caller-provided values (auth tokens, session
    /// IDs, etc.). The tool handle is cloned under a brief read lock so that
    /// long-running tool executions never block writers.
    pub async fn call_tool_with_extensions(
        &self,
        tool_name: &str,
        args: &str,
        extensions: &ToolCallExtensions,
    ) -> Result<String, ToolServerError> {
        let tool = {
            let state = self.0.read().await;
            state.toolset.get(tool_name).cloned()
        };

        match tool {
            Some(tool) => {
                tracing::debug!(target: "rig",
                    "Calling tool {tool_name} with args:\n{}",
                    serde_json::to_string_pretty(&args).unwrap_or_default()
                );
                tool.call_with_extensions(args.to_string(), extensions)
                    .await
                    .map_err(|e| ToolSetError::ToolCallError(e).into())
            }
            None => Err(ToolServerError::ToolsetError(
                ToolSetError::ToolNotFoundError(tool_name.to_string()),
            )),
        }
    }

    /// Look up and execute a tool by name, returning the structured
    /// [`ToolExecutionResult`] (model output + [`ToolOutcome`](crate::tool::ToolOutcome)
    /// + result extensions).
    ///
    /// The structured counterpart of [`call_tool_with_extensions`](Self::call_tool_with_extensions),
    /// and the path the agent loop drives so hooks, tracing, and policies observe
    /// the structured outcome. A missing tool resolves to a
    /// [`NotFound`](crate::tool::ToolFailureKind::NotFound) outcome rather than a
    /// `Result::Err`. The tool handle is cloned under a brief read lock so that
    /// long-running tool executions never block writers.
    pub async fn call_tool_structured(
        &self,
        tool_name: &str,
        args: &str,
        extensions: &ToolCallExtensions,
    ) -> ToolExecutionResult {
        let tool = {
            let state = self.0.read().await;
            state.toolset.get(tool_name).cloned()
        };

        match tool {
            Some(tool) => {
                tracing::debug!(target: "rig",
                    "Calling tool {tool_name} with args:\n{}",
                    serde_json::to_string_pretty(&args).unwrap_or_default()
                );
                tool.call_structured(args.to_string(), extensions).await
            }
            None => ToolExecutionResult::failed(
                format!("tool `{tool_name}` not found"),
                ToolFailure::not_found(format!("no tool named `{tool_name}` is registered")),
            ),
        }
    }

    /// Retrieve the definitions of every registered static tool, advertised to
    /// the model. Rig has no dynamic-tool mechanism — the advertised set is simply
    /// the tools registered on the server.
    pub async fn get_tool_defs(&self) -> Result<Vec<ToolDefinition>, ToolServerError> {
        // Clone the handles under a brief read lock so that a slow tool
        // `definition()` (e.g. an MCP round-trip) never holds the lock across
        // `.await`. Handles come out in registration order.
        let handles: Vec<Arc<dyn ToolDyn>> = {
            let state = self.0.read().await;
            state.toolset.ordered_tools().cloned().collect()
        };

        let mut tools = Vec::with_capacity(handles.len());
        for handle in handles {
            tools.push(handle.definition().await);
        }

        // The toolset is keyed by a tool's registration name (`Tool::NAME`),
        // but the advertised name lives in its `ToolDefinition` — the two are
        // independent, so two distinct registered tools could still emit the
        // same definition name. Providers reject duplicate function
        // declarations, so keep the first and drop exact-name repeats.
        let mut seen = std::collections::HashSet::new();
        tools.retain(|def| {
            let fresh = seen.insert(def.name.clone());
            if !fresh {
                tracing::debug!(
                    tool_name = %def.name,
                    "dropping duplicate tool definition from the request"
                );
            }
            fresh
        });

        Ok(tools)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ToolServerError {
    #[error("Toolset error: {0}")]
    ToolsetError(#[from] ToolSetError),
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use crate::{
        completion::ToolDefinition,
        test_utils::{MockAddTool, MockBarrierTool, MockControlledTool, MockSubtractTool},
        tool::{Tool, ToolCallExtensions, ToolOutcome, ToolSet, server::ToolServer},
    };

    /// A perfectly ordinary user tool that happens to be named `tool_search`. Rig
    /// has no reserved tool names, so it registers, advertises, and executes like
    /// any other tool.
    #[derive(Clone)]
    struct ToolSearchNamedTool;

    #[derive(serde::Deserialize)]
    struct EmptyArgs {}

    #[derive(Debug, thiserror::Error)]
    #[error("tool error")]
    struct ToolSearchErr;

    impl Tool for ToolSearchNamedTool {
        const NAME: &'static str = "tool_search";
        type Error = ToolSearchErr;
        type Args = EmptyArgs;
        type Output = String;

        async fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: Self::NAME.to_string(),
                description: "a user tool named tool_search".to_string(),
                parameters: serde_json::json!({ "type": "object", "properties": {} }),
            }
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok("user tool ran".to_string())
        }
    }

    #[tokio::test]
    pub async fn get_tool_defs_returns_only_registered_static_tools() {
        let handle = ToolServer::new()
            .tool(MockAddTool)
            .tool(MockSubtractTool)
            .run();
        let mut names: Vec<String> = handle
            .get_tool_defs()
            .await
            .unwrap()
            .into_iter()
            .map(|d| d.name)
            .collect();
        names.sort();
        assert_eq!(names, vec!["add".to_string(), "subtract".to_string()]);
    }

    /// Two distinct registration names (`Tool::NAME`) that advertise the *same*
    /// `ToolDefinition.name`. A tool's registration name and its advertised name
    /// are independent, so this collision is possible; providers reject duplicate
    /// function declarations, so `get_tool_defs` must dedup on the advertised name.
    #[derive(Clone)]
    struct DupDefA;
    #[derive(Clone)]
    struct DupDefB;

    macro_rules! dup_def_tool {
        ($ty:ty, $name:literal) => {
            impl Tool for $ty {
                const NAME: &'static str = $name;
                type Error = ToolSearchErr;
                type Args = EmptyArgs;
                type Output = String;

                async fn definition(&self) -> ToolDefinition {
                    ToolDefinition {
                        name: "shared".to_string(),
                        description: concat!("advertised by ", $name).to_string(),
                        parameters: serde_json::json!({ "type": "object", "properties": {} }),
                    }
                }

                async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
                    Ok($name.to_string())
                }
            }
        };
    }
    dup_def_tool!(DupDefA, "dup_a");
    dup_def_tool!(DupDefB, "dup_b");

    #[tokio::test]
    pub async fn get_tool_defs_dedups_by_advertised_definition_name() {
        // `dup_a` and `dup_b` register under distinct keys but both advertise the
        // definition name "shared". Only the first survives, so the request never
        // carries a duplicate function declaration.
        let handle = ToolServer::new().tool(DupDefA).tool(DupDefB).run();
        let names: Vec<String> = handle
            .get_tool_defs()
            .await
            .unwrap()
            .into_iter()
            .map(|d| d.name)
            .collect();
        assert_eq!(
            names,
            vec!["shared".to_string()],
            "colliding advertised names are deduped to the first"
        );
    }

    #[tokio::test]
    pub async fn user_tool_named_tool_search_is_registered_and_executable() {
        // `tool_search` is not reserved: a user tool with that name is advertised
        // and dispatched normally.
        let handle = ToolServer::new().tool(ToolSearchNamedTool).run();

        let names: Vec<String> = handle
            .get_tool_defs()
            .await
            .unwrap()
            .into_iter()
            .map(|d| d.name)
            .collect();
        assert_eq!(
            names,
            vec!["tool_search".to_string()],
            "user tool is advertised"
        );

        let out = handle.call_tool("tool_search", "{}").await.unwrap();
        assert_eq!(out, "user tool ran", "user tool executes via call_tool");

        let structured = handle
            .call_tool_structured("tool_search", "{}", &ToolCallExtensions::EMPTY)
            .await;
        assert!(matches!(structured.outcome(), ToolOutcome::Success));
        assert_eq!(structured.model_output(), "user tool ran");
    }

    #[tokio::test]
    pub async fn removing_a_tool_removes_it_from_defs_and_dispatch() {
        let handle = ToolServer::new().tool(MockAddTool).run();
        handle.remove_tool("add").await.unwrap();

        assert!(
            handle.get_tool_defs().await.unwrap().is_empty(),
            "removed tool is no longer advertised"
        );
        assert!(
            handle.call_tool("add", "{}").await.is_err(),
            "removed tool is no longer dispatchable"
        );
    }

    #[tokio::test]
    pub async fn test_toolserver() {
        let server = ToolServer::new();

        let handle = server.run();

        handle.add_tool(MockAddTool).await.unwrap();
        let res = handle.get_tool_defs().await.unwrap();

        assert_eq!(res.len(), 1);

        let json_args_as_string =
            serde_json::to_string(&serde_json::json!({"x": 2, "y": 5})).unwrap();
        let res = handle.call_tool("add", &json_args_as_string).await.unwrap();
        assert_eq!(res, "7");

        handle.remove_tool("add").await.unwrap();
        let res = handle.get_tool_defs().await.unwrap();

        assert_eq!(res.len(), 0);
    }

    #[tokio::test]
    pub async fn test_toolserver_append_toolset_matches_add_tool() {
        let mut via_add_tool = {
            let handle = ToolServer::new().run();
            handle.add_tool(MockAddTool).await.unwrap();
            handle.add_tool(MockSubtractTool).await.unwrap();
            handle.get_tool_defs().await.unwrap()
        };
        via_add_tool.sort_by(|a, b| a.name.cmp(&b.name));

        let mut via_append_toolset = {
            let handle = ToolServer::new().run();
            let mut toolset = ToolSet::default();
            toolset.add_tool(MockAddTool);
            toolset.add_tool(MockSubtractTool);
            handle.append_toolset(toolset).await.unwrap();
            handle.get_tool_defs().await.unwrap()
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
    pub async fn duplicate_registration_advertises_one_definition() {
        let handle = ToolServer::new().tool(MockAddTool).run();
        handle.add_tool(MockAddTool).await.unwrap();

        let mut toolset = ToolSet::default();
        toolset.add_tool(MockAddTool);
        handle.append_toolset(toolset).await.unwrap();

        let defs = handle.get_tool_defs().await.unwrap();
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
            .map(|_| handle.call_tool("barrier_tool", "{}"))
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
            tokio::spawn(async move { handle_clone.call_tool("controlled", "{}").await });

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
        assert!(add_result.unwrap().is_ok());

        // Allow the background tool to finish and clean up
        allow_finish.notify_one();
        let call_result = call_task.await.unwrap();
        assert_eq!(call_result.unwrap(), "42");
    }

    // --- call_with_extensions tests ---

    #[derive(Clone)]
    struct SessionId(String);

    #[derive(serde::Deserialize, serde::Serialize)]
    struct ExtensionsReader;

    #[derive(Debug, thiserror::Error)]
    #[error("context reader error")]
    struct ExtensionsReaderError;

    impl crate::tool::Tool for ExtensionsReader {
        const NAME: &'static str = "context_reader";
        type Error = ExtensionsReaderError;
        type Args = serde_json::Value;
        type Output = String;

        async fn definition(&self) -> crate::completion::ToolDefinition {
            crate::completion::ToolDefinition {
                name: "context_reader".to_string(),
                description: "Reads SessionId from context".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            }
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok("no context".to_string())
        }

        async fn call_with_extensions(
            &self,
            _args: Self::Args,
            extensions: &crate::tool::ToolCallExtensions,
        ) -> Result<Self::Output, Self::Error> {
            match extensions.get::<SessionId>() {
                Some(session) => Ok(format!("session:{}", session.0)),
                None => Ok("no session".to_string()),
            }
        }
    }

    #[tokio::test]
    async fn test_call_tool_with_extensions_reaches_tool() {
        let server = ToolServer::new().tool(ExtensionsReader);
        let handle = server.run();

        let mut extensions = crate::tool::ToolCallExtensions::new();
        extensions.insert(SessionId("abc-123".to_string()));

        let result = handle
            .call_tool_with_extensions("context_reader", "{}", &extensions)
            .await
            .unwrap();

        assert_eq!(result, "session:abc-123");
    }

    #[tokio::test]
    async fn test_call_tool_without_extensions_uses_default() {
        let server = ToolServer::new().tool(ExtensionsReader);
        let handle = server.run();

        let result = handle.call_tool("context_reader", "{}").await.unwrap();
        assert_eq!(result, "no session");
    }

    #[tokio::test]
    async fn test_tool_ignoring_extensions_still_works() {
        let server = ToolServer::new().tool(MockAddTool);
        let handle = server.run();

        let mut extensions = crate::tool::ToolCallExtensions::new();
        extensions.insert(SessionId("ignored".to_string()));

        let args = serde_json::to_string(&serde_json::json!({"x": 3, "y": 7})).unwrap();
        let result = handle
            .call_tool_with_extensions("add", &args, &extensions)
            .await
            .unwrap();

        assert_eq!(result, "10");
    }

    #[tokio::test]
    async fn call_tool_structured_returns_success_for_a_known_tool() {
        use crate::tool::{ToolCallExtensions, ToolOutcome};

        let handle = ToolServer::new().tool(MockAddTool).run();
        let args = serde_json::to_string(&serde_json::json!({"x": 2, "y": 5})).unwrap();
        let result = handle
            .call_tool_structured("add", &args, &ToolCallExtensions::EMPTY)
            .await;

        assert!(matches!(result.outcome, ToolOutcome::Success));
        assert_eq!(result.model_output, "7");
    }

    #[tokio::test]
    async fn call_tool_structured_classifies_a_missing_tool_as_not_found() {
        use crate::tool::{ToolCallExtensions, ToolFailureKind, ToolOutcome};

        let handle = ToolServer::new().tool(MockAddTool).run();
        let result = handle
            .call_tool_structured("does_not_exist", "{}", &ToolCallExtensions::EMPTY)
            .await;

        match result.outcome {
            ToolOutcome::Error(failure) => assert_eq!(failure.kind, ToolFailureKind::NotFound),
            other => panic!("expected a NotFound error outcome, got {other:?}"),
        }
        assert!(result.model_output.contains("does_not_exist"));
    }
}
