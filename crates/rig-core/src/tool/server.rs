use std::sync::Arc;

use tokio::sync::RwLock;

use crate::{
    completion::{CompletionError, ToolDefinition},
    tool::{Tool, ToolCallContext, ToolDyn, ToolSet, ToolSetError},
    vector_store::{VectorSearchRequest, VectorStoreError, VectorStoreIndexDyn, request::Filter},
};

/// Shared state behind a `ToolServerHandle`.
struct ToolServerState {
    /// Static tool names that persist until explicitly removed.
    static_tool_names: Vec<String>,
    /// Dynamic tools fetched from vector stores on each prompt.
    dynamic_tools: Vec<(usize, Arc<dyn VectorStoreIndexDyn + Send + Sync>)>,
    /// The toolset where tools are registered and executed.
    toolset: ToolSet,
}

/// Builder for constructing a [`ToolServerHandle`].
///
/// Accumulates tools and configuration, then produces a shared handle via
/// [`run()`](ToolServer::run).
pub struct ToolServer {
    static_tool_names: Vec<String>,
    dynamic_tools: Vec<(usize, Arc<dyn VectorStoreIndexDyn + Send + Sync>)>,
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
            static_tool_names: Vec::new(),
            dynamic_tools: Vec::new(),
            toolset: ToolSet::default(),
        }
    }

    pub(crate) fn static_tool_names(mut self, names: Vec<String>) -> Self {
        self.static_tool_names = names;
        self
    }

    pub(crate) fn add_tools(mut self, tools: ToolSet) -> Self {
        self.toolset = tools;
        self
    }

    pub(crate) fn add_dynamic_tools(
        mut self,
        dyn_tools: Vec<(usize, Arc<dyn VectorStoreIndexDyn + Send + Sync>)>,
    ) -> Self {
        self.dynamic_tools = dyn_tools;
        self
    }

    /// Add a static tool to the agent
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        let toolname = tool.name();
        self.toolset.add_tool(tool);
        self.static_tool_names.push(toolname);
        self
    }

    /// Add an MCP tool (from `rmcp`) to the agent
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    #[cfg(feature = "rmcp")]
    pub fn rmcp_tool(mut self, tool: rmcp::model::Tool, client: rmcp::service::ServerSink) -> Self {
        use crate::tool::rmcp::McpTool;
        let toolname = tool.name.clone();
        self.toolset
            .add_tool(McpTool::from_mcp_server(tool, client));
        self.static_tool_names.push(toolname.to_string());
        self
    }

    /// Add some dynamic tools to the agent. On each prompt, `sample` tools from the
    /// dynamic toolset will be inserted in the request.
    pub fn dynamic_tools(
        mut self,
        sample: usize,
        dynamic_tools: impl VectorStoreIndexDyn + Send + Sync + 'static,
        toolset: ToolSet,
    ) -> Self {
        self.dynamic_tools.push((sample, Arc::new(dynamic_tools)));
        self.toolset.add_tools(toolset);
        self
    }

    /// Consume the builder and return a shared [`ToolServerHandle`].
    pub fn run(self) -> ToolServerHandle {
        ToolServerHandle(Arc::new(RwLock::new(ToolServerState {
            static_tool_names: self.static_tool_names,
            dynamic_tools: self.dynamic_tools,
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
    /// Register a new static tool.
    pub async fn add_tool(&self, tool: impl ToolDyn + 'static) -> Result<(), ToolServerError> {
        let mut state = self.0.write().await;
        state.static_tool_names.push(tool.name());
        state.toolset.add_tool_boxed(Box::new(tool));
        Ok(())
    }

    /// Merge an entire toolset into the server. Tool names from `toolset`
    /// are appended to the static-tool list, so the tools become visible
    /// to the LLM via [`Self::get_tool_defs`].
    pub async fn append_toolset(&self, toolset: ToolSet) -> Result<(), ToolServerError> {
        let mut state = self.0.write().await;
        state
            .static_tool_names
            .extend(toolset.tools.keys().cloned());
        state.toolset.add_tools(toolset);
        Ok(())
    }

    /// Remove a tool by name from both the toolset and the static list.
    pub async fn remove_tool(&self, tool_name: &str) -> Result<(), ToolServerError> {
        let mut state = self.0.write().await;
        state.static_tool_names.retain(|x| *x != tool_name);
        state.toolset.delete_tool(tool_name);
        Ok(())
    }

    /// Look up and execute a tool by name.
    ///
    /// The tool handle is cloned under a brief read lock so that
    /// long-running tool executions never block writers.
    pub async fn call_tool(&self, tool_name: &str, args: &str) -> Result<String, ToolServerError> {
        self.call_tool_with_context(tool_name, args, ToolCallContext::new())
            .await
    }

    /// Look up and execute a tool by name with per-call runtime context.
    ///
    /// The context is threaded through to [`Tool::call_with_context`], allowing
    /// tools to access caller-provided values (auth tokens, session IDs, etc.).
    /// The tool handle is cloned under a brief read lock so that long-running
    /// tool executions never block writers.
    pub async fn call_tool_with_context(
        &self,
        tool_name: &str,
        args: &str,
        ctx: ToolCallContext,
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
                tool.call_with_context(args.to_string(), ctx)
                    .await
                    .map_err(|e| ToolSetError::ToolCallError(e).into())
            }
            None => Err(ToolServerError::ToolsetError(
                ToolSetError::ToolNotFoundError(tool_name.to_string()),
            )),
        }
    }

    /// Retrieve tool definitions, optionally using a prompt to select
    /// dynamic tools from configured vector stores.
    pub async fn get_tool_defs(
        &self,
        prompt: Option<String>,
    ) -> Result<Vec<ToolDefinition>, ToolServerError> {
        // Snapshot the metadata we need under a brief read lock
        let (static_tool_names, dynamic_tools) = {
            let state = self.0.read().await;
            (state.static_tool_names.clone(), state.dynamic_tools.clone())
        };

        let mut tools = if let Some(ref text) = prompt {
            // Create a future for each dynamic tool index
            let search_futures = dynamic_tools.iter().map(|(num_sample, index)| {
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
            let dynamic_tool_ids: Vec<String> = futures::future::try_join_all(search_futures)
                .await
                .map_err(|e| {
                    ToolServerError::DefinitionError(CompletionError::RequestError(Box::new(e)))
                })?
                .into_iter()
                .flatten()
                .collect();

            let dynamic_tool_handles: Vec<_> = {
                let state = self.0.read().await;
                dynamic_tool_ids
                    .iter()
                    .filter_map(|doc| {
                        let handle = state.toolset.get(doc).cloned();
                        if handle.is_none() {
                            tracing::warn!("Tool implementation not found in toolset: {}", doc);
                        }
                        handle
                    })
                    .collect()
            };

            let mut tools = Vec::new();
            for tool in dynamic_tool_handles {
                tools.push(tool.definition(text.clone()).await);
            }
            tools
        } else {
            Vec::new()
        };

        let static_tool_handles: Vec<_> = {
            let state = self.0.read().await;
            static_tool_names
                .iter()
                .filter_map(|toolname| {
                    let handle = state.toolset.get(toolname).cloned();
                    if handle.is_none() {
                        tracing::warn!("Tool implementation not found in toolset: {}", toolname);
                    }
                    handle
                })
                .collect()
        };

        for tool in static_tool_handles {
            tools.push(tool.definition(String::new()).await);
        }

        Ok(tools)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ToolServerError {
    #[error("Toolset error: {0}")]
    ToolsetError(#[from] ToolSetError),
    #[error("Failed to retrieve tool definitions: {0}")]
    DefinitionError(CompletionError),
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use crate::{
        test_utils::{
            BarrierMockToolIndex, MockAddTool, MockBarrierTool, MockControlledTool,
            MockSubtractTool, MockToolIndex,
        },
        tool::{ToolSet, server::ToolServer},
    };

    #[tokio::test]
    pub async fn test_toolserver() {
        let server = ToolServer::new();

        let handle = server.run();

        handle.add_tool(MockAddTool).await.unwrap();
        let res = handle.get_tool_defs(None).await.unwrap();

        assert_eq!(res.len(), 1);

        let json_args_as_string =
            serde_json::to_string(&serde_json::json!({"x": 2, "y": 5})).unwrap();
        let res = handle.call_tool("add", &json_args_as_string).await.unwrap();
        assert_eq!(res, "7");

        handle.remove_tool("add").await.unwrap();
        let res = handle.get_tool_defs(None).await.unwrap();

        assert_eq!(res.len(), 0);
    }

    #[tokio::test]
    pub async fn test_toolserver_append_toolset_matches_add_tool() {
        let mut via_add_tool = {
            let handle = ToolServer::new().run();
            handle.add_tool(MockAddTool).await.unwrap();
            handle.add_tool(MockSubtractTool).await.unwrap();
            handle.get_tool_defs(None).await.unwrap()
        };
        via_add_tool.sort_by(|a, b| a.name.cmp(&b.name));

        let mut via_append_toolset = {
            let handle = ToolServer::new().run();
            let mut toolset = ToolSet::default();
            toolset.add_tool(MockAddTool);
            toolset.add_tool(MockSubtractTool);
            handle.append_toolset(toolset).await.unwrap();
            handle.get_tool_defs(None).await.unwrap()
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
    pub async fn test_toolserver_dynamic_tools() {
        // Create a toolset with both tools
        let mut toolset = ToolSet::default();
        toolset.add_tool(MockAddTool);
        toolset.add_tool(MockSubtractTool);

        // Create a mock index that will return "subtract" as the dynamic tool
        let mock_index = MockToolIndex::new(["subtract"]);

        // Build server with static tool "add" and dynamic tools from the mock index
        let server = ToolServer::new().tool(MockAddTool).dynamic_tools(
            1,
            mock_index,
            ToolSet::from_tools(vec![MockSubtractTool]),
        );

        let handle = server.run();

        // Test with None prompt - should only return static tools
        let res = handle.get_tool_defs(None).await.unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].name, "add");

        // Test with Some prompt - should return both static and dynamic tools
        let res = handle
            .get_tool_defs(Some("calculate difference".to_string()))
            .await
            .unwrap();
        assert_eq!(res.len(), 2);

        // Check that both tools are present (order may vary)
        let tool_names: Vec<&str> = res.iter().map(|t| t.name.as_str()).collect();
        assert!(tool_names.contains(&"add"));
        assert!(tool_names.contains(&"subtract"));
    }

    #[tokio::test]
    pub async fn test_toolserver_dynamic_tools_missing_implementation() {
        // Create a mock index that returns a tool ID that doesn't exist in the toolset
        let mock_index = MockToolIndex::new(["nonexistent_tool"]);

        // Build server with only static tool, but dynamic index references missing tool
        let server =
            ToolServer::new()
                .tool(MockAddTool)
                .dynamic_tools(1, mock_index, ToolSet::default());

        let handle = server.run();

        // Test with Some prompt - should only return static tool since dynamic tool is missing
        let res = handle
            .get_tool_defs(Some("some query".to_string()))
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

    #[tokio::test]
    pub async fn test_toolserver_parallel_dynamic_tool_fetching() {
        // We expect exactly 2 parallel searches to hit the barrier at the same time
        let barrier = Arc::new(tokio::sync::Barrier::new(2));

        let index1 = BarrierMockToolIndex::new(barrier.clone(), "add");
        let index2 = BarrierMockToolIndex::new(barrier.clone(), "subtract");

        // Put both tools in the toolset so they resolve correctly
        let mut toolset = ToolSet::default();
        toolset.add_tool(MockAddTool);
        toolset.add_tool(MockSubtractTool);

        let server = ToolServer::new()
            .dynamic_tools(1, index1, ToolSet::default())
            .dynamic_tools(1, index2, toolset);

        let handle = server.run();

        // This will trigger a search across both indices.
        // If fetched sequentially, the first index will wait at the barrier forever.
        let get_defs = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            handle.get_tool_defs(Some("do math".to_string())),
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

    // --- call_with_context tests ---

    #[derive(Clone)]
    struct SessionId(String);

    #[derive(serde::Deserialize, serde::Serialize)]
    struct ContextReader;

    #[derive(Debug, thiserror::Error)]
    #[error("context reader error")]
    struct ContextReaderError;

    impl crate::tool::Tool for ContextReader {
        const NAME: &'static str = "context_reader";
        type Error = ContextReaderError;
        type Args = serde_json::Value;
        type Output = String;

        async fn definition(&self, _prompt: String) -> crate::completion::ToolDefinition {
            crate::completion::ToolDefinition {
                name: "context_reader".to_string(),
                description: "Reads SessionId from context".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            }
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok("no context".to_string())
        }

        async fn call_with_context(
            &self,
            _args: Self::Args,
            ctx: &crate::tool::ToolCallContext,
        ) -> Result<Self::Output, Self::Error> {
            match ctx.get::<SessionId>() {
                Some(session) => Ok(format!("session:{}", session.0)),
                None => Ok("no session".to_string()),
            }
        }
    }

    #[tokio::test]
    async fn test_call_tool_with_context_reaches_tool() {
        let server = ToolServer::new().tool(ContextReader);
        let handle = server.run();

        let mut ctx = crate::tool::ToolCallContext::new();
        ctx.insert(SessionId("abc-123".to_string()));

        let result = handle
            .call_tool_with_context("context_reader", "{}", ctx)
            .await
            .unwrap();

        assert_eq!(result, "\"session:abc-123\"");
    }

    #[tokio::test]
    async fn test_call_tool_without_context_uses_default() {
        let server = ToolServer::new().tool(ContextReader);
        let handle = server.run();

        let result = handle.call_tool("context_reader", "{}").await.unwrap();
        assert_eq!(result, "\"no session\"");
    }

    #[tokio::test]
    async fn test_tool_ignoring_context_still_works() {
        let server = ToolServer::new().tool(MockAddTool);
        let handle = server.run();

        let mut ctx = crate::tool::ToolCallContext::new();
        ctx.insert(SessionId("ignored".to_string()));

        let args = serde_json::to_string(&serde_json::json!({"x": 3, "y": 7})).unwrap();
        let result = handle
            .call_tool_with_context("add", &args, ctx)
            .await
            .unwrap();

        assert_eq!(result, "10");
    }
}
