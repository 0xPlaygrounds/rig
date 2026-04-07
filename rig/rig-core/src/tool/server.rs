use std::sync::Arc;

use tokio::sync::RwLock;

use crate::{
    completion::{CompletionError, ToolDefinition},
    tool::{Tool, ToolDyn, ToolSet, ToolSetError},
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

    /// Merge an entire toolset into the server.
    pub async fn append_toolset(&self, toolset: ToolSet) -> Result<(), ToolServerError> {
        let mut state = self.0.write().await;
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
                tool.call(args.to_string())
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
            let futs: Vec<_> = dynamic_tools
                .into_iter()
                .map(|(num_sample, index)| {
                    let text = text.clone();
                    async move {
                        let req = VectorSearchRequest::builder()
                            .query(text)
                            .samples(num_sample as u64)
                            .build()
                            .expect("Creating VectorSearchRequest here shouldn't fail since the query and samples to return are always present");
                        Ok::<_, VectorStoreError>(
                            index
                                .top_n_ids(req.map_filter(Filter::interpret))
                                .await?
                                .into_iter()
                                .map(|(_, id)| id)
                                .collect::<Vec<String>>(),
                        )
                    }
                })
                .collect();

            let results = futures::future::try_join_all(futs).await.map_err(|e| {
                ToolServerError::DefinitionError(CompletionError::RequestError(Box::new(e)))
            })?;

            let dynamic_tool_ids: Vec<String> = results.into_iter().flatten().collect();

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

    use serde::{Deserialize, Serialize};
    use serde_json::json;

    use crate::{
        completion::ToolDefinition,
        tool::{Tool, ToolSet, server::ToolServer},
        vector_store::{
            VectorStoreError, VectorStoreIndex,
            request::{Filter, VectorSearchRequest},
        },
        wasm_compat::WasmCompatSend,
    };

    #[derive(Deserialize)]
    struct OperationArgs {
        x: i32,
        y: i32,
    }

    #[derive(Debug, thiserror::Error)]
    #[error("Math error")]
    struct MathError;

    #[derive(Deserialize, Serialize)]
    struct Adder;
    impl Tool for Adder {
        const NAME: &'static str = "add";
        type Error = MathError;
        type Args = OperationArgs;
        type Output = i32;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition {
                name: "add".to_string(),
                description: "Add x and y together".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "number",
                            "description": "The first number to add"
                        },
                        "y": {
                            "type": "number",
                            "description": "The second number to add"
                        }
                    },
                    "required": ["x", "y"],
                }),
            }
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            println!("[tool-call] Adding {} and {}", args.x, args.y);
            let result = args.x + args.y;
            Ok(result)
        }
    }

    #[derive(Deserialize, Serialize)]
    struct Subtractor;
    impl Tool for Subtractor {
        const NAME: &'static str = "subtract";
        type Error = MathError;
        type Args = OperationArgs;
        type Output = i32;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition {
                name: "subtract".to_string(),
                description: "Subtract y from x".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "number",
                            "description": "The number to subtract from"
                        },
                        "y": {
                            "type": "number",
                            "description": "The number to subtract"
                        }
                    },
                    "required": ["x", "y"],
                }),
            }
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            let result = args.x - args.y;
            Ok(result)
        }
    }

    /// A mock vector store index that returns a predefined list of tool IDs.
    struct MockToolIndex {
        tool_ids: Vec<String>,
    }

    impl VectorStoreIndex for MockToolIndex {
        type Filter = Filter<serde_json::Value>;

        async fn top_n<T: for<'a> Deserialize<'a> + WasmCompatSend>(
            &self,
            _req: VectorSearchRequest,
        ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
            // Not used by get_tool_definitions, but required by trait
            Ok(vec![])
        }

        async fn top_n_ids(
            &self,
            _req: VectorSearchRequest,
        ) -> Result<Vec<(f64, String)>, VectorStoreError> {
            Ok(self
                .tool_ids
                .iter()
                .enumerate()
                .map(|(i, id)| (1.0 - (i as f64 * 0.1), id.clone()))
                .collect())
        }
    }

    #[tokio::test]
    pub async fn test_toolserver() {
        let server = ToolServer::new();

        let handle = server.run();

        handle.add_tool(Adder).await.unwrap();
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
    pub async fn test_toolserver_dynamic_tools() {
        // Create a toolset with both tools
        let mut toolset = ToolSet::default();
        toolset.add_tool(Adder);
        toolset.add_tool(Subtractor);

        // Create a mock index that will return "subtract" as the dynamic tool
        let mock_index = MockToolIndex {
            tool_ids: vec!["subtract".to_string()],
        };

        // Build server with static tool "add" and dynamic tools from the mock index
        let server = ToolServer::new().tool(Adder).dynamic_tools(
            1,
            mock_index,
            ToolSet::from_tools(vec![Subtractor]),
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
        let mock_index = MockToolIndex {
            tool_ids: vec!["nonexistent_tool".to_string()],
        };

        // Build server with only static tool, but dynamic index references missing tool
        let server = ToolServer::new()
            .tool(Adder)
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

    /// A tool that waits at a barrier to test concurrency of tool execution.
    #[derive(Clone)]
    struct BarrierTool {
        barrier: Arc<tokio::sync::Barrier>,
    }

    #[derive(Debug, thiserror::Error)]
    #[error("Barrier error")]
    struct BarrierError;

    impl Tool for BarrierTool {
        const NAME: &'static str = "barrier_tool";
        type Error = BarrierError;
        type Args = serde_json::Value;
        type Output = String;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition {
                name: "barrier_tool".to_string(),
                description: "Waits at a barrier to test concurrency".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            }
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            // Wait for all concurrent invocations to reach this point
            self.barrier.wait().await;
            Ok("done".to_string())
        }
    }

    #[tokio::test]
    pub async fn test_toolserver_concurrent_tool_execution() {
        let num_calls = 3;
        let barrier = Arc::new(tokio::sync::Barrier::new(num_calls));

        let server = ToolServer::new().tool(BarrierTool {
            barrier: barrier.clone(),
        });
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

    /// A tool that can be controlled to test concurrent writes to the ToolServer.
    #[derive(Clone)]
    struct ControlledTool {
        started: Arc<tokio::sync::Notify>,
        allow_finish: Arc<tokio::sync::Notify>,
    }

    #[derive(Debug, thiserror::Error)]
    #[error("Controlled error")]
    struct ControlledError;

    impl Tool for ControlledTool {
        const NAME: &'static str = "controlled";
        type Error = ControlledError;
        type Args = serde_json::Value;
        type Output = i32;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition {
                name: "controlled".to_string(),
                description: "Test tool".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            }
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            // 1. Signal that we are inside the call (lock should be dropped by now)
            self.started.notify_one();
            // 2. Wait indefinitely until the test allows us to finish
            self.allow_finish.notified().await;
            Ok(42)
        }
    }

    #[tokio::test]
    pub async fn test_toolserver_write_while_tool_running() {
        let started = Arc::new(tokio::sync::Notify::new());
        let allow_finish = Arc::new(tokio::sync::Notify::new());

        // Build server with the controlled tool that waits at a barrier during execution
        let tool = ControlledTool {
            started: started.clone(),
            allow_finish: allow_finish.clone(),
        };

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
        let add_result = tokio::time::timeout(Duration::from_secs(1), handle.add_tool(Adder)).await;

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

    /// A mock vector store index that waits at a barrier to enforce parallel execution
    struct BarrierMockIndex {
        barrier: Arc<tokio::sync::Barrier>,
        tool_id: String,
    }

    impl VectorStoreIndex for BarrierMockIndex {
        type Filter = Filter<serde_json::Value>;

        async fn top_n<T: for<'a> Deserialize<'a> + WasmCompatSend>(
            &self,
            _req: VectorSearchRequest,
        ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
            Ok(vec![])
        }

        async fn top_n_ids(
            &self,
            _req: VectorSearchRequest,
        ) -> Result<Vec<(f64, String)>, VectorStoreError> {
            // Wait for all indices to reach this point simultaneously
            self.barrier.wait().await;
            Ok(vec![(1.0, self.tool_id.clone())])
        }
    }

    #[tokio::test]
    pub async fn test_toolserver_parallel_dynamic_tool_fetching() {
        // We expect exactly 2 parallel searches to hit the barrier at the same time
        let barrier = Arc::new(tokio::sync::Barrier::new(2));

        let index1 = BarrierMockIndex {
            barrier: barrier.clone(),
            tool_id: "add".to_string(),
        };

        let index2 = BarrierMockIndex {
            barrier: barrier.clone(),
            tool_id: "subtract".to_string(),
        };

        // Put both tools in the toolset so they resolve correctly
        let mut toolset = ToolSet::default();
        toolset.add_tool(Adder);
        toolset.add_tool(Subtractor);

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
}
