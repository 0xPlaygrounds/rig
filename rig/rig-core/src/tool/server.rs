use std::sync::Arc;

use futures::{StreamExt, TryStreamExt, channel::oneshot::Canceled, stream};
use tokio::sync::{
    RwLock,
    mpsc::{Sender, error::SendError},
};

use crate::{
    completion::{CompletionError, ToolDefinition},
    tool::{Tool, ToolDyn, ToolError, ToolSet, ToolSetError},
    vector_store::{VectorSearchRequest, VectorStoreError, VectorStoreIndexDyn, request::Filter},
};

pub struct ToolServer {
    /// A list of static tool names.
    /// These tools will always exist on the tool server for as long as they are not deleted.
    static_tool_names: Vec<String>,
    /// Dynamic tools. These tools will be dynamically fetched from a given vector store.
    dynamic_tools: Vec<(usize, Box<dyn VectorStoreIndexDyn + Send + Sync>)>,
    /// The toolset where tools are called (to be executed).
    /// Wrapped in Arc<RwLock<...>> to allow concurrent tool execution.
    toolset: Arc<RwLock<ToolSet>>,
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
            toolset: Arc::new(RwLock::new(ToolSet::default())),
        }
    }

    pub(crate) fn static_tool_names(mut self, names: Vec<String>) -> Self {
        self.static_tool_names = names;
        self
    }

    pub(crate) fn add_tools(mut self, tools: ToolSet) -> Self {
        self.toolset = Arc::new(RwLock::new(tools));
        self
    }

    pub(crate) fn add_dynamic_tools(
        mut self,
        dyn_tools: Vec<(usize, Box<dyn VectorStoreIndexDyn + Send + Sync>)>,
    ) -> Self {
        self.dynamic_tools = dyn_tools;
        self
    }

    /// Add a static tool to the agent
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        let toolname = tool.name();
        // This should be practically impossible to fail: cloning the Arc before calling
        // .tool() is impossible since the toolset field is private, and the server cannot
        // be running prior to run(), which consumes self.
        Arc::get_mut(&mut self.toolset)
            .expect("ToolServer::tool() called after run()")
            .get_mut()
            .add_tool(tool);
        self.static_tool_names.push(toolname);
        self
    }

    // Add an MCP tool (from `rmcp`) to the agent
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    #[cfg(feature = "rmcp")]
    pub fn rmcp_tool(mut self, tool: rmcp::model::Tool, client: rmcp::service::ServerSink) -> Self {
        use crate::tool::rmcp::McpTool;
        let toolname = tool.name.clone();
        // This should be practically impossible to fail: cloning the Arc before calling
        // .rmcp_tool() is impossible since the toolset field is private, and the server cannot
        // be running prior to run(), which consumes self.
        Arc::get_mut(&mut self.toolset)
            .expect("ToolServer::rmcp_tool() called after run()")
            .get_mut()
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
        self.dynamic_tools.push((sample, Box::new(dynamic_tools)));
        // This should be practically impossible to fail: cloning the Arc before calling
        // .dynamic_tools() is impossible since the toolset field is private, and the server cannot
        // be running prior to run(), which consumes self.
        Arc::get_mut(&mut self.toolset)
            .expect("ToolServer::dynamic_tools() called after run()")
            .get_mut()
            .add_tools(toolset);
        self
    }

    pub fn run(mut self) -> ToolServerHandle {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1000);

        #[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
        tokio::spawn(async move {
            while let Some(message) = rx.recv().await {
                self.handle_message(message).await;
            }
        });

        // SAFETY: `rig` currently doesn't compile to WASM without the `worker` feature.
        // Therefore, we can safely assume that the user won't try to compile to wasm without the worker feature.
        #[cfg(all(feature = "wasm", target_arch = "wasm32"))]
        wasm_bindgen_futures::spawn_local(async move {
            while let Some(message) = rx.recv().await {
                self.handle_message(message).await;
            }
        });

        ToolServerHandle(tx)
    }

    pub async fn handle_message(&mut self, message: ToolServerRequest) {
        let ToolServerRequest {
            callback_channel,
            data,
        } = message;

        match data {
            ToolServerRequestMessageKind::AddTool(tool) => {
                self.static_tool_names.push(tool.name());
                self.toolset.write().await.add_tool_boxed(tool);
                callback_channel
                    .send(ToolServerResponse::ToolAdded)
                    .unwrap();
            }
            ToolServerRequestMessageKind::AppendToolset(tools) => {
                self.toolset.write().await.add_tools(tools);
                callback_channel
                    .send(ToolServerResponse::ToolAdded)
                    .unwrap();
            }
            ToolServerRequestMessageKind::RemoveTool { tool_name } => {
                self.static_tool_names.retain(|x| *x != tool_name);
                self.toolset.write().await.delete_tool(&tool_name);
                callback_channel
                    .send(ToolServerResponse::ToolDeleted)
                    .unwrap();
            }
            ToolServerRequestMessageKind::CallTool { name, args } => {
                let toolset = Arc::clone(&self.toolset);

                #[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
                tokio::spawn(async move {
                    match toolset.read().await.call(&name, args.clone()).await {
                        Ok(result) => {
                            let _ =
                                callback_channel.send(ToolServerResponse::ToolExecuted { result });
                        }
                        Err(err) => {
                            let _ = callback_channel.send(ToolServerResponse::ToolError {
                                error: err.to_string(),
                            });
                        }
                    }
                });

                #[cfg(all(feature = "wasm", target_arch = "wasm32"))]
                wasm_bindgen_futures::spawn_local(async move {
                    match toolset.read().await.call(&name, args.clone()).await {
                        Ok(result) => {
                            let _ =
                                callback_channel.send(ToolServerResponse::ToolExecuted { result });
                        }
                        Err(err) => {
                            let _ = callback_channel.send(ToolServerResponse::ToolError {
                                error: err.to_string(),
                            });
                        }
                    }
                });
            }
            ToolServerRequestMessageKind::GetToolDefs { prompt } => {
                let res = self.get_tool_definitions(prompt).await.unwrap();
                callback_channel
                    .send(ToolServerResponse::ToolDefinitions(res))
                    .unwrap();
            }
        }
    }

    pub async fn get_tool_definitions(
        &mut self,
        text: Option<String>,
    ) -> Result<Vec<ToolDefinition>, CompletionError> {
        let static_tool_names = self.static_tool_names.clone();
        let toolset = self.toolset.read().await;

        let mut tools = if let Some(text) = text {
            // First, collect all dynamic tool IDs from vector stores
            let dynamic_tool_ids: Vec<String> = stream::iter(self.dynamic_tools.iter())
                .then(|(num_sample, index)| async {
                    let req = VectorSearchRequest::builder()
                        .query(text.clone())
                        .samples(*num_sample as u64)
                        .build()
                        .expect("Creating VectorSearchRequest here shouldn't fail since the query and samples to return are always present");
                    Ok::<_, VectorStoreError>(
                        index
                            .as_ref()
                            .top_n_ids(req.map_filter(Filter::interpret))
                            .await?
                            .into_iter()
                            .map(|(_, id)| id)
                            .collect::<Vec<String>>(),
                    )
                })
                .try_fold(vec![], |mut acc, docs| async {
                    acc.extend(docs);
                    Ok(acc)
                })
                .await
                .map_err(|e| CompletionError::RequestError(Box::new(e)))?;

            // Then, get tool definitions for each ID
            let mut tools = Vec::new();
            for doc in dynamic_tool_ids {
                if let Some(tool) = toolset.get(&doc) {
                    tools.push(tool.definition(text.clone()).await)
                } else {
                    tracing::warn!("Tool implementation not found in toolset: {}", doc);
                }
            }
            tools
        } else {
            Vec::new()
        };

        for toolname in static_tool_names {
            if let Some(tool) = toolset.get(&toolname) {
                tools.push(tool.definition(String::new()).await)
            } else {
                tracing::warn!("Tool implementation not found in toolset: {}", toolname);
            }
        }

        Ok(tools)
    }
}

#[derive(Clone)]
pub struct ToolServerHandle(Sender<ToolServerRequest>);

impl ToolServerHandle {
    pub async fn add_tool(&self, tool: impl ToolDyn + 'static) -> Result<(), ToolServerError> {
        let tool = Box::new(tool);

        let (tx, rx) = futures::channel::oneshot::channel();

        self.0
            .send(ToolServerRequest {
                callback_channel: tx,
                data: ToolServerRequestMessageKind::AddTool(tool),
            })
            .await?;

        let res = rx.await?;

        let ToolServerResponse::ToolAdded = res else {
            return Err(ToolServerError::InvalidMessage(res));
        };

        Ok(())
    }

    pub async fn append_toolset(&self, toolset: ToolSet) -> Result<(), ToolServerError> {
        let (tx, rx) = futures::channel::oneshot::channel();

        self.0
            .send(ToolServerRequest {
                callback_channel: tx,
                data: ToolServerRequestMessageKind::AppendToolset(toolset),
            })
            .await?;

        let res = rx.await?;

        let ToolServerResponse::ToolAdded = res else {
            return Err(ToolServerError::InvalidMessage(res));
        };

        Ok(())
    }

    pub async fn remove_tool(&self, tool_name: &str) -> Result<(), ToolServerError> {
        let (tx, rx) = futures::channel::oneshot::channel();

        self.0
            .send(ToolServerRequest {
                callback_channel: tx,
                data: ToolServerRequestMessageKind::RemoveTool {
                    tool_name: tool_name.to_string(),
                },
            })
            .await?;

        let res = rx.await?;

        let ToolServerResponse::ToolDeleted = res else {
            return Err(ToolServerError::InvalidMessage(res));
        };

        Ok(())
    }

    pub async fn call_tool(&self, tool_name: &str, args: &str) -> Result<String, ToolServerError> {
        let (tx, rx) = futures::channel::oneshot::channel();

        self.0
            .send(ToolServerRequest {
                callback_channel: tx,
                data: ToolServerRequestMessageKind::CallTool {
                    name: tool_name.to_string(),
                    args: args.to_string(),
                },
            })
            .await?;

        let res = rx.await?;

        match res {
            ToolServerResponse::ToolExecuted { result, .. } => Ok(result),
            ToolServerResponse::ToolError { error } => Err(ToolServerError::ToolsetError(
                ToolSetError::ToolCallError(ToolError::ToolCallError(error.into())),
            )),
            invalid => Err(ToolServerError::InvalidMessage(invalid)),
        }
    }

    pub async fn get_tool_defs(
        &self,
        prompt: Option<String>,
    ) -> Result<Vec<ToolDefinition>, ToolServerError> {
        let (tx, rx) = futures::channel::oneshot::channel();

        self.0
            .send(ToolServerRequest {
                callback_channel: tx,
                data: ToolServerRequestMessageKind::GetToolDefs { prompt },
            })
            .await?;

        let res = rx.await?;

        let ToolServerResponse::ToolDefinitions(tooldefs) = res else {
            return Err(ToolServerError::InvalidMessage(res));
        };

        Ok(tooldefs)
    }
}

pub struct ToolServerRequest {
    callback_channel: futures::channel::oneshot::Sender<ToolServerResponse>,
    data: ToolServerRequestMessageKind,
}

pub enum ToolServerRequestMessageKind {
    AddTool(Box<dyn ToolDyn>),
    AppendToolset(ToolSet),
    RemoveTool { tool_name: String },
    CallTool { name: String, args: String },
    GetToolDefs { prompt: Option<String> },
}

#[derive(PartialEq, Debug)]
pub enum ToolServerResponse {
    ToolAdded,
    ToolDeleted,
    ToolExecuted { result: String },
    ToolError { error: String },
    ToolDefinitions(Vec<ToolDefinition>),
}

#[derive(Debug, thiserror::Error)]
pub enum ToolServerError {
    #[error("Sending message was cancelled")]
    Canceled(#[from] Canceled),
    #[error("Toolset error: {0}")]
    ToolsetError(#[from] ToolSetError),
    #[error("Error while sending message: {0}")]
    SendError(#[from] SendError<ToolServerRequest>),
    #[error("An invalid message type was returned")]
    InvalidMessage(ToolServerResponse),
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use serde::{Deserialize, Serialize};
    use serde_json::json;

    use crate::{
        completion::ToolDefinition,
        tool::{Tool, ToolSet, server::ToolServer},
        vector_store::{VectorStoreError, VectorStoreIndex, request::{Filter, VectorSearchRequest}},
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
        let server = ToolServer::new()
            .tool(Adder)
            .dynamic_tools(1, mock_index, ToolSet::from_tools(vec![Subtractor]));

        let handle = server.run();

        // Test with None prompt - should only return static tools
        let res = handle.get_tool_defs(None).await.unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].name, "add");

        // Test with Some prompt - should return both static and dynamic tools
        let res = handle.get_tool_defs(Some("calculate difference".to_string())).await.unwrap();
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
        let res = handle.get_tool_defs(Some("some query".to_string())).await.unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].name, "add");
    }

    #[derive(Debug, thiserror::Error)]
    #[error("Sleeper error")]
    struct SleeperError;

    /// A tool that sleeps for a configurable duration, used to test concurrent execution.
    #[derive(Deserialize, Serialize, Clone)]
    struct SleeperTool {
        sleep_duration_ms: u64,
    }

    impl SleeperTool {
        fn new(sleep_duration_ms: u64) -> Self {
            Self { sleep_duration_ms }
        }
    }

    impl Tool for SleeperTool {
        const NAME: &'static str = "sleeper";
        type Error = SleeperError;
        type Args = serde_json::Value;
        type Output = u64;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition {
                name: "sleeper".to_string(),
                description: "Sleeps for configured duration".to_string(),
                parameters: json!({"type": "object", "properties": {}}),
            }
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            tokio::time::sleep(Duration::from_millis(self.sleep_duration_ms)).await;
            Ok(self.sleep_duration_ms)
        }
    }

    #[tokio::test]
    pub async fn test_toolserver_concurrent_tool_execution() {
        let sleep_ms: u64 = 100;
        let num_calls: u64 = 3;

        let server = ToolServer::new().tool(SleeperTool::new(sleep_ms));
        let handle = server.run();

        let start = std::time::Instant::now();

        // Make concurrent calls
        let futures: Vec<_> = (0..num_calls)
            .map(|_| handle.call_tool("sleeper", "{}"))
            .collect();
        let results = futures::future::join_all(futures).await;

        let elapsed = start.elapsed();

        // All calls should succeed
        for result in &results {
            assert!(result.is_ok(), "Tool call failed: {:?}", result);
        }

        // If concurrent: elapsed ≈ 100ms (plus overhead)
        // If sequential: elapsed ≈ 300ms
        // Threshold: less than 2x single sleep duration means concurrent execution
        let max_concurrent_time = Duration::from_millis(sleep_ms * 2);
        assert!(
            elapsed < max_concurrent_time,
            "Expected concurrent execution in < {:?}, but took {:?}. Tools may be running sequentially.",
            max_concurrent_time,
            elapsed
        );
    }
}
