use std::sync::Arc;

use tokio::sync::RwLock;

use crate::{
    completion::{CompletionError, ToolDefinition},
    tool::{Tool, ToolContext, ToolExecution, ToolSet, ToolSetError},
    vector_store::{VectorSearchRequest, VectorStoreError, VectorStoreIndexDyn, request::Filter},
};

/// Append `name` to the advertised static-tool list unless already present.
/// Registration is last-wins on the toolset, so the name list only needs
/// first-occurrence order: a re-registered name keeps its original position
/// while the toolset swaps in the new implementation. Providers reject
/// duplicate function declarations, so the list must stay unique.
fn push_unique_name(names: &mut Vec<String>, name: String) {
    if !names.contains(&name) {
        names.push(name);
    }
}

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
        // Last-wins registration replaces the implementation but keeps the
        // original position, so the advertised list dedupes to first
        // occurrence (duplicate declarations are rejected by providers).
        self.static_tool_names = Vec::with_capacity(names.len());
        for name in names {
            push_unique_name(&mut self.static_tool_names, name);
        }
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

    /// Add a static tool to the agent. Re-registering an existing name
    /// replaces the implementation (last wins) and keeps its position.
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        let toolname = self.toolset.add_tool(tool);
        push_unique_name(&mut self.static_tool_names, toolname);
        self
    }

    /// Add a runtime-defined static tool. Re-registering an existing name
    /// replaces the implementation and keeps its position.
    pub fn dynamic_tool(mut self, tool: crate::tool::DynamicTool) -> Self {
        let toolname = self.toolset.add_dynamic_tool(tool);
        push_unique_name(&mut self.static_tool_names, toolname);
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
        let toolname = self
            .toolset
            .add_tool(McpTool::from_mcp_server(tool, client).with_timeout(timeout));
        push_unique_name(&mut self.static_tool_names, toolname);
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
    /// Register a typed static tool.
    pub async fn add_tool<T>(&self, tool: T) -> Result<(), ToolServerError>
    where
        T: Tool + 'static,
    {
        let mut state = self.0.write().await;
        let toolname = state.toolset.add_tool(tool);
        push_unique_name(&mut state.static_tool_names, toolname);
        Ok(())
    }

    /// Register a runtime-defined tool.
    pub async fn add_dynamic_tool(
        &self,
        tool: crate::tool::DynamicTool,
    ) -> Result<(), ToolServerError> {
        let mut state = self.0.write().await;
        let toolname = state.toolset.add_dynamic_tool(tool);
        push_unique_name(&mut state.static_tool_names, toolname);
        Ok(())
    }

    /// Merge an entire toolset into the server. Tool names from `toolset`
    /// are appended to the static-tool list in `toolset`'s registration
    /// order, so the tools become visible to the LLM via
    /// [`Self::get_tool_defs`]. Existing names are replaced (last wins) and
    /// keep their position.
    pub async fn append_toolset(&self, toolset: ToolSet) -> Result<(), ToolServerError> {
        let mut state = self.0.write().await;
        for name in toolset.ordered_names() {
            push_unique_name(&mut state.static_tool_names, name.clone());
        }
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

    /// Execute a tool through the canonical structured dispatch path.
    ///
    /// The registration is cloned under a short read lock; execution never
    /// blocks writers. Missing tools are represented by a classified execution
    /// error in the returned view.
    pub async fn execute(
        &self,
        tool_name: &str,
        args: &str,
        context: ToolContext,
    ) -> ToolExecution {
        let tool = {
            let state = self.0.read().await;
            state.toolset.get(tool_name).cloned()
        };
        match tool {
            Some(tool) => {
                tracing::debug!(
                    target: "rig",
                    tool_name,
                    "executing tool through structured dispatch"
                );
                tool.execute(args.to_string(), context).await
            }
            None => ToolExecution::failed(
                crate::tool::ToolExecutionError::not_found(format!(
                    "no tool named `{tool_name}` is registered"
                )),
                context,
            ),
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
                        handle.map(|handle| (doc.clone(), handle))
                    })
                    .collect()
            };

            dynamic_tool_handles
                .into_iter()
                .map(|(name, tool)| tool.definition_with_name(name))
                .collect()
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
                    handle.map(|handle| (toolname.clone(), handle))
                })
                .collect()
        };

        for (name, tool) in static_tool_handles {
            tools.push(tool.definition_with_name(name));
        }

        // One shared toolset backs both lists, so a name appearing in the
        // dynamic AND static lists (or retrieved by two indexes) refers to
        // the same tool. Keep the first definition and drop exact-name
        // repeats: providers reject duplicate function declarations.
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
    #[error("Failed to retrieve tool definitions: {0}")]
    DefinitionError(CompletionError),
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        time::Duration,
    };

    use serde_json::json;

    use crate::{
        test_utils::{
            BarrierMockToolIndex, MockAddTool, MockBarrierTool, MockControlledTool,
            MockSubtractTool, MockToolIndex,
        },
        tool::{
            DynamicTool, Tool, ToolContext, ToolEmbedding, ToolErrorKind, ToolExecutionError,
            ToolSet,
        },
    };

    use super::ToolServer;

    struct ChangingNameTool {
        calls: AtomicUsize,
    }

    impl ChangingNameTool {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl Tool for ChangingNameTool {
        const NAME: &'static str = "unused";
        type Args = serde_json::Value;
        type Output = String;

        fn name(&self) -> String {
            match self.calls.fetch_add(1, Ordering::SeqCst) {
                0 => "registered_changing".into(),
                _ => "changed_after_registration".into(),
            }
        }

        fn description(&self) -> String {
            "changes name after registration".into()
        }

        fn parameters(&self) -> serde_json::Value {
            json!({"type":"object", "properties":{}})
        }

        async fn call(
            &self,
            _context: &mut ToolContext,
            _args: Self::Args,
        ) -> Result<Self::Output, ToolExecutionError> {
            Ok("ok".into())
        }
    }

    #[derive(Debug, thiserror::Error)]
    #[error("init error")]
    struct InitError;

    impl ToolEmbedding for ChangingNameTool {
        type InitError = InitError;
        type Context = ();
        type State = ();

        fn embedding_docs(&self) -> Vec<String> {
            vec!["changing dynamic tool".into()]
        }

        fn context(&self) -> Self::Context {}

        fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
            Ok(Self::new())
        }
    }

    #[tokio::test]
    async fn registration_execution_and_removal_share_one_handle() {
        let handle = ToolServer::new().run();
        handle.add_tool(MockAddTool).await.unwrap();
        assert_eq!(handle.get_tool_defs(None).await.unwrap().len(), 1);

        let execution = handle
            .execute("add", r#"{"x":2,"y":5}"#, ToolContext::new())
            .await;
        assert_eq!(execution.model_output(), "7");
        assert!(execution.status().is_success());

        handle.remove_tool("add").await.unwrap();
        assert!(handle.get_tool_defs(None).await.unwrap().is_empty());
        let missing = handle.execute("add", "{}", ToolContext::new()).await;
        assert!(missing.status().is_error_kind(ToolErrorKind::NotFound));
    }

    #[tokio::test]
    async fn append_toolset_matches_individual_registration() {
        let individual = ToolServer::new().run();
        individual.add_tool(MockAddTool).await.unwrap();
        individual.add_tool(MockSubtractTool).await.unwrap();

        let appended = ToolServer::new().run();
        let mut set = ToolSet::default();
        set.add_tool(MockAddTool);
        set.add_tool(MockSubtractTool);
        appended.append_toolset(set).await.unwrap();

        let individual_names = individual
            .get_tool_defs(None)
            .await
            .unwrap()
            .into_iter()
            .map(|definition| definition.name)
            .collect::<Vec<_>>();
        let appended_names = appended
            .get_tool_defs(None)
            .await
            .unwrap()
            .into_iter()
            .map(|definition| definition.name)
            .collect::<Vec<_>>();
        assert_eq!(appended_names, individual_names);
    }

    #[tokio::test]
    async fn registered_key_is_source_of_truth() {
        let built = ToolServer::new().tool(ChangingNameTool::new()).run();
        assert_eq!(
            built.get_tool_defs(None).await.unwrap()[0].name,
            "registered_changing"
        );

        let added = ToolServer::new().run();
        added.add_tool(ChangingNameTool::new()).await.unwrap();
        assert_eq!(
            added.get_tool_defs(None).await.unwrap()[0].name,
            "registered_changing"
        );
    }

    #[tokio::test]
    async fn dynamic_retrieval_resolves_registered_key() {
        let set = ToolSet::builder()
            .dynamic_tool(ChangingNameTool::new())
            .build();
        let handle = ToolServer::new()
            .dynamic_tools(1, MockToolIndex::new(["registered_changing"]), set)
            .run();

        let definitions = handle
            .get_tool_defs(Some("use changing tool".into()))
            .await
            .unwrap();
        assert_eq!(definitions.len(), 1);
        assert_eq!(definitions[0].name, "registered_changing");
    }

    #[tokio::test]
    async fn definitions_preserve_order_and_dedupe_replacements() {
        let handle = ToolServer::new().run();
        handle.add_tool(MockSubtractTool).await.unwrap();
        handle.add_tool(MockAddTool).await.unwrap();
        handle.add_tool(MockSubtractTool).await.unwrap();

        let definitions = handle.get_tool_defs(None).await.unwrap();
        assert_eq!(
            definitions
                .iter()
                .map(|definition| definition.name.as_str())
                .collect::<Vec<_>>(),
            ["subtract", "add"]
        );
    }

    #[tokio::test]
    async fn dynamic_and_static_overlap_is_advertised_once() {
        let handle = ToolServer::new()
            .tool(MockAddTool)
            .dynamic_tools(1, MockToolIndex::new(["add"]), ToolSet::default())
            .run();
        let definitions = handle.get_tool_defs(Some("add".into())).await.unwrap();
        assert_eq!(definitions.len(), 1);
        assert_eq!(definitions[0].name, "add");
    }

    #[tokio::test]
    async fn missing_dynamic_implementation_is_ignored() {
        let handle = ToolServer::new()
            .tool(MockAddTool)
            .dynamic_tools(1, MockToolIndex::new(["missing"]), ToolSet::default())
            .run();
        let definitions = handle.get_tool_defs(Some("query".into())).await.unwrap();
        assert_eq!(definitions.len(), 1);
        assert_eq!(definitions[0].name, "add");
    }

    #[tokio::test]
    async fn concurrent_tool_executions_do_not_serialize() {
        let calls = 3;
        let barrier = Arc::new(tokio::sync::Barrier::new(calls));
        let handle = ToolServer::new()
            .tool(MockBarrierTool::new(Arc::clone(&barrier)))
            .run();
        let futures = (0..calls)
            .map(|_| handle.execute("barrier_tool", "{}", ToolContext::new()))
            .collect::<Vec<_>>();

        let executions =
            tokio::time::timeout(Duration::from_secs(1), futures::future::join_all(futures))
                .await
                .expect("tool calls were serialized");
        assert!(
            executions
                .iter()
                .all(|execution| execution.status().is_success())
        );
    }

    #[tokio::test]
    async fn execution_does_not_hold_read_lock() {
        let started = Arc::new(tokio::sync::Notify::new());
        let allow_finish = Arc::new(tokio::sync::Notify::new());
        let handle = ToolServer::new()
            .tool(MockControlledTool::new(
                Arc::clone(&started),
                Arc::clone(&allow_finish),
            ))
            .run();

        let executing = {
            let handle = handle.clone();
            tokio::spawn(
                async move { handle.execute("controlled", "{}", ToolContext::new()).await },
            )
        };
        started.notified().await;
        tokio::time::timeout(Duration::from_secs(1), handle.add_tool(MockAddTool))
            .await
            .expect("writer blocked behind executing tool")
            .unwrap();
        allow_finish.notify_one();
        assert_eq!(executing.await.unwrap().model_output(), "42");
    }

    #[tokio::test]
    async fn dynamic_indexes_are_queried_in_parallel() {
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let mut set = ToolSet::default();
        set.add_tool(MockAddTool);
        set.add_tool(MockSubtractTool);
        let handle = ToolServer::new()
            .dynamic_tools(
                1,
                BarrierMockToolIndex::new(Arc::clone(&barrier), "add"),
                ToolSet::default(),
            )
            .dynamic_tools(
                1,
                BarrierMockToolIndex::new(Arc::clone(&barrier), "subtract"),
                set,
            )
            .run();

        let definitions = tokio::time::timeout(
            Duration::from_secs(1),
            handle.get_tool_defs(Some("math".into())),
        )
        .await
        .expect("dynamic indexes were queried sequentially")
        .unwrap();
        assert_eq!(definitions.len(), 2);
    }

    #[tokio::test]
    async fn context_reaches_tool_and_metadata_returns() {
        #[derive(Clone)]
        struct Session(&'static str);
        #[derive(Clone, Debug, PartialEq)]
        struct RequestId(&'static str);

        struct ContextTool;
        impl Tool for ContextTool {
            const NAME: &'static str = "context";
            type Args = ();
            type Output = String;
            fn description(&self) -> String {
                "context tool".into()
            }
            fn parameters(&self) -> serde_json::Value {
                json!({"type":"null"})
            }
            async fn call(
                &self,
                context: &mut ToolContext,
                _args: Self::Args,
            ) -> Result<Self::Output, ToolExecutionError> {
                let session = context
                    .require::<Session>()
                    .map_err(|error| ToolExecutionError::permission_denied(error.to_string()))?
                    .0;
                context.insert_metadata(RequestId("request-9"));
                Ok(session.into())
            }
        }

        let handle = ToolServer::new().tool(ContextTool).run();
        let mut context = ToolContext::new();
        context.insert(Session("session-9"));
        let execution = handle.execute("context", "null", context).await;
        assert_eq!(execution.model_output(), "session-9");
        assert_eq!(
            execution.metadata::<RequestId>(),
            Some(&RequestId("request-9"))
        );
    }

    #[tokio::test]
    async fn runtime_defined_tool_registers_on_builder_and_handle() {
        fn dynamic(name: &'static str) -> DynamicTool {
            DynamicTool::new(
                name,
                "runtime",
                json!({"type":"object"}),
                |_context, _args| Box::pin(async { Ok(json!("ok")) }),
            )
        }

        let built = ToolServer::new().dynamic_tool(dynamic("built")).run();
        assert_eq!(built.get_tool_defs(None).await.unwrap()[0].name, "built");

        let added = ToolServer::new().run();
        added.add_dynamic_tool(dynamic("added")).await.unwrap();
        assert_eq!(added.get_tool_defs(None).await.unwrap()[0].name, "added");
    }
}
