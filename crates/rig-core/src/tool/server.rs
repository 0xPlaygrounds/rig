use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use tokio::sync::RwLock;

use crate::{
    completion::{CompletionError, ProviderToolDefinition, ToolDefinition},
    runtime::RunContext,
    tool::{
        Tool, ToolCallExtensions, ToolDyn, ToolExecutionResult, ToolFailure, ToolSet, ToolSetError,
    },
    vector_store::{VectorSearchRequest, VectorStoreError, VectorStoreIndexDyn, request::Filter},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
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

/// Runtime kind of an entry in the host tool catalog.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolCatalogKind {
    /// A native Rig tool.
    Native,
    /// A tool backed by MCP.
    Mcp,
    /// A tool selected dynamically from an index.
    Dynamic,
    /// A provider-executed hosted tool.
    ProviderHosted,
}

/// Scheduling hint enforced by hosts when composing tool batches.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ToolScheduling {
    /// No catalog override; preserve the runner's explicit concurrency setting.
    #[default]
    Unspecified,
    /// Force serialized execution even when the runner allows concurrency.
    Serial,
    /// The host declares this tool safe for parallel execution.
    ParallelSafe,
}

/// Host-side provenance and metadata for one catalog entry.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolCatalogMetadata {
    /// Human-readable source or package identifier.
    pub source: Option<String>,
    /// Provenance identifier such as an MCP server URI.
    pub provenance: Option<String>,
    /// Arbitrary host-only metadata. This is never included in provider schemas.
    pub host: serde_json::Map<String, serde_json::Value>,
    /// Host scheduling declaration.
    pub scheduling: ToolScheduling,
    /// Bounded retries for failures explicitly classified retryable.
    pub max_retries: usize,
    /// Whether this tool semantically produces the final run result.
    pub final_result: bool,
}

/// Ordered, introspectable catalog entry.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCatalogEntry {
    /// Catalog name or provider-hosted kind.
    pub name: String,
    /// Runtime kind.
    pub kind: ToolCatalogKind,
    /// Model-facing function schema for executable tools.
    pub definition: Option<ToolDefinition>,
    /// Provider-hosted declaration for provider-executed tools.
    pub provider_definition: Option<ProviderToolDefinition>,
    /// Metadata retained exclusively by the host.
    pub metadata: ToolCatalogMetadata,
}

/// Restrictions for tool calls made from inside another tool.
#[derive(Clone, Debug)]
pub struct NestedToolPolicy {
    /// Maximum ancestry depth after entering the child.
    pub max_depth: usize,
    /// Optional nested target allowlist.
    pub allowlist: Option<HashSet<String>>,
    /// Whether a target already present in ancestry may be entered again.
    pub allow_recursion: bool,
}

impl Default for NestedToolPolicy {
    fn default() -> Self {
        Self {
            max_depth: 8,
            allowlist: None,
            allow_recursion: false,
        }
    }
}

/// Structured result and correlation assigned to a nested dispatch.
#[derive(Debug, Clone)]
pub struct NestedToolResult {
    /// Framework-generated child call ID.
    pub internal_call_id: String,
    /// Parent call ID inherited from the invoking tool.
    pub parent_internal_call_id: Option<String>,
    /// Structured tool result.
    pub result: ToolExecutionResult,
}

pub(crate) trait ScopedToolDispatch: WasmCompatSend + WasmCompatSync {
    #[allow(clippy::too_many_arguments)]
    fn dispatch<'a>(
        &'a self,
        server: &'a ToolServerHandle,
        extensions: ToolCallExtensions,
        context: RunContext,
        name: &'a str,
        args: &'a str,
        internal_call_id: &'a str,
        parent_internal_call_id: Option<&'a str>,
    ) -> WasmBoxedFuture<'a, ToolExecutionResult>;
}

/// Cloneable executor automatically supplied to tools for scoped nested calls.
#[derive(Clone)]
pub struct ScopedToolExecutor {
    server: ToolServerHandle,
    extensions: ToolCallExtensions,
    context: RunContext,
    policy: NestedToolPolicy,
    dispatch: Option<Arc<dyn ScopedToolDispatch>>,
}

impl ScopedToolExecutor {
    pub(crate) fn new(
        server: ToolServerHandle,
        extensions: ToolCallExtensions,
        context: RunContext,
        policy: NestedToolPolicy,
        dispatch: Option<Arc<dyn ScopedToolDispatch>>,
    ) -> Self {
        Self {
            server,
            extensions,
            context,
            policy,
            dispatch,
        }
    }

    /// Dispatch a nested tool with inherited extensions and cancellation.
    pub async fn call(&self, name: &str, args: &str) -> NestedToolResult {
        let child_id = crate::id::generate();
        let parent_id = self.context.current_call_id().map(str::to_owned);
        let rejected = if self.context.should_stop() {
            Some(ToolFailure::cancelled("parent run cancelled"))
        } else if self.context.ancestry().len() + 1 > self.policy.max_depth {
            Some(ToolFailure::permission_denied(
                "nested depth limit exceeded",
            ))
        } else if self
            .policy
            .allowlist
            .as_ref()
            .is_some_and(|set| !set.contains(name))
        {
            Some(ToolFailure::permission_denied(format!(
                "nested tool `{name}` is not allowed"
            )))
        } else if !self.policy.allow_recursion
            && self
                .context
                .ancestry()
                .iter()
                .any(|ancestor| ancestor == name)
        {
            Some(ToolFailure::permission_denied(format!(
                "recursive nested tool `{name}` rejected"
            )))
        } else {
            None
        };
        if let Some(failure) = rejected {
            return NestedToolResult {
                internal_call_id: child_id,
                parent_internal_call_id: parent_id,
                result: ToolExecutionResult::failed(failure.message.clone(), failure),
            };
        }
        let child = self.context.child(name.to_owned(), child_id.clone());
        let mut extensions = self.extensions.clone();
        extensions.insert(child.clone());
        extensions.insert(Self::new(
            self.server.clone(),
            extensions.clone(),
            child.clone(),
            self.policy.clone(),
            self.dispatch.clone(),
        ));
        let call = async {
            if let Some(dispatch) = &self.dispatch {
                dispatch
                    .dispatch(
                        &self.server,
                        extensions,
                        child.clone(),
                        name,
                        args,
                        &child_id,
                        parent_id.as_deref(),
                    )
                    .await
            } else {
                self.server
                    .call_tool_structured(name, args, &extensions)
                    .await
            }
        };
        let stopped = child.stopped();
        futures::pin_mut!(call, stopped);
        let result = match futures::future::select(call, stopped).await {
            futures::future::Either::Left((result, _)) => result,
            futures::future::Either::Right((reason, _)) => ToolExecutionResult::failed(
                format!("nested tool stopped: {reason:?}"),
                ToolFailure::cancelled(format!("nested tool stopped: {reason:?}")),
            ),
        };
        NestedToolResult {
            internal_call_id: child_id.clone(),
            parent_internal_call_id: parent_id.clone(),
            result,
        }
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
    metadata: HashMap<String, (ToolCatalogKind, ToolCatalogMetadata)>,
    provider_tools: Vec<(ProviderToolDefinition, ToolCatalogMetadata)>,
}

/// Builder for constructing a [`ToolServerHandle`].
///
/// Accumulates tools and configuration, then produces a shared handle via
/// [`run()`](ToolServer::run).
pub struct ToolServer {
    static_tool_names: Vec<String>,
    dynamic_tools: Vec<(usize, Arc<dyn VectorStoreIndexDyn + Send + Sync>)>,
    toolset: ToolSet,
    metadata: HashMap<String, (ToolCatalogKind, ToolCatalogMetadata)>,
    provider_tools: Vec<(ProviderToolDefinition, ToolCatalogMetadata)>,
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
            metadata: HashMap::new(),
            provider_tools: Vec::new(),
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

    pub(crate) fn catalog_kinds(mut self, kinds: HashMap<String, ToolCatalogKind>) -> Self {
        for (name, kind) in kinds {
            self.metadata
                .insert(name, (kind, ToolCatalogMetadata::default()));
        }
        self
    }

    pub(crate) fn provider_tools(mut self, tools: Vec<ProviderToolDefinition>) -> Self {
        self.provider_tools = tools
            .into_iter()
            .map(|tool| (tool, ToolCatalogMetadata::default()))
            .collect();
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
        push_unique_name(&mut self.static_tool_names, toolname.clone());
        self.metadata
            .entry(toolname)
            .or_insert_with(|| (ToolCatalogKind::Native, ToolCatalogMetadata::default()));
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
        push_unique_name(&mut self.static_tool_names, toolname.clone());
        self.metadata.insert(
            toolname,
            (ToolCatalogKind::Mcp, ToolCatalogMetadata::default()),
        );
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
        for name in toolset.ordered_names() {
            self.metadata
                .entry(name.clone())
                .or_insert_with(|| (ToolCatalogKind::Dynamic, ToolCatalogMetadata::default()));
        }
        self.toolset.add_tools(toolset);
        self
    }

    /// Consume the builder and return a shared [`ToolServerHandle`].
    pub fn run(self) -> ToolServerHandle {
        let mut metadata = self.metadata;
        for name in &self.static_tool_names {
            metadata
                .entry(name.clone())
                .or_insert_with(|| (ToolCatalogKind::Native, ToolCatalogMetadata::default()));
        }
        ToolServerHandle(Arc::new(RwLock::new(ToolServerState {
            static_tool_names: self.static_tool_names,
            dynamic_tools: self.dynamic_tools,
            toolset: self.toolset,
            metadata,
            provider_tools: self.provider_tools,
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
        let kind = tool.catalog_kind();
        let toolname = state.toolset.add_tool_boxed(Box::new(tool));
        push_unique_name(&mut state.static_tool_names, toolname.clone());
        state
            .metadata
            .insert(toolname, (kind, ToolCatalogMetadata::default()));
        Ok(())
    }

    /// Register or replace a native tool with host-only catalog metadata.
    pub async fn add_tool_with_metadata(
        &self,
        tool: impl ToolDyn + 'static,
        metadata: ToolCatalogMetadata,
    ) -> Result<(), ToolServerError> {
        let mut state = self.0.write().await;
        let kind = tool.catalog_kind();
        let name = state.toolset.add_tool_boxed(Box::new(tool));
        push_unique_name(&mut state.static_tool_names, name.clone());
        state.metadata.insert(name, (kind, metadata));
        Ok(())
    }

    /// Register a provider-hosted tool for introspection. It is never available
    /// to [`call_tool_structured`](Self::call_tool_structured).
    pub async fn add_provider_tool(
        &self,
        tool: ProviderToolDefinition,
        metadata: ToolCatalogMetadata,
    ) {
        let mut state = self.0.write().await;
        if let Some(existing) = state
            .provider_tools
            .iter_mut()
            .find(|(registered, _)| registered.kind == tool.kind)
        {
            *existing = (tool, metadata);
        } else {
            state.provider_tools.push((tool, metadata));
        }
    }

    /// Return an ordered snapshot of executable and provider-hosted entries.
    pub async fn catalog(&self) -> Vec<ToolCatalogEntry> {
        let state = self.0.read().await;
        let mut entries = state
            .toolset
            .ordered_names()
            .filter_map(|name| {
                let tool = state.toolset.get(name)?;
                let (kind, metadata) = state
                    .metadata
                    .get(name)
                    .cloned()
                    .unwrap_or((ToolCatalogKind::Dynamic, ToolCatalogMetadata::default()));
                Some(ToolCatalogEntry {
                    name: name.clone(),
                    kind,
                    definition: Some(tool.definition_with_name(name.clone())),
                    provider_definition: None,
                    metadata,
                })
            })
            .collect::<Vec<_>>();
        entries.extend(
            state
                .provider_tools
                .iter()
                .map(|(tool, metadata)| ToolCatalogEntry {
                    name: tool.kind.clone(),
                    kind: ToolCatalogKind::ProviderHosted,
                    definition: None,
                    provider_definition: Some(tool.clone()),
                    metadata: metadata.clone(),
                }),
        );
        entries
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
            let kind = toolset
                .get(name)
                .map_or(ToolCatalogKind::Native, |tool| tool.catalog_kind());
            state
                .metadata
                .entry(name.clone())
                .or_insert_with(|| (kind, ToolCatalogMetadata::default()));
        }
        state.toolset.add_tools(toolset);
        Ok(())
    }

    /// Remove a tool by name from both the toolset and the static list.
    pub async fn remove_tool(&self, tool_name: &str) -> Result<(), ToolServerError> {
        let mut state = self.0.write().await;
        state.static_tool_names.retain(|x| *x != tool_name);
        state.toolset.delete_tool(tool_name);
        state.metadata.remove(tool_name);
        state
            .provider_tools
            .retain(|(tool, _)| tool.kind != tool_name);
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
        let (tool, max_retries) = {
            let state = self.0.read().await;
            (
                state.toolset.get(tool_name).cloned(),
                state
                    .metadata
                    .get(tool_name)
                    .map_or(0, |(_, metadata)| metadata.max_retries),
            )
        };

        match tool {
            Some(tool) => {
                tracing::debug!(target: "rig",
                    "Calling tool {tool_name} with args:\n{}",
                    serde_json::to_string_pretty(&args).unwrap_or_default()
                );
                let mut attempt = 0;
                loop {
                    let result = tool.call_structured(args.to_string(), extensions).await;
                    let retryable = result
                        .outcome()
                        .failure()
                        .and_then(|failure| failure.retryable)
                        == Some(true);
                    if !retryable || attempt >= max_retries {
                        break result;
                    }
                    attempt += 1;
                }
            }
            None => ToolExecutionResult::failed(
                format!("tool `{tool_name}` not found"),
                ToolFailure::not_found(format!("no tool named `{tool_name}` is registered")),
            ),
        }
    }

    pub(crate) async fn scheduling(&self, name: &str) -> ToolScheduling {
        self.0
            .read()
            .await
            .metadata
            .get(name)
            .map_or(ToolScheduling::Unspecified, |(_, metadata)| {
                metadata.scheduling.clone()
            })
    }

    pub(crate) async fn is_final_result(&self, name: &str) -> bool {
        self.0
            .read()
            .await
            .metadata
            .get(name)
            .is_some_and(|(_, metadata)| metadata.final_result)
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

    use crate::{
        test_utils::{
            BarrierMockToolIndex, MockAddTool, MockBarrierTool, MockControlledTool,
            MockSubtractTool, MockToolError, MockToolIndex,
        },
        tool::{Tool, ToolEmbedding, ToolSet, server::ToolServer},
    };

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
        type Error = MockToolError;
        type Args = serde_json::Value;
        type Output = String;

        fn name(&self) -> String {
            match self.calls.fetch_add(1, Ordering::SeqCst) {
                0 => "registered_changing".to_string(),
                _ => "changed_after_registration".to_string(),
            }
        }

        fn description(&self) -> String {
            "changes name after registration".to_string()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok("ok".to_string())
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
            vec!["changing dynamic tool".to_string()]
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
    pub async fn builder_tool_uses_registered_key_for_static_names() {
        let handle = ToolServer::new().tool(ChangingNameTool::new()).run();

        let defs = handle.get_tool_defs(None).await.unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "registered_changing");
    }

    #[tokio::test]
    pub async fn handle_add_tool_uses_registered_key_for_static_names() {
        let handle = ToolServer::new().run();
        handle.add_tool(ChangingNameTool::new()).await.unwrap();

        let defs = handle.get_tool_defs(None).await.unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "registered_changing");
    }

    #[tokio::test]
    pub async fn dynamic_retrieval_resolves_registered_key() {
        let toolset = ToolSet::builder()
            .dynamic_tool(ChangingNameTool::new())
            .build();
        let handle = ToolServer::new()
            .dynamic_tools(1, MockToolIndex::new(["registered_changing"]), toolset)
            .run();

        let defs = handle
            .get_tool_defs(Some("use the changing tool".to_string()))
            .await
            .unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "registered_changing");
    }

    #[tokio::test]
    pub async fn get_tool_defs_preserves_static_registration_order() {
        let handle = ToolServer::new().run();
        handle.add_tool(MockSubtractTool).await.unwrap();
        handle.add_tool(MockAddTool).await.unwrap();

        let defs = handle.get_tool_defs(None).await.unwrap();
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
            .dynamic_tools(1, MockToolIndex::new(["add"]), ToolSet::default())
            .run();

        let defs = handle
            .get_tool_defs(Some("add two numbers".to_string()))
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
    pub async fn duplicate_registration_advertises_one_definition() {
        let handle = ToolServer::new().tool(MockAddTool).run();
        handle.add_tool(MockAddTool).await.unwrap();

        let mut toolset = ToolSet::default();
        toolset.add_tool(MockAddTool);
        handle.append_toolset(toolset).await.unwrap();

        let defs = handle.get_tool_defs(None).await.unwrap();
        assert_eq!(
            defs.len(),
            1,
            "re-registering a name must not advertise duplicate declarations"
        );
        assert_eq!(defs[0].name, "add");
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

        fn description(&self) -> String {
            "Reads SessionId from context".to_string()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
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
