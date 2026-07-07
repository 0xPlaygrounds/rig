use std::collections::BTreeSet;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::{
    completion::ToolDefinition,
    message::{AssistantContent, Message, ToolResultContent, UserContent},
    tool::{
        Tool, ToolCallExtensions, ToolDyn, ToolError, ToolExecutionResult, ToolFailure, ToolSet,
        ToolSetError,
    },
    wasm_compat::WasmBoxedFuture,
};

/// Name of the built-in meta-tool the model calls to discover deferred tools.
///
/// This name is **reserved**: user tools may not register it. The tool server
/// intercepts calls to it (advertising the built-in only when deferred tools
/// exist), so a user tool with this name could be advertised but never execute —
/// registration is refused instead (with an error on the `Result`-returning
/// [`ToolServerHandle`] methods, or a `tracing::warn!` + skip on the infallible
/// builder methods). See [`is_reserved_tool_name`].
pub const TOOL_SEARCH_NAME: &str = "tool_search";

/// Whether `name` is a reserved built-in tool name that user tools may not use.
///
/// Currently the only reserved name is [`TOOL_SEARCH_NAME`]. Registration paths
/// use this to refuse a colliding user tool rather than silently shadowing it.
pub fn is_reserved_tool_name(name: &str) -> bool {
    name == TOOL_SEARCH_NAME
}

/// Name + description of a deferred tool, handed to a [`ToolSearchFn`] so it can
/// rank candidates without the model paying for their full JSON schemas.
#[derive(Debug, Clone)]
pub struct DeferredToolMeta {
    /// The tool's name (its `tool_search` and dispatch key).
    pub name: String,
    /// The tool's human-readable description.
    pub description: String,
}

/// Strategy the built-in `tool_search` meta-tool uses to rank deferred tools for
/// a set of queries, returning the names to reveal (most relevant first).
///
/// The function is **async**, so a custom strategy can await IO — an embedding
/// similarity query against a vector DB, a BM25 index, or an LLM picker. It takes
/// owned arguments and returns a `'static` boxed future so it can outlive the
/// borrow of the tool server's state. The default ([`default_tool_search_fn`]) is
/// a dependency-free, case-insensitive keyword-overlap scorer over each tool's
/// name + description; swap it via [`ToolServer::tool_search_fn`].
pub type ToolSearchFn = Arc<
    dyn Fn(Vec<String>, Vec<DeferredToolMeta>) -> WasmBoxedFuture<'static, Vec<String>>
        + Send
        + Sync,
>;

/// The default [`ToolSearchFn`]: a keyword-overlap scorer (see
/// [`keyword_overlap_search`]) wrapped in a ready future. The scoring is
/// synchronous — the async signature exists so custom strategies can await IO.
pub fn default_tool_search_fn(
    queries: Vec<String>,
    tools: Vec<DeferredToolMeta>,
) -> WasmBoxedFuture<'static, Vec<String>> {
    Box::pin(async move { keyword_overlap_search(&queries, &tools) })
}

/// Rank deferred tools by how many query tokens appear in each tool's name +
/// description (case-insensitive), returning every tool with a non-zero score,
/// most relevant first. The synchronous core of [`default_tool_search_fn`].
pub fn keyword_overlap_search(queries: &[String], tools: &[DeferredToolMeta]) -> Vec<String> {
    let query_tokens: BTreeSet<String> = queries
        .iter()
        .flat_map(|q| {
            q.to_lowercase()
                .split_whitespace()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .collect();
    if query_tokens.is_empty() {
        return Vec::new();
    }
    let mut scored: Vec<(usize, &str)> = tools
        .iter()
        .filter_map(|tool| {
            let haystack = format!("{} {}", tool.name, tool.description).to_lowercase();
            let score = query_tokens
                .iter()
                .filter(|token| haystack.contains(token.as_str()))
                .count();
            (score > 0).then_some((score, tool.name.as_str()))
        })
        .collect();
    // Higher score first; ties keep first-registered order (stable sort).
    scored.sort_by(|a, b| b.0.cmp(&a.0));
    scored
        .into_iter()
        .map(|(_, name)| name.to_owned())
        .collect()
}

/// Arguments accepted by the built-in `tool_search` meta-tool.
#[derive(Debug, Deserialize)]
struct ToolSearchArgs {
    /// One or more capability queries to search the deferred-tool catalog for.
    queries: Vec<String>,
}

/// A single tool surfaced by a `tool_search` call.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RevealedTool {
    name: String,
    description: String,
}

/// The model-visible result envelope produced by a `tool_search` call. History
/// reconstruction ([`revealed_deferred_tools`]) parses this back out to learn
/// which deferred tools have been unlocked, so reveal state is stateless and
/// resumable (derived from the transcript, never stored on the agent).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ToolSearchResult {
    revealed_tools: Vec<RevealedTool>,
}

/// Failure modes of the built-in `tool_search` meta-tool, mapped to a classified
/// [`ToolExecutionResult`] on the structured path and to a [`ToolServerError`] on
/// the string path.
enum ToolSearchError {
    /// The `tool_search` arguments did not parse (invalid args, not "not found").
    InvalidArgs(serde_json::Error),
    /// The result envelope failed to serialize.
    Serialize(serde_json::Error),
}

/// The [`ToolDefinition`] advertised for the built-in `tool_search` meta-tool.
fn tool_search_definition() -> ToolDefinition {
    ToolDefinition {
        name: TOOL_SEARCH_NAME.to_string(),
        description: "Search for additional tools by capability when the tool you need is not \
                      already listed. Returns matching tool names and descriptions; then call a \
                      returned tool by name on a later step."
            .to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Capability queries describing the tool(s) you are looking for."
                }
            },
            "required": ["queries"]
        }),
    }
}

/// Reconstruct the set of revealed deferred-tool names from a chat transcript.
///
/// Scans for `tool_search` tool calls, then parses the [`ToolSearchResult`]
/// envelope out of the tool-result messages that answer them, unioning the
/// revealed names. Pure (history in → name set out), so it is trivially
/// testable and keeps reveal state append-only and prompt-cache friendly.
pub(crate) fn revealed_deferred_tools(history: &[Message]) -> BTreeSet<String> {
    // Ids of every tool_search call in the transcript (both `id` and `call_id`,
    // since providers correlate results by either).
    let mut search_call_ids: BTreeSet<&str> = BTreeSet::new();
    for msg in history {
        if let Message::Assistant { content, .. } = msg {
            for item in content.iter() {
                if let AssistantContent::ToolCall(call) = item
                    && call.function.name == TOOL_SEARCH_NAME
                {
                    search_call_ids.insert(call.id.as_str());
                    if let Some(cid) = &call.call_id {
                        search_call_ids.insert(cid.as_str());
                    }
                }
            }
        }
    }

    let mut revealed = BTreeSet::new();
    for msg in history {
        if let Message::User { content } = msg {
            for item in content.iter() {
                if let UserContent::ToolResult(result) = item {
                    let answers_search = search_call_ids.contains(result.id.as_str())
                        || result
                            .call_id
                            .as_deref()
                            .is_some_and(|cid| search_call_ids.contains(cid));
                    if !answers_search {
                        continue;
                    }
                    for content in result.content.iter() {
                        if let ToolResultContent::Text(text) = content
                            && let Ok(parsed) =
                                serde_json::from_str::<ToolSearchResult>(text.text())
                        {
                            for tool in parsed.revealed_tools {
                                revealed.insert(tool.name);
                            }
                        }
                    }
                }
            }
        }
    }
    revealed
}

/// Append `name` to the advertised static-tool list unless already present.
/// Registration is last-wins on the toolset, so the name list only needs
/// first-occurrence order: a re-registered name keeps its original position
/// while the toolset swaps in the new implementation. Providers reject
/// duplicate function declarations, so the list must stay unique.
fn push_unique_name(names: &mut Vec<String>, name: String) {
    // The reserved built-in name never enters an advertised name list, so a user
    // tool that collided with it (already refused + warned at the toolset level,
    // or rejected at a `Result`-returning entry point) can never be advertised.
    if is_reserved_tool_name(&name) {
        return;
    }
    if !names.contains(&name) {
        names.push(name);
    }
}

/// Shared state behind a `ToolServerHandle`.
struct ToolServerState {
    /// Static tool names that persist until explicitly removed. Always advertised.
    static_tool_names: Vec<String>,
    /// Deferred tool names: executable (registered in `toolset`) but withheld
    /// from the advertised set until the model reveals them via `tool_search`.
    deferred_tool_names: Vec<String>,
    /// The toolset where tools are registered and executed.
    toolset: ToolSet,
    /// Search strategy backing the built-in `tool_search` meta-tool.
    tool_search_fn: ToolSearchFn,
}

/// Builder for constructing a [`ToolServerHandle`].
///
/// Accumulates tools and configuration, then produces a shared handle via
/// [`run()`](ToolServer::run).
pub struct ToolServer {
    static_tool_names: Vec<String>,
    deferred_tool_names: Vec<String>,
    toolset: ToolSet,
    tool_search_fn: ToolSearchFn,
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
            deferred_tool_names: Vec::new(),
            toolset: ToolSet::default(),
            tool_search_fn: Arc::new(default_tool_search_fn),
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

    pub(crate) fn add_deferred_tool_names(mut self, names: Vec<String>) -> Self {
        for name in names {
            push_unique_name(&mut self.deferred_tool_names, name);
        }
        self
    }

    /// Add a static tool to the agent. Re-registering an existing name
    /// replaces the implementation (last wins) and keeps its position.
    ///
    /// The reserved built-in name ([`TOOL_SEARCH_NAME`]) is refused (logged and
    /// skipped) rather than shadowing the built-in `tool_search` meta-tool.
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        let toolname = tool.name();
        self.toolset.add_tool(tool);
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
        let toolname = tool.name.to_string();
        self.toolset
            .add_tool(McpTool::from_mcp_server(tool, client).with_timeout(timeout));
        push_unique_name(&mut self.static_tool_names, toolname);
        self
    }

    /// Register an executable tool that is **withheld** from the advertised tool
    /// set until the model discovers it via the built-in `tool_search` meta-tool.
    ///
    /// Deferred tools cost zero schema tokens and zero `definition()` calls until
    /// searched, so a catalog far larger than fits in a prompt can be registered
    /// and surfaced on demand. Once revealed (via a `tool_search` call recorded in
    /// the transcript), a deferred tool is advertised and dispatched like any other
    /// tool. Registering any deferred tool auto-advertises `tool_search`.
    ///
    /// The reserved built-in name ([`TOOL_SEARCH_NAME`]) is refused (logged and
    /// skipped).
    pub fn deferred_tool(mut self, tool: impl Tool + 'static) -> Self {
        let toolname = tool.name();
        self.toolset.add_tool(tool);
        push_unique_name(&mut self.deferred_tool_names, toolname);
        self
    }

    /// Override the (async) strategy the built-in `tool_search` meta-tool uses to
    /// rank deferred tools (default: keyword-overlap, see [`default_tool_search_fn`]).
    /// The function may await IO — e.g. an embedding query or an LLM picker (see
    /// [`ToolSearchFn`]).
    pub fn tool_search_fn(
        mut self,
        search: impl Fn(Vec<String>, Vec<DeferredToolMeta>) -> WasmBoxedFuture<'static, Vec<String>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        self.tool_search_fn = Arc::new(search);
        self
    }

    /// Consume the builder and return a shared [`ToolServerHandle`].
    pub fn run(self) -> ToolServerHandle {
        ToolServerHandle(Arc::new(RwLock::new(ToolServerState {
            static_tool_names: self.static_tool_names,
            deferred_tool_names: self.deferred_tool_names,
            toolset: self.toolset,
            tool_search_fn: self.tool_search_fn,
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
    ///
    /// Registering the reserved built-in name ([`TOOL_SEARCH_NAME`]) is rejected
    /// with [`ToolServerError::ReservedToolName`] rather than silently shadowing
    /// the built-in `tool_search` meta-tool.
    pub async fn add_tool(&self, tool: impl ToolDyn + 'static) -> Result<(), ToolServerError> {
        let toolname = tool.name();
        if is_reserved_tool_name(&toolname) {
            return Err(ToolServerError::ReservedToolName(toolname));
        }
        let mut state = self.0.write().await;
        push_unique_name(&mut state.static_tool_names, toolname);
        state.toolset.add_tool_boxed(Box::new(tool));
        Ok(())
    }

    /// Merge an entire toolset into the server. Tool names from `toolset`
    /// are appended to the static-tool list in `toolset`'s registration
    /// order, so the tools become visible to the LLM via
    /// [`Self::get_tool_defs`]. Existing names are replaced (last wins) and
    /// keep their position.
    ///
    /// Rejected with [`ToolServerError::ReservedToolName`] if `toolset` contains
    /// the reserved built-in name ([`TOOL_SEARCH_NAME`]).
    pub async fn append_toolset(&self, toolset: ToolSet) -> Result<(), ToolServerError> {
        for name in toolset.ordered_names() {
            if is_reserved_tool_name(name) {
                return Err(ToolServerError::ReservedToolName(name.clone()));
            }
        }
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
        // The built-in `tool_search` meta-tool is dispatched here too (not just in
        // `call_tool_structured`), so `call_tool` and this method stay consistent
        // with `get_tool_defs` advertising it. Invalid args surface as an error,
        // not a `ToolNotFoundError`.
        if tool_name == TOOL_SEARCH_NAME {
            return self.run_tool_search(args).await.map_err(|e| match e {
                ToolSearchError::InvalidArgs(e) => {
                    ToolSetError::ToolCallError(ToolError::JsonError(e)).into()
                }
                ToolSearchError::Serialize(e) => ToolSetError::JsonError(e).into(),
            });
        }

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
        // The `tool_search` meta-tool is a built-in, not a registered tool: it is
        // advertised by `get_tool_defs` when deferred tools exist and dispatched
        // here (and in `call_tool_with_extensions`). Map its error to a classified
        // outcome so hooks/policies still observe an `InvalidArgs` failure.
        if tool_name == TOOL_SEARCH_NAME {
            return match self.run_tool_search(args).await {
                Ok(json) => ToolExecutionResult::success(json),
                Err(ToolSearchError::InvalidArgs(e)) => ToolExecutionResult::failed(
                    format!("invalid tool_search arguments: {e}"),
                    ToolFailure::invalid_args(e.to_string()),
                ),
                Err(ToolSearchError::Serialize(e)) => ToolExecutionResult::failed(
                    format!("failed to serialize tool_search result: {e}"),
                    ToolFailure::other(e.to_string()),
                ),
            };
        }

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

    /// Run the built-in `tool_search` meta-tool: rank deferred tools for the
    /// requested queries (awaiting the configured [`ToolSearchFn`]) and return the
    /// matches as a [`ToolSearchResult`] JSON envelope (parsed back out of the
    /// transcript by [`revealed_deferred_tools`]). Deferred tool definitions are
    /// resolved lazily — only here, at search time. Shared by
    /// [`call_tool_with_extensions`](Self::call_tool_with_extensions) and
    /// [`call_tool_structured`](Self::call_tool_structured).
    async fn run_tool_search(&self, args: &str) -> Result<String, ToolSearchError> {
        let queries = serde_json::from_str::<ToolSearchArgs>(args)
            .map_err(ToolSearchError::InvalidArgs)?
            .queries;

        let (deferred_handles, search_fn) = {
            let state = self.0.read().await;
            let handles: Vec<(String, _)> = state
                .deferred_tool_names
                .iter()
                .filter_map(|name| state.toolset.get(name).cloned().map(|t| (name.clone(), t)))
                .collect();
            (handles, state.tool_search_fn.clone())
        };

        let mut metas = Vec::with_capacity(deferred_handles.len());
        for (name, handle) in &deferred_handles {
            let def = handle.definition(String::new()).await;
            metas.push(DeferredToolMeta {
                name: name.clone(),
                description: def.description,
            });
        }

        let revealed_tools = search_fn(queries, metas.clone())
            .await
            .into_iter()
            .filter_map(|name| {
                metas.iter().find(|m| m.name == name).map(|m| RevealedTool {
                    name: m.name.clone(),
                    description: m.description.clone(),
                })
            })
            .collect::<Vec<_>>();

        serde_json::to_string(&ToolSearchResult { revealed_tools })
            .map_err(ToolSearchError::Serialize)
    }

    /// Retrieve the tool definitions advertised to the model for a turn.
    ///
    /// The advertised set is: every static tool, the definitions of the deferred
    /// tools named in `revealed` (reconstructed from history by
    /// [`revealed_deferred_tools`]), and the built-in `tool_search` meta-tool when
    /// any deferred tool is registered. Deferred tools that have not been revealed
    /// are omitted, so their schemas never reach the model until searched.
    pub async fn get_tool_defs(
        &self,
        revealed: &BTreeSet<String>,
    ) -> Result<Vec<ToolDefinition>, ToolServerError> {
        let (has_deferred, handles) = {
            let state = self.0.read().await;
            // Static tools + revealed deferred tools; first-occurrence order.
            let mut names: Vec<String> = state.static_tool_names.clone();
            for name in &state.deferred_tool_names {
                if revealed.contains(name) {
                    push_unique_name(&mut names, name.clone());
                }
            }
            let handles: Vec<_> = names
                .iter()
                .filter_map(|name| {
                    let handle = state.toolset.get(name).cloned();
                    if handle.is_none() {
                        tracing::warn!("Tool implementation not found in toolset: {}", name);
                    }
                    handle
                })
                .collect();
            (!state.deferred_tool_names.is_empty(), handles)
        };

        let mut tools = Vec::with_capacity(handles.len() + usize::from(has_deferred));
        for handle in handles {
            tools.push(handle.definition(String::new()).await);
        }

        // Advertise the meta-tool whenever there are deferred tools to discover.
        if has_deferred {
            tools.push(tool_search_definition());
        }

        // One shared toolset backs the lists, so the same name can appear twice
        // (e.g. a re-registered tool). Keep the first definition and drop
        // exact-name repeats: providers reject duplicate function declarations.
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
    #[error(
        "`{0}` is a reserved built-in tool name (the deferred-tool `tool_search` meta-tool); \
         rename the tool"
    )]
    ReservedToolName(String),
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use std::collections::BTreeSet;

    use crate::{
        completion::{Message, ToolDefinition},
        test_utils::{MockAddTool, MockBarrierTool, MockControlledTool, MockSubtractTool},
        tool::{
            Tool, ToolCallExtensions, ToolOutcome, ToolSet,
            server::{ToolServer, ToolServerError, revealed_deferred_tools},
        },
    };

    /// A user tool that (illegally) claims the reserved built-in `tool_search`
    /// name, used to prove such a tool is refused rather than silently shadowed.
    #[derive(Clone)]
    struct ReservedNameTool;

    #[derive(serde::Deserialize)]
    struct EmptyArgs {}

    #[derive(Debug, thiserror::Error)]
    #[error("reserved name tool error")]
    struct ReservedErr;

    impl Tool for ReservedNameTool {
        const NAME: &'static str = super::TOOL_SEARCH_NAME;
        type Error = ReservedErr;
        type Args = EmptyArgs;
        type Output = String;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition {
                name: Self::NAME.to_string(),
                description: "a user tool that should never be registered".to_string(),
                parameters: serde_json::json!({ "type": "object", "properties": {} }),
            }
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok("user tool ran".to_string())
        }
    }

    #[tokio::test]
    pub async fn test_toolserver() {
        let server = ToolServer::new();

        let handle = server.run();

        handle.add_tool(MockAddTool).await.unwrap();
        let res = handle.get_tool_defs(&BTreeSet::new()).await.unwrap();

        assert_eq!(res.len(), 1);

        let json_args_as_string =
            serde_json::to_string(&serde_json::json!({"x": 2, "y": 5})).unwrap();
        let res = handle.call_tool("add", &json_args_as_string).await.unwrap();
        assert_eq!(res, "7");

        handle.remove_tool("add").await.unwrap();
        let res = handle.get_tool_defs(&BTreeSet::new()).await.unwrap();

        assert_eq!(res.len(), 0);
    }

    #[tokio::test]
    pub async fn test_toolserver_append_toolset_matches_add_tool() {
        let mut via_add_tool = {
            let handle = ToolServer::new().run();
            handle.add_tool(MockAddTool).await.unwrap();
            handle.add_tool(MockSubtractTool).await.unwrap();
            handle.get_tool_defs(&BTreeSet::new()).await.unwrap()
        };
        via_add_tool.sort_by(|a, b| a.name.cmp(&b.name));

        let mut via_append_toolset = {
            let handle = ToolServer::new().run();
            let mut toolset = ToolSet::default();
            toolset.add_tool(MockAddTool);
            toolset.add_tool(MockSubtractTool);
            handle.append_toolset(toolset).await.unwrap();
            handle.get_tool_defs(&BTreeSet::new()).await.unwrap()
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
    pub async fn deferred_tool_absent_until_searched() {
        // A deferred tool is executable but withheld from the advertised set;
        // `tool_search` is advertised in its place until the tool is revealed.
        let handle = ToolServer::new()
            .tool(MockAddTool)
            .deferred_tool(MockSubtractTool)
            .run();

        let defs = handle.get_tool_defs(&BTreeSet::new()).await.unwrap();
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"add"), "static tool advertised: {names:?}");
        assert!(
            names.contains(&"tool_search"),
            "tool_search advertised when deferred tools exist: {names:?}"
        );
        assert!(
            !names.contains(&"subtract"),
            "deferred tool withheld until searched: {names:?}"
        );
    }

    #[tokio::test]
    pub async fn tool_search_absent_without_deferred_tools() {
        let handle = ToolServer::new().tool(MockAddTool).run();
        let defs = handle.get_tool_defs(&BTreeSet::new()).await.unwrap();
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, vec!["add"], "no tool_search without deferred tools");
    }

    #[tokio::test]
    pub async fn tool_search_reveals_and_advertises_deferred_tool() {
        let handle = ToolServer::new()
            .tool(MockAddTool)
            .deferred_tool(MockSubtractTool)
            .run();

        // The model calls tool_search; the result envelope names the match.
        let result = handle
            .call_tool_structured(
                "tool_search",
                &serde_json::json!({ "queries": ["subtract"] }).to_string(),
                &ToolCallExtensions::EMPTY,
            )
            .await;
        assert!(matches!(result.outcome(), ToolOutcome::Success));
        assert!(
            result.model_output().contains("subtract"),
            "tool_search result names the match: {}",
            result.model_output()
        );

        // Reveal state reconstructed from a transcript carrying that result.
        let history: Vec<Message> = serde_json::from_value(serde_json::json!([
            {"role": "assistant", "content": [
                {"id": "call-1", "function": {"name": "tool_search", "arguments": {"queries": ["subtract"]}}}
            ]},
            {"role": "user", "content": [
                {"type": "toolresult", "id": "call-1", "content": [{"type": "text", "text": result.model_output()}]}
            ]}
        ]))
        .unwrap();
        let revealed = revealed_deferred_tools(&history);
        assert!(
            revealed.contains("subtract"),
            "reveal reconstructed from history: {revealed:?}"
        );

        // Feeding the reveal set into get_tool_defs advertises the tool, and it
        // dispatches like any other tool.
        let defs = handle.get_tool_defs(&revealed).await.unwrap();
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(
            names.contains(&"subtract"),
            "revealed tool advertised: {names:?}"
        );
        let out = handle
            .call_tool("subtract", &serde_json::json!({"x": 5, "y": 3}).to_string())
            .await
            .unwrap();
        assert_eq!(out, "2");
    }

    #[tokio::test]
    pub async fn tool_search_no_match_reveals_nothing() {
        let handle = ToolServer::new().deferred_tool(MockSubtractTool).run();
        let result = handle
            .call_tool_structured(
                "tool_search",
                &serde_json::json!({ "queries": ["totally unrelated capability"] }).to_string(),
                &ToolCallExtensions::EMPTY,
            )
            .await;
        assert!(matches!(result.outcome(), ToolOutcome::Success));
        let history: Vec<Message> = serde_json::from_value(serde_json::json!([
            {"role": "assistant", "content": [
                {"id": "c1", "function": {"name": "tool_search", "arguments": {"queries": ["x"]}}}
            ]},
            {"role": "user", "content": [
                {"type": "toolresult", "id": "c1", "content": [{"type": "text", "text": result.model_output()}]}
            ]}
        ]))
        .unwrap();
        assert!(
            revealed_deferred_tools(&history).is_empty(),
            "no matches reveals nothing"
        );
    }

    #[tokio::test]
    pub async fn revealed_deferred_tools_ignores_non_search_tool_results() {
        // A normal tool result that happens to be JSON must not be mistaken for a
        // tool_search reveal (correlation is by the tool_search call id).
        let history: Vec<Message> = serde_json::from_value(serde_json::json!([
            {"role": "assistant", "content": [
                {"id": "c1", "function": {"name": "add", "arguments": {"x": 1, "y": 2}}}
            ]},
            {"role": "user", "content": [
                {"type": "toolresult", "id": "c1", "content": [
                    {"type": "text", "text": "{\"revealed_tools\":[{\"name\":\"subtract\",\"description\":\"d\"}]}"}
                ]}
            ]}
        ]))
        .unwrap();
        assert!(
            revealed_deferred_tools(&history).is_empty(),
            "only tool_search results reveal tools"
        );
    }

    #[tokio::test]
    pub async fn tool_search_uses_custom_async_search_fn() {
        use super::{DeferredToolMeta, ToolSearchResult};
        use crate::wasm_compat::WasmBoxedFuture;

        // A custom *async* search fn: it awaits a yield (proving the async
        // signature works) then reveals every deferred tool whose name contains
        // the first query.
        let handle = ToolServer::new()
            .deferred_tool(MockAddTool)
            .deferred_tool(MockSubtractTool)
            .tool_search_fn(
                |queries: Vec<String>,
                 tools: Vec<DeferredToolMeta>|
                 -> WasmBoxedFuture<'static, Vec<String>> {
                    Box::pin(async move {
                        tokio::task::yield_now().await;
                        let needle = queries.first().cloned().unwrap_or_default();
                        tools
                            .into_iter()
                            .filter(|t| t.name.contains(&needle))
                            .map(|t| t.name)
                            .collect()
                    })
                },
            )
            .run();

        let out = handle
            .call_tool(
                "tool_search",
                &serde_json::json!({ "queries": ["sub"] }).to_string(),
            )
            .await
            .unwrap();
        let parsed: ToolSearchResult = serde_json::from_str(&out).unwrap();
        let names: Vec<&str> = parsed
            .revealed_tools
            .iter()
            .map(|t| t.name.as_str())
            .collect();
        assert_eq!(
            names,
            vec!["subtract"],
            "custom async search fn drives the reveal set"
        );
    }

    #[tokio::test]
    pub async fn call_tool_dispatches_tool_search() {
        // The public string path (`call_tool` -> `call_tool_with_extensions`)
        // dispatches the built-in meta-tool and returns the same JSON envelope the
        // structured path produces.
        let handle = ToolServer::new().deferred_tool(MockSubtractTool).run();
        let args = serde_json::json!({ "queries": ["subtract"] }).to_string();

        let via_call = handle.call_tool("tool_search", &args).await.unwrap();
        let via_structured = handle
            .call_tool_structured("tool_search", &args, &ToolCallExtensions::EMPTY)
            .await;

        assert!(matches!(via_structured.outcome(), ToolOutcome::Success));
        assert_eq!(
            via_call,
            via_structured.model_output(),
            "both dispatch paths return the same envelope"
        );
        assert!(via_call.contains("subtract"));
    }

    #[tokio::test]
    pub async fn call_tool_invalid_tool_search_args_is_error_not_not_found() {
        let handle = ToolServer::new().deferred_tool(MockSubtractTool).run();
        let err = handle
            .call_tool("tool_search", "{ not valid json }")
            .await
            .expect_err("invalid tool_search args should error");
        assert!(
            !matches!(
                err,
                super::ToolServerError::ToolsetError(crate::tool::ToolSetError::ToolNotFoundError(
                    _
                ))
            ),
            "invalid args must be an error, not tool-not-found: {err:?}"
        );
    }

    #[tokio::test]
    pub async fn call_tool_structured_invalid_tool_search_args_is_invalid_args() {
        use crate::tool::ToolFailureKind;
        let handle = ToolServer::new().deferred_tool(MockSubtractTool).run();
        let result = handle
            .call_tool_structured("tool_search", "{ not json }", &ToolCallExtensions::EMPTY)
            .await;
        assert!(
            result.outcome().is_error_kind(ToolFailureKind::InvalidArgs),
            "structured path classifies invalid tool_search args as InvalidArgs: {:?}",
            result.outcome()
        );
    }

    #[tokio::test]
    pub async fn reserved_static_tool_search_is_refused_not_shadowed() {
        // A user static tool named `tool_search` must not be registered/advertised
        // (it could never execute — the server intercepts `tool_search`).
        let handle = ToolServer::new()
            .tool(MockAddTool)
            .tool(ReservedNameTool)
            .run();
        let defs = handle.get_tool_defs(&BTreeSet::new()).await.unwrap();
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["add"],
            "the reserved `tool_search` user tool must not be advertised: {names:?}"
        );
    }

    #[tokio::test]
    pub async fn reserved_deferred_tool_search_is_refused() {
        // A deferred tool named `tool_search` is refused: it never registers, so no
        // deferred tools exist and the built-in `tool_search` is not advertised.
        let handle = ToolServer::new().deferred_tool(ReservedNameTool).run();
        let defs = handle.get_tool_defs(&BTreeSet::new()).await.unwrap();
        assert!(
            defs.is_empty(),
            "a refused deferred `tool_search` leaves no tools advertised: {:?}",
            defs.iter().map(|d| d.name.as_str()).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    pub async fn get_tool_defs_never_emits_duplicate_tool_search() {
        // A real deferred tool advertises the built-in `tool_search`; a colliding
        // user tool named `tool_search` is refused, so exactly one is emitted.
        let handle = ToolServer::new()
            .deferred_tool(MockSubtractTool)
            .tool(ReservedNameTool)
            .run();
        // Reveal the deferred tool so both it and `tool_search` are advertised.
        let revealed: BTreeSet<String> = ["subtract".to_string()].into_iter().collect();
        let defs = handle.get_tool_defs(&revealed).await.unwrap();
        let tool_search_count = defs
            .iter()
            .filter(|d| d.name == super::TOOL_SEARCH_NAME)
            .count();
        assert_eq!(
            tool_search_count,
            1,
            "exactly one tool_search definition: {:?}",
            defs.iter().map(|d| d.name.as_str()).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    pub async fn handle_add_tool_reserved_name_errors() {
        let handle = ToolServer::new().run();
        let err = handle
            .add_tool(ReservedNameTool)
            .await
            .expect_err("adding a reserved-name tool must error");
        assert!(
            matches!(&err, ToolServerError::ReservedToolName(name) if name == "tool_search"),
            "expected ReservedToolName error, got {err:?}"
        );
    }

    #[tokio::test]
    pub async fn handle_append_toolset_with_reserved_name_errors() {
        // The reserved name cannot even enter a ToolSet (insert refuses it), so we
        // assert the direct-add error path; append_toolset is guarded the same way.
        let handle = ToolServer::new().run();
        let mut toolset = ToolSet::default();
        toolset.add_tool(MockAddTool);
        // ToolSet::add_tool refuses the reserved name, so the toolset stays clean.
        toolset.add_tool(ReservedNameTool);
        assert!(
            !toolset
                .ordered_names()
                .any(|n| n == super::TOOL_SEARCH_NAME),
            "ToolSet must refuse the reserved name"
        );
        handle
            .append_toolset(toolset)
            .await
            .expect("a toolset without the reserved name appends fine");
        let names: Vec<String> = handle
            .get_tool_defs(&BTreeSet::new())
            .await
            .unwrap()
            .iter()
            .map(|d| d.name.clone())
            .collect();
        assert_eq!(names, vec!["add".to_string()]);
    }

    #[tokio::test]
    pub async fn duplicate_registration_advertises_one_definition() {
        let handle = ToolServer::new().tool(MockAddTool).run();
        handle.add_tool(MockAddTool).await.unwrap();

        let mut toolset = ToolSet::default();
        toolset.add_tool(MockAddTool);
        handle.append_toolset(toolset).await.unwrap();

        let defs = handle.get_tool_defs(&BTreeSet::new()).await.unwrap();
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
