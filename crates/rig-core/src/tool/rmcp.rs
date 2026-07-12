//! MCP (Model Context Protocol) integration via the `rmcp` crate.
//!
//! This module provides:
//! - [`McpTool`]: A wrapper that adapts an `rmcp` tool for use in Rig's tool system.
//! - [`McpClientHandler`]: A client handler that reacts to `notifications/tools/list_changed`
//!   by re-fetching the tool list and updating the [`ToolServer`](super::server::ToolServer).
//!
//! # Example
//!
//! ```rust,ignore
//! use rig_core::tool::rmcp::McpClientHandler;
//! use rig_core::tool::server::ToolServer;
//! use rmcp::ServiceExt;
//!
//! // 1. Create a ToolServer and get a handle
//! let tool_server_handle = ToolServer::new().run();
//!
//! // 2. Create a handler that auto-updates tools on list changes
//! let handler = McpClientHandler::new(client_info, tool_server_handle.clone());
//!
//! // 3. Connect to the MCP server and register initial tools
//! let mcp_service = handler.connect(transport).await?;
//!
//! // 4. Build an agent using the shared tool server handle
//! let agent = openai_client
//!     .agent(openai::GPT_5_2)
//!     .preamble("You are a helpful assistant.")
//!     .tool_server_handle(tool_server_handle)
//!     .build();
//! ```
//!
//! # Per-call metadata
//!
//! [`McpTool`] forwards an [`rmcp::model::Meta`] (re-exported here as [`Meta`])
//! placed in a [`ToolCallExtensions`] as the MCP request's `_meta` (SEP-1319) —
//! the idiomatic channel for per-call values such as auth tokens, session ids,
//! or A2A `context_id`/`task_id`, which the model never sees:
//!
//! ```rust,ignore
//! use rig_core::tool::rmcp::Meta;
//! use rig_core::tool::ToolCallExtensions;
//!
//! let mut meta = Meta::new();
//! meta.0.insert("authorization".into(), serde_json::json!("Bearer …"));
//! let mut extensions = ToolCallExtensions::new();
//! extensions.insert(meta);
//! let answer = agent.prompt("…").tool_extensions(extensions).await?;
//! ```

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Duration;

use rmcp::ServiceExt;
use rmcp::model::ContentBlock;
use tokio::sync::RwLock;

use crate::completion::ToolDefinition;
use crate::tool::server::{ToolServerError, ToolServerHandle};
use crate::tool::{
    ToolCallExtensions, ToolDyn, ToolError, ToolExecutionResult, ToolFailure, ToolFailureKind,
};
use crate::wasm_compat::WasmBoxedFuture;

/// Re-export of [`rmcp::model::Meta`]: place one in a [`ToolCallExtensions`] to have
/// [`McpTool`] forward it as a call's MCP `_meta` (see the module docs).
pub use rmcp::model::Meta;

/// Default per-call timeout applied to MCP tools (see issue #1914).
///
/// MCP tool calls await a response that can be silently lost by the transport
/// (e.g. an rmcp StreamableHttp session re-init dropping an in-flight request),
/// which would otherwise hang the agent forever. A generous default bounds that
/// without disrupting normal, long-running tools. Override per tool with
/// [`McpTool::with_timeout`] (pass `None` to disable, e.g. for tools that may
/// legitimately run longer than this).
pub const DEFAULT_MCP_TOOL_TIMEOUT: Duration = Duration::from_secs(300);

/// A Rig tool adapter wrapping an `rmcp` MCP tool.
///
/// Bridges between the MCP tool protocol and Rig's [`ToolDyn`] trait,
/// allowing MCP tools to be used seamlessly in Rig agents.
#[derive(Clone)]
pub struct McpTool {
    definition: rmcp::model::Tool,
    client: rmcp::service::ServerSink,
    /// Per-call timeout. When `Some`, an MCP `call_tool` that does not complete
    /// within this duration resolves to a [`ToolError`] instead of blocking
    /// forever (see issue #1914). When `None`, the call is unbounded.
    ///
    /// On elapse the call is abandoned **locally** (the future is dropped); the
    /// server is not sent a cancellation, so a still-running tool keeps running
    /// server-side, and rmcp reclaims the request slot when the session closes.
    timeout: Option<Duration>,
}

impl McpTool {
    /// Create a new `McpTool` from an MCP tool definition and server sink.
    ///
    /// Applies [`DEFAULT_MCP_TOOL_TIMEOUT`] so a lost/never-answered response
    /// cannot hang the agent forever (issue #1914). Use [`McpTool::with_timeout`]
    /// to change it, or `with_timeout(None)` to disable it for tools that may
    /// legitimately run longer.
    pub fn from_mcp_server(
        definition: rmcp::model::Tool,
        client: rmcp::service::ServerSink,
    ) -> Self {
        Self {
            definition,
            client,
            timeout: Some(DEFAULT_MCP_TOOL_TIMEOUT),
        }
    }

    /// Set (or clear) the per-call timeout, consuming and returning the tool.
    ///
    /// Pass a [`Duration`] to bound calls, or `None` to make them unbounded.
    /// On timeout the call resolves to a [`ToolError`] (which the agent loop
    /// surfaces to the model as a tool result, so the agent can recover rather
    /// than hang). Note the timeout abandons the call locally and does **not**
    /// send a cancellation to the MCP server — see the [`McpTool::timeout`]
    /// field docs.
    pub fn with_timeout(mut self, timeout: impl Into<Option<Duration>>) -> Self {
        self.timeout = timeout.into();
        self
    }

    /// The per-call timeout, if any.
    pub fn timeout(&self) -> Option<Duration> {
        self.timeout
    }
}

impl From<&rmcp::model::Tool> for ToolDefinition {
    fn from(val: &rmcp::model::Tool) -> Self {
        let mut definition = Self::new(
            val.name.to_string(),
            val.description.clone().unwrap_or(Cow::from("")).to_string(),
            val.schema_as_json_value(),
        );
        definition.output_schema = val
            .output_schema
            .as_ref()
            .map(|schema| serde_json::Value::Object((**schema).clone()));
        definition.metadata.kind = crate::completion::ToolKind::Mcp;
        definition
    }
}

impl From<rmcp::model::Tool> for ToolDefinition {
    fn from(val: rmcp::model::Tool) -> Self {
        Self::from(&val)
    }
}

/// Error returned by an [`McpTool`] call.
///
/// Carries a structured [`ToolFailureKind`] so an MCP timeout, transport
/// failure, or tool-reported error reaches hooks and telemetry as a classified
/// [`ToolFailure`] (via [`McpTool`]'s
/// [`ToolDyn::call_structured`]) instead
/// of an opaque string.
#[derive(Debug, thiserror::Error)]
#[error("MCP tool error: {message}")]
pub struct McpToolError {
    message: String,
    kind: ToolFailureKind,
}

impl McpToolError {
    fn new(kind: ToolFailureKind, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind,
        }
    }

    /// The structured classification of this MCP error.
    pub fn kind(&self) -> ToolFailureKind {
        self.kind
    }

    /// Convert into a structured [`ToolFailure`] carrying the kind's default
    /// `retryable` hint (e.g. an MCP timeout is retryable).
    fn into_failure(self) -> ToolFailure {
        ToolFailure::of_kind(self.kind, self.message)
    }
}

impl From<McpToolError> for ToolError {
    fn from(e: McpToolError) -> Self {
        ToolError::ToolCallError(Box::new(e))
    }
}

/// Parse the JSON `args` string into MCP call arguments.
///
/// Returns `Ok(None)` for empty input or valid-but-non-object JSON (`null`, an
/// array, a scalar) — all of which carry no MCP arguments — and `Ok(Some(obj))`
/// for a JSON object. **Malformed JSON is a hard error** (`Err`): LLMs
/// occasionally emit invalid JSON, and it must surface as a
/// [`ToolFailureKind::InvalidArgs`] failure rather than being silently coerced
/// into a no-argument call that reaches the server.
fn parse_mcp_arguments(args: &str) -> Result<Option<rmcp::model::JsonObject>, serde_json::Error> {
    let trimmed = args.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let value: serde_json::Value = serde_json::from_str(trimmed)?;
    match value {
        serde_json::Value::Object(_) => Ok(Some(serde_json::from_value(value)?)),
        _ => Ok(None),
    }
}

impl McpTool {
    /// Shared executor for [`ToolDyn::call`] and [`ToolDyn::call_with_extensions`].
    ///
    /// `meta`, when present, is attached as the MCP request's `_meta`
    /// (SEP-1319) — the idiomatic channel for per-call metadata such as auth
    /// tokens, session ids, or A2A `context_id`/`task_id`. It is supplied by a
    /// caller that places an [`rmcp::model::Meta`] into the
    /// [`ToolCallExtensions`]; otherwise the call behaves exactly as before.
    fn execute(
        &self,
        args: String,
        meta: Option<rmcp::model::Meta>,
    ) -> WasmBoxedFuture<'_, Result<String, McpToolError>> {
        let name = self.definition.name.clone();

        Box::pin(async move {
            // Validate the JSON arguments before contacting the server: malformed
            // JSON must surface as an InvalidArgs failure, not a silent no-arg call.
            let arguments = parse_mcp_arguments(&args).map_err(|err| {
                McpToolError::new(
                    ToolFailureKind::InvalidArgs,
                    format!("MCP tool '{name}' received invalid JSON arguments: {err}"),
                )
            })?;
            let mut request = arguments
                .map(|arguments| {
                    rmcp::model::CallToolRequestParams::new(name.clone()).with_arguments(arguments)
                })
                .unwrap_or_else(|| rmcp::model::CallToolRequestParams::new(name));
            request.meta = meta;

            let call = self.client.call_tool(request);
            // Bound the call so a never-answered request (see issue #1914)
            // becomes a recoverable error instead of an unbounded await.
            let call_result = match self.timeout {
                Some(timeout) => {
                    crate::wasm_compat::timeout(timeout, call)
                        .await
                        .map_err(|_| {
                            McpToolError::new(
                                ToolFailureKind::Timeout,
                                format!(
                                    "MCP tool '{}' timed out after {timeout:?}",
                                    self.definition.name
                                ),
                            )
                        })?
                }
                None => call.await,
            };
            // A transport/service error before the tool produced a result.
            let result = call_result.map_err(|e| {
                McpToolError::new(
                    ToolFailureKind::Provider,
                    format!("Tool returned an error: {e}"),
                )
            })?;

            if let Some(true) = result.is_error {
                let error_msg = result
                    .content
                    .into_iter()
                    .map(|x| x.as_text().map(|y| y.text.clone()))
                    .collect::<Option<Vec<String>>>();

                // The MCP tool ran and reported its own error result — a handled
                // tool failure rather than a transport/timeout condition.
                let error_message = error_msg.map(|x| x.join("\n"));
                if let Some(error_message) = error_message {
                    return Err(McpToolError::new(ToolFailureKind::Other, error_message));
                } else {
                    return Err(McpToolError::new(
                        ToolFailureKind::Other,
                        "No message returned".to_string(),
                    ));
                }
            };

            let mut content = String::new();

            for item in result.content {
                let chunk = match item {
                    ContentBlock::Text(raw) => raw.text,
                    ContentBlock::Image(raw) => {
                        format!("data:{};base64,{}", raw.mime_type, raw.data)
                    }
                    ContentBlock::Resource(raw) => match raw.resource {
                        rmcp::model::ResourceContents::TextResourceContents {
                            uri,
                            mime_type,
                            text,
                            ..
                        } => {
                            format!(
                                "{mime_type}{uri}:{text}",
                                mime_type =
                                    mime_type.map(|m| format!("data:{m};")).unwrap_or_default(),
                            )
                        }
                        rmcp::model::ResourceContents::BlobResourceContents {
                            uri,
                            mime_type,
                            blob,
                            ..
                        } => format!(
                            "{mime_type}{uri}:{blob}",
                            mime_type = mime_type.map(|m| format!("data:{m};")).unwrap_or_default(),
                        ),
                        thing => {
                            return Err(McpToolError::new(
                                ToolFailureKind::Other,
                                format!(
                                    "MCP tool returned unsupported resource contents: {thing:?}"
                                ),
                            ));
                        }
                    },
                    ContentBlock::Audio(_) => {
                        return Err(McpToolError::new(
                            ToolFailureKind::Other,
                            "MCP tool returned audio content, which Rig does not support yet"
                                .to_string(),
                        ));
                    }
                    thing => {
                        return Err(McpToolError::new(
                            ToolFailureKind::Other,
                            format!("MCP tool returned unsupported content: {thing:?}"),
                        ));
                    }
                };

                content.push_str(&chunk);
            }

            Ok(content)
        })
    }
}

impl ToolDyn for McpTool {
    fn name(&self) -> String {
        self.definition.name.to_string()
    }

    fn description(&self) -> String {
        self.definition
            .description
            .clone()
            .unwrap_or(Cow::from(""))
            .to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        self.definition.schema_as_json_value()
    }

    fn output_schema(&self) -> Option<serde_json::Value> {
        self.definition
            .output_schema
            .as_ref()
            .map(|schema| serde_json::Value::Object((**schema).clone()))
    }

    fn metadata(&self) -> crate::completion::ToolMetadata {
        crate::completion::ToolMetadata {
            kind: crate::completion::ToolKind::Mcp,
            execution: crate::completion::ToolExecutionPolicy::ParallelSafe,
            source: None,
            attributes: Default::default(),
        }
    }

    fn call(&self, args: String) -> WasmBoxedFuture<'_, Result<String, ToolError>> {
        Box::pin(async move { self.execute(args, None).await.map_err(ToolError::from) })
    }

    /// Forwards an [`rmcp::model::Meta`] from the [`ToolCallExtensions`], if present,
    /// as the MCP request's `_meta`. This lets callers attach per-call metadata
    /// (auth, session, A2A `context_id`/`task_id`) to MCP tool invocations
    /// without exposing it to the model. Absent a `Meta`, behaves like
    /// [`call`](ToolDyn::call).
    fn call_with_extensions<'a>(
        &'a self,
        args: String,
        extensions: &'a ToolCallExtensions,
    ) -> WasmBoxedFuture<'a, Result<String, ToolError>> {
        let meta = extensions.get::<rmcp::model::Meta>().cloned();
        Box::pin(async move { self.execute(args, meta).await.map_err(ToolError::from) })
    }

    /// Surfaces MCP failures as structured outcomes: a per-call timeout (issue
    /// #1914) becomes a [`Timeout`](ToolFailureKind::Timeout) failure, a
    /// transport error a [`Provider`](ToolFailureKind::Provider) failure, and a
    /// tool-reported error result an [`Other`](ToolFailureKind::Other) failure —
    /// all with the MCP error text as the model-visible output.
    fn call_structured<'a>(
        &'a self,
        args: String,
        extensions: &'a ToolCallExtensions,
    ) -> WasmBoxedFuture<'a, ToolExecutionResult> {
        let meta = extensions.get::<rmcp::model::Meta>().cloned();
        Box::pin(async move {
            match self.execute(args, meta).await {
                Ok(output) => ToolExecutionResult::success(output),
                Err(err) => {
                    let failure = err.into_failure();
                    ToolExecutionResult::failed(failure.message.clone(), failure)
                }
            }
        })
    }
}

/// Error type for [`McpClientHandler`] operations.
#[derive(Debug, thiserror::Error)]
pub enum McpClientError {
    /// Failed to establish the MCP connection or complete the handshake.
    #[error("MCP connection error: {0}")]
    ConnectionError(String),

    /// Failed to fetch the tool list from the MCP server.
    #[error("Failed to fetch MCP tool list: {0}")]
    ToolFetchError(#[from] rmcp::ServiceError),

    /// Failed to update the tool server with new tools.
    #[error("Tool server error: {0}")]
    ToolServerError(#[from] ToolServerError),
}

/// An MCP client handler that automatically re-fetches the tool list when the
/// server sends a `notifications/tools/list_changed` notification.
///
/// This handler implements [`rmcp::ClientHandler`] and bridges the MCP
/// notification lifecycle with Rig's [`ToolServer`](super::server::ToolServer).
/// When the MCP server's available tools change, this handler:
/// 1. Removes previously registered MCP tools from the tool server
/// 2. Re-fetches the full tool list from the MCP server
/// 3. Registers the updated tools with the tool server
///
/// # Usage
///
/// Use [`McpClientHandler::connect`] for a streamlined setup that handles
/// connection, initial tool fetch, and registration in one call:
///
/// ```rust,ignore
/// let tool_server_handle = ToolServer::new().run();
/// let handler = McpClientHandler::new(client_info, tool_server_handle.clone());
/// let mcp_service = handler.connect(transport).await?;
/// ```
///
/// The returned `RunningService` keeps the MCP connection alive. When the
/// server updates its tools, the handler automatically syncs with the tool server.
pub struct McpClientHandler {
    client_info: rmcp::model::ClientInfo,
    tool_server_handle: ToolServerHandle,
    /// Per-call timeout applied to every MCP tool this handler registers
    /// (see issue #1914). Defaults to [`DEFAULT_MCP_TOOL_TIMEOUT`].
    timeout: Option<Duration>,
    /// Tracks which tool names were registered by this handler so they
    /// can be removed and replaced on list-change notifications.
    managed_tool_names: Arc<RwLock<Vec<String>>>,
}

impl McpClientHandler {
    /// Create a new handler with the given client info and tool server handle.
    ///
    /// The `tool_server_handle` should be a clone of the handle used by the agent,
    /// so that tool updates are reflected in agent requests. Registered tools get
    /// [`DEFAULT_MCP_TOOL_TIMEOUT`]; change it with [`McpClientHandler::with_timeout`].
    pub fn new(client_info: rmcp::model::ClientInfo, tool_server_handle: ToolServerHandle) -> Self {
        Self {
            client_info,
            tool_server_handle,
            timeout: Some(DEFAULT_MCP_TOOL_TIMEOUT),
            managed_tool_names: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Set (or clear) the per-call timeout applied to every MCP tool this handler
    /// registers. Pass a [`Duration`] to bound calls, or `None` to disable.
    ///
    /// See [`McpTool::with_timeout`].
    pub fn with_timeout(mut self, timeout: impl Into<Option<Duration>>) -> Self {
        self.timeout = timeout.into();
        self
    }

    /// Build an [`McpTool`], applying this handler's configured timeout.
    fn build_tool(&self, tool: rmcp::model::Tool, client: rmcp::service::ServerSink) -> McpTool {
        McpTool::from_mcp_server(tool, client).with_timeout(self.timeout)
    }

    /// Connect to an MCP server, fetch the initial tool list, and register
    /// all tools with the tool server.
    ///
    /// Returns the running MCP service. The connection stays alive as long as the
    /// returned `RunningService` is held. When the server sends
    /// `notifications/tools/list_changed`, this handler automatically re-fetches
    /// and re-registers tools.
    ///
    /// # Errors
    ///
    /// Returns [`McpClientError`] if the connection fails, the initial tool fetch
    /// fails, or tool registration with the tool server fails.
    pub async fn connect<T, E, A>(
        self,
        transport: T,
    ) -> Result<rmcp::service::RunningService<rmcp::service::RoleClient, Self>, McpClientError>
    where
        T: rmcp::transport::IntoTransport<rmcp::service::RoleClient, E, A>,
        E: std::error::Error + Send + Sync + 'static,
    {
        let service = ServiceExt::serve(self, transport)
            .await
            .map_err(|e| McpClientError::ConnectionError(e.to_string()))?;

        let tools = service.peer().list_all_tools().await?;

        {
            let handler = service.service();
            let mut managed = handler.managed_tool_names.write().await;

            for tool in tools {
                let tool_name = tool.name.to_string();
                let mcp_tool = handler.build_tool(tool, service.peer().clone());
                handler.tool_server_handle.add_tool(mcp_tool).await?;
                managed.push(tool_name);
            }
        }

        Ok(service)
    }
}

impl rmcp::handler::client::ClientHandler for McpClientHandler {
    fn get_info(&self) -> rmcp::model::ClientInfo {
        self.client_info.clone()
    }

    async fn on_tool_list_changed(
        &self,
        context: rmcp::service::NotificationContext<rmcp::service::RoleClient>,
    ) {
        let tools = match context.peer.list_all_tools().await {
            Ok(tools) => tools,
            Err(e) => {
                tracing::error!("Failed to re-fetch MCP tool list: {e}");
                return;
            }
        };

        let mut managed = self.managed_tool_names.write().await;

        for name in managed.drain(..) {
            if let Err(e) = self.tool_server_handle.remove_tool(&name).await {
                tracing::warn!("Failed to remove MCP tool '{name}' during refresh: {e}");
            }
        }

        for tool in tools {
            let tool_name = tool.name.to_string();
            let mcp_tool = self.build_tool(tool, context.peer.clone());
            match self.tool_server_handle.add_tool(mcp_tool).await {
                Ok(()) => {
                    managed.push(tool_name);
                }
                Err(e) => {
                    tracing::error!("Failed to register MCP tool '{tool_name}': {e}");
                }
            }
        }

        tracing::info!(
            tool_count = managed.len(),
            "MCP tool list refreshed successfully"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use rmcp::handler::client::ClientHandler;
    use rmcp::model::*;
    use rmcp::service::RequestContext;
    use rmcp::{RoleServer, ServerHandler, ServiceExt};
    use serde_json::json;
    use tokio::sync::RwLock;

    use super::McpClientHandler;
    use crate::tool::server::ToolServer;

    /// An MCP server whose tool list can be swapped at runtime.
    #[derive(Clone)]
    struct DynamicToolServer {
        tools: Arc<RwLock<Vec<Tool>>>,
    }

    impl DynamicToolServer {
        fn new(tools: Vec<Tool>) -> Self {
            Self {
                tools: Arc::new(RwLock::new(tools)),
            }
        }

        async fn set_tools(&self, tools: Vec<Tool>) {
            *self.tools.write().await = tools;
        }
    }

    impl ServerHandler for DynamicToolServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
                .with_protocol_version(ProtocolVersion::LATEST)
                .with_server_info(Implementation::new("test-dynamic-server", "0.1.0"))
        }

        async fn list_tools(
            &self,
            _request: Option<PaginatedRequestParams>,
            _context: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            let tools = self.tools.read().await.clone();
            Ok(ListToolsResult::with_all_items(tools))
        }

        async fn call_tool(
            &self,
            request: CallToolRequestParams,
            _context: RequestContext<RoleServer>,
        ) -> Result<CallToolResult, ErrorData> {
            Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "called {}",
                request.name
            ))]))
        }
    }

    fn make_tool(name: &str, description: &str) -> Tool {
        Tool::new(
            name.to_string(),
            description.to_string(),
            Arc::new(serde_json::Map::new()),
        )
    }

    /// An MCP server that advertises one tool whose `call_tool` handler never
    /// returns, so no `CallToolResult` is ever sent back to the client.
    ///
    /// This models the failure in <https://github.com/0xPlaygrounds/rig/issues/1914>:
    /// in the wild, rmcp 1.7.0's StreamableHttp transport can drop an in-flight
    /// tool response during transparent session re-initialization (server
    /// returns HTTP 404 -> the worker calls `streams.abort_all()`, cancelling
    /// the SSE task carrying the outstanding response -> `JoinError::Cancelled`).
    /// The request is then permanently orphaned: it never receives a response
    /// and never errors. A handler that simply never returns produces the same
    /// observable client-side behavior, deterministically and without a network.
    #[derive(Clone)]
    struct HangingToolServer;

    impl ServerHandler for HangingToolServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
                .with_protocol_version(ProtocolVersion::LATEST)
                .with_server_info(Implementation::new("hanging-server", "0.1.0"))
        }

        async fn list_tools(
            &self,
            _request: Option<PaginatedRequestParams>,
            _context: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            Ok(ListToolsResult::with_all_items(vec![make_tool(
                "hang_forever",
                "A tool whose handler never returns",
            )]))
        }

        async fn call_tool(
            &self,
            _request: CallToolRequestParams,
            _context: RequestContext<RoleServer>,
        ) -> Result<CallToolResult, ErrorData> {
            // Never resolves: the crux of the reproduction. No response is ever
            // sent, so the client's `call_tool` future (and therefore rig's
            // `McpTool::call`) never completes.
            std::future::pending::<Result<CallToolResult, ErrorData>>().await
        }
    }

    #[tokio::test]
    async fn test_mcp_client_handler_initial_tool_registration() {
        let initial_tools = vec![
            make_tool("tool_a", "First tool"),
            make_tool("tool_b", "Second tool"),
        ];

        let server = DynamicToolServer::new(initial_tools);
        let tool_server_handle = ToolServer::new().run();

        let (client_to_server, server_from_client) = tokio::io::duplex(8192);
        let (server_to_client, client_from_server) = tokio::io::duplex(8192);

        let server_clone = server.clone();
        tokio::spawn(async move {
            let _service = server_clone
                .serve((server_from_client, server_to_client))
                .await
                .expect("server failed to start");
            _service.waiting().await.expect("server error");
        });

        let client_info = ClientInfo::default();
        let handler = McpClientHandler::new(client_info, tool_server_handle.clone());

        let _mcp_service = handler
            .connect((client_from_server, client_to_server))
            .await
            .expect("connect failed");

        let defs = tool_server_handle.get_tool_defs(None).await.unwrap();
        assert_eq!(defs.len(), 2);

        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"tool_a"));
        assert!(names.contains(&"tool_b"));
    }

    #[tokio::test]
    async fn test_mcp_client_handler_refreshes_on_tool_list_changed() {
        let initial_tools = vec![make_tool("alpha", "Alpha tool")];

        let server = DynamicToolServer::new(initial_tools);
        let tool_server_handle = ToolServer::new().run();

        let (client_to_server, server_from_client) = tokio::io::duplex(8192);
        let (server_to_client, client_from_server) = tokio::io::duplex(8192);

        let server_clone = server.clone();
        let server_service_handle = tokio::spawn(async move {
            server_clone
                .serve((server_from_client, server_to_client))
                .await
                .expect("server failed to start")
        });

        let client_info = ClientInfo::default();
        let handler = McpClientHandler::new(client_info, tool_server_handle.clone());

        let _mcp_service = handler
            .connect((client_from_server, client_to_server))
            .await
            .expect("connect failed");

        // Verify initial state
        let defs = tool_server_handle.get_tool_defs(None).await.unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "alpha");

        // Update the server's tool list
        server
            .set_tools(vec![
                make_tool("beta", "Beta tool"),
                make_tool("gamma", "Gamma tool"),
            ])
            .await;

        // Send the notification from the server side
        let server_service = server_service_handle.await.unwrap();
        server_service
            .peer()
            .notify_tool_list_changed()
            .await
            .expect("failed to send notification");

        // The handler processes the notification asynchronously, so give it
        // a moment to re-fetch and re-register tools.
        tokio::time::sleep(Duration::from_millis(200)).await;

        let defs = tool_server_handle.get_tool_defs(None).await.unwrap();
        assert_eq!(defs.len(), 2);

        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"beta"), "expected 'beta' in {names:?}");
        assert!(names.contains(&"gamma"), "expected 'gamma' in {names:?}");
        // The old tool must be gone
        assert!(
            !names.contains(&"alpha"),
            "expected 'alpha' to be removed, found {names:?}"
        );
    }

    #[tokio::test]
    async fn test_mcp_client_handler_get_info_delegates() {
        let client_info = ClientInfo::new(
            ClientCapabilities::default(),
            Implementation::new("test-client", "1.0.0"),
        );

        let tool_server_handle = ToolServer::new().run();
        let handler = McpClientHandler::new(client_info.clone(), tool_server_handle);

        let returned = handler.get_info();
        assert_eq!(returned.client_info.name, "test-client");
        assert_eq!(returned.client_info.version, "1.0.0");
    }

    #[tokio::test]
    async fn mcp_tool_exposes_flattened_metadata() {
        use super::McpTool;
        use crate::tool::{ToolDyn, tool_definition};

        let mut schema = serde_json::Map::new();
        schema.insert("type".to_string(), json!("object"));
        schema.insert(
            "properties".to_string(),
            json!({ "query": { "type": "string" } }),
        );
        let server_tool = Tool::new(
            "search_docs".to_string(),
            "Search the docs".to_string(),
            Arc::new(schema),
        );

        let (client_to_server, server_from_client) = tokio::io::duplex(8192);
        let (server_to_client, client_from_server) = tokio::io::duplex(8192);
        let server = DynamicToolServer::new(vec![server_tool.clone()]);
        let server_task = tokio::spawn(async move {
            let running = server
                .serve((server_from_client, server_to_client))
                .await
                .expect("server failed to start");
            running.waiting().await.expect("server error");
        });

        let client = ClientInfo::default()
            .serve((client_from_server, client_to_server))
            .await
            .expect("client connect failed");
        let mcp_tool = McpTool::from_mcp_server(server_tool, client.peer().clone());
        let definition = tool_definition(&mcp_tool);

        assert_eq!(mcp_tool.name(), "search_docs");
        assert_eq!(definition.name, "search_docs");
        assert_eq!(definition.description, "Search the docs");
        assert_eq!(
            definition.parameters["properties"]["query"]["type"],
            "string"
        );

        client.cancel().await.expect("client cancel failed");
        server_task.abort();
    }

    /// Documents the unbounded escape hatch and the underlying issue #1914 hazard.
    ///
    /// `McpTool::call` awaits `self.client.call_tool(request)`; if the MCP request
    /// is orphaned (no response, no error — e.g. an rmcp StreamableHttp session
    /// re-init dropping an in-flight request), that `.await` never completes and
    /// the agent loop wedges (the loop turns a tool *error* into a tool result,
    /// but cannot recover from a call that never returns). That is exactly why
    /// the default now applies [`DEFAULT_MCP_TOOL_TIMEOUT`].
    ///
    /// Here we opt **out** with `with_timeout(None)` and show the call stays
    /// unbounded (does not resolve within the window). The outer `timeout` exists
    /// only so this test terminates; it elapsing is the *intended* unbounded
    /// behavior of the disabled-timeout path, not a bug. The bounded paths are
    /// covered by `mcp_tool_call_with_timeout_errors_instead_of_hanging`.
    #[tokio::test]
    async fn mcp_tool_call_without_timeout_is_unbounded() {
        use super::McpTool;
        use crate::tool::ToolDyn;

        let (client_to_server, server_from_client) = tokio::io::duplex(8192);
        let (server_to_client, client_from_server) = tokio::io::duplex(8192);

        let server_task = tokio::spawn(async move {
            let running = HangingToolServer
                .serve((server_from_client, server_to_client))
                .await
                .expect("server failed to start");
            running.waiting().await.expect("server error");
        });

        // A bare client (`ClientInfo` implements `ClientHandler`); `.peer()` is
        // the `ServerSink` that rig stores inside every `McpTool`.
        let client = ClientInfo::default()
            .serve((client_from_server, client_to_server))
            .await
            .expect("client connect failed");

        let tools = client
            .peer()
            .list_all_tools()
            .await
            .expect("list_tools failed");
        assert_eq!(tools.len(), 1, "expected exactly one advertised tool");

        // `from_mcp_server` applies the generous default out of the box...
        let mcp_tool = McpTool::from_mcp_server(tools[0].clone(), client.peer().clone());
        assert_eq!(mcp_tool.timeout(), Some(super::DEFAULT_MCP_TOOL_TIMEOUT));
        // ...and callers can explicitly disable it to opt into unbounded behavior.
        let mcp_tool = mcp_tool.with_timeout(None);
        assert_eq!(mcp_tool.timeout(), None);

        let timed =
            tokio::time::timeout(Duration::from_millis(150), mcp_tool.call("{}".to_string())).await;

        assert!(
            timed.is_err(),
            "with the timeout disabled, McpTool::call must stay unbounded; got {:?}",
            timed.ok(),
        );

        server_task.abort();
    }

    /// Regression test for the fix to <https://github.com/0xPlaygrounds/rig/issues/1914>.
    ///
    /// With a per-call timeout configured, `McpTool::call` against a server that
    /// never responds resolves to a `ToolError` (which the agent loop surfaces
    /// to the model) instead of hanging forever. The outer `timeout` is only a
    /// safety net so a regression cannot wedge the test runner; the inner 200ms
    /// timeout fires first.
    #[tokio::test]
    async fn mcp_tool_call_with_timeout_errors_instead_of_hanging() {
        use super::McpTool;
        use crate::tool::ToolDyn;

        let (client_to_server, server_from_client) = tokio::io::duplex(8192);
        let (server_to_client, client_from_server) = tokio::io::duplex(8192);

        let server_task = tokio::spawn(async move {
            let running = HangingToolServer
                .serve((server_from_client, server_to_client))
                .await
                .expect("server failed to start");
            running.waiting().await.expect("server error");
        });

        let client = ClientInfo::default()
            .serve((client_from_server, client_to_server))
            .await
            .expect("client connect failed");

        let tools = client
            .peer()
            .list_all_tools()
            .await
            .expect("list_tools failed");

        // The fix: a per-call timeout bounds the otherwise-unbounded await.
        let mcp_tool = McpTool::from_mcp_server(tools[0].clone(), client.peer().clone())
            .with_timeout(Duration::from_millis(200));

        let timed =
            tokio::time::timeout(Duration::from_secs(5), mcp_tool.call("{}".to_string())).await;

        let result = timed.expect(
            "regression: McpTool::call hung past the safety timeout; the per-call \
             timeout did not fire (issue #1914 fix is broken)",
        );
        let err =
            result.expect_err("call should resolve to an error when the server never responds");
        // "timed out" mirrors the McpToolError format string in McpTool::call.
        assert!(
            err.to_string().contains("timed out"),
            "expected a timeout error, got: {err}"
        );

        server_task.abort();
    }

    /// Success path with a timeout *configured*: a responsive tool resolves with
    /// its result (exercising the `Some(timeout)` arm of `McpTool::call` and the
    /// `wasm_compat::timeout` "completed" branch, with a real `CallToolResult`
    /// flowing through content parsing). Also guards that the bound in the
    /// hanging-server tests is meaningful — a healthy call returns well inside it.
    #[tokio::test]
    async fn mcp_tool_call_returns_promptly_for_responsive_server() {
        use super::McpTool;
        use crate::tool::ToolDyn;

        let server = DynamicToolServer::new(vec![make_tool("ping", "responds immediately")]);

        let (client_to_server, server_from_client) = tokio::io::duplex(8192);
        let (server_to_client, client_from_server) = tokio::io::duplex(8192);

        let server_clone = server.clone();
        let server_task = tokio::spawn(async move {
            let running = server_clone
                .serve((server_from_client, server_to_client))
                .await
                .expect("server failed to start");
            running.waiting().await.expect("server error");
        });

        let client = ClientInfo::default()
            .serve((client_from_server, client_to_server))
            .await
            .expect("client connect failed");

        let tools = client
            .peer()
            .list_all_tools()
            .await
            .expect("list_tools failed");
        let mcp_tool = McpTool::from_mcp_server(tools[0].clone(), client.peer().clone())
            .with_timeout(Duration::from_secs(2));

        let timed =
            tokio::time::timeout(Duration::from_secs(5), mcp_tool.call("{}".to_string())).await;

        let result = timed
            .expect("responsive tool should resolve within the safety window")
            .expect("tool call should succeed");
        assert!(result.contains("ping"), "unexpected tool output: {result}");

        server_task.abort();
    }

    /// `McpClientHandler::with_timeout` is applied to every tool it registers:
    /// calling a registered tool from the shared `ToolServerHandle` (the path the
    /// agent loop uses) surfaces a timeout error instead of hanging.
    #[tokio::test]
    async fn mcp_client_handler_with_timeout_bounds_registered_tools() {
        let tool_server_handle = ToolServer::new().run();

        let (client_to_server, server_from_client) = tokio::io::duplex(8192);
        let (server_to_client, client_from_server) = tokio::io::duplex(8192);

        let server_task = tokio::spawn(async move {
            let running = HangingToolServer
                .serve((server_from_client, server_to_client))
                .await
                .expect("server failed to start");
            running.waiting().await.expect("server error");
        });

        let handler = McpClientHandler::new(ClientInfo::default(), tool_server_handle.clone())
            .with_timeout(Duration::from_millis(200));
        let _mcp_service = handler
            .connect((client_from_server, client_to_server))
            .await
            .expect("connect failed");

        // Call through the shared handle exactly as the agent loop does.
        let timed = tokio::time::timeout(
            Duration::from_secs(5),
            tool_server_handle.call_tool("hang_forever", "{}"),
        )
        .await;

        let result = timed.expect("handler-registered tool hung past the safety timeout");
        let err = result.expect_err("call should time out when the server never responds");
        assert!(
            err.to_string().contains("timed out"),
            "expected a timeout error, got: {err}"
        );

        server_task.abort();
    }

    /// `ToolServer::rmcp_tool_with_timeout` bounds the registered tool: calling it
    /// through the `ToolServerHandle` surfaces a timeout error instead of hanging.
    #[tokio::test]
    async fn tool_server_rmcp_tool_with_timeout_bounds_calls() {
        let (client_to_server, server_from_client) = tokio::io::duplex(8192);
        let (server_to_client, client_from_server) = tokio::io::duplex(8192);

        let server_task = tokio::spawn(async move {
            let running = HangingToolServer
                .serve((server_from_client, server_to_client))
                .await
                .expect("server failed to start");
            running.waiting().await.expect("server error");
        });

        let client = ClientInfo::default()
            .serve((client_from_server, client_to_server))
            .await
            .expect("client connect failed");

        // The tool definition is constructed directly; the peer routes the call
        // to the hanging server, which never responds.
        let handle = ToolServer::new()
            .rmcp_tool_with_timeout(
                make_tool("hang_forever", "never returns"),
                client.peer().clone(),
                Duration::from_millis(200),
            )
            .run();

        let timed = tokio::time::timeout(
            Duration::from_secs(5),
            handle.call_tool("hang_forever", "{}"),
        )
        .await;

        let result = timed.expect("ToolServer-registered tool hung past the safety timeout");
        let err = result.expect_err("call should time out when the server never responds");
        assert!(
            err.to_string().contains("timed out"),
            "expected a timeout error, got: {err}"
        );

        server_task.abort();
    }

    /// An MCP server that records the `_meta` of the last `call_tool` request, so
    /// a test can assert what metadata reached the server.
    #[derive(Clone)]
    struct MetaCapturingServer {
        seen_meta: Arc<RwLock<Option<Meta>>>,
    }

    impl ServerHandler for MetaCapturingServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
                .with_protocol_version(ProtocolVersion::LATEST)
                .with_server_info(Implementation::new("meta-capturing-server", "0.1.0"))
        }

        async fn list_tools(
            &self,
            _request: Option<PaginatedRequestParams>,
            _context: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            Ok(ListToolsResult::with_all_items(vec![make_tool(
                "echo_meta",
                "records the request _meta",
            )]))
        }

        async fn call_tool(
            &self,
            _request: CallToolRequestParams,
            context: RequestContext<RoleServer>,
        ) -> Result<CallToolResult, ErrorData> {
            // rmcp's router moves the request's `_meta` into `RequestContext.meta`
            // (a `std::mem::swap`), so the forwarded metadata lands here.
            *self.seen_meta.write().await = Some(context.meta.clone());
            Ok(CallToolResult::success(vec![ContentBlock::text("ok")]))
        }
    }

    /// `McpTool::call_with_extensions` forwards an [`rmcp::model::Meta`] placed in the
    /// [`ToolCallExtensions`] as the request's `_meta`, so callers can attach per-call
    /// auth/session metadata to MCP tool invocations (the A2A use-case).
    #[tokio::test]
    async fn mcp_tool_forwards_meta_from_context() {
        use super::McpTool;
        use crate::tool::{ToolCallExtensions, ToolDyn};

        let seen_meta = Arc::new(RwLock::new(None));
        let server = MetaCapturingServer {
            seen_meta: seen_meta.clone(),
        };

        let (client_to_server, server_from_client) = tokio::io::duplex(8192);
        let (server_to_client, client_from_server) = tokio::io::duplex(8192);

        let server_clone = server.clone();
        let server_task = tokio::spawn(async move {
            let running = server_clone
                .serve((server_from_client, server_to_client))
                .await
                .expect("server failed to start");
            running.waiting().await.expect("server error");
        });

        let client = ClientInfo::default()
            .serve((client_from_server, client_to_server))
            .await
            .expect("client connect failed");

        let tools = client
            .peer()
            .list_all_tools()
            .await
            .expect("list_tools failed");
        let mcp_tool = McpTool::from_mcp_server(tools[0].clone(), client.peer().clone());

        // Caller-supplied per-call metadata, the kind an A2A integration carries.
        let mut meta = Meta::new();
        meta.0
            .insert("authorization".to_string(), json!("Bearer xyz"));
        let mut extensions = ToolCallExtensions::new();
        extensions.insert(meta);

        let out = mcp_tool
            .call_with_extensions("{}".to_string(), &extensions)
            .await
            .expect("call should succeed");
        assert_eq!(out, "ok");

        let received = seen_meta
            .read()
            .await
            .clone()
            .expect("server should have observed a request");
        assert_eq!(received.0.get("authorization"), Some(&json!("Bearer xyz")));

        // Without a Meta in the context, the caller metadata is not forwarded.
        *seen_meta.write().await = None;
        let empty_ctx = ToolCallExtensions::new();
        let out = mcp_tool
            .call_with_extensions("{}".to_string(), &empty_ctx)
            .await
            .expect("call should succeed");
        assert_eq!(out, "ok");
        let received = seen_meta
            .read()
            .await
            .clone()
            .expect("server should have observed a request");
        assert!(
            received.0.get("authorization").is_none(),
            "no caller metadata should be forwarded when the context carries none"
        );

        server_task.abort();
    }

    /// `parse_mcp_arguments` distinguishes malformed JSON (a hard error) from
    /// valid-but-argument-free inputs (empty / `null` / non-object) which map to
    /// `None`, while a JSON object round-trips.
    #[test]
    fn parse_mcp_arguments_classifies_inputs() {
        use super::parse_mcp_arguments;

        assert!(parse_mcp_arguments("").expect("empty is no-args").is_none());
        assert!(
            parse_mcp_arguments("   ")
                .expect("whitespace is no-args")
                .is_none()
        );
        assert!(
            parse_mcp_arguments("null")
                .expect("null is no-args")
                .is_none()
        );
        assert!(
            parse_mcp_arguments("[1,2]")
                .expect("array is no-args")
                .is_none()
        );
        assert!(parse_mcp_arguments("{}").expect("empty object").is_some());
        let obj = parse_mcp_arguments("{\"a\":1}")
            .expect("valid object")
            .expect("object present");
        assert_eq!(obj.get("a"), Some(&json!(1)));

        // Malformed JSON is a hard error, not a silent no-arg call.
        assert!(parse_mcp_arguments("{not valid json").is_err());
        assert!(parse_mcp_arguments("{\"a\":").is_err());
    }

    /// Malformed JSON arguments are classified as [`ToolFailureKind::InvalidArgs`]
    /// and short-circuit **before** the MCP server is contacted — proven by
    /// pointing the tool at a server that never responds and asserting the call
    /// returns fast with a structured invalid-args outcome instead of hanging.
    #[tokio::test]
    async fn mcp_tool_invalid_json_args_short_circuit_as_invalid_args() {
        use super::McpTool;
        use crate::tool::{ToolCallExtensions, ToolDyn, ToolFailureKind, ToolOutcome};

        let (client_to_server, server_from_client) = tokio::io::duplex(8192);
        let (server_to_client, client_from_server) = tokio::io::duplex(8192);

        let server_task = tokio::spawn(async move {
            let running = HangingToolServer
                .serve((server_from_client, server_to_client))
                .await
                .expect("server failed to start");
            running.waiting().await.expect("server error");
        });

        let client = ClientInfo::default()
            .serve((client_from_server, client_to_server))
            .await
            .expect("client connect failed");

        let tools = client
            .peer()
            .list_all_tools()
            .await
            .expect("list_tools failed");

        // A generous per-call timeout: if invalid args wrongly reached the hanging
        // server, the call would take this long; the safety timeout below is much
        // shorter, so the test fails fast on a regression rather than short-circuiting.
        let mcp_tool = McpTool::from_mcp_server(tools[0].clone(), client.peer().clone())
            .with_timeout(Duration::from_secs(30));

        let structured = tokio::time::timeout(
            Duration::from_secs(2),
            mcp_tool.call_structured("{not valid json".to_string(), &ToolCallExtensions::new()),
        )
        .await
        .expect(
            "invalid JSON args must be classified before contacting the (hanging) server; \
             the call reached the server and hung",
        );

        match &structured.outcome {
            ToolOutcome::Error(failure) => {
                assert_eq!(
                    failure.kind,
                    ToolFailureKind::InvalidArgs,
                    "malformed MCP args must classify as InvalidArgs, got {:?}",
                    failure.kind
                );
            }
            other => panic!("expected an InvalidArgs error outcome, got {other:?}"),
        }
        assert!(
            structured.model_output.contains("invalid JSON"),
            "the model-visible output should explain the parse failure, got {:?}",
            structured.model_output
        );

        // The string `call` path surfaces the same failure as a `ToolError`.
        let err = tokio::time::timeout(
            Duration::from_secs(2),
            mcp_tool.call("{not valid json".to_string()),
        )
        .await
        .expect("invalid JSON args must short-circuit on the string path too")
        .expect_err("malformed args must be an error");
        assert!(err.to_string().contains("invalid JSON"), "got: {err}");

        server_task.abort();
    }
}
