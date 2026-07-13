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
//! placed in a [`ToolContext`] as the MCP request's `_meta` (SEP-1319) —
//! the idiomatic channel for per-call values such as auth tokens, session ids,
//! or A2A `context_id`/`task_id`, which the model never sees:
//!
//! ```rust,ignore
//! use rig_core::tool::rmcp::Meta;
//! use rig_core::tool::ToolContext;
//!
//! let mut meta = Meta::new();
//! meta.0.insert("authorization".into(), serde_json::json!("Bearer …"));
//! let mut context = ToolContext::new();
//! context.insert(meta);
//! let answer = agent.prompt("…").tool_context(context).await?;
//! ```

use std::borrow::Cow;
#[cfg(not(target_family = "wasm"))]
use std::sync::Arc;
use std::time::Duration;

#[cfg(not(target_family = "wasm"))]
use rmcp::ServiceExt;
use rmcp::model::ContentBlock;
#[cfg(not(target_family = "wasm"))]
use tokio::sync::RwLock;

use crate::completion::ToolDefinition;
use crate::tool::server::ToolServerError;
#[cfg(not(target_family = "wasm"))]
use crate::tool::server::ToolServerHandle;
use crate::tool::{DynamicTool, ErasedTool, ToolContext, ToolExecutionError};
use crate::wasm_compat::WasmBoxedFuture;

/// Re-export of [`rmcp::model::Meta`]: place one in a [`ToolContext`] to have
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
/// Adapts the MCP tool protocol to Rig's canonical tool execution system,
/// allowing MCP tools to be used seamlessly in Rig agents.
#[derive(Clone)]
pub struct McpTool {
    definition: rmcp::model::Tool,
    client: rmcp::service::ServerSink,
    /// Per-call timeout. When `Some`, an MCP `call_tool` that does not complete
    /// within this duration resolves to a [`ToolExecutionError`] instead of blocking
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
    /// On timeout the call resolves to a [`ToolExecutionError`] (which the agent loop
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

    /// Convert this adapter into the public runtime-defined tool type.
    ///
    /// Use this when registering an MCP tool through an API that accepts a
    /// [`DynamicTool`] rather than the dedicated `rmcp_tool` builders.
    pub fn into_dynamic_tool(self) -> DynamicTool {
        DynamicTool::from_erased(self)
    }
}

impl From<McpTool> for DynamicTool {
    fn from(tool: McpTool) -> Self {
        tool.into_dynamic_tool()
    }
}

impl From<&rmcp::model::Tool> for ToolDefinition {
    fn from(val: &rmcp::model::Tool) -> Self {
        Self {
            name: val.name.to_string(),
            description: val.description.clone().unwrap_or(Cow::from("")).to_string(),
            parameters: val.schema_as_json_value(),
        }
    }
}

impl From<rmcp::model::Tool> for ToolDefinition {
    fn from(val: rmcp::model::Tool) -> Self {
        Self {
            name: val.name.to_string(),
            description: val.description.clone().unwrap_or(Cow::from("")).to_string(),
            parameters: val.schema_as_json_value(),
        }
    }
}

/// Parse the JSON `args` string into MCP call arguments.
///
/// Empty input denotes an omitted arguments object. Any supplied JSON value must
/// be an object; silently dropping arrays, scalars, or `null` could execute a
/// remote tool with different arguments than the model requested.
fn parse_mcp_arguments(args: &str) -> Result<Option<rmcp::model::JsonObject>, serde_json::Error> {
    let trimmed = args.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    match serde_json::from_str(trimmed)? {
        serde_json::Value::Object(map) => Ok(Some(map)),
        other => Err(<serde_json::Error as serde::de::Error>::custom(format!(
            "MCP tool arguments must be a JSON object, got {}",
            match other {
                serde_json::Value::Null => "null",
                serde_json::Value::Bool(_) => "a boolean",
                serde_json::Value::Number(_) => "a number",
                serde_json::Value::String(_) => "a string",
                serde_json::Value::Array(_) => "an array",
                serde_json::Value::Object(_) => "an object",
            }
        ))),
    }
}

impl McpTool {
    fn execute_mcp(
        &self,
        args: &str,
        meta: Option<rmcp::model::Meta>,
    ) -> WasmBoxedFuture<'_, Result<String, ToolExecutionError>> {
        let name = self.definition.name.clone();
        let args = args.to_string();
        Box::pin(async move {
            let arguments = parse_mcp_arguments(&args).map_err(|error| {
                ToolExecutionError::invalid_args(format!(
                    "MCP tool '{name}' received invalid JSON arguments: {error}"
                ))
                .with_source(error)
            })?;
            let mut request = arguments
                .map(|arguments| {
                    rmcp::model::CallToolRequestParams::new(name.clone()).with_arguments(arguments)
                })
                .unwrap_or_else(|| rmcp::model::CallToolRequestParams::new(name));
            request.meta = meta;

            let call = self.client.call_tool(request);
            let call_result = match self.timeout {
                Some(timeout) => {
                    crate::wasm_compat::timeout(timeout, call)
                        .await
                        .map_err(|_| {
                            ToolExecutionError::timeout(format!(
                                "MCP tool '{}' timed out after {timeout:?}",
                                self.definition.name
                            ))
                        })?
                }
                None => call.await,
            };
            let result = call_result.map_err(|source| {
                ToolExecutionError::provider(format!("Tool returned an error: {source}"))
                    .with_source(source)
            })?;

            if result.is_error == Some(true) {
                let message = result
                    .content
                    .into_iter()
                    .filter_map(|item| item.as_text().map(|text| text.text.clone()))
                    .collect::<Vec<_>>()
                    .join("\n");
                let message = if message.is_empty() {
                    "No message returned".to_string()
                } else {
                    message
                };
                return Err(ToolExecutionError::other(message.clone()).with_model_feedback(message));
            }

            let mut content = String::new();
            let mut image_parts = Vec::new();
            for item in result.content {
                let chunk = match item {
                    ContentBlock::Text(raw) => raw.text,
                    ContentBlock::Image(raw) => {
                        image_parts.push(serde_json::json!({
                            "type": "image",
                            "data": raw.data,
                            "mimeType": raw.mime_type,
                        }));
                        continue;
                    }
                    ContentBlock::Resource(raw) => match raw.resource {
                        rmcp::model::ResourceContents::TextResourceContents {
                            uri,
                            mime_type,
                            text,
                            ..
                        } => format!(
                            "{mime_type}{uri}:{text}",
                            mime_type = mime_type
                                .map(|mime| format!("data:{mime};"))
                                .unwrap_or_default(),
                        ),
                        rmcp::model::ResourceContents::BlobResourceContents {
                            uri,
                            mime_type,
                            blob,
                            ..
                        } => format!(
                            "{mime_type}{uri}:{blob}",
                            mime_type = mime_type
                                .map(|mime| format!("data:{mime};"))
                                .unwrap_or_default(),
                        ),
                        other => {
                            return Err(ToolExecutionError::other(format!(
                                "MCP tool returned unsupported resource contents: {other:?}"
                            )));
                        }
                    },
                    ContentBlock::Audio(_) => {
                        return Err(ToolExecutionError::other(
                            "MCP tool returned audio content, which Rig does not support yet",
                        ));
                    }
                    other => {
                        return Err(ToolExecutionError::other(format!(
                            "MCP tool returned unsupported content: {other:?}"
                        )));
                    }
                };
                content.push_str(&chunk);
            }

            match image_parts.as_slice() {
                [] => Ok(content),
                [image] if content.is_empty() => Ok(image.to_string()),
                _ if content.is_empty() => Ok(serde_json::json!({
                    "parts": image_parts,
                })
                .to_string()),
                _ => Ok(serde_json::json!({
                    "response": content,
                    "parts": image_parts,
                })
                .to_string()),
            }
        })
    }
}

impl ErasedTool for McpTool {
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

    fn execute<'a>(
        &'a self,
        context: &'a mut ToolContext,
        args: &'a str,
    ) -> WasmBoxedFuture<'a, Result<String, ToolExecutionError>> {
        let meta = context.get::<rmcp::model::Meta>().cloned();
        self.execute_mcp(args, meta)
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
#[cfg(not(target_family = "wasm"))]
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

#[cfg(not(target_family = "wasm"))]
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
    #[cfg(not(target_family = "wasm"))]
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
                handler
                    .tool_server_handle
                    .add_dynamic_tool(DynamicTool::from_erased(mcp_tool))
                    .await?;
                managed.push(tool_name);
            }
        }

        Ok(service)
    }
}

#[cfg(not(target_family = "wasm"))]
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
            match self
                .tool_server_handle
                .add_dynamic_tool(DynamicTool::from_erased(mcp_tool))
                .await
            {
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

    use super::{McpClientHandler, McpTool};
    use crate::tool::server::ToolServer;
    use crate::tool::{ErasedTool, ToolContext, ToolExecutionError, ToolExecutionErrorKind};

    async fn execute_mcp(
        tool: &McpTool,
        args: &str,
        context: &mut ToolContext,
    ) -> Result<String, ToolExecutionError> {
        ErasedTool::execute(tool, context, args).await
    }

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
            let content = if request.name == "image_tool" {
                vec![
                    ContentBlock::text("caption"),
                    ContentBlock::image("aW1hZ2U=", "image/png"),
                ]
            } else {
                vec![ContentBlock::text(format!("called {}", request.name))]
            };
            Ok(CallToolResult::success(content))
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
            // MCP tool execution never completes.
            std::future::pending::<Result<CallToolResult, ErrorData>>().await
        }
    }

    const TOOL_REPORTED_ERROR: &str = "the remote lookup found no matching record";

    /// An MCP server whose tool executes but reports a caller-visible failure.
    #[derive(Clone)]
    struct ToolReportedErrorServer;

    impl ServerHandler for ToolReportedErrorServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
                .with_protocol_version(ProtocolVersion::LATEST)
                .with_server_info(Implementation::new("tool-error-server", "0.1.0"))
        }

        async fn list_tools(
            &self,
            _request: Option<PaginatedRequestParams>,
            _context: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            Ok(ListToolsResult::with_all_items(vec![make_tool(
                "reported_error",
                "reports a caller-visible tool failure",
            )]))
        }

        async fn call_tool(
            &self,
            _request: CallToolRequestParams,
            _context: RequestContext<RoleServer>,
        ) -> Result<CallToolResult, ErrorData> {
            Ok(CallToolResult::error(vec![ContentBlock::text(
                TOOL_REPORTED_ERROR,
            )]))
        }
    }

    const PROVIDER_ERROR: &str = "the remote MCP service could not route the tool call";

    /// An MCP server that rejects `call_tool` at the protocol/provider boundary.
    #[derive(Clone)]
    struct ProviderErrorServer;

    impl ServerHandler for ProviderErrorServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
                .with_protocol_version(ProtocolVersion::LATEST)
                .with_server_info(Implementation::new("provider-error-server", "0.1.0"))
        }

        async fn list_tools(
            &self,
            _request: Option<PaginatedRequestParams>,
            _context: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            Ok(ListToolsResult::with_all_items(vec![make_tool(
                "provider_error",
                "fails at the provider boundary",
            )]))
        }

        async fn call_tool(
            &self,
            _request: CallToolRequestParams,
            _context: RequestContext<RoleServer>,
        ) -> Result<CallToolResult, ErrorData> {
            Err(ErrorData::internal_error(PROVIDER_ERROR, None))
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
        let definition = mcp_tool.clone().into_dynamic_tool().definition();

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
    /// `McpTool` execution awaits `self.client.call_tool(request)`; if the MCP request
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

        let timed = tokio::time::timeout(
            Duration::from_millis(150),
            execute_mcp(&mcp_tool, "{}", &mut ToolContext::new()),
        )
        .await;

        assert!(
            timed.is_err(),
            "with the timeout disabled, MCP tool execution must stay unbounded; got {:?}",
            timed.ok(),
        );

        server_task.abort();
    }

    /// Regression test for the fix to <https://github.com/0xPlaygrounds/rig/issues/1914>.
    ///
    /// With a per-call timeout configured, MCP tool execution against a server that
    /// never responds resolves to a `ToolExecutionError` (which the agent loop surfaces
    /// to the model) instead of hanging forever. The outer `timeout` is only a
    /// safety net so a regression cannot wedge the test runner; the inner 200ms
    /// timeout fires first.
    #[tokio::test]
    async fn mcp_tool_call_with_timeout_errors_instead_of_hanging() {
        use super::McpTool;
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

        let timed = tokio::time::timeout(
            Duration::from_secs(5),
            execute_mcp(&mcp_tool, "{}", &mut ToolContext::new()),
        )
        .await;

        let result = timed.expect(
            "regression: MCP tool execution hung past the safety timeout; the per-call \
             timeout did not fire (issue #1914 fix is broken)",
        );
        let err =
            result.expect_err("call should resolve to an error when the server never responds");
        assert_eq!(err.kind(), ToolExecutionErrorKind::Timeout);
        // The structured error retains the configured timeout diagnostic.
        assert!(
            err.to_string().contains("timed out"),
            "expected a timeout error, got: {err}"
        );

        server_task.abort();
    }

    #[tokio::test]
    async fn mcp_tool_preserves_tool_reported_error() {
        let (client_to_server, server_from_client) = tokio::io::duplex(8192);
        let (server_to_client, client_from_server) = tokio::io::duplex(8192);

        let server_task = tokio::spawn(async move {
            let running = ToolReportedErrorServer
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
        let tool = McpTool::from_mcp_server(tools[0].clone(), client.peer().clone());

        let error = execute_mcp(&tool, "{}", &mut ToolContext::new())
            .await
            .expect_err("the MCP tool reported a failure");
        assert_eq!(error.kind(), ToolExecutionErrorKind::Other);
        assert_eq!(error.message(), TOOL_REPORTED_ERROR);
        assert_eq!(error.model_feedback(), Some(TOOL_REPORTED_ERROR));

        client.cancel().await.expect("client cancel failed");
        server_task.abort();
    }

    #[tokio::test]
    async fn mcp_tool_preserves_provider_error_and_source() {
        let (client_to_server, server_from_client) = tokio::io::duplex(8192);
        let (server_to_client, client_from_server) = tokio::io::duplex(8192);

        let server_task = tokio::spawn(async move {
            let running = ProviderErrorServer
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
        let tool = McpTool::from_mcp_server(tools[0].clone(), client.peer().clone());

        let error = execute_mcp(&tool, "{}", &mut ToolContext::new())
            .await
            .expect_err("the MCP provider rejected the call");
        assert_eq!(error.kind(), ToolExecutionErrorKind::Provider);
        assert!(
            error.message().contains("Tool returned an error")
                && error.message().contains(PROVIDER_ERROR),
            "operator diagnostic should retain the MCP provider error: {error:?}"
        );
        assert_eq!(error.model_feedback(), None);
        #[cfg(not(target_family = "wasm"))]
        assert!(
            error.downcast_source_ref::<rmcp::ServiceError>().is_some(),
            "the concrete rmcp::ServiceError source should remain downcastable"
        );

        client.cancel().await.expect("client cancel failed");
        server_task.abort();
    }

    /// Success path with a timeout *configured*: a responsive tool resolves with
    /// its result (exercising the configured-timeout execution branch and the
    /// `wasm_compat::timeout` "completed" branch, with a real `CallToolResult`
    /// flowing through content parsing). Also guards that the bound in the
    /// hanging-server tests is meaningful — a healthy call returns well inside it.
    #[tokio::test]
    async fn mcp_image_result_preserves_multimodal_content() {
        use crate::message::{DocumentSourceKind, ToolResultContent};

        let server_tool = make_tool("image_tool", "Returns text and an image");
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
        let tool = McpTool::from_mcp_server(server_tool, client.peer().clone());

        let output = execute_mcp(&tool, "{}", &mut ToolContext::new())
            .await
            .expect("MCP tool output");
        let content = ToolResultContent::from_tool_output(output);
        let parts = content.iter().collect::<Vec<_>>();
        assert!(matches!(
            parts.as_slice(),
            [ToolResultContent::Text(text), ToolResultContent::Image(image)]
                if text.text == "caption"
                    && image.data == DocumentSourceKind::Base64("aW1hZ2U=".to_string())
        ));

        client.cancel().await.expect("client cancel failed");
        server_task.abort();
    }

    #[tokio::test]
    async fn mcp_tool_call_returns_promptly_for_responsive_server() {
        use super::McpTool;
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

        let timed = tokio::time::timeout(
            Duration::from_secs(5),
            execute_mcp(&mcp_tool, "{}", &mut ToolContext::new()),
        )
        .await;

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
        let timed = tokio::time::timeout(Duration::from_secs(5), async {
            let mut context = ToolContext::new();
            tool_server_handle
                .execute_tool("hang_forever", "{}", &mut context)
                .await
        })
        .await;

        let result = timed.expect("handler-registered tool hung past the safety timeout");
        let err = result.expect_err("call should time out when the server never responds");
        assert_eq!(err.kind(), ToolExecutionErrorKind::Timeout);
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

        let timed = tokio::time::timeout(Duration::from_secs(5), async {
            let mut context = ToolContext::new();
            handle
                .execute_tool("hang_forever", "{}", &mut context)
                .await
        })
        .await;

        let result = timed.expect("ToolServer-registered tool hung past the safety timeout");
        let err = result.expect_err("call should time out when the server never responds");
        assert_eq!(err.kind(), ToolExecutionErrorKind::Timeout);
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

    /// `McpTool::execute` forwards an [`rmcp::model::Meta`] placed in the
    /// [`ToolContext`] as the request's `_meta`, so callers can attach per-call
    /// auth/session metadata to MCP tool invocations (the A2A use-case).
    #[tokio::test]
    async fn mcp_tool_forwards_meta_from_context() {
        use super::McpTool;
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
        let mut context = ToolContext::new();
        context.insert(meta);

        let out = mcp_tool
            .execute(&mut context, "{}")
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
        let mut empty_ctx = ToolContext::new();
        let out = mcp_tool
            .execute(&mut empty_ctx, "{}")
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

    /// `parse_mcp_arguments` accepts omitted or object arguments and rejects
    /// malformed or non-object JSON before dispatch.
    #[test]
    fn parse_mcp_arguments_classifies_inputs() {
        use super::parse_mcp_arguments;

        assert!(parse_mcp_arguments("").expect("empty is no-args").is_none());
        assert!(
            parse_mcp_arguments("   ")
                .expect("whitespace is no-args")
                .is_none()
        );
        for invalid in ["null", "[1,2]", "true", "42", r#"\"text\""#] {
            assert!(
                parse_mcp_arguments(invalid).is_err(),
                "non-object arguments must be rejected: {invalid}"
            );
        }
        assert!(parse_mcp_arguments("{}").expect("empty object").is_some());
        let obj = parse_mcp_arguments("{\"a\":1}")
            .expect("valid object")
            .expect("object present");
        assert_eq!(obj.get("a"), Some(&json!(1)));

        // Malformed JSON is a hard error, not a silent no-arg call.
        assert!(parse_mcp_arguments("{not valid json").is_err());
        assert!(parse_mcp_arguments("{\"a\":").is_err());
    }

    /// Malformed JSON arguments are classified as [`ToolExecutionErrorKind::InvalidArgs`]
    /// and short-circuit **before** the MCP server is contacted — proven by
    /// pointing the tool at a server that never responds and asserting the call
    /// returns fast with a structured invalid-args outcome instead of hanging.
    #[tokio::test]
    async fn mcp_tool_invalid_json_args_short_circuit_as_invalid_args() {
        use crate::tool::ToolExecutionErrorKind;

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

        let error = tokio::time::timeout(
            Duration::from_secs(2),
            execute_mcp(&mcp_tool, "{not valid json", &mut ToolContext::new()),
        )
        .await
        .expect(
            "invalid JSON args must be classified before contacting the (hanging) server; \
             the call reached the server and hung",
        )
        .expect_err("malformed MCP args must be an error");
        assert_eq!(error.kind(), ToolExecutionErrorKind::InvalidArgs);
        assert!(
            error.message().contains("invalid JSON"),
            "the model-visible output should explain the parse failure, got {error:?}"
        );

        server_task.abort();
    }
}
