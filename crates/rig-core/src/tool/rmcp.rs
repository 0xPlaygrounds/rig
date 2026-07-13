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
use std::sync::Arc;
use std::time::Duration;

use rmcp::ServiceExt;
use rmcp::model::ContentBlock;
use tokio::sync::RwLock;

use crate::completion::ToolDefinition;
use crate::tool::server::{ToolServerError, ToolServerHandle};
use crate::tool::{ErasedTool, ToolContext, ToolExecutionError, ToolResult};
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

/// A Rig adapter wrapping an `rmcp` MCP tool.
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
/// Returns `Ok(None)` for empty input or valid-but-non-object JSON (`null`, an
/// array, a scalar) — all of which carry no MCP arguments — and `Ok(Some(obj))`
/// for a JSON object. **Malformed JSON is a hard error** (`Err`): LLMs
/// occasionally emit invalid JSON, and it must surface as a
/// [`ToolErrorKind::InvalidArgs`] failure rather than being silently coerced
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
    /// Execute one MCP request.
    ///
    /// `meta`, when present, is attached as the MCP request's `_meta`
    /// (SEP-1319) — the idiomatic channel for per-call metadata such as auth
    /// tokens, session ids, or A2A `context_id`/`task_id`. It is supplied by a
    /// caller that places an [`rmcp::model::Meta`] into the
    /// [`ToolContext`]; otherwise the call behaves exactly as before.
    fn execute_mcp(
        &self,
        args: String,
        meta: Option<rmcp::model::Meta>,
    ) -> WasmBoxedFuture<'_, Result<String, ToolExecutionError>> {
        let name = self.definition.name.clone();

        Box::pin(async move {
            // Validate the JSON arguments before contacting the server: malformed
            // JSON must surface as an InvalidArgs failure, not a silent no-arg call.
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
            // Bound the call so a never-answered request (see issue #1914)
            // becomes a recoverable error instead of an unbounded await.
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
            // A transport/service error before the tool produced a result.
            let result = call_result.map_err(|error| {
                ToolExecutionError::provider(format!("Tool returned an error: {error}"))
                    .with_source(error)
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
                    return Err(ToolExecutionError::other(error_message));
                } else {
                    return Err(ToolExecutionError::other("No message returned"));
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
                            return Err(ToolExecutionError::other(format!(
                                "MCP tool returned unsupported resource contents: {thing:?}"
                            )));
                        }
                    },
                    ContentBlock::Audio(_) => {
                        return Err(ToolExecutionError::other(
                            "MCP tool returned audio content, which Rig does not support yet",
                        ));
                    }
                    thing => {
                        return Err(ToolExecutionError::other(format!(
                            "MCP tool returned unsupported content: {thing:?}"
                        )));
                    }
                };

                content.push_str(&chunk);
            }

            Ok(content)
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
        args: String,
        context: &'a mut ToolContext,
    ) -> WasmBoxedFuture<'a, ToolResult> {
        let meta = context.get::<rmcp::model::Meta>().cloned();
        Box::pin(async move {
            match self.execute_mcp(args, meta).await {
                Ok(output) => ToolResult::success(output),
                Err(error) => ToolResult::failed(error),
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
                handler
                    .tool_server_handle
                    .add_erased_tool(Arc::new(mcp_tool))
                    .await?;
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
            match self
                .tool_server_handle
                .add_erased_tool(Arc::new(mcp_tool))
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
    use std::{sync::Arc, time::Duration};

    use rmcp::model::*;
    use rmcp::service::RequestContext;
    use rmcp::{RoleServer, ServerHandler, ServiceExt};
    use serde_json::json;
    use tokio::{sync::RwLock, task::JoinHandle};

    use super::*;
    use crate::tool::{
        ToolErrorKind,
        server::{ToolServer, ToolServerHandle},
    };

    #[derive(Clone)]
    enum Scenario {
        Success,
        Hang,
        ServiceError,
        ToolReportedError,
    }

    #[derive(Clone)]
    struct ScenarioServer {
        scenario: Scenario,
        seen: Arc<RwLock<Option<Meta>>>,
    }

    impl ServerHandler for ScenarioServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
                .with_protocol_version(ProtocolVersion::LATEST)
                .with_server_info(Implementation::new("rig-mcp-test", "0.1.0"))
        }

        async fn call_tool(
            &self,
            _request: CallToolRequestParams,
            context: RequestContext<RoleServer>,
        ) -> Result<CallToolResult, ErrorData> {
            *self.seen.write().await = Some(context.meta.clone());
            match self.scenario {
                Scenario::Success => Ok(CallToolResult::success(vec![ContentBlock::text("ok")])),
                Scenario::Hang => std::future::pending::<Result<CallToolResult, ErrorData>>().await,
                Scenario::ServiceError => {
                    Err(ErrorData::internal_error("fixture service failed", None))
                }
                Scenario::ToolReportedError => Ok(CallToolResult::error(vec![ContentBlock::text(
                    "tool reported exact failure",
                )])),
            }
        }
    }

    struct Fixture {
        handle: ToolServerHandle,
        seen: Arc<RwLock<Option<Meta>>>,
        _client: rmcp::service::RunningService<rmcp::service::RoleClient, ClientInfo>,
        server_task: JoinHandle<()>,
    }

    async fn fixture(scenario: Scenario, timeout: Option<Duration>) -> Fixture {
        let seen = Arc::new(RwLock::new(None));
        let (client_to_server, server_from_client) = tokio::io::duplex(8192);
        let (server_to_client, client_from_server) = tokio::io::duplex(8192);
        let server = ScenarioServer {
            scenario,
            seen: seen.clone(),
        };
        let server_task = tokio::spawn(async move {
            let running = server
                .serve((server_from_client, server_to_client))
                .await
                .expect("server start");
            running.waiting().await.expect("server error");
        });
        let client = ClientInfo::default()
            .serve((client_from_server, client_to_server))
            .await
            .expect("client connect");
        let definition = Tool::new(
            "fixture_tool".to_string(),
            "fixture".to_string(),
            Arc::new(serde_json::Map::new()),
        );
        let handle = ToolServer::new()
            .rmcp_tool_with_timeout(definition, client.peer().clone(), timeout)
            .run();
        Fixture {
            handle,
            seen,
            _client: client,
            server_task,
        }
    }

    async fn execute(fixture: &Fixture, args: &str, context: &mut ToolContext) -> ToolResult {
        tokio::time::timeout(
            Duration::from_secs(5),
            fixture.handle.execute("fixture_tool", args, context),
        )
        .await
        .expect("MCP dispatch exceeded the outer safety timeout")
    }

    #[tokio::test]
    async fn canonical_dispatch_forwards_context_meta() {
        let fixture = fixture(Scenario::Success, Some(Duration::from_secs(1))).await;
        let mut meta = Meta::new();
        meta.0.insert("authorization".into(), json!("Bearer test"));
        let mut context = ToolContext::new();
        context.insert(meta);

        let result = execute(&fixture, "{}", &mut context).await;
        assert!(result.is_success());
        assert_eq!(
            fixture
                .seen
                .read()
                .await
                .as_ref()
                .expect("server observed metadata")
                .0
                .get("authorization"),
            Some(&json!("Bearer test"))
        );
        fixture.server_task.abort();
    }

    #[tokio::test]
    async fn canonical_dispatch_classifies_timeout() {
        let fixture = fixture(Scenario::Hang, Some(Duration::from_millis(25))).await;
        let result = execute(&fixture, "{}", &mut ToolContext::new()).await;
        assert!(result.is_error_kind(ToolErrorKind::Timeout));
        assert!(result.model_output().contains("timed out"));
        fixture.server_task.abort();
    }

    #[tokio::test]
    async fn canonical_dispatch_classifies_service_error_and_preserves_source() {
        let fixture = fixture(Scenario::ServiceError, Some(Duration::from_secs(1))).await;
        let result = execute(&fixture, "{}", &mut ToolContext::new()).await;
        let error = result.error().expect("structured MCP service error");
        assert_eq!(error.kind(), ToolErrorKind::Provider);
        assert!(error.is::<rmcp::ServiceError>());
        fixture.server_task.abort();
    }

    #[tokio::test]
    async fn canonical_dispatch_preserves_tool_reported_error_message() {
        let fixture = fixture(Scenario::ToolReportedError, Some(Duration::from_secs(1))).await;
        let result = execute(&fixture, "{}", &mut ToolContext::new()).await;
        assert!(result.is_error_kind(ToolErrorKind::Other));
        assert_eq!(result.model_output(), "tool reported exact failure");
        fixture.server_task.abort();
    }

    #[tokio::test]
    async fn canonical_dispatch_classifies_invalid_json_and_preserves_source() {
        let fixture = fixture(Scenario::Success, Some(Duration::from_secs(1))).await;
        let result = execute(&fixture, "{", &mut ToolContext::new()).await;
        let error = result.error().expect("structured argument error");
        assert_eq!(error.kind(), ToolErrorKind::InvalidArgs);
        assert!(error.is::<serde_json::Error>());
        fixture.server_task.abort();
    }
}

#[cfg(test)]
mod migrated_tests {
    use super::McpClientHandler;
    use crate::tool::server::ToolServer;
    use rmcp::{
        RoleServer, ServerHandler, ServiceExt, handler::client::ClientHandler, model::*,
        service::RequestContext,
    };
    use std::{sync::Arc, time::Duration};
    use tokio::sync::RwLock;

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
            _: Option<PaginatedRequestParams>,
            _: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            Ok(ListToolsResult::with_all_items(
                self.tools.read().await.clone(),
            ))
        }
        async fn call_tool(
            &self,
            request: CallToolRequestParams,
            _: RequestContext<RoleServer>,
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

    async fn connect(
        server: DynamicToolServer,
        handle: crate::tool::server::ToolServerHandle,
    ) -> (
        rmcp::service::RunningService<rmcp::RoleClient, McpClientHandler>,
        tokio::task::JoinHandle<rmcp::service::RunningService<rmcp::RoleServer, DynamicToolServer>>,
    ) {
        let (c2s, sfc) = tokio::io::duplex(8192);
        let (s2c, cfs) = tokio::io::duplex(8192);
        let server_task =
            tokio::spawn(async move { server.serve((sfc, s2c)).await.expect("server start") });
        let service = McpClientHandler::new(ClientInfo::default(), handle)
            .connect((cfs, c2s))
            .await
            .expect("connect");
        (service, server_task)
    }

    #[tokio::test]
    async fn client_handler_registers_initial_tools() {
        let server = DynamicToolServer::new(vec![
            make_tool("tool_a", "First"),
            make_tool("tool_b", "Second"),
        ]);
        let handle = ToolServer::new().run();
        let (client, task) = connect(server, handle.clone()).await;
        let defs = handle.get_tool_defs(None).await.unwrap();
        assert_eq!(
            defs.iter().map(|d| d.name.as_str()).collect::<Vec<_>>(),
            vec!["tool_a", "tool_b"]
        );
        client.cancel().await.unwrap();
        task.abort();
    }

    #[tokio::test]
    async fn client_handler_refreshes_on_tool_list_changed() {
        let server = DynamicToolServer::new(vec![make_tool("alpha", "Alpha")]);
        let handle = ToolServer::new().run();
        let (c2s, sfc) = tokio::io::duplex(8192);
        let (s2c, cfs) = tokio::io::duplex(8192);
        let copy = server.clone();
        let task = tokio::spawn(async move { copy.serve((sfc, s2c)).await.expect("server start") });
        let client = McpClientHandler::new(ClientInfo::default(), handle.clone())
            .connect((cfs, c2s))
            .await
            .unwrap();
        assert_eq!(handle.get_tool_defs(None).await.unwrap()[0].name, "alpha");
        server
            .set_tools(vec![make_tool("beta", "Beta"), make_tool("gamma", "Gamma")])
            .await;
        let running = task.await.unwrap();
        running.peer().notify_tool_list_changed().await.unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let defs = handle.get_tool_defs(None).await.unwrap();
                if defs.len() == 2 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("refresh");
        let names = handle
            .get_tool_defs(None)
            .await
            .unwrap()
            .into_iter()
            .map(|d| d.name)
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["beta", "gamma"]);
        client.cancel().await.unwrap();
    }

    #[test]
    fn client_handler_get_info_delegates() {
        let info = ClientInfo::new(
            ClientCapabilities::default(),
            Implementation::new("test-client", "1.0.0"),
        );
        let handler = McpClientHandler::new(info, ToolServer::new().run());
        let returned = handler.get_info();
        assert_eq!(returned.client_info.name, "test-client");
        assert_eq!(returned.client_info.version, "1.0.0");
    }

    #[tokio::test]
    async fn mcp_tool_preserves_provider_definition() {
        let tool = make_tool("search_docs", "Search the docs");
        let server = DynamicToolServer::new(vec![tool.clone()]);
        let (c2s, sfc) = tokio::io::duplex(8192);
        let (s2c, cfs) = tokio::io::duplex(8192);
        let task = tokio::spawn(async move {
            let running = server.serve((sfc, s2c)).await.unwrap();
            running.waiting().await.unwrap();
        });
        let client = ClientInfo::default().serve((cfs, c2s)).await.unwrap();
        let handle = ToolServer::new()
            .rmcp_tool(tool, client.peer().clone())
            .run();
        let defs = handle.get_tool_defs(None).await.unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "search_docs");
        assert_eq!(defs[0].description, "Search the docs");
        client.cancel().await.unwrap();
        task.abort();
    }
}
