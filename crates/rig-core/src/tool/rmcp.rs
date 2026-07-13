//! MCP (Model Context Protocol) integration via the `rmcp` crate.
//!
//! This module provides [`McpClientHandler`], a client handler that reacts to
//! `notifications/tools/list_changed` by re-fetching the tool list and updating
//! the [`ToolServer`](super::server::ToolServer). Individual MCP tools are
//! registered through the agent and tool-server `rmcp_tool` builder methods.
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
//! Rig's MCP adapter forwards an [`rmcp::model::Meta`] (re-exported here as
//! [`Meta`]) placed in a [`ToolContext`] as the MCP request's `_meta`
//! (SEP-1319) — the idiomatic channel for per-call values such as auth tokens,
//! session ids, or A2A `context_id`/`task_id`, which the model never sees:
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
//!
//! # Response metadata
//!
//! MCP responses retain their protocol data in the per-dispatch
//! [`ToolContext`]. Result hooks can inspect the untouched
//! [`rmcp::model::CallToolResult`], its `structuredContent` as a
//! [`serde_json::Value`], and response [`Meta`] with
//! `event.tool_context.result::<T>()`. These values are host-only; only the
//! response's ordered presentation content is sent to the model.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use rmcp::ServiceExt;
use rmcp::model::{CallToolResult, ContentBlock, ResourceContents};
use tokio::sync::RwLock;

use crate::OneOrMany;
use crate::completion::ToolDefinition;
use crate::message::{ImageMediaType, MimeType, ToolResultContent};
use crate::tool::server::{ManagedToolToken, ToolServerHandle};
use crate::tool::{ErasedTool, ToolContext, ToolExecutionError, ToolOutput, ToolResult};
use crate::wasm_compat::WasmBoxedFuture;

/// Re-export of [`rmcp::model::Meta`]: place one in a [`ToolContext`] to have
/// Rig's MCP registration methods forward it as a call's `_meta`.
pub use rmcp::model::Meta;

/// Default per-call timeout applied to MCP tools (see issue #1914).
///
/// MCP tool calls await a response that can be silently lost by the transport
/// (e.g. an rmcp StreamableHttp session re-init dropping an in-flight request),
/// which would otherwise hang the agent forever. A generous default bounds that
/// without disrupting normal, long-running tools. The agent and tool-server
/// `rmcp_tool_with_timeout` builders can override or disable it.
pub const DEFAULT_MCP_TOOL_TIMEOUT: Duration = Duration::from_secs(300);

/// Crate-private adapter used by Rig's public MCP registration methods.
#[derive(Clone)]
pub(crate) struct McpTool {
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
    /// Create an adapter from an MCP tool definition and server sink.
    ///
    /// Applies [`DEFAULT_MCP_TOOL_TIMEOUT`] so a lost/never-answered response
    /// cannot hang the agent forever (issue #1914).
    pub(crate) fn from_mcp_server(
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
    /// send a cancellation to the MCP server.
    pub(crate) fn with_timeout(mut self, timeout: impl Into<Option<Duration>>) -> Self {
        self.timeout = timeout.into();
        self
    }

    /// The per-call timeout, if any.
    #[cfg(test)]
    pub(crate) fn timeout(&self) -> Option<Duration> {
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
    ) -> WasmBoxedFuture<'_, Result<CallToolResult, ToolExecutionError>> {
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
            call_result.map_err(|error| {
                ToolExecutionError::provider(format!(
                    "MCP tool '{}' request failed: {error}",
                    self.definition.name
                ))
                .with_source(error)
            })
        })
    }
}

fn mcp_content_block_as_json(
    content: &ContentBlock,
) -> Result<ToolResultContent, ToolExecutionError> {
    serde_json::to_value(content)
        .map(ToolResultContent::json)
        .map_err(|error| {
            ToolExecutionError::provider(format!(
                "failed to preserve an MCP content block as JSON: {error}"
            ))
            .with_source(error)
        })
}

fn mcp_content_block_to_tool_content(
    content: &ContentBlock,
) -> Result<ToolResultContent, ToolExecutionError> {
    match content {
        ContentBlock::Text(text) => Ok(ToolResultContent::text(text.text.clone())),
        ContentBlock::Image(image) => match ImageMediaType::from_mime_type(&image.mime_type) {
            Some(media_type) => Ok(ToolResultContent::image_base64(
                image.data.clone(),
                Some(media_type),
                None,
            )),
            None => mcp_content_block_as_json(content),
        },
        ContentBlock::Resource(resource) => match &resource.resource {
            // Rig has no resource-content variant. Serializing the complete MCP
            // block keeps its URI, MIME type, metadata, annotations, and body
            // together instead of presenting only the body to the model.
            ResourceContents::TextResourceContents { .. } => mcp_content_block_as_json(content),
            ResourceContents::BlobResourceContents {
                mime_type, blob, ..
            } => match mime_type
                .as_deref()
                .and_then(ImageMediaType::from_mime_type)
            {
                Some(media_type) => Ok(ToolResultContent::image_base64(
                    blob.clone(),
                    Some(media_type),
                    None,
                )),
                _ => mcp_content_block_as_json(content),
            },
            _ => mcp_content_block_as_json(content),
        },
        ContentBlock::ResourceLink(_) | ContentBlock::Audio(_) => {
            mcp_content_block_as_json(content)
        }
        // ContentBlock is non-exhaustive. Preserve future protocol variants in
        // full rather than replacing them with a lossy placeholder.
        _ => mcp_content_block_as_json(content),
    }
}

/// Build the model presentation without flattening or reparsing MCP blocks.
fn mcp_result_output(result: &CallToolResult) -> Result<ToolOutput, ToolExecutionError> {
    let mut content = result.content.iter().map(mcp_content_block_to_tool_content);

    if let Some(first) = content.next() {
        let mut ordered = OneOrMany::one(first?);
        for block in content {
            ordered.push(block?);
        }
        return Ok(ToolOutput::content(ordered));
    }

    if let Some(structured) = result.structured_content.clone() {
        // `structuredContent` is explicitly typed JSON by MCP. Preserve that
        // type even when its JSON value happens to be a string.
        return Ok(ToolOutput::Json(structured));
    }

    if result.is_error == Some(true) {
        Ok(ToolOutput::text("the MCP tool reported an error"))
    } else {
        Ok(ToolOutput::text(""))
    }
}

fn preserve_mcp_result(context: &mut ToolContext, result: &CallToolResult) {
    if let Some(structured) = result.structured_content.clone() {
        context.insert_result(structured);
    }
    if let Some(meta) = result.meta.clone() {
        context.insert_result(meta);
    }
    context.insert_result(result.clone());
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
                Ok(result) => {
                    let is_error = result.is_error == Some(true);
                    preserve_mcp_result(context, &result);
                    let output = match mcp_result_output(&result) {
                        Ok(output) => output,
                        Err(error) => return ToolResult::failed(error),
                    };

                    if is_error {
                        ToolResult::failed_with_output(
                            ToolExecutionError::other(format!(
                                "MCP tool '{}' reported an execution error",
                                self.definition.name
                            )),
                            output,
                        )
                    } else {
                        ToolResult::success(output)
                    }
                }
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
}

/// An MCP client handler that automatically re-fetches the tool list when the
/// server sends a `notifications/tools/list_changed` notification.
///
/// This handler implements [`rmcp::ClientHandler`] and bridges the MCP
/// notification lifecycle with Rig's [`ToolServer`](super::server::ToolServer).
/// When the MCP server's available tools change, this handler:
/// 1. Re-fetches the full tool list from the MCP server
/// 2. Replaces or removes registrations still owned by this handler
/// 3. Leaves newer local and peer-handler same-name registrations intact
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
    /// Tracks the exact registry generation installed for each tool. Refreshes
    /// only mutate a name while this generation remains current, so a newer
    /// local or peer-handler registration cannot be deleted or overwritten.
    managed_tools: Arc<RwLock<HashMap<String, ManagedToolToken>>>,
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
            managed_tools: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Set (or clear) the per-call timeout applied to every MCP tool this handler
    /// registers. Pass a [`Duration`] to bound calls, or `None` to disable.
    ///
    /// This applies the same setting to every tool managed by the handler.
    pub fn with_timeout(mut self, timeout: impl Into<Option<Duration>>) -> Self {
        self.timeout = timeout.into();
        self
    }

    /// Build the internal MCP adapter with this handler's configured timeout.
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
    /// Returns [`McpClientError`] if the connection or initial tool fetch fails.
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

        {
            let handler = service.service();
            // Serialize the initial fetch with list-changed callbacks. Without
            // this guard, a callback can install a newer list while this fetch
            // is in flight, only for connect to overwrite it with stale tools.
            let mut managed = handler.managed_tools.write().await;
            let tools = service.peer().list_all_tools().await?;
            let tools = tools
                .into_iter()
                .map(|tool| {
                    Arc::new(handler.build_tool(tool, service.peer().clone()))
                        as Arc<dyn ErasedTool>
                })
                .collect();
            // Use the same managed-registry -> tool-server lock order as
            // refreshes. A list-changed notification cannot observe installed
            // tools before their ownership generations are recorded.
            let registered = handler
                .tool_server_handle
                .add_managed_erased_tools(tools)
                .await;
            *managed = registered;
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
        // Serialize the fetch as well as the registry commit. If two
        // notifications fetch concurrently, an older, slower response can
        // otherwise commit after a newer response and roll the registry back.
        let mut managed = self.managed_tools.write().await;
        let tools = match context.peer.list_all_tools().await {
            Ok(tools) => tools,
            Err(e) => {
                tracing::error!("Failed to re-fetch MCP tool list: {e}");
                return;
            }
        };

        let tools = tools
            .into_iter()
            .map(|tool| {
                Arc::new(self.build_tool(tool, context.peer.clone())) as Arc<dyn ErasedTool>
            })
            .collect();
        // Keep the current generations recorded until reconciliation commits.
        // If this notification future is cancelled while awaiting the server
        // lock, the next refresh can still update or remove its registrations.
        let expected = managed.clone();
        *managed = self
            .tool_server_handle
            .reconcile_managed_erased_tools(expected, tools)
            .await;

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
    use crate::message::ToolResultContent as RigToolResultContent;
    use crate::tool::{
        ToolErrorKind,
        server::{ToolServer, ToolServerHandle},
    };

    #[derive(Clone)]
    enum Scenario {
        Success,
        StructuredSuccess,
        StructuredOnly,
        Hang,
        ServiceError,
        ToolReportedError,
        ImageToolReportedError,
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
                Scenario::StructuredSuccess => {
                    let mut response = CallToolResult::success(vec![
                        ContentBlock::text("before"),
                        ContentBlock::image("aGVsbG8=", "image/png"),
                        ContentBlock::text("after"),
                    ]);
                    response.structured_content = Some(json!({
                        "answer": 42,
                        "source": "fixture"
                    }));
                    let mut meta = Meta::new();
                    meta.0.insert("response-id".into(), json!("response-123"));
                    response.meta = Some(meta);
                    Ok(response)
                }
                Scenario::StructuredOnly => {
                    let mut response = CallToolResult::structured(json!({"answer": 42}));
                    response.content.clear();
                    Ok(response)
                }
                Scenario::Hang => std::future::pending::<Result<CallToolResult, ErrorData>>().await,
                Scenario::ServiceError => {
                    Err(ErrorData::internal_error("fixture service failed", None))
                }
                Scenario::ToolReportedError => Ok(CallToolResult::error(vec![ContentBlock::text(
                    "tool reported exact failure",
                )])),
                Scenario::ImageToolReportedError => {
                    Ok(CallToolResult::error(vec![ContentBlock::image(
                        "ZXJyb3ItaW1hZ2U=",
                        "image/png",
                    )]))
                }
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

    #[test]
    fn model_presentation_preserves_unrepresentable_mcp_blocks_as_json() {
        let blocks = vec![
            ContentBlock::resource(ResourceContents::TextResourceContents {
                uri: "file:///reports/summary.txt".to_string(),
                mime_type: Some("text/plain".to_string()),
                text: "full report".to_string(),
                meta: None,
            }),
            ContentBlock::resource(ResourceContents::BlobResourceContents {
                uri: "file:///reports/raw.bin".to_string(),
                mime_type: Some("application/octet-stream".to_string()),
                blob: "AAEC".to_string(),
                meta: None,
            }),
            ContentBlock::audio("UklGRg==", "audio/wav"),
            ContentBlock::resource_link(
                Resource::new("file:///reports/linked.txt", "linked.txt")
                    .with_mime_type("text/plain"),
            ),
            ContentBlock::image("YXZpZg==", "image/avif"),
            ContentBlock::resource(ResourceContents::BlobResourceContents {
                uri: "file:///images/chart.avif".to_string(),
                mime_type: Some("image/avif".to_string()),
                blob: "YmxvYi1hdmlm".to_string(),
                meta: None,
            }),
        ];
        let expected = blocks
            .iter()
            .map(|block| {
                RigToolResultContent::json(
                    serde_json::to_value(block).expect("MCP block is JSON serializable"),
                )
            })
            .collect::<Vec<_>>();

        let result = CallToolResult::success(blocks);
        let content = mcp_result_output(&result)
            .expect("MCP content mapping")
            .into_content()
            .into_iter()
            .collect::<Vec<_>>();

        assert_eq!(content, expected);
        assert!(matches!(
            &content[0],
            RigToolResultContent::Json { value }
                if value["resource"]["uri"] == "file:///reports/summary.txt"
                    && value["resource"]["mimeType"] == "text/plain"
                    && value["resource"]["text"] == "full report"
        ));
        assert!(matches!(
            &content[1],
            RigToolResultContent::Json { value }
                if value["resource"]["uri"] == "file:///reports/raw.bin"
                    && value["resource"]["mimeType"] == "application/octet-stream"
                    && value["resource"]["blob"] == "AAEC"
        ));
        assert!(matches!(
            &content[2],
            RigToolResultContent::Json { value }
                if value["mimeType"] == "audio/wav" && value["data"] == "UklGRg=="
        ));
        assert!(matches!(
            &content[4],
            RigToolResultContent::Json { value }
                if value["mimeType"] == "image/avif" && value["data"] == "YXZpZg=="
        ));
        assert!(matches!(
            &content[5],
            RigToolResultContent::Json { value }
                if value["resource"]["uri"] == "file:///images/chart.avif"
                    && value["resource"]["mimeType"] == "image/avif"
                    && value["resource"]["blob"] == "YmxvYi1hdmlm"
        ));
    }

    #[test]
    fn image_resource_blob_maps_to_an_image_block() {
        let result = CallToolResult::success(vec![ContentBlock::resource(
            ResourceContents::BlobResourceContents {
                uri: "file:///images/chart.png".to_string(),
                mime_type: Some("image/png".to_string()),
                blob: "aW1hZ2U=".to_string(),
                meta: None,
            },
        )]);

        assert_eq!(
            mcp_result_output(&result).expect("MCP content mapping"),
            ToolOutput::one(RigToolResultContent::image_base64(
                "aW1hZ2U=",
                Some(ImageMediaType::PNG),
                None,
            ))
        );
    }

    #[test]
    fn string_valued_structured_content_remains_json() {
        let mut result = CallToolResult::structured(json!("forty-two"));
        result.content.clear();

        assert_eq!(
            mcp_result_output(&result).expect("MCP content mapping"),
            ToolOutput::Json(json!("forty-two"))
        );
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
        assert_eq!(result.output().as_text(), Some("tool execution timed out"));
        fixture.server_task.abort();
    }

    #[tokio::test]
    async fn canonical_dispatch_classifies_service_error_and_preserves_source() {
        let fixture = fixture(Scenario::ServiceError, Some(Duration::from_secs(1))).await;
        let result = execute(&fixture, "{}", &mut ToolContext::new()).await;
        let error = result.error().expect("structured MCP service error");
        assert_eq!(error.kind(), ToolErrorKind::Provider);
        assert!(error.is::<rmcp::ServiceError>());
        assert!(error.message().contains("fixture service failed"));
        assert_eq!(result.output().as_text(), Some("the tool provider failed"));
        assert!(!result.output().render().contains("fixture service failed"));
        fixture.server_task.abort();
    }

    #[tokio::test]
    async fn canonical_dispatch_preserves_tool_reported_error_message() {
        let fixture = fixture(Scenario::ToolReportedError, Some(Duration::from_secs(1))).await;
        let result = execute(&fixture, "{}", &mut ToolContext::new()).await;
        assert!(result.is_error_kind(ToolErrorKind::Other));
        assert_eq!(
            result.output(),
            &ToolOutput::one(RigToolResultContent::text("tool reported exact failure"))
        );
        assert_eq!(
            result.error().map(ToolExecutionError::message),
            Some("MCP tool 'fixture_tool' reported an execution error")
        );
        fixture.server_task.abort();
    }

    #[tokio::test]
    async fn canonical_dispatch_preserves_non_text_tool_error_content() {
        let fixture = fixture(
            Scenario::ImageToolReportedError,
            Some(Duration::from_secs(1)),
        )
        .await;
        let mut context = ToolContext::new();
        let result = execute(&fixture, "{}", &mut context).await;

        assert!(result.is_error_kind(ToolErrorKind::Other));
        assert_eq!(
            result.output(),
            &ToolOutput::one(RigToolResultContent::image_base64(
                "ZXJyb3ItaW1hZ2U=",
                Some(ImageMediaType::PNG),
                None,
            ))
        );
        let raw = context
            .result::<CallToolResult>()
            .expect("raw MCP error result metadata");
        assert_eq!(raw.is_error, Some(true));
        assert!(matches!(raw.content.as_slice(), [ContentBlock::Image(_)]));
        fixture.server_task.abort();
    }

    #[tokio::test]
    async fn canonical_dispatch_preserves_ordered_content_and_response_metadata() {
        let fixture = fixture(Scenario::StructuredSuccess, Some(Duration::from_secs(1))).await;
        let mut context = ToolContext::new();
        let result = execute(&fixture, "{}", &mut context).await;

        let mut expected_content = OneOrMany::one(RigToolResultContent::text("before"));
        expected_content.push(RigToolResultContent::image_base64(
            "aGVsbG8=",
            Some(ImageMediaType::PNG),
            None,
        ));
        expected_content.push(RigToolResultContent::text("after"));
        assert_eq!(result.output(), &ToolOutput::content(expected_content));

        let raw = context
            .result::<CallToolResult>()
            .expect("raw MCP result metadata");
        assert_eq!(raw.content.len(), 3);
        assert_eq!(
            raw.structured_content,
            Some(json!({"answer": 42, "source": "fixture"}))
        );
        assert_eq!(
            context.result::<serde_json::Value>(),
            Some(&json!({"answer": 42, "source": "fixture"}))
        );
        assert_eq!(
            context
                .result::<Meta>()
                .and_then(|meta| meta.0.get("response-id")),
            Some(&json!("response-123"))
        );
        fixture.server_task.abort();
    }

    #[tokio::test]
    async fn canonical_dispatch_uses_structured_content_when_blocks_are_empty() {
        let fixture = fixture(Scenario::StructuredOnly, Some(Duration::from_secs(1))).await;
        let mut context = ToolContext::new();
        let result = execute(&fixture, "{}", &mut context).await;

        assert_eq!(result.output(), &ToolOutput::json(json!({"answer": 42})));
        assert_eq!(
            context.result::<serde_json::Value>(),
            Some(&json!({"answer": 42}))
        );
        fixture.server_task.abort();
    }

    #[tokio::test]
    async fn canonical_dispatch_classifies_invalid_json_and_preserves_source() {
        let fixture = fixture(Scenario::Success, Some(Duration::from_secs(1))).await;
        let result = execute(&fixture, "{", &mut ToolContext::new()).await;
        let error = result.error().expect("structured argument error");
        assert_eq!(error.kind(), ToolErrorKind::InvalidArgs);
        assert!(error.is::<serde_json::Error>());
        assert_eq!(
            result.output().as_text(),
            Some("tool arguments were invalid")
        );
        fixture.server_task.abort();
    }
}

#[cfg(test)]
mod migrated_tests {
    use super::McpClientHandler;
    use crate::tool::{DynamicTool, ToolOutput, server::ToolServer};
    use rmcp::{
        RoleServer, ServerHandler, ServiceExt, handler::client::ClientHandler, model::*,
        service::RequestContext,
    };
    use std::{
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        time::Duration,
    };
    use tokio::sync::{Notify, RwLock};

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

    #[derive(Clone)]
    struct OrderedRefreshServer {
        tools: Arc<RwLock<Vec<Tool>>>,
        list_calls: Arc<AtomicUsize>,
        first_refresh_started: Arc<Notify>,
        release_first_refresh: Arc<Notify>,
    }

    impl OrderedRefreshServer {
        fn new(tools: Vec<Tool>) -> Self {
            Self {
                tools: Arc::new(RwLock::new(tools)),
                list_calls: Arc::new(AtomicUsize::new(0)),
                first_refresh_started: Arc::new(Notify::new()),
                release_first_refresh: Arc::new(Notify::new()),
            }
        }

        async fn set_tools(&self, tools: Vec<Tool>) {
            *self.tools.write().await = tools;
        }
    }

    impl ServerHandler for OrderedRefreshServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
                .with_protocol_version(ProtocolVersion::LATEST)
                .with_server_info(Implementation::new("test-ordered-refresh-server", "0.1.0"))
        }

        async fn list_tools(
            &self,
            _: Option<PaginatedRequestParams>,
            _: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            let call = self.list_calls.fetch_add(1, Ordering::SeqCst);
            let tools = self.tools.read().await.clone();

            // Call zero is connect's initial fetch. Hold the first notification's
            // stale snapshot so a second notification is concurrent with it.
            if call == 1 {
                self.first_refresh_started.notify_one();
                self.release_first_refresh.notified().await;
            }

            Ok(ListToolsResult::with_all_items(tools))
        }
    }

    fn make_tool(name: &str, description: &str) -> Tool {
        Tool::new(
            name.to_string(),
            description.to_string(),
            Arc::new(serde_json::Map::new()),
        )
    }

    fn make_dynamic_tool(name: &str, description: &str) -> DynamicTool {
        DynamicTool::new(
            name,
            description,
            serde_json::json!({"type": "object", "properties": {}}),
            |_context, _args| Box::pin(async { Ok(ToolOutput::text("local")) }),
        )
    }

    async fn connect<S>(
        server: S,
        handle: crate::tool::server::ToolServerHandle,
    ) -> (
        rmcp::service::RunningService<rmcp::RoleClient, McpClientHandler>,
        tokio::task::JoinHandle<rmcp::service::RunningService<rmcp::RoleServer, S>>,
    )
    where
        S: ServerHandler,
    {
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

    #[tokio::test]
    async fn concurrent_refreshes_cannot_roll_back_a_newer_tool_list() {
        let server = OrderedRefreshServer::new(vec![make_tool("stale", "Stale snapshot")]);
        let server_control = server.clone();
        let handle = ToolServer::new().run();
        let (client, server_task) = connect(server, handle.clone()).await;
        let running_server = server_task.await.unwrap();

        running_server
            .peer()
            .notify_tool_list_changed()
            .await
            .unwrap();
        tokio::time::timeout(
            Duration::from_secs(2),
            server_control.first_refresh_started.notified(),
        )
        .await
        .expect("first refresh fetch started");

        // The refresh must own the managed-registry lock before it fetches.
        // Otherwise another notification can fetch and commit a newer list,
        // then be rolled back when this deliberately delayed response arrives.
        assert!(
            client.service().managed_tools.try_write().is_err(),
            "the first refresh fetched without serializing registry commits"
        );

        server_control
            .set_tools(vec![make_tool("newest", "Newest snapshot")])
            .await;
        running_server
            .peer()
            .notify_tool_list_changed()
            .await
            .unwrap();
        server_control.release_first_refresh.notify_one();

        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let defs = handle.get_tool_defs(None).await.unwrap();
                if defs.len() == 1 && defs[0].name == "newest" {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("newest refresh committed after delayed stale refresh");

        assert_eq!(server_control.list_calls.load(Ordering::SeqCst), 3);
        client.cancel().await.unwrap();
    }

    #[tokio::test]
    async fn refresh_does_not_replace_a_newer_local_registration() {
        let server = DynamicToolServer::new(vec![make_tool("alpha", "MCP alpha")]);
        let server_control = server.clone();
        let handle = ToolServer::new().run();
        let (client, server_task) = connect(server, handle.clone()).await;

        handle
            .add_dynamic_tool(make_dynamic_tool("alpha", "Local alpha"))
            .await;
        server_control
            .set_tools(vec![make_tool("refresh_complete", "Refresh sentinel")])
            .await;
        let running_server = server_task.await.unwrap();
        running_server
            .peer()
            .notify_tool_list_changed()
            .await
            .unwrap();

        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let defs = handle.get_tool_defs(None).await.unwrap();
                if defs
                    .iter()
                    .any(|definition| definition.name == "refresh_complete")
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("MCP refresh completed");

        let defs = handle.get_tool_defs(None).await.unwrap();
        let alpha = defs
            .iter()
            .find(|definition| definition.name == "alpha")
            .expect("alpha remains registered");
        assert_eq!(alpha.description, "Local alpha");

        let result = handle
            .execute("alpha", "{}", &mut crate::tool::ToolContext::new())
            .await;
        assert_eq!(result.output(), &ToolOutput::text("local"));
        client.cancel().await.unwrap();
    }

    #[tokio::test]
    async fn one_handler_refresh_does_not_replace_a_newer_peer_handler_registration() {
        let server_a = DynamicToolServer::new(vec![make_tool("alpha", "Handler A")]);
        let server_a_control = server_a.clone();
        let server_b = DynamicToolServer::new(vec![make_tool("alpha", "Handler B")]);
        let handle = ToolServer::new().run();

        let (client_a, server_task_a) = connect(server_a, handle.clone()).await;
        let (client_b, server_task_b) = connect(server_b, handle.clone()).await;

        server_a_control
            .set_tools(vec![
                make_tool("alpha", "Refreshed handler A"),
                make_tool("a_refresh_complete", "Refresh sentinel"),
            ])
            .await;
        let running_server_a = server_task_a.await.unwrap();
        let _running_server_b = server_task_b.await.unwrap();
        running_server_a
            .peer()
            .notify_tool_list_changed()
            .await
            .unwrap();

        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let defs = handle.get_tool_defs(None).await.unwrap();
                if defs
                    .iter()
                    .any(|definition| definition.name == "a_refresh_complete")
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("handler A refresh completed");

        let defs = handle.get_tool_defs(None).await.unwrap();
        let alpha = defs
            .iter()
            .find(|definition| definition.name == "alpha")
            .expect("alpha remains registered");
        assert_eq!(alpha.description, "Handler B");

        client_a.cancel().await.unwrap();
        client_b.cancel().await.unwrap();
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
