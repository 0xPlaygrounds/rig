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
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::Duration;

use rmcp::ServiceExt;
use rmcp::model::{
    CallToolRequest, CallToolResult, ClientRequest, ContentBlock, ListToolsRequest,
    PaginatedRequestParams, ResourceContents, ServerResult,
};
use rmcp::service::PeerRequestOptions;
use tokio::sync::{Mutex, RwLock};

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

/// Default deadline for fetching an MCP server's complete tool list.
///
/// Refreshes are versioned as well as bounded: a slow older fetch may finish,
/// but it can never roll the registry back after a newer snapshot commits.
pub const DEFAULT_MCP_REFRESH_TIMEOUT: Duration = Duration::from_secs(30);

/// Crate-private adapter used by Rig's public MCP registration methods.
#[derive(Clone)]
pub(crate) struct McpTool {
    definition: rmcp::model::Tool,
    client: rmcp::service::ServerSink,
    /// Per-call timeout. When `Some`, an MCP `call_tool` that does not complete
    /// within this duration resolves to a [`ToolExecutionError`] instead of blocking
    /// forever (see issue #1914). When `None`, the call is unbounded.
    ///
    /// On elapse RMCP sends a cancellation notification so both peers can
    /// release request-scoped resources.
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
    /// than hang). RMCP sends a cancellation notification when the deadline
    /// elapses.
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
/// Argument decoding failure at the MCP object boundary.
#[derive(Debug, thiserror::Error)]
enum McpArgumentError {
    /// Malformed JSON.
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// Valid JSON that cannot be represented by MCP's object-valued arguments.
    #[error("expected a JSON object or null, got {0}")]
    NonObject(&'static str),
}

fn json_value_kind(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Returns no argument map for empty input or explicit JSON `null`, and an MCP
/// argument map for a JSON object. Other valid JSON shapes are rejected: silently
/// turning an array or scalar into a no-argument request can execute a different
/// operation than the model requested.
fn parse_mcp_arguments(args: &str) -> Result<Option<rmcp::model::JsonObject>, McpArgumentError> {
    let trimmed = args.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let value: serde_json::Value = serde_json::from_str(trimmed)?;
    match value {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Object(_) => Ok(Some(serde_json::from_value(value)?)),
        value => Err(McpArgumentError::NonObject(json_value_kind(&value))),
    }
}

async fn call_mcp_tool(
    peer: &rmcp::service::ServerSink,
    params: rmcp::model::CallToolRequestParams,
    timeout: Option<Duration>,
) -> Result<CallToolResult, rmcp::ServiceError> {
    let deadline = timeout.map(|timeout| (tokio::time::Instant::now() + timeout, timeout));
    let response = send_mcp_request(
        peer,
        ClientRequest::CallToolRequest(CallToolRequest::new(params)),
        deadline,
    )
    .await?;

    match response {
        ServerResult::CallToolResult(result) => Ok(result),
        _ => Err(rmcp::ServiceError::UnexpectedResponse),
    }
}

async fn send_mcp_request(
    peer: &rmcp::service::ServerSink,
    request: ClientRequest,
    deadline: Option<(tokio::time::Instant, Duration)>,
) -> Result<ServerResult, rmcp::ServiceError> {
    let handle = match deadline {
        Some((deadline, timeout)) => {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return Err(rmcp::ServiceError::Timeout { timeout });
            }
            crate::wasm_compat::timeout(
                remaining,
                peer.send_cancellable_request(request, PeerRequestOptions::no_options()),
            )
            .await
            .map_err(|_| rmcp::ServiceError::Timeout { timeout })??
        }
        None => {
            peer.send_cancellable_request(request, PeerRequestOptions::no_options())
                .await?
        }
    };

    let Some((deadline, timeout)) = deadline else {
        return handle.await_response().await;
    };
    let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
    let mut handle = handle;
    match crate::wasm_compat::timeout(remaining, &mut handle.rx).await {
        Ok(response) => response.map_err(|_| rmcp::ServiceError::TransportClosed)?,
        Err(_) => {
            cancel_timed_out_request(handle);
            Err(rmcp::ServiceError::Timeout { timeout })
        }
    }
}

/// Keep cancellation delivery out of the caller's deadline. RMCP's cancellation
/// notification uses the same bounded outbound queue as requests, so awaiting it
/// inline could exceed the timeout precisely when that queue is saturated.
fn cancel_timed_out_request(handle: rmcp::service::RequestHandle<rmcp::service::RoleClient>) {
    let cancellation = async move {
        let _ = handle
            .cancel(Some(
                rmcp::service::RequestHandle::<rmcp::service::RoleClient>::REQUEST_TIMEOUT_REASON
                    .to_owned(),
            ))
            .await;
    };

    #[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
    tokio::spawn(cancellation);

    #[cfg(all(feature = "wasm", target_arch = "wasm32"))]
    wasm_bindgen_futures::spawn_local(cancellation);
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
                    "MCP tool '{name}' received invalid arguments: {error}"
                ))
                .with_source(error)
            })?;
            let mut request = arguments
                .map(|arguments| {
                    rmcp::model::CallToolRequestParams::new(name.clone()).with_arguments(arguments)
                })
                .unwrap_or_else(|| rmcp::model::CallToolRequestParams::new(name));
            request.meta = meta;

            match call_mcp_tool(&self.client, request, self.timeout).await {
                Ok(result) => Ok(result),
                Err(
                    error @ rmcp::ServiceError::Timeout {
                        timeout: elapsed_timeout,
                    },
                ) => {
                    let timeout = self.timeout.unwrap_or(elapsed_timeout);
                    Err(ToolExecutionError::timeout(format!(
                        "MCP tool '{}' timed out after {timeout:?}",
                        self.definition.name
                    ))
                    .with_source(error))
                }
                // A transport/service error before the tool produced a result.
                Err(error) => Err(ToolExecutionError::provider(format!(
                    "MCP tool '{}' request failed: {error}",
                    self.definition.name
                ))
                .with_source(error)),
            }
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
    let structured = result.structured_content.as_ref();
    let canonical_fallback = structured.map(serde_json::Value::to_string);
    let mut replaced_fallback = false;
    let mut mapped = Vec::with_capacity(result.content.len());

    for block in &result.content {
        let fallback_structured = if !replaced_fallback {
            match (block, canonical_fallback.as_deref(), structured) {
                (ContentBlock::Text(text), Some(fallback), Some(structured))
                    if text.text == fallback =>
                {
                    Some(structured)
                }
                _ => None,
            }
        } else {
            None
        };
        if let Some(structured) = fallback_structured {
            // rmcp's `structured`/`structured_error` constructors include this
            // text block solely for older clients. Replace it in place with the
            // typed value; do not duplicate it as model-visible text.
            mapped.push(ToolResultContent::json(structured.clone()));
            replaced_fallback = true;
        } else {
            mapped.push(mcp_content_block_to_tool_content(block)?);
        }
    }

    if let Some(structured) = structured
        && !replaced_fallback
    {
        // A server may provide genuine text/rich content in addition to its
        // structured result. Keep every real block and place the typed value
        // first deterministically; only the canonical compatibility text is
        // replaced rather than duplicated.
        mapped.insert(0, ToolResultContent::json(structured.clone()));
    }

    let mut mapped = mapped.into_iter();
    if let Some(first) = mapped.next() {
        let mut ordered = OneOrMany::one(first);
        for block in mapped {
            ordered.push(block);
        }
        return Ok(ToolOutput::content(ordered));
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

    fn is_live(&self) -> bool {
        !self.client.is_transport_closed()
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
                        ToolResult::failed(
                            ToolExecutionError::other(format!(
                                "MCP tool '{}' reported an execution error",
                                self.definition.name
                            ))
                            .with_model_output(output),
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

    /// The server did not finish returning its tool list before the deadline.
    #[error("Timed out fetching MCP tool list after {0:?}")]
    ToolFetchTimeout(Duration),
}

#[derive(Default)]
struct ManagedToolsState {
    registrations: HashMap<String, ManagedToolToken>,
    committed_refresh: u64,
}

#[derive(Default)]
struct RefreshActivity {
    active: usize,
    dirty: bool,
}

const MAX_CONCURRENT_REFRESHES: usize = 2;

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
    /// Deadline for initial and list-changed tool-list fetches.
    refresh_timeout: Duration,
    /// Tracks the exact registry generation installed for each tool. Refreshes
    /// only mutate a name while this generation remains current, so a newer
    /// local or peer-handler registration cannot be deleted or overwritten.
    managed_tools: Arc<RwLock<ManagedToolsState>>,
    /// Bounds notification-driven list fetches and coalesces excess signals.
    refresh_activity: Arc<Mutex<RefreshActivity>>,
    /// Monotonic identity assigned when each tool-list fetch begins.
    next_refresh: Arc<AtomicU64>,
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
            refresh_timeout: DEFAULT_MCP_REFRESH_TIMEOUT,
            managed_tools: Arc::new(RwLock::new(ManagedToolsState::default())),
            refresh_activity: Arc::new(Mutex::new(RefreshActivity::default())),
            next_refresh: Arc::new(AtomicU64::new(0)),
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

    /// Set the deadline for initial and list-changed tool-list fetches.
    pub fn with_refresh_timeout(mut self, timeout: Duration) -> Self {
        self.refresh_timeout = timeout;
        self
    }

    /// Build the internal MCP adapter with this handler's configured timeout.
    fn build_tool(&self, tool: rmcp::model::Tool, client: rmcp::service::ServerSink) -> McpTool {
        McpTool::from_mcp_server(tool, client).with_timeout(self.timeout)
    }

    fn begin_refresh(&self) -> u64 {
        self.next_refresh.fetch_add(1, Ordering::SeqCst) + 1
    }

    async fn fetch_tools(
        &self,
        peer: &rmcp::service::ServerSink,
    ) -> Result<Vec<Arc<dyn ErasedTool>>, McpClientError> {
        let deadline = tokio::time::Instant::now() + self.refresh_timeout;
        let mut tools = Vec::new();
        let mut cursor = None;

        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return Err(McpClientError::ToolFetchTimeout(self.refresh_timeout));
            }
            let mut params = PaginatedRequestParams::default();
            params.cursor = cursor;
            let response = send_mcp_request(
                peer,
                ClientRequest::ListToolsRequest(ListToolsRequest::with_param(params)),
                Some((deadline, self.refresh_timeout)),
            )
            .await
            .map_err(|error| match error {
                rmcp::ServiceError::Timeout { .. } => {
                    McpClientError::ToolFetchTimeout(self.refresh_timeout)
                }
                error => McpClientError::ToolFetchError(error),
            })?;
            let page = match response {
                ServerResult::ListToolsResult(page) => page,
                _ => {
                    return Err(McpClientError::ToolFetchError(
                        rmcp::ServiceError::UnexpectedResponse,
                    ));
                }
            };
            tools.extend(page.tools);
            cursor = page.next_cursor;
            if cursor.is_none() {
                break;
            }
        }

        Ok(tools
            .into_iter()
            .map(|tool| Arc::new(self.build_tool(tool, peer.clone())) as Arc<dyn ErasedTool>)
            .collect())
    }

    async fn try_start_refresh(&self) -> bool {
        let mut activity = self.refresh_activity.lock().await;
        if activity.active >= MAX_CONCURRENT_REFRESHES {
            activity.dirty = true;
            false
        } else {
            activity.active += 1;
            true
        }
    }

    async fn finish_or_restart_refresh(&self) -> bool {
        let mut activity = self.refresh_activity.lock().await;
        if activity.dirty {
            activity.dirty = false;
            true
        } else {
            activity.active -= 1;
            false
        }
    }

    async fn commit_initial(&self, refresh: u64, tools: Vec<Arc<dyn ErasedTool>>) {
        let mut managed = self.managed_tools.write().await;
        if refresh <= managed.committed_refresh {
            tracing::debug!(refresh, "discarding stale initial MCP tool list");
            return;
        }
        managed.registrations = self
            .tool_server_handle
            .add_managed_erased_tools(tools)
            .await;
        managed.committed_refresh = refresh;
    }

    async fn commit_refresh(&self, refresh: u64, tools: Vec<Arc<dyn ErasedTool>>) -> bool {
        let mut managed = self.managed_tools.write().await;
        if refresh <= managed.committed_refresh {
            tracing::debug!(refresh, "discarding stale MCP tool-list response");
            return false;
        }
        let expected = managed.registrations.clone();
        managed.registrations = self
            .tool_server_handle
            .reconcile_managed_erased_tools(expected, tools)
            .await;
        managed.committed_refresh = refresh;
        true
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

        let handler = service.service();
        let refresh = handler.begin_refresh();
        let tools = handler.fetch_tools(service.peer()).await?;
        handler.commit_initial(refresh, tools).await;

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
        if !self.try_start_refresh().await {
            return;
        }

        loop {
            let refresh = self.begin_refresh();
            // Network IO is deliberately outside the ownership lock. Up to two
            // fetches may overlap so a newer snapshot can bypass one stalled
            // request; further notifications coalesce into one follow-up fetch.
            match self.fetch_tools(&context.peer).await {
                Ok(tools) => {
                    if self.commit_refresh(refresh, tools).await {
                        let tool_count = self.managed_tools.read().await.registrations.len();
                        tracing::info!(tool_count, "MCP tool list refreshed successfully");
                    }
                }
                Err(error) => tracing::error!("Failed to re-fetch MCP tool list: {error}"),
            }

            if !self.finish_or_restart_refresh().await {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use rmcp::model::*;
    use rmcp::service::RequestContext;
    use rmcp::{RoleServer, ServerHandler, ServiceExt};
    use serde_json::json;
    use tokio::{
        sync::{Notify, RwLock},
        task::JoinHandle,
    };

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
        cancelled: Arc<Notify>,
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
                Scenario::Hang => {
                    context.ct.cancelled().await;
                    self.cancelled.notify_one();
                    Err(ErrorData::internal_error("fixture request cancelled", None))
                }
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
        cancelled: Arc<Notify>,
        _client: rmcp::service::RunningService<rmcp::service::RoleClient, ClientInfo>,
        server_task: JoinHandle<()>,
    }

    async fn fixture(scenario: Scenario, timeout: Option<Duration>) -> Fixture {
        let seen = Arc::new(RwLock::new(None));
        let cancelled = Arc::new(Notify::new());
        let (client_to_server, server_from_client) = tokio::io::duplex(8192);
        let (server_to_client, client_from_server) = tokio::io::duplex(8192);
        let server = ScenarioServer {
            scenario,
            seen: seen.clone(),
            cancelled: cancelled.clone(),
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
            cancelled,
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
            ToolOutput::json(json!("forty-two"))
        );
    }

    #[test]
    fn structured_constructors_replace_their_canonical_text_fallback() {
        let value = json!({"answer": 42});
        for result in [
            CallToolResult::structured(value.clone()),
            CallToolResult::structured_error(value.clone()),
        ] {
            assert_eq!(
                mcp_result_output(&result).expect("MCP structured output"),
                ToolOutput::json(value.clone())
            );
        }
    }

    #[test]
    fn structured_content_is_kept_alongside_real_rich_blocks() {
        let value = json!({"answer": 42});
        let mut result = CallToolResult::structured(value.clone());
        result
            .content
            .push(ContentBlock::image("aW1hZ2U=", "image/png"));
        result
            .content
            .push(ContentBlock::text("human-readable note"));

        let mut expected = OneOrMany::one(RigToolResultContent::json(value));
        expected.push(RigToolResultContent::image_base64(
            "aW1hZ2U=",
            Some(ImageMediaType::PNG),
            None,
        ));
        expected.push(RigToolResultContent::text("human-readable note"));
        assert_eq!(
            mcp_result_output(&result).expect("MCP structured rich output"),
            ToolOutput::content(expected)
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
        assert_eq!(
            result.output().as_text(),
            Some("MCP tool 'fixture_tool' timed out after 25ms")
        );
        tokio::time::timeout(Duration::from_secs(1), fixture.cancelled.notified())
            .await
            .expect("the timed-out MCP request should be cancelled at the peer");
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
        let output = result.output().render();
        assert!(output.contains("MCP tool 'fixture_tool' request failed"));
        assert!(output.contains("fixture service failed"));
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

        let mut expected_content = OneOrMany::one(RigToolResultContent::json(json!({
            "answer": 42,
            "source": "fixture"
        })));
        expected_content.push(RigToolResultContent::text("before"));
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
        assert!(matches!(
            error.downcast_ref::<McpArgumentError>(),
            Some(McpArgumentError::Json(_))
        ));
        let output = result.output().render();
        assert!(output.contains("MCP tool 'fixture_tool' received invalid arguments"));
        assert!(output.contains("invalid JSON"));
        fixture.server_task.abort();
    }

    #[tokio::test]
    async fn canonical_dispatch_rejects_non_object_arguments() {
        let fixture = fixture(Scenario::Success, Some(Duration::from_secs(1))).await;
        for args in [r#"[1,2]"#, r#""text""#, "7", "true"] {
            let result = execute(&fixture, args, &mut ToolContext::new()).await;
            assert!(
                result.is_error_kind(ToolErrorKind::InvalidArgs),
                "{args} must not be coerced into an argument-less MCP call"
            );
        }

        // Empty input and explicit null remain the documented no-argument forms.
        for args in ["", "null"] {
            let result = execute(&fixture, args, &mut ToolContext::new()).await;
            assert!(
                result.is_success(),
                "{args:?} should remain a no-argument call"
            );
        }
        fixture.server_task.abort();
    }
}

#[cfg(test)]
mod migrated_tests {
    use super::{MAX_CONCURRENT_REFRESHES, McpClientError, McpClientHandler};
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
        first_refresh_returned: Arc<Notify>,
    }

    impl OrderedRefreshServer {
        fn new(tools: Vec<Tool>) -> Self {
            Self {
                tools: Arc::new(RwLock::new(tools)),
                list_calls: Arc::new(AtomicUsize::new(0)),
                first_refresh_started: Arc::new(Notify::new()),
                release_first_refresh: Arc::new(Notify::new()),
                first_refresh_returned: Arc::new(Notify::new()),
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
                self.first_refresh_returned.notify_one();
            }

            Ok(ListToolsResult::with_all_items(tools))
        }
    }

    #[derive(Clone)]
    struct HangingListServer;

    impl ServerHandler for HangingListServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
                .with_protocol_version(ProtocolVersion::LATEST)
                .with_server_info(Implementation::new("test-hanging-list-server", "0.1.0"))
        }

        async fn list_tools(
            &self,
            _: Option<PaginatedRequestParams>,
            _: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            std::future::pending().await
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
    async fn disconnected_handler_tools_are_retired_on_snapshot() {
        let server = DynamicToolServer::new(vec![make_tool("tool_a", "First")]);
        let handle = ToolServer::new().run();
        let (client, task) = connect(server, handle.clone()).await;
        assert_eq!(handle.get_tool_defs(None).await.unwrap().len(), 1);

        client.cancel().await.unwrap();

        let defs = handle.get_tool_defs(None).await.unwrap();
        assert!(
            defs.is_empty(),
            "a disconnected sole owner must not remain provider-visible"
        );
        task.abort();
    }

    #[tokio::test]
    async fn disconnected_handler_tools_are_retired_on_direct_dispatch() {
        let server = DynamicToolServer::new(vec![make_tool("tool_a", "First")]);
        let handle = ToolServer::new().run();
        let (client, task) = connect(server, handle.clone()).await;
        assert_eq!(handle.get_tool_defs(None).await.unwrap().len(), 1);

        client.cancel().await.unwrap();

        let result = handle
            .execute("tool_a", "{}", &mut crate::tool::ToolContext::new())
            .await;
        assert_eq!(
            result.error().expect("disconnected tool must fail").kind(),
            crate::tool::ToolErrorKind::NotFound
        );
        task.abort();
    }

    #[tokio::test]
    async fn initial_tool_fetch_is_bounded_by_the_refresh_timeout() {
        let (c2s, sfc) = tokio::io::duplex(8192);
        let (s2c, cfs) = tokio::io::duplex(8192);
        let server_task = tokio::spawn(async move {
            HangingListServer
                .serve((sfc, s2c))
                .await
                .expect("server start")
        });
        let refresh_timeout = Duration::from_millis(25);
        let result = McpClientHandler::new(ClientInfo::default(), ToolServer::new().run())
            .with_refresh_timeout(refresh_timeout)
            .connect((cfs, c2s))
            .await;

        assert!(matches!(
            result,
            Err(McpClientError::ToolFetchTimeout(timeout)) if timeout == refresh_timeout
        ));
        server_task.abort();
    }

    #[tokio::test]
    async fn refresh_activity_is_bounded_and_coalesces_excess_notifications() {
        let handler = McpClientHandler::new(ClientInfo::default(), ToolServer::new().run());

        assert!(handler.try_start_refresh().await);
        assert!(handler.try_start_refresh().await);
        assert!(!handler.try_start_refresh().await);
        {
            let activity = handler.refresh_activity.lock().await;
            assert_eq!(activity.active, MAX_CONCURRENT_REFRESHES);
            assert!(activity.dirty);
        }

        assert!(handler.finish_or_restart_refresh().await);
        assert!(!handler.finish_or_restart_refresh().await);
        assert!(!handler.finish_or_restart_refresh().await);
        let activity = handler.refresh_activity.lock().await;
        assert_eq!(activity.active, 0);
        assert!(!activity.dirty);
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

        assert!(
            client.service().managed_tools.try_write().is_ok(),
            "a hung network fetch must not hold the managed-registry lock"
        );

        server_control
            .set_tools(vec![make_tool("newest", "Newest snapshot")])
            .await;
        running_server
            .peer()
            .notify_tool_list_changed()
            .await
            .unwrap();

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
        .expect("newest refresh committed while the older fetch remained hung");

        // Let the stale response arrive after the newer snapshot committed. Its
        // lower refresh version must be discarded rather than rolling back.
        server_control.release_first_refresh.notify_one();
        tokio::time::timeout(
            Duration::from_secs(2),
            server_control.first_refresh_returned.notified(),
        )
        .await
        .expect("delayed refresh response returned");
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        let defs = handle.get_tool_defs(None).await.unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "newest");

        assert_eq!(server_control.list_calls.load(Ordering::SeqCst), 3);
        client.cancel().await.unwrap();
    }

    #[tokio::test]
    async fn refresh_rebuilds_owned_tools_in_latest_server_order() {
        let server =
            DynamicToolServer::new(vec![make_tool("alpha", "Alpha"), make_tool("beta", "Beta")]);
        let server_control = server.clone();
        let handle = ToolServer::new().run();
        let (client, server_task) = connect(server, handle.clone()).await;
        server_control
            .set_tools(vec![
                make_tool("beta", "Beta refreshed"),
                make_tool("gamma", "Gamma"),
                make_tool("alpha", "Alpha refreshed"),
            ])
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
                let names = defs
                    .iter()
                    .map(|definition| definition.name.as_str())
                    .collect::<Vec<_>>();
                if names == ["beta", "gamma", "alpha"] && defs[0].description == "Beta refreshed" {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("latest MCP order committed");
        client.cancel().await.unwrap();
    }

    #[tokio::test]
    async fn one_refresh_reclaims_a_name_after_a_peer_owner_disappears() {
        let handle = ToolServer::new().run();
        let first_server = DynamicToolServer::new(vec![make_tool("shared", "First owner")]);
        let first_control = first_server.clone();
        let (first_client, first_server_task) = connect(first_server, handle.clone()).await;
        let first_running_server = first_server_task.await.unwrap();

        let second_server = DynamicToolServer::new(vec![make_tool("shared", "Second owner")]);
        let second_control = second_server.clone();
        let (second_client, second_server_task) = connect(second_server, handle.clone()).await;
        let second_running_server = second_server_task.await.unwrap();
        assert_eq!(
            handle.get_tool_defs(None).await.unwrap()[0].description,
            "Second owner"
        );

        second_control.set_tools(Vec::new()).await;
        second_running_server
            .peer()
            .notify_tool_list_changed()
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if handle.get_tool_defs(None).await.unwrap().is_empty() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("second owner removed its registration");

        // The first handler still has a stale generation token for `shared`.
        // One full-list refresh must reclaim the now-empty slot rather than
        // requiring a second notification to converge.
        first_control
            .set_tools(vec![make_tool("shared", "First owner refreshed")])
            .await;
        first_running_server
            .peer()
            .notify_tool_list_changed()
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let defs = handle.get_tool_defs(None).await.unwrap();
                if defs.len() == 1 && defs[0].description == "First owner refreshed" {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("one refresh reclaimed the empty slot");

        second_client.cancel().await.unwrap();
        first_client.cancel().await.unwrap();
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
    async fn one_handler_refresh_protects_live_peer_and_reclaims_after_disconnect() {
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

        // Once B disconnects, its generation must no longer shield the dead
        // registration from A. Otherwise the registry keeps advertising B and
        // execution fails with `Transport closed` indefinitely.
        client_b.cancel().await.unwrap();
        server_a_control
            .set_tools(vec![make_tool("alpha", "Reclaimed handler A")])
            .await;
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
                    .any(|definition| definition.description == "Reclaimed handler A")
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("handler A reclaimed the disconnected peer's registration");

        let result = handle
            .execute("alpha", "{}", &mut crate::tool::ToolContext::new())
            .await;
        assert!(
            result.is_success(),
            "reclaimed tool should execute: {result:?}"
        );

        client_a.cancel().await.unwrap();
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

        let defs = handle.get_tool_defs(None).await.unwrap();
        assert!(
            defs.is_empty(),
            "a disconnected directly registered MCP tool must not remain provider-visible"
        );
        task.abort();
    }

    #[tokio::test]
    async fn disconnected_directly_registered_mcp_tool_is_retired_on_dispatch() {
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

        client.cancel().await.unwrap();

        let result = handle
            .execute("search_docs", "{}", &mut crate::tool::ToolContext::new())
            .await;
        assert_eq!(
            result.error().expect("disconnected tool must fail").kind(),
            crate::tool::ToolErrorKind::NotFound
        );
        task.abort();
    }
}
