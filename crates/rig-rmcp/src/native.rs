//! Native-only body of `rig-rmcp` (see the crate root for why).

use std::sync::Arc;
use std::time::Duration;

use rmcp::model::{
    CallToolRequest, CallToolResult, ClientRequest, ContentBlock, ResourceContents, ServerResult,
};
use rmcp::service::PeerRequestOptions;

use rig_core::message::{ImageMediaType, MimeType, ToolResultContent};
use rig_core::tool::{PortableDynamicTool, ToolContext, ToolExecutionError, ToolOutput};
use rig_core::wasm_compat::WasmBoxedFuture;

/// Re-export of [`rmcp::model::Meta`]: place one in the per-call [`ToolContext`]
/// to have MCP tools forward it as the request's `_meta`.
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

/// Maximum time spent delivering a best-effort cancellation after a request
/// has already exceeded its caller-visible deadline.
const MCP_CANCELLATION_GRACE_PERIOD: Duration = Duration::from_secs(1);

/// One MCP server tool, callable through an rmcp [`ServerSink`](rmcp::service::ServerSink).
///
/// Construct with [`McpTool::from_mcp_server`] (or [`tools_from_server`] for a
/// whole list). Use it as a rig-core [`PortableDynamicTool`] via `From`, or —
/// with the `agent` feature — register it directly in rig-agent's tool server
/// (it implements rig-agent's contextual `ErasedTool`, which additionally
/// forwards MCP `_meta` from the `ToolContext` and preserves the raw result).
#[derive(Clone)]
pub struct McpTool {
    pub(crate) definition: rmcp::model::Tool,
    pub(crate) client: rmcp::service::ServerSink,
    /// Per-call timeout. When `Some`, an MCP `call_tool` that does not complete
    /// within this duration resolves to a [`ToolExecutionError`] instead of blocking
    /// forever (see issue #1914). When `None`, the call is unbounded.
    ///
    /// On elapse RMCP sends a cancellation notification so both peers can
    /// release request-scoped resources.
    pub(crate) timeout: Option<Duration>,
}

impl McpTool {
    /// Create an adapter from an MCP tool definition and server sink.
    ///
    /// Applies [`DEFAULT_MCP_TOOL_TIMEOUT`] so a lost/never-answered response
    /// cannot hang the agent forever (issue #1914).
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
    /// than hang). RMCP sends a cancellation notification when the deadline
    /// elapses.
    pub fn with_timeout(mut self, timeout: impl Into<Option<Duration>>) -> Self {
        self.timeout = timeout.into();
        self
    }

    /// The per-call timeout, if any.
    pub fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    /// The MCP tool definition this adapter wraps.
    pub fn definition(&self) -> &rmcp::model::Tool {
        &self.definition
    }
}

/// Parse the JSON `args` string into MCP call arguments.
///
/// Argument decoding failure at the MCP object boundary.
#[derive(Debug, thiserror::Error)]
pub(crate) enum McpArgumentError {
    /// Malformed JSON.
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// Valid JSON that cannot be represented by MCP's object-valued arguments.
    #[error("expected a JSON object or null, got {0}")]
    NonObject(&'static str),
}

pub(crate) fn json_value_kind(value: &serde_json::Value) -> &'static str {
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
pub(crate) fn parse_mcp_arguments(
    args: &str,
) -> Result<Option<rmcp::model::JsonObject>, McpArgumentError> {
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

pub(crate) async fn call_mcp_tool(
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

pub(crate) async fn send_mcp_request(
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
            rig_core::wasm_compat::timeout(
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
    match rig_core::wasm_compat::timeout(remaining, &mut handle.rx).await {
        Ok(response) => response.map_err(|_| rmcp::ServiceError::TransportClosed)?,
        Err(_) => {
            cancel_timed_out_request(handle);
            Err(rmcp::ServiceError::Timeout { timeout })
        }
    }
}

/// Keep cancellation delivery out of the caller's deadline. RMCP's cancellation
/// notification uses the same bounded outbound queue as requests, so awaiting it
/// inline could exceed the timeout precisely when that queue is saturated. The
/// detached delivery is itself bounded so a stalled transport cannot retain one
/// task and request handle for every timed-out call indefinitely.
pub(crate) fn cancel_timed_out_request(
    handle: rmcp::service::RequestHandle<rmcp::service::RoleClient>,
) {
    let cancellation = async move {
        bounded_best_effort_cancellation(
            handle.cancel(Some(
                rmcp::service::RequestHandle::<rmcp::service::RoleClient>::REQUEST_TIMEOUT_REASON
                    .to_owned(),
            )),
            MCP_CANCELLATION_GRACE_PERIOD,
        )
        .await;
    };

    // This crate is native-only (see the `compile_error!` at the crate root), so
    // there is no `spawn_local` branch to pick: `tokio::spawn` is always right
    // here.
    tokio::spawn(cancellation);
}

pub(crate) async fn bounded_best_effort_cancellation(
    cancellation: impl std::future::Future<Output = Result<(), rmcp::ServiceError>>,
    grace_period: Duration,
) {
    let _ = rig_core::wasm_compat::timeout(grace_period, cancellation).await;
}

impl McpTool {
    /// Execute one MCP request.
    ///
    /// `meta`, when present, is attached as the MCP request's `_meta`
    /// (SEP-1319) — the idiomatic channel for per-call metadata such as auth
    /// tokens, session ids, or A2A `context_id`/`task_id`. It is supplied by a
    /// caller that places an [`rmcp::model::Meta`] into the per-call
    /// [`ToolContext`]; otherwise the call behaves exactly as before.
    pub fn execute_mcp(
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

pub(crate) fn mcp_content_block_as_json(
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

pub(crate) fn mcp_content_block_to_tool_content(
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
pub fn mcp_result_output(result: &CallToolResult) -> Result<ToolOutput, ToolExecutionError> {
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

    if !mapped.is_empty() {
        return ToolOutput::content(mapped);
    }

    // A content-less MCP result normalizes to one empty text block. This is
    // deliberately *not* what the native path does — a native tool returning an
    // empty `Vec<ToolResultContent>` gets an eager `ToolExecutionError`,
    // because that shape is the tool author's own type choice and fixable in
    // one read. An empty MCP result is protocol-legal and outside the caller's
    // control, so erroring here would fail tools the author cannot fix; the
    // empty block keeps the result sendable without inventing text.
    if result.is_error == Some(true) {
        Ok(ToolOutput::text("the MCP tool reported an error"))
    } else {
        Ok(ToolOutput::text(""))
    }
}

/// Error type for MCP client operations (connection, tool-list fetch).
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

/// Wrap every tool of an MCP server's list as an [`McpTool`] sharing one
/// [`ServerSink`](rmcp::service::ServerSink), each with the given per-call
/// timeout (`None` = unbounded).
pub fn tools_from_server(
    tools: impl IntoIterator<Item = rmcp::model::Tool>,
    client: &rmcp::service::ServerSink,
    timeout: impl Into<Option<Duration>>,
) -> Vec<McpTool> {
    let timeout = timeout.into();
    tools
        .into_iter()
        .map(|tool| McpTool::from_mcp_server(tool, client.clone()).with_timeout(timeout))
        .collect()
}

/// An MCP tool as a context-free rig-core dynamic tool, with a liveness probe
/// bound to the MCP transport so registries can retire it on disconnect.
///
/// The call is made with no MCP `_meta` and the raw [`CallToolResult`] is not
/// retained (callers that need either drive [`McpTool::execute_mcp`]
/// themselves). A tool that reports `is_error` becomes a failed call whose
/// error carries the tool's output.
/// Keep the MCP response's protocol data on the per-call [`ToolContext`] for
/// result hooks: the `structuredContent` value, the response [`Meta`], and the
/// untouched [`CallToolResult`]. Host-only; the model sees only the ordered
/// presentation content.
pub fn preserve_mcp_result(context: &mut ToolContext, result: CallToolResult) {
    if let Some(structured) = result.structured_content.clone() {
        context.insert_result(structured);
    }
    if let Some(meta) = result.meta.clone() {
        context.insert_result(meta);
    }
    context.insert_result(result);
}

/// An MCP tool as a context-aware rig-core dynamic tool, with a liveness probe
/// bound to the MCP transport so registries can retire it on disconnect.
///
/// Per call: an [`rmcp::model::Meta`] placed in the [`ToolContext`] (re-exported
/// here as [`Meta`]) is forwarded as the request's `_meta` (SEP-1319), and the
/// response's `structuredContent`, response `Meta`, and raw [`CallToolResult`]
/// are published to the context's result map ([`preserve_mcp_result`]). A tool
/// that reports `is_error` becomes a failed call whose error carries the tool's
/// output.
impl From<McpTool> for PortableDynamicTool {
    fn from(tool: McpTool) -> Self {
        let name = tool.definition.name.to_string();
        let description = tool
            .definition
            .description
            .as_deref()
            .unwrap_or("")
            .to_string();
        let parameters = tool.definition.schema_as_json_value();
        let liveness_client = tool.client.clone();
        let tool = Arc::new(tool);
        PortableDynamicTool::new_with_context(
            name,
            description,
            parameters,
            move |context: &mut ToolContext, args: serde_json::Value| {
                let tool = Arc::clone(&tool);
                let meta = context.get::<rmcp::model::Meta>().cloned();
                Box::pin(async move {
                    let result = tool.execute_mcp(args.to_string(), meta).await?;
                    let is_error = result.is_error == Some(true);
                    let output = mcp_result_output(&result);
                    preserve_mcp_result(context, result);
                    let output = output?;
                    if is_error {
                        Err(ToolExecutionError::other(format!(
                            "MCP tool '{}' reported an execution error",
                            tool.definition.name
                        ))
                        .with_model_output(output))
                    } else {
                        Ok(output)
                    }
                })
            },
        )
        .with_liveness(move || !liveness_client.is_transport_closed())
    }
}

// Compile-time thread-safety contract: an `McpTool` is handed to the agent's
// tool registry and executed from whichever thread the host runs tools on.
const _: fn() = || {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<McpTool>();
};
