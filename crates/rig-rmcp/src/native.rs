//! Native MCP calls, metadata preservation, and dynamic tool conversion.
//!
//! ```
//! use rig_rmcp::{McpMeta, Meta};
//!
//! let metadata = McpMeta(Meta::default());
//! ```

use std::sync::Arc;
use std::time::Duration;

use rmcp::model::{
    CallToolRequest, CallToolResult, ClientRequest, ContentBlock, ResourceContents, ServerResult,
};
use rmcp::service::PeerRequestOptions;

use rig_core::message::{ImageMediaType, MimeType, ToolResultContent};
use rig_core::tool::{
    ContextValue, DynamicTool, ToolContext, ToolContextError, ToolExecutionError, ToolOutput,
};
use rig_core::wasm_compat::WasmBoxedFuture;

/// Re-export of [`rmcp::model::Meta`]: wrap one in [`McpMeta`] and place it
/// in the per-call [`ToolContext`] to have MCP tools forward it as the
/// request's `_meta`.
pub use rmcp::model::Meta;

/// The request `_meta` a caller places in the [`ToolContext`] for an MCP
/// tool, forwarded as the call's `_meta` (SEP-1319). A newtype because the
/// context stores values under declared keys and `rmcp::model::Meta` is
/// not this crate's to implement [`ContextValue`] for.
#[derive(
    Debug, Clone, Default, PartialEq, rig_core::serde::Serialize, rig_core::serde::Deserialize,
)]
#[serde(crate = "rig_core::serde", transparent)]
pub struct McpMeta(pub Meta);

impl ContextValue for McpMeta {
    const KEY: &'static str = "rmcp.meta";
}

/// The `structuredContent` an MCP tool answered with, on the context's
/// result map for result hooks.
#[derive(Debug, Clone, PartialEq, rig_core::serde::Serialize, rig_core::serde::Deserialize)]
#[serde(crate = "rig_core::serde", transparent)]
pub struct McpStructuredContent(pub serde_json::Value);

impl ContextValue for McpStructuredContent {
    const KEY: &'static str = "rmcp.structured_content";
}

/// The response `_meta` an MCP tool answered with, on the context's result
/// map for result hooks.
#[derive(
    Debug, Clone, Default, PartialEq, rig_core::serde::Serialize, rig_core::serde::Deserialize,
)]
#[serde(crate = "rig_core::serde", transparent)]
pub struct McpResponseMeta(pub Meta);

impl ContextValue for McpResponseMeta {
    const KEY: &'static str = "rmcp.response_meta";
}

/// The untouched [`CallToolResult`], on the context's result map for
/// result hooks.
#[derive(Debug, Clone, PartialEq, rig_core::serde::Serialize, rig_core::serde::Deserialize)]
#[serde(crate = "rig_core::serde", transparent)]
pub struct McpCallToolResult(pub CallToolResult);

impl ContextValue for McpCallToolResult {
    const KEY: &'static str = "rmcp.call_tool_result";
}

/// Default MCP tool-call deadline, overridable through [`McpTool::with_timeout`].
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
/// Construct with [`Self::from_mcp_server`] or [`tools_from_server`]. Conversion
/// to [`DynamicTool`] forwards [`McpMeta`] from context, publishes raw
/// results, and binds a transport liveness probe.
#[derive(Clone)]
pub struct McpTool {
    pub(crate) definition: rmcp::model::Tool,
    pub(crate) client: rmcp::service::ServerSink,
    /// Optional per-call deadline. Timeout triggers best-effort cancellation
    /// for requests with an acquired handle; `None` leaves calls unbounded.
    pub(crate) timeout: Option<Duration>,
}

impl McpTool {
    /// Create an adapter from an MCP tool definition and server sink.
    ///
    /// Applies [`DEFAULT_MCP_TOOL_TIMEOUT`].
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
    /// Timeout returns [`ToolExecutionError`] and attempts cancellation when a
    /// request handle is available. Remote cancellation is not guaranteed.
    #[must_use = "the setting applies to the returned value"]
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

/// Spawns bounded best-effort cancellation without extending the caller's deadline.
/// Detaching avoids waiting on a saturated outbound queue; the grace period
/// bounds retention of the task and request handle.
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

    // Native-only compilation permits a Send cancellation task.
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
    /// Attaches `meta` as request `_meta`. Rejects invalid or non-object JSON
    /// arguments before dispatch; empty input and `null` mean no arguments.
    /// Returns timeout or provider errors for failed requests.
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

    // Empty MCP content is legal; normalize it to sendable text, with a
    // diagnostic when the tool explicitly reports failure.
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
/// [`ServerSink`](rmcp::service::ServerSink). Each tool carries the same
/// [`DEFAULT_MCP_TOOL_TIMEOUT`] as [`McpTool::from_mcp_server`]; override it
/// per tool with [`McpTool::with_timeout`].
pub fn tools_from_server(
    tools: impl IntoIterator<Item = rmcp::model::Tool>,
    client: &rmcp::service::ServerSink,
) -> Vec<McpTool> {
    tools
        .into_iter()
        .map(|tool| McpTool::from_mcp_server(tool, client.clone()))
        .collect()
}

/// Publishes structured content, response metadata, and the raw result to context.
/// Returns context insertion errors; earlier insertions are not rolled back.
/// These values are host-visible and are not automatically sent to the model.
pub fn preserve_mcp_result(
    context: &mut ToolContext,
    result: CallToolResult,
) -> Result<(), ToolContextError> {
    if let Some(structured) = result.structured_content.clone() {
        context.insert_result(McpStructuredContent(structured))?;
    }
    if let Some(meta) = result.meta.clone() {
        context.insert_result(McpResponseMeta(meta))?;
    }
    context.insert_result(McpCallToolResult(result))?;
    Ok(())
}

/// An MCP tool as a context-aware rig-core dynamic tool, with a liveness probe
/// bound to the MCP transport so registries can retire it on disconnect.
///
/// Per call: [`McpMeta`] in the [`ToolContext`] supplies request `_meta`, and the
/// response's `structuredContent`, response `Meta`, and raw [`CallToolResult`]
/// are published to the context's result map ([`preserve_mcp_result`]). A tool
/// that reports `is_error` becomes a failed call whose error carries the tool's
/// output.
impl From<McpTool> for DynamicTool {
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
        DynamicTool::new_with_context(
            name,
            description,
            parameters,
            move |context: &mut ToolContext, args: serde_json::Value| {
                let tool = Arc::clone(&tool);
                let meta = context.get::<McpMeta>();
                Box::pin(async move {
                    let meta = meta?.map(|meta| meta.0);
                    let result = tool.execute_mcp(args.to_string(), meta).await?;
                    let is_error = result.is_error == Some(true);
                    let output = mcp_result_output(&result);
                    preserve_mcp_result(context, result)?;
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
