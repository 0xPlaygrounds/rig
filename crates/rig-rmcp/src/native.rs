//! Native MCP calls, metadata preservation, and dynamic tool conversion.
//!
//! ```
//! use rig_rmcp::McpMeta;
//! use rig_rmcp::rmcp::model::RequestMetaObject;
//!
//! let metadata = McpMeta(RequestMetaObject::new());
//! ```

use std::sync::Arc;
use std::time::Duration;

use rmcp::model::{
    CallToolRequest, CallToolResult, ClientRequest, ContentBlock, JsonObject, MetaObject,
    RequestMetaObject, ResourceContents, ServerResult,
};
use rmcp::service::{PeerRequestOptions, ServerSink};

use rig_core::completion::ToolDefinition;
use rig_core::message::{EmptyToolName, ImageMediaType, MimeType, ToolName, ToolResultContent};
use rig_core::tool::{
    ContextValue, DynamicTool, Schema, SchemaError, ToolContext, ToolContextError,
    ToolExecutionError, ToolOutput,
};
use rig_core::wasm_compat::WasmBoxedFuture;

use crate::McpConnection;

/// The request `_meta` a caller places in the [`ToolContext`] for an MCP
/// tool, forwarded as the call's `_meta` (SEP-1319). A newtype because the
/// context stores values under declared keys and `RequestMetaObject` is
/// not this crate's to implement [`ContextValue`] for.
#[derive(
    Debug, Clone, Default, PartialEq, rig_core::serde::Serialize, rig_core::serde::Deserialize,
)]
#[serde(crate = "rig_core::serde", transparent)]
pub struct McpMeta(pub RequestMetaObject);

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
pub struct McpResponseMeta(pub MetaObject);

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

/// One MCP tool call the model chose, as a [`McpTool::with_call_meta`]
/// function sees it before the request is sent.
#[non_exhaustive]
pub struct McpToolCall<'a> {
    /// The tool's MCP name, the one sent to the server.
    pub tool: &'a str,
    /// The exact arguments sent to the server, after any hook rewrites.
    pub arguments: Option<&'a JsonObject>,
    /// The call's context.
    pub context: &'a ToolContext,
}

/// Derives request `_meta` from one tool call; see [`McpTool::with_call_meta`].
pub(crate) type CallMetaFn =
    Arc<dyn Fn(&McpToolCall<'_>) -> Option<RequestMetaObject> + Send + Sync>;

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

/// One MCP server tool.
///
/// Build it with [`Self::new`] or [`McpConnection::tools`]. A tool keeps its
/// [`McpConnection`] open, so registering it is enough to keep it usable.
/// Conversion to [`DynamicTool`] forwards [`McpMeta`] from context, publishes
/// raw results, and binds a transport liveness probe.
#[derive(Clone)]
pub struct McpTool {
    pub(crate) definition: rmcp::model::Tool,
    /// The model-visible name, when it differs from the MCP name.
    pub(crate) name: Option<ToolName>,
    pub(crate) peer: ServerSink,
    /// Keeps a directly built tool's connection open. Tools a
    /// [`McpClientHandler`](crate::McpClientHandler) manages leave it to the
    /// connection its `connect` returns, which owns them through the sink.
    pub(crate) _connection: Option<McpConnection>,
    /// Optional per-call deadline. Timeout triggers best-effort cancellation
    /// for requests with an acquired handle; `None` leaves calls unbounded.
    pub(crate) timeout: Option<Duration>,
    pub(crate) call_meta: Option<CallMetaFn>,
}

impl McpTool {
    /// The tool `definition` of the server behind `connection`, which the
    /// tool keeps open. Applies [`DEFAULT_MCP_TOOL_TIMEOUT`].
    pub fn new(definition: rmcp::model::Tool, connection: &McpConnection) -> Self {
        Self {
            _connection: Some(connection.clone()),
            ..Self::managed(definition, connection.peer().clone())
        }
    }

    /// A tool whose connection someone else keeps open.
    pub(crate) fn managed(definition: rmcp::model::Tool, peer: ServerSink) -> Self {
        Self {
            definition,
            name: None,
            peer,
            _connection: None,
            timeout: Some(DEFAULT_MCP_TOOL_TIMEOUT),
            call_meta: None,
        }
    }

    /// Offer the tool to the model as `name`. Calls still send the server
    /// the tool's MCP name, so one server's tools can be namespaced or two
    /// servers' same-named tools told apart.
    #[must_use = "the setting applies to the returned value"]
    pub fn with_name(mut self, name: ToolName) -> Self {
        self.name = Some(name);
        self
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

    /// Derive request `_meta` from each call, for values bound to the chosen
    /// arguments such as a signature or an idempotency key. The entries `meta`
    /// returns are added over the context's [`McpMeta`], replacing keys both set.
    /// The model sees neither.
    #[must_use = "the setting applies to the returned value"]
    pub fn with_call_meta<F>(mut self, meta: F) -> Self
    where
        F: Fn(&McpToolCall<'_>) -> Option<RequestMetaObject> + Send + Sync + 'static,
    {
        self.call_meta = Some(Arc::new(meta));
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

    /// The name the model sees: the one set by [`Self::with_name`], or the MCP name.
    pub fn name(&self) -> &str {
        self.name
            .as_ref()
            .map_or(&self.definition.name, ToolName::as_str)
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

/// Returns no argument map for explicit JSON `null`, and an MCP argument map
/// for a JSON object. Other shapes are rejected: silently turning an array or
/// scalar into a no-argument request can execute a different operation than
/// the model requested.
pub(crate) fn mcp_arguments(
    value: serde_json::Value,
) -> Result<Option<JsonObject>, McpArgumentError> {
    match value {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Object(arguments) => Ok(Some(arguments)),
        value => Err(McpArgumentError::NonObject(json_value_kind(&value))),
    }
}

/// [`mcp_arguments`] over JSON text, where empty input also means no arguments.
pub(crate) fn parse_mcp_arguments(args: &str) -> Result<Option<JsonObject>, McpArgumentError> {
    let trimmed = args.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    mcp_arguments(serde_json::from_str(trimmed)?)
}

pub(crate) async fn call_mcp_tool(
    peer: &ServerSink,
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
    peer: &ServerSink,
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
        meta: Option<RequestMetaObject>,
    ) -> WasmBoxedFuture<'_, Result<CallToolResult, ToolExecutionError>> {
        Box::pin(async move {
            // Malformed JSON must surface as an InvalidArgs failure, not a
            // silent no-argument call.
            let arguments = parse_mcp_arguments(&args).map_err(|error| self.invalid(error))?;
            self.call(arguments, meta).await
        })
    }

    fn invalid(&self, error: McpArgumentError) -> ToolExecutionError {
        ToolExecutionError::invalid_args(format!(
            "MCP tool '{}' received invalid arguments: {error}",
            self.definition.name
        ))
        .with_source(error)
    }

    /// The request `_meta` for one call: the context's [`McpMeta`] with the
    /// [`Self::with_call_meta`] entries over it.
    fn request_meta(
        &self,
        arguments: Option<&JsonObject>,
        context: &ToolContext,
    ) -> Result<Option<RequestMetaObject>, ToolContextError> {
        let mut meta = context.get::<McpMeta>()?.map(|meta| meta.0);
        let call = McpToolCall {
            tool: &self.definition.name,
            arguments,
            context,
        };
        if let Some(derived) = self.call_meta.as_ref().and_then(|derive| derive(&call)) {
            meta.get_or_insert_with(RequestMetaObject::new)
                .0
                .extend(derived.0);
        }
        Ok(meta)
    }

    async fn call(
        &self,
        arguments: Option<JsonObject>,
        meta: Option<RequestMetaObject>,
    ) -> Result<CallToolResult, ToolExecutionError> {
        let name = &self.definition.name;
        let mut request = rmcp::model::CallToolRequestParams::new(name.clone());
        request.arguments = arguments;
        request.meta = meta;

        match call_mcp_tool(&self.peer, request, self.timeout).await {
            Ok(result) => Ok(result),
            Err(
                error @ rmcp::ServiceError::Timeout {
                    timeout: elapsed_timeout,
                },
            ) => {
                let timeout = self.timeout.unwrap_or(elapsed_timeout);
                Err(ToolExecutionError::timeout(format!(
                    "MCP tool '{name}' timed out after {timeout:?}"
                ))
                .with_source(error))
            }
            // A transport/service error before the tool produced a result.
            Err(error) => Err(ToolExecutionError::provider(format!(
                "MCP tool '{name}' request failed: {error}"
            ))
            .with_source(error)),
        }
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
    let mut replaced_fallback = false;
    let mut mapped = Vec::with_capacity(result.content.len());

    for block in &result.content {
        let fallback_structured = if !replaced_fallback {
            // The compatibility text is the structured value serialized,
            // however the server spaced it.
            match (block, structured) {
                (ContentBlock::Text(text), Some(structured))
                    if serde_json::from_str::<serde_json::Value>(&text.text)
                        .ok()
                        .as_ref()
                        == Some(structured) =>
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

/// Why an MCP tool cannot become a rig tool.
#[derive(Debug, thiserror::Error)]
pub enum McpToolError {
    /// The server gave the tool no name.
    #[error(transparent)]
    EmptyName(#[from] EmptyToolName),
    /// The server gave the tool an input or output schema that is not well
    /// formed.
    #[error("MCP tool `{tool}`: {error}")]
    InvalidSchema {
        /// The tool's name.
        tool: String,
        /// What is wrong with the schema.
        #[source]
        error: SchemaError,
    },
}

/// Error type for MCP client operations: connecting, listing, and reading.
#[derive(Debug, thiserror::Error)]
pub enum McpClientError {
    /// Failed to establish the MCP connection or complete the handshake.
    #[error("MCP connection error: {0}")]
    Connection(#[from] rmcp::service::ClientInitializeError),

    /// An MCP request failed.
    #[error("MCP request failed: {0}")]
    Request(#[from] rmcp::ServiceError),

    /// The server did not answer before the deadline.
    #[error("MCP request timed out after {0:?}")]
    Timeout(Duration),
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
/// Per call: [`McpMeta`] in the [`ToolContext`] and [`McpTool::with_call_meta`]
/// supply request `_meta`, and the response's `structuredContent`, response
/// `_meta`, and raw [`CallToolResult`] are published to the context's result
/// map ([`preserve_mcp_result`]). A tool that declares an output schema
/// answers the model with its structured content alone; otherwise the model
/// gets [`mcp_result_output`]. A tool that reports `is_error` becomes a
/// failed call whose error carries the tool's output. The tool's
/// `outputSchema` becomes its definition's output schema. Fails with
/// [`McpToolError`] when the server gave the tool no name, or an input or
/// output schema that is not well formed.
impl TryFrom<McpTool> for DynamicTool {
    type Error = McpToolError;

    fn try_from(tool: McpTool) -> Result<Self, McpToolError> {
        let name = match &tool.name {
            Some(name) => name.clone(),
            None => ToolName::new(tool.definition.name.to_string())?,
        };
        let description = tool
            .definition
            .description
            .as_deref()
            .unwrap_or("")
            .to_string();
        let schema = |object: &rmcp::model::JsonObject| {
            Schema::new(serde_json::Value::Object(object.clone())).map_err(|error| {
                McpToolError::InvalidSchema {
                    tool: name.to_string(),
                    error,
                }
            })
        };
        let parameters = schema(&tool.definition.input_schema)?;
        let output_schema = tool
            .definition
            .output_schema
            .as_deref()
            .map(schema)
            .transpose()?;
        let mut definition = ToolDefinition::new(name, description, parameters)
            .with_annotations(annotations(&tool.definition));
        definition.output_schema = output_schema;
        let liveness_peer = tool.peer.clone();
        let tool = Arc::new(tool);
        Ok(DynamicTool::new_with_context(
            definition,
            move |context: &mut ToolContext, args: serde_json::Value| {
                let tool = Arc::clone(&tool);
                Box::pin(async move {
                    let arguments = mcp_arguments(args).map_err(|error| tool.invalid(error))?;
                    let meta = tool.request_meta(arguments.as_ref(), context)?;
                    let result = tool.call(arguments, meta).await?;
                    let is_error = result.is_error == Some(true);
                    let output = match &result.structured_content {
                        // With a declared output schema the structured value is the
                        // result; MCP's content blocks only repeat it for older clients.
                        Some(structured)
                            if tool.definition.output_schema.is_some() && !is_error =>
                        {
                            Ok(ToolOutput::json(structured.clone()))
                        }
                        _ => mcp_result_output(&result),
                    };
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
        .with_liveness(move || !liveness_peer.is_transport_closed()))
    }
}

/// The hints an MCP server gives about what calling `tool` does, as rig's
/// tool annotations.
pub(crate) fn annotations(tool: &rmcp::model::Tool) -> rig_core::completion::ToolAnnotations {
    let mut annotations = rig_core::completion::ToolAnnotations::default();
    if let Some(hints) = &tool.annotations {
        annotations.title = hints.title.clone();
        annotations.read_only_hint = hints.read_only_hint;
        annotations.destructive_hint = hints.destructive_hint;
        annotations.idempotent_hint = hints.idempotent_hint;
        annotations.open_world_hint = hints.open_world_hint;
    }
    annotations
}

// Compile-time thread-safety contract: an `McpTool` is handed to the agent's
// tool registry and executed from whichever thread the host runs tools on.
const _: fn() = || {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<McpTool>();
};
