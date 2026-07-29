//! The host-owned MCP toolset.
//!
//! Timeout, argument-validation, and content-mapping semantics are ported from
//! `rig-agent/src/tool/rmcp.rs` (crate-private there); helpers copied verbatim
//! carry attribution comments.

use std::time::Duration;

use rmcp::model::{
    CallToolRequest, CallToolResult, ClientRequest, ContentBlock, ListToolsRequest,
    PaginatedRequestParams, ResourceContents, ServerResult,
};
use rmcp::service::PeerRequestOptions;

use rig_core::OneOrMany;
use rig_core::completion::ToolDefinition;
use rig_core::message::{ImageMediaType, MimeType, ToolResultContent};
use rig_core::tool::{ToolExecutionError, ToolOutput, ToolResult};

/// Default per-call timeout applied to MCP tool calls (see rig#1914).
///
/// MCP tool calls await a response that can be silently lost by the transport
/// (e.g. an rmcp StreamableHttp session re-init dropping an in-flight request),
/// which would otherwise hang the caller forever. A generous default bounds
/// that without disrupting normal, long-running tools.
pub const DEFAULT_MCP_TOOL_TIMEOUT: Duration = Duration::from_secs(300);

/// Default deadline for fetching an MCP server's complete (paginated) tool list.
pub const DEFAULT_MCP_REFRESH_TIMEOUT: Duration = Duration::from_secs(30);

/// Maximum time spent delivering a best-effort cancellation after a request
/// has already exceeded its caller-visible deadline.
// Copied from rig-agent/src/tool/rmcp.rs (MCP_CANCELLATION_GRACE_PERIOD).
const MCP_CANCELLATION_GRACE_PERIOD: Duration = Duration::from_secs(1);

/// Error type for [`McpToolset`] operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum McpError {
    /// The named tool is not in this toolset's current snapshot.
    #[error("MCP tool '{0}' is not in this toolset")]
    NotFound(String),

    /// The arguments cannot be represented by MCP's object-valued arguments.
    #[error("invalid MCP tool arguments: {0}")]
    InvalidArguments(String),

    /// The call did not complete within the configured per-call timeout.
    ///
    /// RMCP delivers a best-effort cancellation notification to the server so
    /// both peers can release request-scoped resources.
    #[error("MCP tool '{tool}' timed out after {timeout:?}")]
    Timeout {
        /// The tool that was being called.
        tool: String,
        /// The elapsed per-call timeout.
        timeout: Duration,
    },

    /// A transport or service failure before the tool produced a result.
    #[error("MCP transport error: {0}")]
    Transport(#[from] rmcp::ServiceError),

    /// The server did not finish returning its tool list before the deadline.
    #[error("timed out fetching MCP tool list after {0:?}")]
    ToolFetchTimeout(Duration),

    /// An MCP content block could not be preserved as JSON for the model.
    #[error("failed to preserve an MCP content block as JSON: {0}")]
    ContentMapping(#[source] serde_json::Error),
}

/// The typed outcome of one MCP tool call.
#[derive(Debug)]
pub struct McpCallOutcome {
    /// The model-visible result, mapped with the same `ContentBlock` →
    /// [`ToolResultContent`] rules as the classic runtime: text and images stay
    /// typed, unrepresentable blocks are preserved as JSON (never stringified),
    /// and `is_error: true` becomes a failed [`ToolResult`] whose model output
    /// is the server-provided error content.
    pub result: ToolResult,
    /// The untouched wire result, including `structuredContent` and `_meta`.
    pub raw: CallToolResult,
}

/// A host-owned set of tools served by one connected MCP server.
///
/// The tool list is a snapshot taken at construction ([`McpToolset::from_sink`])
/// and re-taken only when the host calls [`McpToolset::refresh`] — there is no
/// notification-driven reconciliation here; that push model belongs to the
/// classic runtime.
pub struct McpToolset {
    client: rmcp::service::ServerSink,
    tools: Vec<rmcp::model::Tool>,
    /// Per-call timeout. When `Some`, a `call_tool` that does not complete
    /// within this duration resolves to [`McpError::Timeout`] instead of
    /// blocking forever (see rig#1914). When `None`, calls are unbounded.
    timeout: Option<Duration>,
    /// Deadline for complete (paginated) tool-list fetches.
    refresh_timeout: Duration,
}

impl McpToolset {
    /// Wrap an already-connected rmcp service and fetch its tool list
    /// (paginated, bounded by [`DEFAULT_MCP_REFRESH_TIMEOUT`]).
    ///
    /// Calls get [`DEFAULT_MCP_TOOL_TIMEOUT`]; change it with
    /// [`McpToolset::with_timeout`].
    pub async fn from_sink(client: rmcp::service::ServerSink) -> Result<Self, McpError> {
        let mut toolset = Self {
            client,
            tools: Vec::new(),
            timeout: Some(DEFAULT_MCP_TOOL_TIMEOUT),
            refresh_timeout: DEFAULT_MCP_REFRESH_TIMEOUT,
        };
        toolset.refresh().await?;
        Ok(toolset)
    }

    /// Set the per-call timeout, consuming and returning the toolset.
    ///
    /// On timeout, [`McpToolset::call`] resolves to [`McpError::Timeout`] and
    /// RMCP delivers a best-effort cancellation notification to the server.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Remove the per-call timeout, making calls unbounded.
    pub fn without_timeout(mut self) -> Self {
        self.timeout = None;
        self
    }

    /// Set the deadline for [`McpToolset::refresh`] tool-list fetches.
    pub fn with_refresh_timeout(mut self, timeout: Duration) -> Self {
        self.refresh_timeout = timeout;
        self
    }

    /// Re-fetch the tool list from the server, replacing the current snapshot.
    ///
    /// This is the host-initiated replacement for the classic runtime's
    /// `notifications/tools/list_changed` reconciliation: call it when your
    /// host learns (or suspects) the server's tools changed. On error the
    /// previous snapshot is kept.
    pub async fn refresh(&mut self) -> Result<(), McpError> {
        let tools = fetch_tools(&self.client, self.refresh_timeout).await?;
        tracing::debug!(tool_count = tools.len(), "MCP tool list refreshed");
        self.tools = tools;
        Ok(())
    }

    /// The current tool snapshot as model-facing definitions.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.iter().map(tool_definition).collect()
    }

    /// The current tool snapshot as raw rmcp tools.
    pub fn tools(&self) -> &[rmcp::model::Tool] {
        &self.tools
    }

    /// Whether the underlying transport is still open.
    pub fn is_live(&self) -> bool {
        !self.client.is_transport_closed()
    }

    /// Whether the current snapshot contains a tool with this name.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.iter().any(|tool| tool.name == name)
    }

    /// Call a tool by name.
    ///
    /// `args` must be a JSON object or `null` (MCP arguments are object-valued;
    /// `null` means a no-argument call). Other JSON shapes are rejected with
    /// [`McpError::InvalidArguments`] rather than silently coerced into a
    /// no-argument request that could execute a different operation than the
    /// one requested.
    ///
    /// `meta`, when present, is attached as the MCP request's `_meta`
    /// (SEP-1319) — the idiomatic channel for per-call values such as auth
    /// tokens, session ids, or A2A `context_id`/`task_id`, which the model
    /// never sees.
    pub async fn call(
        &self,
        name: &str,
        args: &serde_json::Value,
        meta: Option<rmcp::model::Meta>,
    ) -> Result<McpCallOutcome, McpError> {
        if !self.contains(name) {
            return Err(McpError::NotFound(name.to_string()));
        }
        let arguments = validate_mcp_arguments(args)?;

        let mut request = rmcp::model::CallToolRequestParams::new(name.to_string());
        request.arguments = arguments;
        request.meta = meta;

        let raw = call_mcp_tool(&self.client, request, self.timeout)
            .await
            .map_err(|error| match error {
                rmcp::ServiceError::Timeout {
                    timeout: elapsed_timeout,
                } => McpError::Timeout {
                    tool: name.to_string(),
                    timeout: self.timeout.unwrap_or(elapsed_timeout),
                },
                error => McpError::Transport(error),
            })?;

        let output = mcp_result_output(&raw)?;
        let result = if raw.is_error == Some(true) {
            ToolResult::failed(
                ToolExecutionError::other(format!(
                    "MCP tool '{name}' reported an execution error"
                ))
                .with_model_output(output),
            )
        } else {
            ToolResult::success(output)
        };

        Ok(McpCallOutcome { result, raw })
    }
}

/// Map one rmcp tool to a model-facing definition.
fn tool_definition(tool: &rmcp::model::Tool) -> ToolDefinition {
    ToolDefinition {
        name: tool.name.to_string(),
        description: tool
            .description
            .as_deref()
            .unwrap_or_default()
            .to_string(),
        parameters: tool.schema_as_json_value(),
    }
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

/// Returns no argument map for explicit JSON `null` and an MCP argument map for
/// a JSON object. Other JSON shapes are rejected: silently turning an array or
/// scalar into a no-argument request can execute a different operation than
/// the one requested.
///
/// Adapted from `parse_mcp_arguments` in rig-agent/src/tool/rmcp.rs; the
/// session runtime already holds a typed `serde_json::Value`, so the raw-JSON
/// parsing half of the classic helper is unnecessary here.
fn validate_mcp_arguments(
    args: &serde_json::Value,
) -> Result<Option<rmcp::model::JsonObject>, McpError> {
    match args {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Object(map) => Ok(Some(map.clone())),
        other => Err(McpError::InvalidArguments(format!(
            "expected a JSON object or null, got {}",
            json_value_kind(other)
        ))),
    }
}

// Copied from rig-agent/src/tool/rmcp.rs (call_mcp_tool), unchanged apart from
// the timeout primitive: this crate is native-only, so `tokio::time::timeout`
// replaces `rig_core::wasm_compat::timeout`.
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

// Copied from rig-agent/src/tool/rmcp.rs (send_mcp_request). Two-phase
// deadline: the send itself and the response wait share one deadline, and a
// timed-out response wait detaches a bounded best-effort cancellation.
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
            tokio::time::timeout(
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
    match tokio::time::timeout(remaining, &mut handle.rx).await {
        Ok(response) => response.map_err(|_| rmcp::ServiceError::TransportClosed)?,
        Err(_) => {
            cancel_timed_out_request(handle);
            Err(rmcp::ServiceError::Timeout { timeout })
        }
    }
}

// Copied from rig-agent/src/tool/rmcp.rs (cancel_timed_out_request).
//
// Keep cancellation delivery out of the caller's deadline. RMCP's cancellation
// notification uses the same bounded outbound queue as requests, so awaiting it
// inline could exceed the timeout precisely when that queue is saturated. The
// detached delivery is itself bounded so a stalled transport cannot retain one
// task and request handle for every timed-out call indefinitely.
fn cancel_timed_out_request(handle: rmcp::service::RequestHandle<rmcp::service::RoleClient>) {
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

    // This crate is native-only (the whole library is cfg'd off the wasm
    // graph), so `tokio::spawn` is always right here.
    tokio::spawn(cancellation);
}

// Copied from rig-agent/src/tool/rmcp.rs (bounded_best_effort_cancellation).
async fn bounded_best_effort_cancellation(
    cancellation: impl std::future::Future<Output = Result<(), rmcp::ServiceError>>,
    grace_period: Duration,
) {
    let _ = tokio::time::timeout(grace_period, cancellation).await;
}

// Copied from rig-agent/src/tool/rmcp.rs (fetch_tools), reshaped to return the
// raw `rmcp::model::Tool` list instead of erased registry adapters.
async fn fetch_tools(
    peer: &rmcp::service::ServerSink,
    refresh_timeout: Duration,
) -> Result<Vec<rmcp::model::Tool>, McpError> {
    let deadline = tokio::time::Instant::now() + refresh_timeout;
    let mut tools = Vec::new();
    let mut cursor = None;

    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return Err(McpError::ToolFetchTimeout(refresh_timeout));
        }
        let mut params = PaginatedRequestParams::default();
        params.cursor = cursor;
        let response = send_mcp_request(
            peer,
            ClientRequest::ListToolsRequest(ListToolsRequest::with_param(params)),
            Some((deadline, refresh_timeout)),
        )
        .await
        .map_err(|error| match error {
            rmcp::ServiceError::Timeout { .. } => McpError::ToolFetchTimeout(refresh_timeout),
            error => McpError::Transport(error),
        })?;
        let page = match response {
            ServerResult::ListToolsResult(page) => page,
            _ => {
                return Err(McpError::Transport(rmcp::ServiceError::UnexpectedResponse));
            }
        };
        tools.extend(page.tools);
        cursor = page.next_cursor;
        if cursor.is_none() {
            break;
        }
    }

    Ok(tools)
}

fn mcp_content_block_as_json(content: &ContentBlock) -> Result<ToolResultContent, McpError> {
    serde_json::to_value(content)
        .map(ToolResultContent::json)
        .map_err(McpError::ContentMapping)
}

// Copied from rig-agent/src/tool/rmcp.rs (mcp_content_block_to_tool_content),
// with the error type swapped for this crate's `McpError`.
fn mcp_content_block_to_tool_content(content: &ContentBlock) -> Result<ToolResultContent, McpError> {
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

// Copied from rig-agent/src/tool/rmcp.rs (mcp_result_output), with the error
// type swapped for this crate's `McpError`.
//
// Build the model presentation without flattening or reparsing MCP blocks.
fn mcp_result_output(result: &CallToolResult) -> Result<ToolOutput, McpError> {
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rmcp::model::{Resource, Tool};
    use serde_json::json;

    use super::*;

    fn fixture_tool(name: &str, description: Option<&str>) -> Tool {
        let schema: serde_json::Map<String, serde_json::Value> = serde_json::from_value(json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }))
        .expect("fixture schema is an object");
        let mut tool = Tool::new(
            name.to_string(),
            description.unwrap_or_default().to_string(),
            Arc::new(schema),
        );
        if description.is_none() {
            tool.description = None;
        }
        tool
    }

    #[test]
    fn definitions_map_name_description_and_schema() {
        let definition = tool_definition(&fixture_tool("get_weather", Some("Weather lookup")));
        assert_eq!(definition.name, "get_weather");
        assert_eq!(definition.description, "Weather lookup");
        assert_eq!(definition.parameters["type"], "object");
        assert_eq!(
            definition.parameters["properties"]["city"]["type"],
            "string"
        );
    }

    #[test]
    fn definitions_tolerate_a_missing_description() {
        let definition = tool_definition(&fixture_tool("bare", None));
        assert_eq!(definition.description, "");
    }

    #[test]
    fn arguments_accept_objects_and_null() {
        assert_eq!(
            validate_mcp_arguments(&json!(null)).expect("null is a no-argument call"),
            None
        );
        let map = validate_mcp_arguments(&json!({"city": "Lisbon"}))
            .expect("objects are valid arguments")
            .expect("object arguments are forwarded");
        assert_eq!(map.get("city"), Some(&json!("Lisbon")));
    }

    #[test]
    fn arguments_reject_non_object_shapes() {
        for args in [json!([1, 2]), json!("text"), json!(7), json!(true)] {
            assert!(
                matches!(
                    validate_mcp_arguments(&args),
                    Err(McpError::InvalidArguments(_))
                ),
                "{args} must not be coerced into an argument-less MCP call"
            );
        }
    }

    #[test]
    fn text_and_image_blocks_stay_typed() {
        let result = CallToolResult::success(vec![
            ContentBlock::text("before"),
            ContentBlock::image("aGVsbG8=", "image/png"),
        ]);
        let mut expected = OneOrMany::one(ToolResultContent::text("before"));
        expected.push(ToolResultContent::image_base64(
            "aGVsbG8=",
            Some(ImageMediaType::PNG),
            None,
        ));
        assert_eq!(
            mcp_result_output(&result).expect("MCP content mapping"),
            ToolOutput::content(expected)
        );
    }

    #[test]
    fn unrepresentable_blocks_are_preserved_as_json_not_stringified() {
        let block = ContentBlock::resource(ResourceContents::TextResourceContents {
            uri: "file:///reports/summary.txt".to_string(),
            mime_type: Some("text/plain".to_string()),
            text: "full report".to_string(),
            meta: None,
        });
        let content = mcp_result_output(&CallToolResult::success(vec![block.clone()]))
            .expect("MCP content mapping")
            .into_content()
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(
            content,
            vec![ToolResultContent::json(
                serde_json::to_value(&block).expect("MCP block is JSON serializable")
            )]
        );
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
            ToolOutput::one(ToolResultContent::image_base64(
                "aW1hZ2U=",
                Some(ImageMediaType::PNG),
                None,
            ))
        );
    }

    #[test]
    fn resource_links_and_audio_are_preserved_as_json() {
        let blocks = vec![
            ContentBlock::audio("UklGRg==", "audio/wav"),
            ContentBlock::resource_link(
                Resource::new("file:///reports/linked.txt", "linked.txt")
                    .with_mime_type("text/plain"),
            ),
        ];
        let expected = blocks
            .iter()
            .map(|block| {
                ToolResultContent::json(
                    serde_json::to_value(block).expect("MCP block is JSON serializable"),
                )
            })
            .collect::<Vec<_>>();
        let content = mcp_result_output(&CallToolResult::success(blocks))
            .expect("MCP content mapping")
            .into_content()
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(content, expected);
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

        let mut expected = OneOrMany::one(ToolResultContent::json(value));
        expected.push(ToolResultContent::image_base64(
            "aW1hZ2U=",
            Some(ImageMediaType::PNG),
            None,
        ));
        expected.push(ToolResultContent::text("human-readable note"));
        assert_eq!(
            mcp_result_output(&result).expect("MCP structured rich output"),
            ToolOutput::content(expected)
        );
    }

    #[test]
    fn empty_error_result_gets_a_placeholder_presentation() {
        let result = CallToolResult::error(vec![]);
        assert_eq!(
            mcp_result_output(&result).expect("MCP content mapping"),
            ToolOutput::text("the MCP tool reported an error")
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
}
