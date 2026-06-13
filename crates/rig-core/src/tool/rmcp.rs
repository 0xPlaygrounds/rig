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

use std::borrow::Cow;
use std::sync::Arc;

use rmcp::ServiceExt;
use rmcp::model::RawContent;
use tokio::sync::RwLock;

use crate::completion::ToolDefinition;
use crate::tool::ToolDyn;
use crate::tool::ToolError;
use crate::tool::server::{OwnedToolRegistration, OwnerId, ToolServerError, ToolServerHandle};
use crate::wasm_compat::WasmBoxedFuture;

/// A Rig tool adapter wrapping an `rmcp` MCP tool.
///
/// Bridges between the MCP tool protocol and Rig's [`ToolDyn`] trait,
/// allowing MCP tools to be used seamlessly in Rig agents.
#[derive(Clone)]
pub struct McpTool {
    definition: rmcp::model::Tool,
    client: rmcp::service::ServerSink,
    /// Optional name override under which this tool is advertised to and
    /// dispatched by Rig (e.g. a prefixed name to avoid collisions). The MCP
    /// server is still called with the original `definition.name`, so the
    /// override never reaches the wire.
    advertised_name: Option<String>,
}

impl McpTool {
    /// Create a new `McpTool` from an MCP tool definition and server sink.
    pub fn from_mcp_server(
        definition: rmcp::model::Tool,
        client: rmcp::service::ServerSink,
    ) -> Self {
        Self {
            definition,
            client,
            advertised_name: None,
        }
    }

    /// Advertise this tool to Rig under a different name than the MCP server
    /// uses. The override is what Rig registers, advertises to the model, and
    /// dispatches on; the MCP `call_tool` request still uses the original
    /// name. Used by [`McpClientHandler::with_tool_prefix`] for namespacing.
    pub fn with_advertised_name(mut self, name: impl Into<String>) -> Self {
        self.advertised_name = Some(name.into());
        self
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

/// Error from an MCP tool call. Displays as the bare message so the text a
/// model sees as a tool result carries no internal wrapper prefix.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct McpToolError(String);

impl From<McpToolError> for ToolError {
    fn from(e: McpToolError) -> Self {
        ToolError::ToolCallError(Box::new(e))
    }
}

/// A human-readable name for an MCP content variant, for diagnostics.
fn content_kind(raw: &RawContent) -> &'static str {
    match raw {
        RawContent::Text(_) => "text",
        RawContent::Image(_) => "image",
        RawContent::Resource(_) => "resource",
        RawContent::Audio(_) => "audio",
        _ => "unsupported",
    }
}

/// Render an MCP resource as a labeled, separated string the model can read,
/// e.g. `resource file:///x (text/plain):\n<body>` (mime omitted when absent).
fn labeled_resource(uri: &str, mime_type: Option<&str>, body: &str) -> String {
    match mime_type {
        Some(mime) => format!("resource {uri} ({mime}):\n{body}"),
        None => format!("resource {uri}:\n{body}"),
    }
}

impl ToolDyn for McpTool {
    fn name(&self) -> String {
        self.advertised_name
            .clone()
            .unwrap_or_else(|| self.definition.name.to_string())
    }

    fn definition(&self, _prompt: String) -> WasmBoxedFuture<'_, ToolDefinition> {
        // Delegate to the `From` impl so the schema has exactly one
        // conversion path (`schema_as_json_value`), keeping wire parity with
        // every other place an rmcp tool is rendered; then apply the advertised
        // name override if present.
        Box::pin(async move {
            let mut def = ToolDefinition::from(&self.definition);
            if let Some(name) = &self.advertised_name {
                def.name = name.clone();
            }
            def
        })
    }

    fn call(&self, args: String) -> WasmBoxedFuture<'_, Result<String, ToolError>> {
        let name = self.definition.name.clone();
        let arguments: Option<rmcp::model::JsonObject> =
            serde_json::from_str(&args).unwrap_or_default();

        Box::pin(async move {
            let request = arguments
                .map(|arguments| {
                    rmcp::model::CallToolRequestParams::new(name.clone()).with_arguments(arguments)
                })
                .unwrap_or_else(|| rmcp::model::CallToolRequestParams::new(name));

            let result = self
                .client
                .call_tool(request)
                .await
                .map_err(|e| McpToolError(format!("Tool returned an error: {e}")))?;

            if let Some(true) = result.is_error {
                if result.content.is_empty() {
                    return Err(
                        McpToolError("tool returned an error with no content".to_string()).into(),
                    );
                }
                // Collect text parts permissively: a mixed error (text + image)
                // must still surface its text, not collapse to a placeholder.
                let mut texts = Vec::new();
                let mut dropped: Vec<&'static str> = Vec::new();
                for item in &result.content {
                    match item.raw.as_text() {
                        Some(text) => texts.push(text.text.clone()),
                        None => {
                            let kind = content_kind(&item.raw);
                            if !dropped.contains(&kind) {
                                dropped.push(kind);
                            }
                        }
                    }
                }
                let message = if texts.is_empty() {
                    format!(
                        "tool returned an error with non-text content: {}",
                        dropped.join(", ")
                    )
                } else {
                    texts.join("\n")
                };
                return Err(McpToolError(message).into());
            };

            // Content is collected as ordered parts so multi-part results
            // keep their original text/image interleaving, mapped onto the
            // JSON shapes `ToolResultContent::from_tool_output` converts
            // into text and image tool-result parts.
            let mut texts: Vec<String> = Vec::new();
            let mut parts: Vec<serde_json::Value> = Vec::new();
            let mut image_count = 0usize;

            for item in result.content {
                let chunk = match item.raw {
                    rmcp::model::RawContent::Text(raw) => raw.text,
                    rmcp::model::RawContent::Image(raw) => {
                        image_count += 1;
                        parts.push(serde_json::json!({
                            "type": "image",
                            "data": raw.data,
                            "mimeType": raw.mime_type,
                        }));
                        continue;
                    }
                    rmcp::model::RawContent::Resource(raw) => match raw.resource {
                        rmcp::model::ResourceContents::TextResourceContents {
                            uri,
                            mime_type,
                            text,
                            ..
                        } => labeled_resource(&uri, mime_type.as_deref(), &text),
                        rmcp::model::ResourceContents::BlobResourceContents {
                            uri,
                            mime_type,
                            blob,
                            ..
                        } => labeled_resource(&uri, mime_type.as_deref(), &blob),
                    },
                    RawContent::Audio(raw) => {
                        // Rig has no audio tool-result part yet; degrade to a
                        // text placeholder so co-returned content survives
                        // instead of failing the whole call.
                        tracing::warn!(
                            "MCP tool returned audio content ({}); representing it as a text placeholder",
                            raw.mime_type
                        );
                        format!("[unsupported audio content ({})]", raw.mime_type)
                    }
                    thing => {
                        return Err(McpToolError(format!(
                            "MCP tool returned unsupported content: {thing:?}"
                        ))
                        .into());
                    }
                };

                parts.push(serde_json::json!({ "type": "text", "text": chunk.clone() }));
                texts.push(chunk);
            }

            if image_count == 0 {
                // Text-only results stay verbatim, as before.
                return Ok(texts.concat());
            }
            if parts.len() == 1
                && let Some(image) = parts.pop()
            {
                // A lone image maps to the top-level image shape.
                return Ok(image.to_string());
            }
            // Mixed or multi-image results map to the ordered-parts shape,
            // preserving the original content interleaving.
            Ok(serde_json::json!({ "parts": parts }).to_string())
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
    /// Tracks which tool names were registered by this handler so they
    /// can be removed and replaced on list-change notifications. Holds the
    /// *advertised* names (prefixed, when a prefix is set).
    managed_tool_names: Arc<RwLock<Vec<String>>>,
    /// Ownership identity for tools this handler registers, so a refresh
    /// removes only its own tools and never a user's directly-registered one.
    owner: OwnerId,
    /// Optional namespace prefix for advertised tool names (`"{prefix}_{name}"`),
    /// mirroring pydantic-ai's `MCPServer.tool_prefix`.
    tool_prefix: Option<String>,
}

impl McpClientHandler {
    /// Create a new handler with the given client info and tool server handle.
    ///
    /// The `tool_server_handle` should be a clone of the handle used by the agent,
    /// so that tool updates are reflected in agent requests.
    pub fn new(client_info: rmcp::model::ClientInfo, tool_server_handle: ToolServerHandle) -> Self {
        Self {
            client_info,
            tool_server_handle,
            managed_tool_names: Arc::new(RwLock::new(Vec::new())),
            owner: OwnerId::new(),
            tool_prefix: None,
        }
    }

    /// The ownership identity this handler registers tools under.
    ///
    /// Persist this *before* [`Self::connect`] (which consumes the handler) and
    /// pass it to a replacement handler via [`Self::with_owner`] so the
    /// replacement can reclaim and refresh this handler's stale registrations
    /// after a reconnect.
    pub fn owner_id(&self) -> OwnerId {
        self.owner
    }

    /// Reuse an existing ownership identity instead of a fresh one. Pair with
    /// [`Self::owner_id`] to let a replacement handler adopt a prior handler's
    /// registrations on reconnect.
    pub fn with_owner(mut self, owner: OwnerId) -> Self {
        self.owner = owner;
        self
    }

    /// Advertise this server's tools under `"{prefix}_{name}"`, avoiding
    /// collisions with other tools (mirrors pydantic-ai's `MCPServer.tool_prefix`).
    /// The MCP server is still called with the original tool names.
    pub fn with_tool_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.tool_prefix = Some(prefix.into());
        self
    }

    /// The name this tool is advertised under (prefixed when a prefix is set).
    fn advertised_name(&self, original: &str) -> String {
        match &self.tool_prefix {
            Some(prefix) => format!("{prefix}_{original}"),
            None => original.to_string(),
        }
    }

    /// Register one MCP tool under this handler's ownership, applying the
    /// prefix. Returns the advertised name to track, or `None` if the
    /// registration was skipped (name conflict) or errored.
    async fn register_tool(
        &self,
        tool: rmcp::model::Tool,
        peer: rmcp::service::ServerSink,
    ) -> Option<String> {
        let advertised = self.advertised_name(&tool.name);
        let mut mcp_tool = McpTool::from_mcp_server(tool, peer);
        if self.tool_prefix.is_some() {
            mcp_tool = mcp_tool.with_advertised_name(advertised.clone());
        }
        match self
            .tool_server_handle
            .add_tool_owned(mcp_tool, self.owner)
            .await
        {
            Ok(OwnedToolRegistration::Added | OwnedToolRegistration::Replaced) => Some(advertised),
            Ok(OwnedToolRegistration::SkippedConflict) => None,
            Err(e) => {
                tracing::error!("Failed to register MCP tool '{advertised}': {e}");
                None
            }
        }
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
            let mut registered = Vec::new();
            for tool in tools {
                if let Some(advertised) = handler.register_tool(tool, service.peer().clone()).await
                {
                    registered.push(advertised);
                }
            }
            *handler.managed_tool_names.write().await = registered;
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

        // Remove only tools we still own: a name a user has since taken over
        // via a direct registration is now unowned and must not be evicted.
        for name in managed.drain(..) {
            self.tool_server_handle
                .remove_tool_owned(&name, self.owner)
                .await;
        }

        for tool in tools {
            if let Some(advertised) = self.register_tool(tool, context.peer.clone()).await {
                managed.push(advertised);
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
    use tokio::sync::RwLock;

    use super::McpClientHandler;
    use crate::test_utils::MockAddTool;
    use crate::tool::server::ToolServer;

    /// Serve `server` over an in-memory duplex transport and connect `handler`,
    /// returning the running client service (hold it to keep the connection
    /// alive and to receive list-change notifications).
    async fn serve_and_connect(
        server: DynamicToolServer,
        handler: McpClientHandler,
    ) -> rmcp::service::RunningService<rmcp::RoleClient, McpClientHandler> {
        let (client_to_server, server_from_client) = tokio::io::duplex(8192);
        let (server_to_client, client_from_server) = tokio::io::duplex(8192);
        tokio::spawn(async move {
            let service = server
                .serve((server_from_client, server_to_client))
                .await
                .expect("server failed to start");
            let _ = service.waiting().await;
        });
        handler
            .connect((client_from_server, client_to_server))
            .await
            .expect("connect failed")
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
            match request.name.as_ref() {
                "single_image" => Ok(CallToolResult::success(vec![Content::image(
                    "aGVsbG8=",
                    "image/png",
                )])),
                "text_and_image" => Ok(CallToolResult::success(vec![
                    Content::text("the badge is attached"),
                    Content::image("aGVsbG8=", "image/png"),
                ])),
                "text_image_text" => Ok(CallToolResult::success(vec![
                    Content::text("before the image"),
                    Content::image("aGVsbG8=", "image/png"),
                    Content::text("after the image"),
                ])),
                "text_resource" => Ok(CallToolResult::success(vec![Content::embedded_text(
                    "file:///notes.txt",
                    "resource body",
                )])),
                "audio_and_text" => Ok(CallToolResult::success(vec![
                    Annotated::new(
                        RawContent::Audio(RawAudioContent {
                            data: "QUJD".to_string(),
                            mime_type: "audio/wav".to_string(),
                        }),
                        None,
                    ),
                    Content::text("transcript follows"),
                ])),
                "error_text_and_image" => Ok(CallToolResult::error(vec![
                    Content::text("boom"),
                    Content::image("aGVsbG8=", "image/png"),
                ])),
                "error_image_only" => Ok(CallToolResult::error(vec![Content::image(
                    "aGVsbG8=",
                    "image/png",
                )])),
                "error_empty" => Ok(CallToolResult::error(vec![])),
                name => Ok(CallToolResult::success(vec![Content::text(format!(
                    "called {name}"
                ))])),
            }
        }
    }

    fn make_tool(name: &str, description: &str) -> Tool {
        Tool::new(
            name.to_string(),
            description.to_string(),
            Arc::new(serde_json::Map::new()),
        )
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

    /// Spawn `server` over an in-memory duplex transport and return a tool
    /// bound to it, plus the running client service that keeps it alive.
    async fn connect_mcp_tool(
        server: DynamicToolServer,
        tool: Tool,
    ) -> (
        super::McpTool,
        rmcp::service::RunningService<rmcp::RoleClient, McpClientHandler>,
    ) {
        let tool_server_handle = ToolServer::new().run();
        let (client_to_server, server_from_client) = tokio::io::duplex(8192);
        let (server_to_client, client_from_server) = tokio::io::duplex(8192);

        tokio::spawn(async move {
            let service = server
                .serve((server_from_client, server_to_client))
                .await
                .expect("server failed to start");
            let _ = service.waiting().await;
        });

        let handler = McpClientHandler::new(ClientInfo::default(), tool_server_handle);
        let service = handler
            .connect((client_from_server, client_to_server))
            .await
            .expect("connect failed");
        let mcp_tool = super::McpTool::from_mcp_server(tool, service.peer().clone());
        (mcp_tool, service)
    }

    #[tokio::test]
    async fn image_content_maps_to_image_tool_output() {
        let tool = make_tool("single_image", "Returns one image");
        let server = DynamicToolServer::new(vec![tool.clone()]);
        let (mcp_tool, _service) = connect_mcp_tool(server, tool).await;

        let output = crate::tool::ToolDyn::call(&mcp_tool, "{}".to_string())
            .await
            .expect("image tool call should succeed");

        let json: serde_json::Value =
            serde_json::from_str(&output).expect("image output should be JSON");
        assert_eq!(
            json,
            serde_json::json!({
                "type": "image",
                "data": "aGVsbG8=",
                "mimeType": "image/png"
            })
        );

        let content = crate::completion::message::ToolResultContent::from_tool_output(output);
        assert!(
            matches!(
                content.first(),
                crate::completion::message::ToolResultContent::Image(_)
            ),
            "the output should parse into an image tool-result part"
        );
    }

    #[tokio::test]
    async fn mixed_content_maps_to_ordered_parts() {
        let tool = make_tool("text_and_image", "Returns text and an image");
        let server = DynamicToolServer::new(vec![tool.clone()]);
        let (mcp_tool, _service) = connect_mcp_tool(server, tool).await;

        let output = crate::tool::ToolDyn::call(&mcp_tool, "{}".to_string())
            .await
            .expect("mixed tool call should succeed");

        let json: serde_json::Value =
            serde_json::from_str(&output).expect("mixed output should be JSON");
        assert_eq!(
            json["parts"][0],
            serde_json::json!({ "type": "text", "text": "the badge is attached" })
        );
        assert_eq!(json["parts"][1]["type"], "image");

        let content = crate::completion::message::ToolResultContent::from_tool_output(output);
        assert_eq!(content.len(), 2, "mixed output should split into two parts");
    }

    #[tokio::test]
    async fn interleaved_content_preserves_part_order() {
        use crate::completion::message::ToolResultContent;

        let tool = make_tool("text_image_text", "Returns text, image, text");
        let server = DynamicToolServer::new(vec![tool.clone()]);
        let (mcp_tool, _service) = connect_mcp_tool(server, tool).await;

        let output = crate::tool::ToolDyn::call(&mcp_tool, "{}".to_string())
            .await
            .expect("interleaved tool call should succeed");

        let content = ToolResultContent::from_tool_output(output);
        let kinds: Vec<&str> = content
            .iter()
            .map(|part| match part {
                ToolResultContent::Text(text) => text.text.as_str(),
                ToolResultContent::Image(_) => "<image>",
            })
            .collect();
        assert_eq!(
            kinds,
            vec!["before the image", "<image>", "after the image"],
            "the original text/image interleaving should be preserved"
        );
    }

    #[tokio::test]
    async fn is_error_with_text_and_image_keeps_text() {
        let tool = make_tool("error_text_and_image", "errors with text + image");
        let server = DynamicToolServer::new(vec![tool.clone()]);
        let (mcp_tool, _service) = connect_mcp_tool(server, tool).await;

        let err = crate::tool::ToolDyn::call(&mcp_tool, "{}".to_string())
            .await
            .expect_err("an is_error result should surface as an error");
        // The text must survive; the non-text part is dropped silently.
        assert!(err.to_string().contains("boom"), "got {err}");
    }

    #[tokio::test]
    async fn is_error_with_only_non_text_names_kinds() {
        let tool = make_tool("error_image_only", "errors with image only");
        let server = DynamicToolServer::new(vec![tool.clone()]);
        let (mcp_tool, _service) = connect_mcp_tool(server, tool).await;

        let err = crate::tool::ToolDyn::call(&mcp_tool, "{}".to_string())
            .await
            .expect_err("an is_error result should surface as an error");
        assert!(
            err.to_string().contains("non-text content: image"),
            "got {err}"
        );
    }

    #[tokio::test]
    async fn is_error_with_empty_content() {
        let tool = make_tool("error_empty", "errors with no content");
        let server = DynamicToolServer::new(vec![tool.clone()]);
        let (mcp_tool, _service) = connect_mcp_tool(server, tool).await;

        let err = crate::tool::ToolDyn::call(&mcp_tool, "{}".to_string())
            .await
            .expect_err("an is_error result should surface as an error");
        assert!(err.to_string().contains("no content"), "got {err}");
    }

    #[tokio::test]
    async fn text_resource_uses_labeled_format() {
        let tool = make_tool("text_resource", "returns a text resource");
        let server = DynamicToolServer::new(vec![tool.clone()]);
        let (mcp_tool, _service) = connect_mcp_tool(server, tool).await;

        let output = crate::tool::ToolDyn::call(&mcp_tool, "{}".to_string())
            .await
            .expect("resource tool call should succeed");
        assert_eq!(output, "resource file:///notes.txt (text):\nresource body");
    }

    #[tokio::test]
    async fn audio_content_degrades_to_placeholder() {
        let tool = make_tool("audio_and_text", "returns audio + text");
        let server = DynamicToolServer::new(vec![tool.clone()]);
        let (mcp_tool, _service) = connect_mcp_tool(server, tool).await;

        // Audio no longer fails the whole call: it degrades to a placeholder,
        // and the co-returned text survives.
        let output = crate::tool::ToolDyn::call(&mcp_tool, "{}".to_string())
            .await
            .expect("audio tool call should succeed");
        assert!(
            output.contains("[unsupported audio content (audio/wav)]"),
            "got {output}"
        );
        assert!(output.contains("transcript follows"), "got {output}");
    }

    #[tokio::test]
    async fn mcp_tool_does_not_evict_colliding_static_tool() {
        // A user's directly-registered "add" must survive an MCP server that
        // also exposes "add": the MCP registration is skipped, not co-opted.
        let handle = ToolServer::new().tool(MockAddTool).run();
        let server = DynamicToolServer::new(vec![make_tool("add", "mcp add")]);
        let _svc = serve_and_connect(
            server,
            McpClientHandler::new(ClientInfo::default(), handle.clone()),
        )
        .await;

        let defs = handle.get_tool_defs(None).await.unwrap();
        assert_eq!(defs.len(), 1, "the colliding MCP tool must be skipped");
        // The static implementation survived: it sums, the MCP one would not.
        let out = handle.call_tool("add", r#"{"x":2,"y":5}"#).await.unwrap();
        assert_eq!(
            out, "7",
            "the user's static tool must still be the live impl"
        );
    }

    #[tokio::test]
    async fn refresh_does_not_evict_user_tool_that_took_over_the_name() {
        // MCP registers "add", the user then overwrites it via add_tool, and a
        // later refresh that drops "add" must NOT delete the user's tool.
        let handle = ToolServer::new().run();
        let server = DynamicToolServer::new(vec![make_tool("add", "mcp add")]);

        let (client_to_server, server_from_client) = tokio::io::duplex(8192);
        let (server_to_client, client_from_server) = tokio::io::duplex(8192);
        let server_clone = server.clone();
        let server_handle = tokio::spawn(async move {
            server_clone
                .serve((server_from_client, server_to_client))
                .await
                .expect("server failed to start")
        });
        let _svc = McpClientHandler::new(ClientInfo::default(), handle.clone())
            .connect((client_from_server, client_to_server))
            .await
            .expect("connect failed");

        // The user takes over "add" with a direct (unowned) registration.
        handle.add_tool(MockAddTool).await.unwrap();

        // The MCP server drops "add"; refresh drains its managed names.
        server
            .set_tools(vec![make_tool("other", "unrelated")])
            .await;
        let server_service = server_handle.await.unwrap();
        server_service
            .peer()
            .notify_tool_list_changed()
            .await
            .expect("notify failed");
        tokio::time::sleep(Duration::from_millis(200)).await;

        // The user's tool survived (owned-removal saw the cleared tag and
        // skipped it); the new MCP tool was added alongside.
        let out = handle.call_tool("add", r#"{"x":2,"y":5}"#).await.unwrap();
        assert_eq!(out, "7", "the user's tool must survive an MCP refresh");
        let names: Vec<String> = handle
            .get_tool_defs(None)
            .await
            .unwrap()
            .into_iter()
            .map(|d| d.name)
            .collect();
        assert!(names.contains(&"add".to_string()));
        assert!(names.contains(&"other".to_string()));
    }

    #[tokio::test]
    async fn replacement_handler_reclaims_tools_via_shared_owner() {
        // A replacement handler reusing the prior owner id re-registers
        // (Replace) instead of skipping, rebinding the tool to the live peer.
        let handle = ToolServer::new().run();

        let handler1 = McpClientHandler::new(ClientInfo::default(), handle.clone());
        let owner = handler1.owner_id();
        let svc1 = serve_and_connect(
            DynamicToolServer::new(vec![make_tool("add", "mcp add")]),
            handler1,
        )
        .await;
        assert_eq!(handle.get_tool_defs(None).await.unwrap().len(), 1);

        // The first connection dies; its tool stays registered, owned by `owner`.
        drop(svc1);

        // A replacement handler adopts the same owner and reconnects.
        let handler2 =
            McpClientHandler::new(ClientInfo::default(), handle.clone()).with_owner(owner);
        let _svc2 = serve_and_connect(
            DynamicToolServer::new(vec![make_tool("add", "mcp add")]),
            handler2,
        )
        .await;

        // "add" is callable through the NEW connection: it was reclaimed
        // (Replaced), not skipped — a skip would leave it bound to the dead peer.
        let out = handle.call_tool("add", "{}").await.unwrap();
        assert_eq!(out, "called add");
    }

    #[tokio::test]
    async fn tool_prefix_advertises_prefixed_name_and_calls_original() {
        let handle = ToolServer::new().run();
        let _svc = serve_and_connect(
            DynamicToolServer::new(vec![make_tool("add", "mcp add")]),
            McpClientHandler::new(ClientInfo::default(), handle.clone()).with_tool_prefix("mcp"),
        )
        .await;

        let names: Vec<String> = handle
            .get_tool_defs(None)
            .await
            .unwrap()
            .into_iter()
            .map(|d| d.name)
            .collect();
        assert_eq!(
            names,
            vec!["mcp_add".to_string()],
            "advertised under the prefix"
        );

        // Dispatch under the prefixed name; the MCP server is called with "add".
        let out = handle.call_tool("mcp_add", "{}").await.unwrap();
        assert_eq!(out, "called add");
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
}
