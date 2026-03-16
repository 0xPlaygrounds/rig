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
//! use rig::tool::rmcp::McpClientHandler;
//! use rig::tool::server::ToolServer;
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
use crate::tool::server::{ToolServerError, ToolServerHandle};
use crate::wasm_compat::WasmBoxedFuture;

/// A Rig tool adapter wrapping an `rmcp` MCP tool.
///
/// Bridges between the MCP tool protocol and Rig's [`ToolDyn`] trait,
/// allowing MCP tools to be used seamlessly in Rig agents.
#[derive(Clone)]
pub struct McpTool {
    definition: rmcp::model::Tool,
    client: rmcp::service::ServerSink,
}

impl McpTool {
    /// Create a new `McpTool` from an MCP tool definition and server sink.
    pub fn from_mcp_server(
        definition: rmcp::model::Tool,
        client: rmcp::service::ServerSink,
    ) -> Self {
        Self { definition, client }
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

#[derive(Debug, thiserror::Error)]
#[error("MCP tool error: {0}")]
pub struct McpToolError(String);

impl From<McpToolError> for ToolError {
    fn from(e: McpToolError) -> Self {
        ToolError::ToolCallError(Box::new(e))
    }
}

impl ToolDyn for McpTool {
    fn name(&self) -> String {
        self.definition.name.to_string()
    }

    fn definition(&self, _prompt: String) -> WasmBoxedFuture<'_, ToolDefinition> {
        Box::pin(async move {
            ToolDefinition {
                name: self.definition.name.to_string(),
                description: self
                    .definition
                    .description
                    .clone()
                    .unwrap_or(Cow::from(""))
                    .to_string(),
                parameters: serde_json::to_value(&self.definition.input_schema).unwrap_or_default(),
            }
        })
    }

    fn call(&self, args: String) -> WasmBoxedFuture<'_, Result<String, ToolError>> {
        let name = self.definition.name.clone();
        let arguments = serde_json::from_str(&args).unwrap_or_default();

        Box::pin(async move {
            let result = self
                .client
                .call_tool(rmcp::model::CallToolRequestParams {
                    name,
                    arguments,
                    meta: None,
                    task: None,
                })
                .await
                .map_err(|e| McpToolError(format!("Tool returned an error: {e}")))?;

            if let Some(true) = result.is_error {
                let error_msg = result
                    .content
                    .into_iter()
                    .map(|x| x.raw.as_text().map(|y| y.to_owned()))
                    .map(|x| x.map(|x| x.clone().text))
                    .collect::<Option<Vec<String>>>();

                let error_message = error_msg.map(|x| x.join("\n"));
                if let Some(error_message) = error_message {
                    return Err(McpToolError(error_message).into());
                } else {
                    return Err(McpToolError("No message returned".to_string()).into());
                }
            };

            Ok(result
                .content
                .into_iter()
                .map(|c| match c.raw {
                    rmcp::model::RawContent::Text(raw) => raw.text,
                    rmcp::model::RawContent::Image(raw) => {
                        format!("data:{};base64,{}", raw.mime_type, raw.data)
                    }
                    rmcp::model::RawContent::Resource(raw) => match raw.resource {
                        rmcp::model::ResourceContents::TextResourceContents {
                            uri,
                            mime_type,
                            text,
                            ..
                        } => {
                            format!(
                                "{mime_type}{uri}:{text}",
                                mime_type = mime_type
                                    .map(|m| format!("data:{m};"))
                                    .unwrap_or_default(),
                            )
                        }
                        rmcp::model::ResourceContents::BlobResourceContents {
                            uri,
                            mime_type,
                            blob,
                            ..
                        } => format!(
                            "{mime_type}{uri}:{blob}",
                            mime_type = mime_type
                                .map(|m| format!("data:{m};"))
                                .unwrap_or_default(),
                        ),
                    },
                    RawContent::Audio(_) => {
                        panic!("Support for audio results from an MCP tool is currently unimplemented. Come back later!")
                    }
                    thing => {
                        panic!("Unsupported type found: {thing:?}")
                    }
                })
                .collect::<String>())
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
    /// can be removed and replaced on list-change notifications.
    managed_tool_names: Arc<RwLock<Vec<String>>>,
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
            let mut managed = handler.managed_tool_names.write().await;

            for tool in tools {
                let tool_name = tool.name.to_string();
                let mcp_tool = McpTool::from_mcp_server(tool, service.peer().clone());
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
            let mcp_tool = McpTool::from_mcp_server(tool, context.peer.clone());
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
            ServerInfo {
                protocol_version: ProtocolVersion::V_2024_11_05,
                capabilities: ServerCapabilities::builder().enable_tools().build(),
                server_info: Implementation {
                    name: "test-dynamic-server".to_string(),
                    version: "0.1.0".to_string(),
                    ..Default::default()
                },
                instructions: None,
            }
        }

        async fn list_tools(
            &self,
            _request: Option<PaginatedRequestParams>,
            _context: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            let tools = self.tools.read().await.clone();
            Ok(ListToolsResult {
                tools,
                next_cursor: None,
                meta: None,
            })
        }

        async fn call_tool(
            &self,
            request: CallToolRequestParams,
            _context: RequestContext<RoleServer>,
        ) -> Result<CallToolResult, ErrorData> {
            Ok(CallToolResult::success(vec![Content::text(format!(
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
        let client_info = ClientInfo {
            protocol_version: Default::default(),
            capabilities: ClientCapabilities::default(),
            client_info: Implementation {
                name: "test-client".to_string(),
                version: "1.0.0".to_string(),
                ..Default::default()
            },
            meta: None,
        };

        let tool_server_handle = ToolServer::new().run();
        let handler = McpClientHandler::new(client_info.clone(), tool_server_handle);

        let returned = handler.get_info();
        assert_eq!(returned.client_info.name, "test-client");
        assert_eq!(returned.client_info.version, "1.0.0");
    }
}
