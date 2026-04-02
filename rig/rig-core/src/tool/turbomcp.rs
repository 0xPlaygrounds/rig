//! MCP (Model Context Protocol) integration via the `turbomcp` crate.
//!
//! This module provides:
//! - [`TurboMcpTool`]: A wrapper that adapts a TurboMCP tool for use in Rig's tool system.
//! - [`TurboMcpClientHandler`]: A client handler that reacts to `notifications/tools/list_changed`
//!   by re-fetching the tool list and updating the [`ToolServer`](super::server::ToolServer).
//!
//! # Example
//!
//! ```rust,ignore
//! use rig::tool::turbomcp::TurboMcpClientHandler;
//! use rig::tool::server::ToolServer;
//! use turbomcp_client::{ClientBuilder, StreamableHttpClientConfig, StreamableHttpClientTransport};
//!
//! // 1. Create a ToolServer and get a handle
//! let tool_server_handle = ToolServer::new().run();
//!
//! // 2. Create a TurboMCP client
//! let transport = StreamableHttpClientTransport::new(config);
//! let client = ClientBuilder::new().with_tools(true).build_sync(transport);
//!
//! // 3. Create a handler that auto-updates tools on list changes
//! let handler = TurboMcpClientHandler::new(client.clone(), tool_server_handle.clone());
//!
//! // 4. Connect to the MCP server and register initial tools
//! let client = handler.connect().await?;
//!
//! // 5. Build an agent using the shared tool server handle
//! let agent = openai_client
//!     .agent(openai::GPT_4O)
//!     .preamble("You are a helpful assistant.")
//!     .tool_server_handle(tool_server_handle)
//!     .build();
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;
use turbomcp_client::{
    CallToolResult, Client, ContentBlock, ResourceContent, Result as McpResult, Tool, Transport,
};

use crate::completion::ToolDefinition;
use crate::tool::ToolDyn;
use crate::tool::ToolError;
use crate::tool::server::{ToolServerError, ToolServerHandle};
use crate::wasm_compat::WasmBoxedFuture;

// ============================================================================
// TOOL CALLER TRAIT (type erasure for transport-agnostic tool calling)
// ============================================================================

/// Trait for abstracting over TurboMCP client tool calling.
///
/// This trait allows storing a type-erased client that can call tools
/// regardless of the underlying transport type (HTTP, WebSocket, TCP, etc.).
pub trait TurboMcpToolCaller: Send + Sync {
    fn call_tool(
        &self,
        name: &str,
        arguments: Option<HashMap<String, serde_json::Value>>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = McpResult<CallToolResult>> + Send + '_>,
    >;
}

impl<T> TurboMcpToolCaller for Client<T>
where
    T: Transport + 'static,
{
    fn call_tool(
        &self,
        name: &str,
        arguments: Option<HashMap<String, serde_json::Value>>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = McpResult<CallToolResult>> + Send + '_>,
    > {
        let name = name.to_string();
        Box::pin(async move { self.call_tool(&name, arguments, None).await })
    }
}

// ============================================================================
// TOOL ADAPTER
// ============================================================================

/// A Rig tool adapter wrapping a TurboMCP tool.
///
/// Bridges between TurboMCP's tool system and Rig's [`ToolDyn`] trait,
/// allowing TurboMCP tools to be used seamlessly in Rig agents.
#[derive(Clone)]
pub struct TurboMcpTool {
    definition: Tool,
    client: Arc<dyn TurboMcpToolCaller>,
}

impl TurboMcpTool {
    /// Create a new `TurboMcpTool` from a tool definition and a concrete client.
    pub fn from_mcp_server<T>(definition: Tool, client: Client<T>) -> Self
    where
        T: Transport + 'static,
    {
        Self {
            definition,
            client: Arc::new(client),
        }
    }

    /// Create a new `TurboMcpTool` from a tool definition and a type-erased client.
    pub fn from_client_arc(definition: Tool, client: Arc<dyn TurboMcpToolCaller>) -> Self {
        Self { definition, client }
    }
}

impl From<&Tool> for ToolDefinition {
    fn from(val: &Tool) -> Self {
        Self {
            name: val.name.clone(),
            description: val.description.clone().unwrap_or_default(),
            parameters: serde_json::to_value(&val.input_schema).unwrap_or_default(),
        }
    }
}

impl From<Tool> for ToolDefinition {
    fn from(val: Tool) -> Self {
        Self {
            name: val.name.clone(),
            description: val.description.clone().unwrap_or_default(),
            parameters: serde_json::to_value(&val.input_schema).unwrap_or_default(),
        }
    }
}

// ============================================================================
// ERROR TYPES
// ============================================================================

/// Error type for TurboMCP tool call failures.
#[derive(Debug, thiserror::Error)]
#[error("TurboMCP tool error: {0}")]
pub struct TurboMcpToolError(String);

impl From<TurboMcpToolError> for ToolError {
    fn from(e: TurboMcpToolError) -> Self {
        ToolError::ToolCallError(Box::new(e))
    }
}

/// Error type for [`TurboMcpClientHandler`] operations.
#[derive(Debug, thiserror::Error)]
pub enum TurboMcpClientError {
    /// Failed to initialize the MCP connection or complete the handshake.
    #[error("TurboMCP connection error: {0}")]
    ConnectionError(turbomcp_client::Error),

    /// Failed to fetch the tool list from the MCP server.
    #[error("Failed to fetch TurboMCP tool list: {0}")]
    ToolFetchError(turbomcp_client::Error),

    /// Failed to update the tool server with new tools.
    #[error("Tool server error: {0}")]
    ToolServerError(#[from] ToolServerError),
}

// ============================================================================
// TOOL DYN IMPLEMENTATION
// ============================================================================

impl ToolDyn for TurboMcpTool {
    fn name(&self) -> String {
        self.definition.name.clone()
    }

    fn definition(&self, _prompt: String) -> WasmBoxedFuture<'_, ToolDefinition> {
        Box::pin(async move {
            ToolDefinition {
                name: self.definition.name.clone(),
                description: self.definition.description.clone().unwrap_or_default(),
                parameters: serde_json::to_value(&self.definition.input_schema)
                    .unwrap_or_default(),
            }
        })
    }

    fn call(&self, args: String) -> WasmBoxedFuture<'_, Result<String, ToolError>> {
        let name = self.definition.name.clone();
        let arguments: Option<HashMap<String, serde_json::Value>> =
            serde_json::from_str(&args).ok();

        Box::pin(async move {
            let result = self
                .client
                .call_tool(&name, arguments)
                .await
                .map_err(|e| TurboMcpToolError(format!("Tool returned an error: {e}")))?;

            if let Some(true) = result.is_error {
                let error_msg = result
                    .content
                    .iter()
                    .filter_map(|c| {
                        if let ContentBlock::Text(text) = c {
                            Some(text.text.clone())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                if !error_msg.is_empty() {
                    return Err(TurboMcpToolError(error_msg).into());
                } else {
                    return Err(
                        TurboMcpToolError("No error message returned".to_string()).into()
                    );
                }
            }

            Ok(result
                .content
                .into_iter()
                .map(|c| match c {
                    ContentBlock::Text(text) => text.text,
                    ContentBlock::Image(img) => {
                        format!("data:{};base64,{}", img.mime_type, img.data)
                    }
                    ContentBlock::Resource(res) => match res.resource {
                        ResourceContent::Text(t) => {
                            let mime = t
                                .mime_type
                                .as_deref()
                                .map(|m| format!("data:{m};"))
                                .unwrap_or_default();
                            format!("{mime}{}:{}", t.uri, t.text)
                        }
                        ResourceContent::Blob(b) => {
                            let mime = b
                                .mime_type
                                .as_deref()
                                .map(|m| format!("data:{m};"))
                                .unwrap_or_default();
                            format!("{mime}{}:{}", b.uri, b.blob)
                        }
                    },
                    ContentBlock::ResourceLink(link) => {
                        format!("resource-link:{}", link.uri)
                    }
                    ContentBlock::Audio(audio) => {
                        format!("data:{};base64,{}", audio.mime_type, audio.data)
                    }
                    ContentBlock::ToolUse(tu) => {
                        format!("tool-use:{}:{}", tu.name, tu.id)
                    }
                    ContentBlock::ToolResult(tr) => {
                        format!("tool-result:{}", tr.tool_use_id)
                    }
                })
                .collect::<String>())
        })
    }
}

// ============================================================================
// CLIENT HANDLER (auto-updating tool registration)
// ============================================================================

/// A TurboMCP client handler that automatically re-fetches the tool list when the
/// server sends a `notifications/tools/list_changed` notification.
///
/// This handler bridges the TurboMCP notification lifecycle with Rig's
/// [`ToolServer`](super::server::ToolServer). When the MCP server's available
/// tools change, this handler:
/// 1. Removes previously registered TurboMCP tools from the tool server
/// 2. Re-fetches the full tool list from the MCP server
/// 3. Registers the updated tools with the tool server
///
/// # Usage
///
/// Use [`TurboMcpClientHandler::connect`] for a streamlined setup that handles
/// initialization, initial tool fetch, and registration in one call:
///
/// ```rust,ignore
/// let tool_server_handle = ToolServer::new().run();
/// let client = ClientBuilder::new().with_tools(true).build_sync(transport);
/// let handler = TurboMcpClientHandler::new(client, tool_server_handle.clone());
/// let client = handler.connect().await?;
/// ```
///
/// The returned `Client` keeps the MCP connection alive. When the server updates
/// its tools, the handler automatically syncs with the tool server.
pub struct TurboMcpClientHandler<T: Transport + 'static> {
    client: Client<T>,
    tool_server_handle: ToolServerHandle,
    /// Tracks which tool names were registered by this handler so they
    /// can be removed and replaced on list-change notifications.
    managed_tool_names: Arc<RwLock<Vec<String>>>,
}

impl<T: Transport + 'static> TurboMcpClientHandler<T> {
    /// Create a new handler with the given client and tool server handle.
    ///
    /// The `tool_server_handle` should be a clone of the handle used by the agent,
    /// so that tool updates are reflected in agent requests.
    pub fn new(client: Client<T>, tool_server_handle: ToolServerHandle) -> Self {
        Self {
            client,
            tool_server_handle,
            managed_tool_names: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Connect to the MCP server, fetch the initial tool list, and register
    /// all tools with the tool server.
    ///
    /// This method:
    /// 1. Initializes the MCP connection (handshake + capability negotiation)
    /// 2. Fetches the initial tool list from the server
    /// 3. Registers all tools with the tool server
    /// 4. Installs a [`ToolListChangedHandler`](turbomcp_client::ToolListChangedHandler)
    ///    so future changes are automatically synced
    ///
    /// Returns the initialized `Client<T>`, which should be kept alive for the
    /// duration of the session. Dropping the client closes the connection.
    ///
    /// # Errors
    ///
    /// Returns [`TurboMcpClientError`] if the connection fails, the initial tool
    /// fetch fails, or tool registration with the tool server fails.
    pub async fn connect(self) -> Result<Client<T>, TurboMcpClientError> {
        // Phase 1: Initialize the MCP connection
        self.client
            .initialize()
            .await
            .map_err(TurboMcpClientError::ConnectionError)?;

        // Phase 2: Fetch initial tools and register them
        let tools = self
            .client
            .list_tools()
            .await
            .map_err(TurboMcpClientError::ToolFetchError)?;

        {
            let mut managed = self.managed_tool_names.write().await;
            for tool in tools {
                let tool_name = tool.name.clone();
                let mcp_tool =
                    TurboMcpTool::from_mcp_server(tool, self.client.clone());
                if let Err(e) = self.tool_server_handle.add_tool(mcp_tool).await {
                    // Roll back already-registered tools before propagating
                    for name in managed.drain(..) {
                        let _ = self.tool_server_handle.remove_tool(&name).await;
                    }
                    return Err(e.into());
                }
                managed.push(tool_name);
            }
        }

        tracing::info!(
            tool_count = self.managed_tool_names.read().await.len(),
            "TurboMCP initial tool registration complete"
        );

        // Phase 3: Install the tool-list-changed handler for auto-refresh.
        //
        // The handler holds its own clone of the client (cheaply cloneable Arc
        // wrapper) and the shared state needed to sync tools.
        let refresh_handler = ToolRefreshHandler {
            client: self.client.clone(),
            tool_server_handle: self.tool_server_handle.clone(),
            managed_tool_names: self.managed_tool_names.clone(),
        };

        self.client
            .set_tool_list_changed_handler(Arc::new(refresh_handler));

        Ok(self.client)
    }
}

/// Internal handler that implements [`ToolListChangedHandler`] for auto-refresh.
///
/// Registered on the client after initial connection. When the MCP server sends
/// `notifications/tools/list_changed`, this handler removes stale tools from the
/// [`ToolServer`](super::server::ToolServer), re-fetches the full list, and
/// re-registers them.
struct ToolRefreshHandler<T: Transport + 'static> {
    client: Client<T>,
    tool_server_handle: ToolServerHandle,
    managed_tool_names: Arc<RwLock<Vec<String>>>,
}

impl<T: Transport + 'static> std::fmt::Debug for ToolRefreshHandler<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRefreshHandler")
            .field("managed_tool_names", &self.managed_tool_names)
            .finish_non_exhaustive()
    }
}

impl<T: Transport + 'static> turbomcp_client::ToolListChangedHandler for ToolRefreshHandler<T> {
    fn handle_tool_list_changed(
        &self,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = turbomcp_client::HandlerResult<()>> + Send + '_>,
    > {
        Box::pin(async move {
            // Re-fetch the tool list from the server
            let tools = match self.client.list_tools().await {
                Ok(tools) => tools,
                Err(e) => {
                    tracing::error!("Failed to re-fetch TurboMCP tool list: {e}");
                    return Err(turbomcp_client::HandlerError::Generic {
                        message: format!("Failed to re-fetch tool list: {e}"),
                    });
                }
            };

            let mut managed = self.managed_tool_names.write().await;

            // Remove all previously registered tools
            for name in managed.drain(..) {
                if let Err(e) = self.tool_server_handle.remove_tool(&name).await {
                    tracing::warn!(
                        "Failed to remove TurboMCP tool '{name}' during refresh: {e}"
                    );
                }
            }

            // Register the updated tool list
            for tool in tools {
                let tool_name = tool.name.clone();
                let mcp_tool = TurboMcpTool::from_mcp_server(tool, self.client.clone());
                match self.tool_server_handle.add_tool(mcp_tool).await {
                    Ok(()) => {
                        managed.push(tool_name);
                    }
                    Err(e) => {
                        tracing::error!(
                            "Failed to register TurboMCP tool '{tool_name}': {e}"
                        );
                    }
                }
            }

            tracing::info!(
                tool_count = managed.len(),
                "TurboMCP tool list refreshed successfully"
            );

            Ok(())
        })
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use turbomcp_client::{CallToolResult, ContentBlock, Tool};
    use turbomcp_protocol::types::content::{
        AudioContent, ImageContent, ResourceLink, TextContent,
    };

    /// Mock TurboMcpToolCaller for testing without a real MCP server
    struct MockToolCaller {
        response: CallToolResult,
    }

    impl MockToolCaller {
        fn success(text: &str) -> Self {
            Self {
                response: CallToolResult {
                    content: vec![ContentBlock::Text(TextContent {
                        text: text.to_string(),
                        annotations: None,
                        meta: None,
                    })],
                    is_error: None,
                    structured_content: None,
                    _meta: None,
                    task_id: None,
                },
            }
        }

        fn error(msg: &str) -> Self {
            Self {
                response: CallToolResult {
                    content: vec![ContentBlock::Text(TextContent {
                        text: msg.to_string(),
                        annotations: None,
                        meta: None,
                    })],
                    is_error: Some(true),
                    structured_content: None,
                    _meta: None,
                    task_id: None,
                },
            }
        }

        fn with_content(content: Vec<ContentBlock>) -> Self {
            Self {
                response: CallToolResult {
                    content,
                    is_error: None,
                    structured_content: None,
                    _meta: None,
                    task_id: None,
                },
            }
        }
    }

    impl TurboMcpToolCaller for MockToolCaller {
        fn call_tool(
            &self,
            _name: &str,
            _arguments: Option<HashMap<String, serde_json::Value>>,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = turbomcp_client::Result<CallToolResult>>
                    + Send
                    + '_,
            >,
        > {
            let result = self.response.clone();
            Box::pin(async move { Ok(result) })
        }
    }

    fn make_tool(name: &str, description: &str) -> Tool {
        serde_json::from_value(serde_json::json!({
            "name": name,
            "description": description,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "x": { "type": "number" },
                    "y": { "type": "number" }
                },
                "required": ["x", "y"]
            }
        }))
        .expect("valid tool JSON")
    }

    #[test]
    fn test_tool_definition_from_ref() {
        let tool = make_tool("add", "Add two numbers");
        let def: ToolDefinition = (&tool).into();

        assert_eq!(def.name, "add");
        assert_eq!(def.description, "Add two numbers");
        assert!(def.parameters.is_object());
    }

    #[test]
    fn test_tool_definition_from_owned() {
        let tool = make_tool("multiply", "Multiply two numbers");
        let def: ToolDefinition = tool.into();

        assert_eq!(def.name, "multiply");
        assert_eq!(def.description, "Multiply two numbers");
    }

    #[test]
    fn test_tool_definition_empty_description() {
        let tool: Tool = serde_json::from_value(serde_json::json!({
            "name": "no_desc",
            "inputSchema": { "type": "object" }
        }))
        .expect("valid tool JSON");
        let def: ToolDefinition = tool.into();

        assert_eq!(def.name, "no_desc");
        assert_eq!(def.description, "");
    }

    #[tokio::test]
    async fn test_turbomcp_tool_name() {
        let tool = make_tool("my_tool", "A test tool");
        let caller = MockToolCaller::success("ok");
        let mcp_tool = TurboMcpTool::from_client_arc(tool, Arc::new(caller));

        assert_eq!(ToolDyn::name(&mcp_tool), "my_tool");
    }

    #[tokio::test]
    async fn test_turbomcp_tool_definition() {
        let tool = make_tool("calc", "Calculate things");
        let caller = MockToolCaller::success("ok");
        let mcp_tool = TurboMcpTool::from_client_arc(tool, Arc::new(caller));

        let def = ToolDyn::definition(&mcp_tool, "prompt".into()).await;
        assert_eq!(def.name, "calc");
        assert_eq!(def.description, "Calculate things");
    }

    #[tokio::test]
    async fn test_turbomcp_tool_call_text_result() {
        let tool = make_tool("add", "Add");
        let caller = MockToolCaller::success("42");
        let mcp_tool = TurboMcpTool::from_client_arc(tool, Arc::new(caller));

        let result = ToolDyn::call(&mcp_tool, r#"{"x": 1, "y": 2}"#.to_string()).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "42");
    }

    #[tokio::test]
    async fn test_turbomcp_tool_call_error_result() {
        let tool = make_tool("fail", "Fail");
        let caller = MockToolCaller::error("something went wrong");
        let mcp_tool = TurboMcpTool::from_client_arc(tool, Arc::new(caller));

        let result = ToolDyn::call(&mcp_tool, r#"{}"#.to_string()).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("something went wrong"));
    }

    #[tokio::test]
    async fn test_turbomcp_tool_call_error_no_message() {
        let caller = MockToolCaller {
            response: CallToolResult {
                content: vec![ContentBlock::Image(ImageContent {
                    data: "abc".into(),
                    mime_type: "image/png".into(),
                    annotations: None,
                    meta: None,
                })],
                is_error: Some(true),
                structured_content: None,
                _meta: None,
                task_id: None,
            },
        };
        let tool = make_tool("fail2", "Fail2");
        let mcp_tool = TurboMcpTool::from_client_arc(tool, Arc::new(caller));

        let result = ToolDyn::call(&mcp_tool, r#"{}"#.to_string()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No error message"));
    }

    #[tokio::test]
    async fn test_turbomcp_tool_call_image_content() {
        let tool = make_tool("img", "Image");
        let caller = MockToolCaller::with_content(vec![ContentBlock::Image(ImageContent {
            data: "base64data".into(),
            mime_type: "image/png".into(),
            annotations: None,
            meta: None,
        })]);
        let mcp_tool = TurboMcpTool::from_client_arc(tool, Arc::new(caller));

        let result = ToolDyn::call(&mcp_tool, r#"{}"#.to_string()).await.unwrap();
        assert_eq!(result, "data:image/png;base64,base64data");
    }

    #[tokio::test]
    async fn test_turbomcp_tool_call_audio_content() {
        let tool = make_tool("audio", "Audio");
        let caller = MockToolCaller::with_content(vec![ContentBlock::Audio(AudioContent {
            data: "audiodata".into(),
            mime_type: "audio/wav".into(),
            annotations: None,
            meta: None,
        })]);
        let mcp_tool = TurboMcpTool::from_client_arc(tool, Arc::new(caller));

        let result = ToolDyn::call(&mcp_tool, r#"{}"#.to_string()).await.unwrap();
        assert_eq!(result, "data:audio/wav;base64,audiodata");
    }

    #[tokio::test]
    async fn test_turbomcp_tool_call_resource_link_content() {
        let tool = make_tool("reslink", "ResLink");
        let link: ResourceLink = serde_json::from_value(serde_json::json!({
            "uri": "file:///tmp/test.txt",
            "name": "test"
        }))
        .expect("valid resource link JSON");
        let caller = MockToolCaller::with_content(vec![ContentBlock::ResourceLink(link)]);
        let mcp_tool = TurboMcpTool::from_client_arc(tool, Arc::new(caller));

        let result = ToolDyn::call(&mcp_tool, r#"{}"#.to_string()).await.unwrap();
        assert_eq!(result, "resource-link:file:///tmp/test.txt");
    }

    #[tokio::test]
    async fn test_turbomcp_tool_call_resource_text_with_mime() {
        use turbomcp_protocol::types::content::{EmbeddedResource, TextResourceContents};

        let tool = make_tool("res", "Resource");
        let caller = MockToolCaller::with_content(vec![ContentBlock::Resource(EmbeddedResource {
            resource: ResourceContent::Text(TextResourceContents {
                uri: "file:///data.csv".into(),
                mime_type: Some("text/csv".into()),
                text: "a,b,c".to_string(),
                meta: None,
            }),
            annotations: None,
            meta: None,
        })]);
        let mcp_tool = TurboMcpTool::from_client_arc(tool, Arc::new(caller));

        let result = ToolDyn::call(&mcp_tool, r#"{}"#.to_string()).await.unwrap();
        assert_eq!(result, "data:text/csv;file:///data.csv:a,b,c");
    }

    #[tokio::test]
    async fn test_turbomcp_tool_call_resource_text_without_mime() {
        use turbomcp_protocol::types::content::{EmbeddedResource, TextResourceContents};

        let tool = make_tool("res", "Resource");
        let caller = MockToolCaller::with_content(vec![ContentBlock::Resource(EmbeddedResource {
            resource: ResourceContent::Text(TextResourceContents {
                uri: "file:///data.txt".into(),
                mime_type: None,
                text: "hello".to_string(),
                meta: None,
            }),
            annotations: None,
            meta: None,
        })]);
        let mcp_tool = TurboMcpTool::from_client_arc(tool, Arc::new(caller));

        let result = ToolDyn::call(&mcp_tool, r#"{}"#.to_string()).await.unwrap();
        assert_eq!(result, "file:///data.txt:hello");
    }

    #[tokio::test]
    async fn test_turbomcp_tool_call_mixed_content() {
        let tool = make_tool("mixed", "Mixed");
        let caller = MockToolCaller::with_content(vec![
            ContentBlock::Text(TextContent {
                text: "Hello ".to_string(),
                annotations: None,
                meta: None,
            }),
            ContentBlock::Text(TextContent {
                text: "World".to_string(),
                annotations: None,
                meta: None,
            }),
        ]);
        let mcp_tool = TurboMcpTool::from_client_arc(tool, Arc::new(caller));

        let result = ToolDyn::call(&mcp_tool, r#"{}"#.to_string()).await.unwrap();
        assert_eq!(result, "Hello World");
    }

    #[tokio::test]
    async fn test_turbomcp_tool_call_invalid_json_args() {
        let tool = make_tool("t", "T");
        let caller = MockToolCaller::success("ok");
        let mcp_tool = TurboMcpTool::from_client_arc(tool, Arc::new(caller));

        // Invalid JSON args should still work (parsed as None)
        let result = ToolDyn::call(&mcp_tool, "not json".to_string()).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "ok");
    }

    #[test]
    fn test_turbomcp_tool_clone() {
        let tool = make_tool("cloneable", "Clone");
        let caller = MockToolCaller::success("ok");
        let mcp_tool = TurboMcpTool::from_client_arc(tool, Arc::new(caller));

        let cloned = mcp_tool.clone();
        assert_eq!(ToolDyn::name(&cloned), "cloneable");
    }

    #[tokio::test]
    async fn test_turbomcp_tool_in_toolset() {
        use crate::tool::ToolSet;

        let tool = make_tool("set_tool", "Tool in set");
        let caller = MockToolCaller::success("result");
        let mcp_tool = TurboMcpTool::from_client_arc(tool, Arc::new(caller));

        let toolset = ToolSet::from_tools(vec![mcp_tool]);
        assert!(toolset.contains("set_tool"));

        let result = toolset.call("set_tool", r#"{}"#.to_string()).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "result");
    }

    #[tokio::test]
    async fn test_turbomcp_multiple_tools_shared_client() {
        use crate::tool::ToolSet;

        let caller: Arc<dyn TurboMcpToolCaller> = Arc::new(MockToolCaller::success("shared"));

        let tool1 = TurboMcpTool::from_client_arc(make_tool("t1", "Tool 1"), caller.clone());
        let tool2 = TurboMcpTool::from_client_arc(make_tool("t2", "Tool 2"), caller.clone());

        let toolset = ToolSet::from_tools(vec![tool1, tool2]);
        assert!(toolset.contains("t1"));
        assert!(toolset.contains("t2"));
        assert_eq!(toolset.tools.len(), 2);

        let r1 = toolset.call("t1", r#"{}"#.to_string()).await.unwrap();
        let r2 = toolset.call("t2", r#"{}"#.to_string()).await.unwrap();
        assert_eq!(r1, "shared");
        assert_eq!(r2, "shared");
    }

    #[tokio::test]
    async fn test_turbomcp_tool_definitions_in_toolset() {
        use crate::tool::ToolSet;

        let caller: Arc<dyn TurboMcpToolCaller> = Arc::new(MockToolCaller::success("x"));

        let tool1 = TurboMcpTool::from_client_arc(make_tool("alpha", "Alpha tool"), caller.clone());
        let tool2 = TurboMcpTool::from_client_arc(make_tool("beta", "Beta tool"), caller);

        let toolset = ToolSet::from_tools(vec![tool1, tool2]);
        let defs = toolset.get_tool_definitions().await.unwrap();

        assert_eq!(defs.len(), 2);
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));
    }
}

/// Integration tests that exercise the real TurboMCP protocol path using an
/// in-process channel transport. These complement the mock tests above by
/// verifying the full JSON-RPC round-trip.
#[cfg(test)]
mod integration_tests {
    use std::future::Future;
    use std::sync::Arc;

    use tokio::sync::RwLock;
    use turbomcp::{
        McpError, McpHandler, McpResult, Prompt, PromptResult, Resource, ResourceResult,
        RequestContext as CoreRequestContext, ServerInfo, Tool, ToolResult,
    };

    use super::TurboMcpClientHandler;
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

    impl McpHandler for DynamicToolServer {
        fn server_info(&self) -> ServerInfo {
            ServerInfo::new("test-dynamic-server", "0.1.0")
        }

        fn list_tools(&self) -> Vec<Tool> {
            // McpHandler::list_tools is sync, so we can't await here.
            // Use try_read to avoid blocking; if contended, return empty.
            match self.tools.try_read() {
                Ok(tools) => tools.clone(),
                Err(_) => vec![],
            }
        }

        fn list_resources(&self) -> Vec<Resource> {
            vec![]
        }

        fn list_prompts(&self) -> Vec<Prompt> {
            vec![]
        }

        fn call_tool<'a>(
            &'a self,
            name: &'a str,
            _args: serde_json::Value,
            _ctx: &'a CoreRequestContext,
        ) -> impl Future<Output = McpResult<ToolResult>> + Send + 'a {
            let name = name.to_string();
            async move {
                Ok(ToolResult::text(format!("called {name}")))
            }
        }

        fn read_resource<'a>(
            &'a self,
            uri: &'a str,
            _ctx: &'a CoreRequestContext,
        ) -> impl Future<Output = McpResult<ResourceResult>> + Send + 'a {
            let uri = uri.to_string();
            async move { Err(McpError::resource_not_found(&uri)) }
        }

        fn get_prompt<'a>(
            &'a self,
            name: &'a str,
            _args: Option<serde_json::Value>,
            _ctx: &'a CoreRequestContext,
        ) -> impl Future<Output = McpResult<PromptResult>> + Send + 'a {
            let name = name.to_string();
            async move { Err(McpError::prompt_not_found(&name)) }
        }
    }

    #[tokio::test]
    async fn test_turbomcp_client_handler_initial_tool_registration() {
        let initial_tools = vec![
            Tool::new("tool_a", "First tool"),
            Tool::new("tool_b", "Second tool"),
        ];

        let server = DynamicToolServer::new(initial_tools);
        let tool_server_handle = ToolServer::new().run();

        let (client_transport, server_handle) =
            turbomcp_server::transport::channel::run_in_process(&server)
                .await
                .expect("failed to start in-process server");

        let client = turbomcp_client::Client::new(client_transport);
        let handler = TurboMcpClientHandler::new(client, tool_server_handle.clone());

        // connect() initializes the MCP connection, fetches tools, registers them
        let client = handler.connect().await.expect("connect failed");

        // Verify tools were registered in the ToolServer
        let defs = tool_server_handle.get_tool_defs(None).await.unwrap();
        assert_eq!(defs.len(), 2);

        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"tool_a"));
        assert!(names.contains(&"tool_b"));

        // Verify the tools are callable, not just registered
        let result = tool_server_handle
            .call_tool("tool_a", r#"{}"#)
            .await
            .unwrap();
        assert_eq!(result, "called tool_a");

        // Graceful teardown: drop client first (closes channel), then await server
        let _ = client.shutdown().await;
        server_handle.abort();
    }

    #[tokio::test]
    async fn test_turbomcp_client_handler_tool_call_through_tool_server() {
        let server = DynamicToolServer::new(vec![Tool::new("echo", "Echo tool")]);
        let tool_server_handle = ToolServer::new().run();

        let (client_transport, server_handle) =
            turbomcp_server::transport::channel::run_in_process(&server)
                .await
                .expect("failed to start in-process server");

        let client = turbomcp_client::Client::new(client_transport);
        let handler = TurboMcpClientHandler::new(client, tool_server_handle.clone());
        let client = handler.connect().await.expect("connect failed");

        // Verify we can call a tool through the ToolServer -> TurboMcpTool -> Client -> Server path
        let result = tool_server_handle.call_tool("echo", r#"{}"#).await.unwrap();
        assert_eq!(result, "called echo");

        // Verify a non-existent tool produces an error
        let err = tool_server_handle.call_tool("nonexistent", r#"{}"#).await;
        assert!(err.is_err());

        let _ = client.shutdown().await;
        server_handle.abort();
    }

    #[tokio::test]
    async fn test_turbomcp_client_handler_refreshes_on_tool_list_changed() {
        let initial_tools = vec![Tool::new("alpha", "Alpha tool")];
        let server = DynamicToolServer::new(initial_tools);
        let tool_server_handle = ToolServer::new().run();

        let (client_transport, server_handle) =
            turbomcp_server::transport::channel::run_in_process(&server)
                .await
                .expect("failed to start in-process server");

        let client = turbomcp_client::Client::new(client_transport);
        let handler = TurboMcpClientHandler::new(client, tool_server_handle.clone());
        let client = handler.connect().await.expect("connect failed");

        // Verify initial state
        let defs = tool_server_handle.get_tool_defs(None).await.unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "alpha");

        // Verify the handler was installed
        assert!(client.has_tool_list_changed_handler());

        // Update the server's tool list
        server
            .set_tools(vec![
                Tool::new("beta", "Beta tool"),
                Tool::new("gamma", "Gamma tool"),
            ])
            .await;

        // Trigger the handler directly. trigger_tool_list_changed() is fully
        // awaited inline — no sleep needed. The handler re-fetches tools via the
        // real protocol (client -> server), exercising the full JSON-RPC path.
        client
            .trigger_tool_list_changed()
            .await
            .expect("trigger_tool_list_changed failed");

        let defs = tool_server_handle.get_tool_defs(None).await.unwrap();
        assert_eq!(defs.len(), 2);

        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"beta"), "expected 'beta' in {names:?}");
        assert!(names.contains(&"gamma"), "expected 'gamma' in {names:?}");
        assert!(
            !names.contains(&"alpha"),
            "expected 'alpha' to be removed, found {names:?}"
        );

        // Verify the new tools are actually callable
        let result = tool_server_handle.call_tool("beta", r#"{}"#).await.unwrap();
        assert_eq!(result, "called beta");

        let _ = client.shutdown().await;
        server_handle.abort();
    }

    #[tokio::test]
    async fn test_turbomcp_client_handler_empty_tool_list() {
        let server = DynamicToolServer::new(vec![]);
        let tool_server_handle = ToolServer::new().run();

        let (client_transport, server_handle) =
            turbomcp_server::transport::channel::run_in_process(&server)
                .await
                .expect("failed to start in-process server");

        let client = turbomcp_client::Client::new(client_transport);
        let handler = TurboMcpClientHandler::new(client, tool_server_handle.clone());
        let client = handler.connect().await.expect("connect failed");

        // Empty tool list should register zero tools
        let defs = tool_server_handle.get_tool_defs(None).await.unwrap();
        assert_eq!(defs.len(), 0);

        // Verify refresh from empty to non-empty works
        server.set_tools(vec![Tool::new("new_tool", "New tool")]).await;
        client
            .trigger_tool_list_changed()
            .await
            .expect("trigger_tool_list_changed failed");

        let defs = tool_server_handle.get_tool_defs(None).await.unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "new_tool");

        let _ = client.shutdown().await;
        server_handle.abort();
    }
}
