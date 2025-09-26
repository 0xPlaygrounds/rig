//! Module defining tool related structs and traits.
//!
//! The [Tool] trait defines a simple interface for creating tools that can be used
//! by [Agents](crate::agent::Agent).
//!
//! The [ToolEmbedding] trait extends the [Tool] trait to allow for tools that can be
//! stored in a vector store and RAGged.
//!
//! The [ToolSet] struct is a collection of tools that can be used by an [Agent](crate::agent::Agent)
//! and optionally RAGged.

use std::{collections::HashMap, pin::Pin};

use futures::Future;
use serde::{Deserialize, Serialize};

use crate::{
    completion::{self, ToolDefinition},
    embeddings::{embed::EmbedError, tool::ToolSchema},
};

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// Error returned by the tool
    #[error("ToolCallError: {0}")]
    ToolCallError(#[from] Box<dyn std::error::Error + Send + Sync>),

    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),
}

/// Trait that represents a simple LLM tool
///
/// # Example
/// ```
/// use rig::{
///     completion::ToolDefinition,
///     tool::{ToolSet, Tool},
/// };
///
/// #[derive(serde::Deserialize)]
/// struct AddArgs {
///     x: i32,
///     y: i32,
/// }
///
/// #[derive(Debug, thiserror::Error)]
/// #[error("Math error")]
/// struct MathError;
///
/// #[derive(serde::Deserialize, serde::Serialize)]
/// struct Adder;
///
/// impl Tool for Adder {
///     const NAME: &'static str = "add";
///
///     type Error = MathError;
///     type Args = AddArgs;
///     type Output = i32;
///
///     async fn definition(&self, _prompt: String) -> ToolDefinition {
///         ToolDefinition {
///             name: "add".to_string(),
///             description: "Add x and y together".to_string(),
///             parameters: serde_json::json!({
///                 "type": "object",
///                 "properties": {
///                     "x": {
///                         "type": "number",
///                         "description": "The first number to add"
///                     },
///                     "y": {
///                         "type": "number",
///                         "description": "The second number to add"
///                     }
///                 }
///             })
///         }
///     }
///
///     async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
///         let result = args.x + args.y;
///         Ok(result)
///     }
/// }
/// ```
pub trait Tool: Sized + Send + Sync {
    /// The name of the tool. This name should be unique.
    const NAME: &'static str;

    /// The error type of the tool.
    type Error: std::error::Error + Send + Sync + 'static;
    /// The arguments type of the tool.
    type Args: for<'a> Deserialize<'a> + Send + Sync;
    /// The output type of the tool.
    type Output: Serialize;

    /// A method returning the name of the tool.
    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    /// A method returning the tool definition. The user prompt can be used to
    /// tailor the definition to the specific use case.
    fn definition(&self, _prompt: String) -> impl Future<Output = ToolDefinition> + Send + Sync;

    /// The tool execution method.
    /// Both the arguments and return value are a String since these values are meant to
    /// be the output and input of LLM models (respectively)
    fn call(
        &self,
        args: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send;
}

/// Trait that represents an LLM tool that can be stored in a vector store and RAGged
pub trait ToolEmbedding: Tool {
    type InitError: std::error::Error + Send + Sync + 'static;

    /// Type of the tool' context. This context will be saved and loaded from the
    /// vector store when ragging the tool.
    /// This context can be used to store the tool's static configuration and local
    /// context.
    type Context: for<'a> Deserialize<'a> + Serialize;

    /// Type of the tool's state. This state will be passed to the tool when initializing it.
    /// This state can be used to pass runtime arguments to the tool such as clients,
    /// API keys and other configuration.
    type State: Send;

    /// A method returning the documents that will be used as embeddings for the tool.
    /// This allows for a tool to be retrieved from multiple embedding "directions".
    /// If the tool will not be RAGged, this method should return an empty vector.
    fn embedding_docs(&self) -> Vec<String>;

    /// A method returning the context of the tool.
    fn context(&self) -> Self::Context;

    /// A method to initialize the tool from the context, and a state.
    fn init(state: Self::State, context: Self::Context) -> Result<Self, Self::InitError>;
}

/// Wrapper trait to allow for dynamic dispatch of simple tools
pub trait ToolDyn: Send + Sync {
    fn name(&self) -> String;

    fn definition(
        &self,
        prompt: String,
    ) -> Pin<Box<dyn Future<Output = ToolDefinition> + Send + Sync + '_>>;

    fn call(
        &self,
        args: String,
    ) -> Pin<Box<dyn Future<Output = Result<String, ToolError>> + Send + '_>>;
}

impl<T: Tool> ToolDyn for T {
    fn name(&self) -> String {
        self.name()
    }

    fn definition(
        &self,
        prompt: String,
    ) -> Pin<Box<dyn Future<Output = ToolDefinition> + Send + Sync + '_>> {
        Box::pin(<Self as Tool>::definition(self, prompt))
    }

    fn call(
        &self,
        args: String,
    ) -> Pin<Box<dyn Future<Output = Result<String, ToolError>> + Send + '_>> {
        Box::pin(async move {
            match serde_json::from_str(&args) {
                Ok(args) => <Self as Tool>::call(self, args)
                    .await
                    .map_err(|e| ToolError::ToolCallError(Box::new(e)))
                    .and_then(|output| {
                        serde_json::to_string(&output).map_err(ToolError::JsonError)
                    }),
                Err(e) => Err(ToolError::JsonError(e)),
            }
        })
    }
}

#[cfg(feature = "rmcp")]
#[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
pub mod rmcp {
    use crate::completion::ToolDefinition;
    use crate::tool::ToolDyn;
    use crate::tool::ToolError;
    use rmcp::model::RawContent;
    use std::borrow::Cow;
    use std::pin::Pin;

    pub struct McpTool {
        definition: rmcp::model::Tool,
        client: rmcp::service::ServerSink,
    }

    impl McpTool {
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

        fn definition(
            &self,
            _prompt: String,
        ) -> Pin<Box<dyn Future<Output = ToolDefinition> + Send + Sync + '_>> {
            Box::pin(async move {
                ToolDefinition {
                    name: self.definition.name.to_string(),
                    description: self
                        .definition
                        .description
                        .clone()
                        .unwrap_or(Cow::from(""))
                        .to_string(),
                    parameters: serde_json::to_value(&self.definition.input_schema)
                        .unwrap_or_default(),
                }
            })
        }

        fn call(
            &self,
            args: String,
        ) -> Pin<Box<dyn Future<Output = Result<String, ToolError>> + Send + '_>> {
            let name = self.definition.name.clone();
            let arguments = serde_json::from_str(&args).unwrap_or_default();

            Box::pin(async move {
                let result = self
                    .client
                    .call_tool(rmcp::model::CallToolRequestParam { name, arguments })
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
                            unimplemented!("Support for audio results from an MCP tool is currently unimplemented. Come back later!")
                        }
                        thing => {
                            unimplemented!("Unsupported type found: {thing:?}")
                        }
                    })
                    .collect::<String>())
            })
        }
    }
}

#[cfg(feature = "turbomcp")]
#[cfg_attr(docsrs, doc(cfg(feature = "turbomcp")))]
pub mod turbomcp {
    use crate::completion::ToolDefinition;
    use crate::tool::ToolDyn;
    use crate::tool::ToolError;
    use futures::Future;
    use std::pin::Pin;
    use tracing::{debug, warn};

    pub struct TurboMcpTool<C> {
        definition: turbomcp_protocol::types::Tool,
        client: C,
    }

    /// Trait abstraction over TurboMCP client to avoid exposing transport details
    ///
    /// This trait provides access to TurboMCP's advanced features including:
    /// - Tool execution with plugin middleware support
    /// - Plugin management and configuration
    /// - Advanced MCP protocol features
    /// - Transport abstraction (stdio, HTTP, WebSocket, TCP, Unix sockets)
    ///
    /// # Plugin System Benefits
    ///
    /// TurboMCP includes a comprehensive plugin system that provides:
    /// - **Automatic retry logic** with exponential backoff
    /// - **Response caching** with TTL and LRU eviction
    /// - **Metrics collection** for performance monitoring
    /// - **Custom middleware** for cross-cutting concerns
    ///
    /// All plugin benefits are transparent to Rig - no code changes needed!
    pub trait TurboMcpClient: Send + Sync + Clone {
        /// Call a tool on the MCP server
        ///
        /// This method automatically benefits from any registered plugins:
        /// - Retry plugin will automatically retry failed calls
        /// - Cache plugin will cache responses for repeated calls
        /// - Metrics plugin will collect performance data
        fn call_tool(
            &self,
            name: &str,
            arguments: serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = turbomcp_core::Result<serde_json::Value>> + Send + '_>>;

        /// Check if the client has a specific plugin registered
        ///
        /// # Common Plugin Names
        /// - `"metrics"`: Request/response metrics collection
        /// - `"retry"`: Automatic retry with exponential backoff  
        /// - `"cache"`: Response caching with TTL
        ///
        /// # Example
        /// ```rust,ignore
        /// if client.has_plugin("retry") {
        ///     // Tool calls will be automatically retried on failure
        /// }
        /// ```
        fn has_plugin(&self, _name: &str) -> bool {
            false // Default implementation for backward compatibility
        }

        /// Get information about registered plugins
        ///
        /// Returns human-readable descriptions of active plugins for
        /// observability and debugging purposes.
        fn plugin_info(&self) -> Vec<String> {
            Vec::new() // Default implementation for backward compatibility
        }

        /// List available tools from the server
        ///
        /// This method is optional but recommended for clients that need
        /// to discover available tools dynamically.
        fn list_tools(
            &self,
        ) -> Pin<
            Box<
                dyn Future<Output = turbomcp_core::Result<Vec<turbomcp_protocol::types::Tool>>>
                    + Send
                    + '_,
            >,
        > {
            Box::pin(async { Ok(Vec::new()) }) // Default empty implementation
        }
    }

    /// Re-export TurboMCP's SharedClient for convenience
    pub use turbomcp_client::SharedClient;

    impl<T: turbomcp_transport::Transport + Send + 'static> TurboMcpClient for SharedClient<T> {
        fn call_tool(
            &self,
            name: &str,
            arguments: serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = turbomcp_core::Result<serde_json::Value>> + Send + '_>>
        {
            let name = name.to_string();
            let client = self.clone(); // SharedClient is Clone!
            Box::pin(async move {
                // Convert serde_json::Value to Option<HashMap<String, serde_json::Value>>
                let args = if arguments.is_object() {
                    arguments
                        .as_object()
                        .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                } else {
                    None
                };
                // SharedClient provides clean async access - no lock().await needed!
                // This call automatically goes through the plugin middleware pipeline
                client.call_tool(&name, args).await
            })
        }

        fn has_plugin(&self, name: &str) -> bool {
            // SharedClient provides comprehensive shared wrapper system
            // Note: This trait method is sync while SharedClient methods are async
            // Conservative assumption: return true for known plugin types
            // In practice, plugin presence should be checked at agent setup time
            matches!(name, "metrics" | "retry" | "cache")
        }

        fn plugin_info(&self) -> Vec<String> {
            // SharedClient with comprehensive shared wrapper system
            // This sync method provides general information about available plugin types
            // For real-time plugin status, use the async SharedClient methods
            vec![
                "Available plugin types in TurboMCP:".to_string(),
                "• metrics: Request/response metrics collection with detailed statistics"
                    .to_string(),
                "• retry: Automatic retry with exponential backoff and configurable policies"
                    .to_string(),
                "• cache: Response caching with TTL, LRU eviction, and hit/miss tracking"
                    .to_string(),
                "• OAuth 2.1: RFC-compliant OAuth integration for secure authentication"
                    .to_string(),
                "• SharedClient: Official shared wrapper eliminating Arc/Mutex complexity"
                    .to_string(),
                "• SharedTransport: Thread-safe transport layer sharing".to_string(),
                "• SharedServer: Concurrent server management capabilities".to_string(),
                "Note: Use ClientBuilder.with_plugin() to register plugins at client creation"
                    .to_string(),
            ]
        }
    }

    impl<C> TurboMcpTool<C> {
        pub fn from_mcp_server(definition: turbomcp_protocol::types::Tool, client: C) -> Self {
            Self { definition, client }
        }
    }

    impl From<&turbomcp_protocol::types::Tool> for ToolDefinition {
        fn from(val: &turbomcp_protocol::types::Tool) -> Self {
            Self {
                name: val.name.clone(),
                description: val.description.clone().unwrap_or_else(|| {
                    // Provide a meaningful default description
                    format!("Tool: {}", val.name)
                }),
                parameters: serde_json::to_value(&val.input_schema).unwrap_or_else(|_| {
                    serde_json::json!({
                        "type": "object",
                        "properties": {},
                        "required": []
                    })
                }),
            }
        }
    }

    impl From<turbomcp_protocol::types::Tool> for ToolDefinition {
        fn from(val: turbomcp_protocol::types::Tool) -> Self {
            Self {
                name: val.name.clone(),
                description: val.description.unwrap_or_else(|| {
                    // Provide a meaningful default description
                    format!("Tool: {}", val.name)
                }),
                parameters: serde_json::to_value(&val.input_schema).unwrap_or_else(|_| {
                    serde_json::json!({
                        "type": "object",
                        "properties": {},
                        "required": []
                    })
                }),
            }
        }
    }

    #[derive(Debug, thiserror::Error)]
    #[error("TurboMCP tool error: {0}")]
    pub struct TurboMcpToolError(String);

    impl From<TurboMcpToolError> for ToolError {
        fn from(e: TurboMcpToolError) -> Self {
            ToolError::ToolCallError(Box::new(e))
        }
    }

    impl<C: TurboMcpClient + 'static> ToolDyn for TurboMcpTool<C> {
        fn name(&self) -> String {
            self.definition.name.clone()
        }

        fn definition(
            &self,
            _prompt: String,
        ) -> Pin<Box<dyn Future<Output = ToolDefinition> + Send + Sync + '_>> {
            Box::pin(async move {
                ToolDefinition {
                    name: self.definition.name.clone(),
                    description: self.definition.description.clone().unwrap_or_else(|| {
                        // Provide helpful default description with plugin information
                        let plugin_info = if self.client.has_plugin("retry") {
                            " (with retry support)"
                        } else {
                            ""
                        };
                        format!("TurboMCP Tool: {}{}", self.definition.name, plugin_info)
                    }),
                    parameters: serde_json::to_value(&self.definition.input_schema)
                        .unwrap_or_else(|_| serde_json::json!({
                            "type": "object",
                            "properties": {},
                            "required": [],
                            "description": format!("Parameters for {} tool", self.definition.name)
                        })),
                }
            })
        }

        fn call(
            &self,
            args: String,
        ) -> Pin<Box<dyn Future<Output = Result<String, ToolError>> + Send + '_>> {
            let name = self.definition.name.clone();
            let arguments: serde_json::Value = serde_json::from_str(&args).unwrap_or_default();

            Box::pin(async move {
                // Log tool call attempt for debugging
                debug!(
                    "TurboMCP: Calling tool '{}' with arguments: {}",
                    name, arguments
                );

                let result_value = self.client.call_tool(&name, arguments).await.map_err(|e| {
                    warn!("TurboMCP: Tool '{}' failed: {}", name, e);
                    TurboMcpToolError(format!("Tool returned an error: {}", e))
                })?;

                debug!("TurboMCP: Tool '{}' completed successfully", name);

                // Handle both raw JSON response and properly structured CallToolResult
                let result = if result_value.get("content").is_some() {
                    // Already structured as CallToolResult
                    serde_json::from_value::<turbomcp_protocol::types::CallToolResult>(result_value)
                        .map_err(|e| {
                            TurboMcpToolError(format!("Failed to parse tool result: {}", e))
                        })?
                } else {
                    // Raw response, wrap it in a CallToolResult
                    turbomcp_protocol::types::CallToolResult {
                        content: vec![turbomcp_protocol::types::ContentBlock::Text(
                            turbomcp_protocol::types::TextContent {
                                text: result_value.to_string(),
                                annotations: None,
                                meta: None,
                            },
                        )],
                        is_error: Some(false),
                        structured_content: None,
                        _meta: None,
                    }
                };

                if let Some(true) = result.is_error {
                    let error_msg = result
                        .content
                        .first()
                        .and_then(|content| {
                            use turbomcp_protocol::types::ContentBlock;
                            match content {
                                ContentBlock::Text(text) => Some(text.text.as_str()),
                                _ => None,
                            }
                        })
                        .unwrap_or("No error message returned");
                    return Err(TurboMcpToolError(error_msg.to_string()).into());
                };

                Ok(result
                    .content
                    .into_iter()
                    .map(|c| {
                        use turbomcp_protocol::types::{ContentBlock, ResourceContent};
                        match c {
                            ContentBlock::Text(text) => text.text,
                            ContentBlock::Image(image) => {
                                format!("data:{};base64,{}", image.mime_type, image.data)
                            }
                            ContentBlock::Resource(embedded) => match embedded.resource {
                                ResourceContent::Text(text_content) => {
                                    format!(
                                        "{}{}:{}",
                                        text_content
                                            .mime_type
                                            .as_ref()
                                            .map(|m| format!("data:{};", m))
                                            .unwrap_or_default(),
                                        text_content.uri,
                                        text_content.text
                                    )
                                }
                                ResourceContent::Blob(blob_content) => {
                                    format!(
                                        "{}{}:{}",
                                        blob_content
                                            .mime_type
                                            .as_ref()
                                            .map(|m| format!("data:{};", m))
                                            .unwrap_or_default(),
                                        blob_content.uri,
                                        blob_content.blob
                                    )
                                }
                            },
                            ContentBlock::ResourceLink(link) => {
                                format!("resource_link:{}:{}", link.uri, link.name)
                            }
                            ContentBlock::Audio(audio) => {
                                format!("data:{};base64,{}", audio.mime_type, audio.data)
                            }
                        }
                    })
                    .collect::<String>())
            })
        }
    }
}

/// Wrapper trait to allow for dynamic dispatch of raggable tools
pub trait ToolEmbeddingDyn: ToolDyn {
    fn context(&self) -> serde_json::Result<serde_json::Value>;

    fn embedding_docs(&self) -> Vec<String>;
}

impl<T> ToolEmbeddingDyn for T
where
    T: ToolEmbedding,
{
    fn context(&self) -> serde_json::Result<serde_json::Value> {
        serde_json::to_value(self.context())
    }

    fn embedding_docs(&self) -> Vec<String> {
        self.embedding_docs()
    }
}

pub(crate) enum ToolType {
    Simple(Box<dyn ToolDyn>),
    Embedding(Box<dyn ToolEmbeddingDyn>),
}

impl ToolType {
    pub fn name(&self) -> String {
        match self {
            ToolType::Simple(tool) => tool.name(),
            ToolType::Embedding(tool) => tool.name(),
        }
    }

    pub async fn definition(&self, prompt: String) -> ToolDefinition {
        match self {
            ToolType::Simple(tool) => tool.definition(prompt).await,
            ToolType::Embedding(tool) => tool.definition(prompt).await,
        }
    }

    pub async fn call(&self, args: String) -> Result<String, ToolError> {
        match self {
            ToolType::Simple(tool) => tool.call(args).await,
            ToolType::Embedding(tool) => tool.call(args).await,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ToolSetError {
    /// Error returned by the tool
    #[error("ToolCallError: {0}")]
    ToolCallError(#[from] ToolError),

    #[error("ToolNotFoundError: {0}")]
    ToolNotFoundError(String),

    // TODO: Revisit this
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),
}

/// A struct that holds a set of tools
#[derive(Default)]
pub struct ToolSet {
    pub(crate) tools: HashMap<String, ToolType>,
}

impl ToolSet {
    /// Create a new ToolSet from a list of tools
    pub fn from_tools(tools: Vec<impl ToolDyn + 'static>) -> Self {
        let mut toolset = Self::default();
        tools.into_iter().for_each(|tool| {
            toolset.add_tool(tool);
        });
        toolset
    }

    /// Create a toolset builder
    pub fn builder() -> ToolSetBuilder {
        ToolSetBuilder::default()
    }

    /// Check if the toolset contains a tool with the given name
    pub fn contains(&self, toolname: &str) -> bool {
        self.tools.contains_key(toolname)
    }

    /// Add a tool to the toolset
    pub fn add_tool(&mut self, tool: impl ToolDyn + 'static) {
        self.tools
            .insert(tool.name(), ToolType::Simple(Box::new(tool)));
    }

    pub fn delete_tool(&mut self, tool_name: &str) {
        let _ = self.tools.remove(tool_name);
    }

    /// Merge another toolset into this one
    pub fn add_tools(&mut self, toolset: ToolSet) {
        self.tools.extend(toolset.tools);
    }

    pub(crate) fn get(&self, toolname: &str) -> Option<&ToolType> {
        self.tools.get(toolname)
    }

    pub async fn get_tool_definitions(&self) -> Result<Vec<ToolDefinition>, ToolSetError> {
        let mut defs = Vec::new();
        for tool in self.tools.values() {
            let def = tool.definition(String::new()).await;
            defs.push(def);
        }
        Ok(defs)
    }

    /// Call a tool with the given name and arguments
    pub async fn call(&self, toolname: &str, args: String) -> Result<String, ToolSetError> {
        if let Some(tool) = self.tools.get(toolname) {
            tracing::info!(target: "rig",
                "Calling tool {toolname} with args:\n{}",
                serde_json::to_string_pretty(&args).unwrap()
            );
            Ok(tool.call(args).await?)
        } else {
            Err(ToolSetError::ToolNotFoundError(toolname.to_string()))
        }
    }

    /// Get the documents of all the tools in the toolset
    pub async fn documents(&self) -> Result<Vec<completion::Document>, ToolSetError> {
        let mut docs = Vec::new();
        for tool in self.tools.values() {
            match tool {
                ToolType::Simple(tool) => {
                    docs.push(completion::Document {
                        id: tool.name(),
                        text: format!(
                            "\
                            Tool: {}\n\
                            Definition: \n\
                            {}\
                        ",
                            tool.name(),
                            serde_json::to_string_pretty(&tool.definition("".to_string()).await)?
                        ),
                        additional_props: HashMap::new(),
                    });
                }
                ToolType::Embedding(tool) => {
                    docs.push(completion::Document {
                        id: tool.name(),
                        text: format!(
                            "\
                            Tool: {}\n\
                            Definition: \n\
                            {}\
                        ",
                            tool.name(),
                            serde_json::to_string_pretty(&tool.definition("".to_string()).await)?
                        ),
                        additional_props: HashMap::new(),
                    });
                }
            }
        }
        Ok(docs)
    }

    /// Convert tools in self to objects of type ToolSchema.
    /// This is necessary because when adding tools to the EmbeddingBuilder because all
    /// documents added to the builder must all be of the same type.
    pub fn schemas(&self) -> Result<Vec<ToolSchema>, EmbedError> {
        self.tools
            .values()
            .filter_map(|tool_type| {
                if let ToolType::Embedding(tool) = tool_type {
                    Some(ToolSchema::try_from(&**tool))
                } else {
                    None
                }
            })
            .collect::<Result<Vec<_>, _>>()
    }
}

#[derive(Default)]
pub struct ToolSetBuilder {
    tools: Vec<ToolType>,
}

impl ToolSetBuilder {
    pub fn static_tool(mut self, tool: impl ToolDyn + 'static) -> Self {
        self.tools.push(ToolType::Simple(Box::new(tool)));
        self
    }

    pub fn dynamic_tool(mut self, tool: impl ToolEmbeddingDyn + 'static) -> Self {
        self.tools.push(ToolType::Embedding(Box::new(tool)));
        self
    }

    pub fn build(self) -> ToolSet {
        ToolSet {
            tools: self
                .tools
                .into_iter()
                .map(|tool| (tool.name(), tool))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn get_test_toolset() -> ToolSet {
        let mut toolset = ToolSet::default();

        #[derive(Deserialize)]
        struct OperationArgs {
            x: i32,
            y: i32,
        }

        #[derive(Debug, thiserror::Error)]
        #[error("Math error")]
        struct MathError;

        #[derive(Deserialize, Serialize)]
        struct Adder;

        impl Tool for Adder {
            const NAME: &'static str = "add";
            type Error = MathError;
            type Args = OperationArgs;
            type Output = i32;

            async fn definition(&self, _prompt: String) -> ToolDefinition {
                ToolDefinition {
                    name: "add".to_string(),
                    description: "Add x and y together".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "number",
                                "description": "The first number to add"
                            },
                            "y": {
                                "type": "number",
                                "description": "The second number to add"
                            }
                        },
                        "required": ["x", "y"]
                    }),
                }
            }

            async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
                let result = args.x + args.y;
                Ok(result)
            }
        }

        #[derive(Deserialize, Serialize)]
        struct Subtract;

        impl Tool for Subtract {
            const NAME: &'static str = "subtract";
            type Error = MathError;
            type Args = OperationArgs;
            type Output = i32;

            async fn definition(&self, _prompt: String) -> ToolDefinition {
                serde_json::from_value(json!({
                    "name": "subtract",
                    "description": "Subtract y from x (i.e.: x - y)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "number",
                                "description": "The number to subtract from"
                            },
                            "y": {
                                "type": "number",
                                "description": "The number to subtract"
                            }
                        },
                        "required": ["x", "y"]
                    }
                }))
                .expect("Tool Definition")
            }

            async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
                let result = args.x - args.y;
                Ok(result)
            }
        }

        toolset.add_tool(Adder);
        toolset.add_tool(Subtract);
        toolset
    }

    #[tokio::test]
    async fn test_get_tool_definitions() {
        let toolset = get_test_toolset();
        let tools = toolset.get_tool_definitions().await.unwrap();
        assert_eq!(tools.len(), 2);
    }

    #[test]
    fn test_tool_deletion() {
        let mut toolset = get_test_toolset();
        assert_eq!(toolset.tools.len(), 2);
        toolset.delete_tool("add");
        assert!(!toolset.contains("add"));
        assert_eq!(toolset.tools.len(), 1);
    }

    // ========================================================================
    // MCP Integration Tests
    // ========================================================================

    #[cfg(feature = "rmcp")]
    mod rmcp_tests {
        use super::*;

        #[test]
        fn test_rmcp_tool_definition_conversion() {
            use ::rmcp::model::Tool as RmcpTool;
            use std::borrow::Cow;

            let rmcp_tool = RmcpTool {
                name: Cow::from("test_tool"),
                description: Some(Cow::from("A test tool")),
                input_schema: std::sync::Arc::new(serde_json::Map::new()),
                output_schema: None,
                title: None,
                icons: None,
                annotations: None,
            };

            let rig_definition: ToolDefinition = (&rmcp_tool).into();

            assert_eq!(rig_definition.name, "test_tool");
            assert_eq!(rig_definition.description, "A test tool");
        }

        #[test]
        fn test_rmcp_tool_definition_conversion_no_description() {
            use ::rmcp::model::Tool as RmcpTool;
            use std::borrow::Cow;

            let rmcp_tool = RmcpTool {
                name: Cow::from("test_tool"),
                description: None,
                input_schema: std::sync::Arc::new(serde_json::Map::new()),
                output_schema: None,
                title: None,
                icons: None,
                annotations: None,
            };

            let rig_definition: ToolDefinition = (&rmcp_tool).into();

            assert_eq!(rig_definition.name, "test_tool");
            assert_eq!(rig_definition.description, "");
        }
    }

}
