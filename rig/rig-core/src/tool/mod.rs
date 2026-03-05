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

pub mod server;
use std::collections::HashMap;
use std::fmt;

use futures::Future;
use serde::{Deserialize, Serialize};

use crate::{
    completion::{self, ToolDefinition},
    embeddings::{embed::EmbedError, tool::ToolSchema},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[cfg(not(target_family = "wasm"))]
    /// Error returned by the tool
    ToolCallError(#[from] Box<dyn std::error::Error + Send + Sync>),

    #[cfg(target_family = "wasm")]
    /// Error returned by the tool
    ToolCallError(#[from] Box<dyn std::error::Error>),
    /// Error caused by a de/serialization fail
    JsonError(#[from] serde_json::Error),
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolError::ToolCallError(e) => {
                let error_str = e.to_string();
                // This is required due to being able to use agents as tools
                // which means it is possible to get recursive tool call errors
                if error_str.starts_with("ToolCallError: ") {
                    write!(f, "{}", error_str)
                } else {
                    write!(f, "ToolCallError: {}", error_str)
                }
            }
            ToolError::JsonError(e) => write!(f, "JsonError: {e}"),
        }
    }
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
pub trait Tool: Sized + WasmCompatSend + WasmCompatSync {
    /// The name of the tool. This name should be unique.
    const NAME: &'static str;

    /// The error type of the tool.
    type Error: std::error::Error + WasmCompatSend + WasmCompatSync + 'static;
    /// The arguments type of the tool.
    type Args: for<'a> Deserialize<'a> + WasmCompatSend + WasmCompatSync;
    /// The output type of the tool.
    type Output: Serialize;

    /// A method returning the name of the tool.
    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    /// A method returning the tool definition. The user prompt can be used to
    /// tailor the definition to the specific use case.
    fn definition(
        &self,
        _prompt: String,
    ) -> impl Future<Output = ToolDefinition> + WasmCompatSend + WasmCompatSync;

    /// The tool execution method.
    /// Both the arguments and return value are a String since these values are meant to
    /// be the output and input of LLM models (respectively)
    fn call(
        &self,
        args: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + WasmCompatSend;
}

/// Trait that represents an LLM tool that can be stored in a vector store and RAGged
pub trait ToolEmbedding: Tool {
    type InitError: std::error::Error + WasmCompatSend + WasmCompatSync + 'static;

    /// Type of the tool' context. This context will be saved and loaded from the
    /// vector store when ragging the tool.
    /// This context can be used to store the tool's static configuration and local
    /// context.
    type Context: for<'a> Deserialize<'a> + Serialize;

    /// Type of the tool's state. This state will be passed to the tool when initializing it.
    /// This state can be used to pass runtime arguments to the tool such as clients,
    /// API keys and other configuration.
    type State: WasmCompatSend;

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
pub trait ToolDyn: WasmCompatSend + WasmCompatSync {
    fn name(&self) -> String;

    fn definition<'a>(&'a self, prompt: String) -> WasmBoxedFuture<'a, ToolDefinition>;

    fn call<'a>(&'a self, args: String) -> WasmBoxedFuture<'a, Result<String, ToolError>>;
}

impl<T: Tool> ToolDyn for T {
    fn name(&self) -> String {
        self.name()
    }

    fn definition<'a>(&'a self, prompt: String) -> WasmBoxedFuture<'a, ToolDefinition> {
        Box::pin(<Self as Tool>::definition(self, prompt))
    }

    fn call<'a>(&'a self, args: String) -> WasmBoxedFuture<'a, Result<String, ToolError>> {
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
pub mod rmcp;

#[cfg(feature = "turbomcp")]
#[cfg_attr(docsrs, doc(cfg(feature = "turbomcp")))]
pub mod turbomcp {
    use crate::completion::ToolDefinition;
    use crate::tool::ToolDyn;
    use crate::tool::ToolError;
    use crate::wasm_compat::WasmBoxedFuture;
    use std::collections::HashMap;
    use std::sync::Arc;
    use turbomcp_client::{
        CallToolResult, Client, ContentBlock, ResourceContent, Result as McpResult, Tool,
        Transport,
    };

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

    /// A wrapper around a TurboMCP tool that implements Rig's ToolDyn trait.
    ///
    /// Bridges TurboMCP's tool system with Rig's tool abstraction,
    /// allowing TurboMCP tools to be used seamlessly with Rig agents.
    #[derive(Clone)]
    pub struct TurboMcpTool {
        definition: Tool,
        client: Arc<dyn TurboMcpToolCaller>,
    }

    impl TurboMcpTool {
        pub fn from_mcp_server<T>(definition: Tool, client: Client<T>) -> Self
        where
            T: Transport + 'static,
        {
            Self {
                definition,
                client: Arc::new(client),
            }
        }

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

    #[derive(Debug, thiserror::Error)]
    #[error("TurboMCP tool error: {0}")]
    pub struct TurboMcpToolError(String);

    impl From<TurboMcpToolError> for ToolError {
        fn from(e: TurboMcpToolError) -> Self {
            ToolError::ToolCallError(Box::new(e))
        }
    }

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
                            TurboMcpToolError("No error message returned".to_string()).into(),
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
                            ResourceContent::Text(text_res) => {
                                format!("resource:{}", text_res.uri)
                            }
                            ResourceContent::Blob(blob_res) => {
                                format!("resource:{}", blob_res.uri)
                            }
                        },
                        ContentBlock::ResourceLink(link) => {
                            format!("resource-link:{}", link.uri)
                        }
                        ContentBlock::Audio(audio) => {
                            format!("data:{};base64,{}", audio.mime_type, audio.data)
                        }
                        // ToolUse and ToolResult are MCP 2025-11-25 content types
                        // for nested tool invocations; convert to descriptive text
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
}

/// Wrapper trait to allow for dynamic dispatch of raggable tools
pub trait ToolEmbeddingDyn: ToolDyn {
    fn context(&self) -> serde_json::Result<serde_json::Value>;

    fn embedding_docs(&self) -> Vec<String>;
}

impl<T> ToolEmbeddingDyn for T
where
    T: ToolEmbedding + 'static,
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

    /// Could not find a tool
    #[error("ToolNotFoundError: {0}")]
    ToolNotFoundError(String),

    // TODO: Revisit this
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Tool call was interrupted. Primarily useful for agent multi-step/turn prompting.
    #[error("Tool call interrupted")]
    Interrupted,
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

    pub fn from_tools_boxed(tools: Vec<Box<dyn ToolDyn + 'static>>) -> Self {
        let mut toolset = Self::default();
        tools.into_iter().for_each(|tool| {
            toolset.add_tool_boxed(tool);
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

    /// Adds a boxed tool to the toolset. Useful for situations when dynamic dispatch is required.
    pub fn add_tool_boxed(&mut self, tool: Box<dyn ToolDyn>) {
        self.tools.insert(tool.name(), ToolType::Simple(tool));
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
            tracing::debug!(target: "rig",
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
}

#[cfg(all(test, feature = "turbomcp"))]
mod turbomcp_tests {
    use super::turbomcp::*;
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;
    use turbomcp_client::{CallToolResult, ContentBlock, Tool};
    use turbomcp_protocol::types::content::{
        AudioContent, EmbeddedResource, ImageContent, ResourceLink, TextContent,
    };
    use turbomcp_protocol::types::tools::ToolInputSchema;

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
                    data: "abc".to_string(),
                    mime_type: "image/png".to_string(),
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
            data: "base64data".to_string(),
            mime_type: "image/png".to_string(),
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
            data: "audiodata".to_string(),
            mime_type: "audio/wav".to_string(),
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
