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
    #[error("ToolCallError: {0}")]
    ToolCallError(#[from] Box<dyn std::error::Error + Send + Sync>),

    #[cfg(target_family = "wasm")]
    /// Error returned by the tool
    #[error("ToolCallError: {0}")]
    ToolCallError(#[from] Box<dyn std::error::Error>),

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
pub mod rmcp {
    use crate::completion::ToolDefinition;
    use crate::tool::ToolDyn;
    use crate::tool::ToolError;
    use crate::wasm_compat::WasmBoxedFuture;
    use rmcp::model::RawContent;
    use std::borrow::Cow;

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
                    parameters: serde_json::to_value(&self.definition.input_schema)
                        .unwrap_or_default(),
                }
            })
        }

        fn call(&self, args: String) -> WasmBoxedFuture<'_, Result<String, ToolError>> {
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
}
