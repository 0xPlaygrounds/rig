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
use std::sync::Arc;

use futures::Future;
use indexmap::IndexMap;
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
/// use rig_core::{
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
    /// The name of the tool. This name should be unique within a single
    /// [`ToolSet`] or other registration scope that dispatches tools by name.
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
    /// Error returned when reconstructing a dynamic tool from stored context.
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
    /// Returns the tool name used for dispatch.
    fn name(&self) -> String;

    /// Returns the provider-facing tool schema.
    fn definition<'a>(&'a self, prompt: String) -> WasmBoxedFuture<'a, ToolDefinition>;

    /// Calls the tool with JSON-encoded arguments and returns model-facing text.
    fn call<'a>(&'a self, args: String) -> WasmBoxedFuture<'a, Result<String, ToolError>>;
}

fn serialize_tool_output(output: impl Serialize) -> serde_json::Result<String> {
    match serde_json::to_value(output)? {
        serde_json::Value::String(text) => Ok(text),
        value => Ok(value.to_string()),
    }
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
            // LLMs frequently send `null` for tools whose arguments are all optional.
            // `serde_json::from_str::<T>("null")` fails for struct types even when
            // every field is `Option<_>`, because JSON null does not deserialize to an
            // empty object. Preserve any args type that already accepts `null` (such as
            // `()` or `Option<T>`) and fall back to `{}` only after the original parse
            // fails.
            let args = match serde_json::from_str(&args) {
                Ok(args) => Ok(args),
                Err(err) if args.trim() == "null" => serde_json::from_str("{}").map_err(|_| err),
                Err(err) => Err(err),
            };
            match args {
                Ok(args) => <Self as Tool>::call(self, args)
                    .await
                    .map_err(|e| ToolError::ToolCallError(Box::new(e)))
                    .and_then(|output| serialize_tool_output(output).map_err(ToolError::JsonError)),
                Err(e) => Err(ToolError::JsonError(e)),
            }
        })
    }
}

#[cfg(feature = "rmcp")]
#[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
pub mod rmcp;

/// Wrapper trait to allow for dynamic dispatch of raggable tools
pub trait ToolEmbeddingDyn: ToolDyn {
    /// Serializes context needed to reconstruct this dynamic tool.
    fn context(&self) -> serde_json::Result<serde_json::Value>;

    /// Returns text fragments used to retrieve this tool from a vector store.
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

#[derive(Clone)]
pub(crate) enum ToolType {
    Simple(Arc<dyn ToolDyn>),
    Embedding(Arc<dyn ToolEmbeddingDyn>),
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

    /// JSON serialization or deserialization failed while preparing tool data.
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Tool call was interrupted. Primarily useful for agent multi-step/turn prompting.
    #[error("Tool call interrupted")]
    Interrupted,
}

/// A struct that holds a set of tools.
///
/// Tools are stored in an [`IndexMap`] keyed by name, so iteration
/// (definitions, documents, schemas) follows registration order and the tool
/// list sent to providers is deterministic across processes. Re-registering an
/// existing name replaces the implementation but keeps its original position.
#[derive(Default)]
pub struct ToolSet {
    pub(crate) tools: IndexMap<String, ToolType>,
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

    /// Create a new `ToolSet` from boxed dynamically-dispatched tools.
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
        self.insert(ToolType::Simple(Arc::new(tool)));
    }

    /// Adds a boxed tool to the toolset. Useful for situations when dynamic dispatch is required.
    pub fn add_tool_boxed(&mut self, tool: Box<dyn ToolDyn>) {
        self.insert(ToolType::Simple(Arc::from(tool)));
    }

    pub(crate) fn insert(&mut self, tool: ToolType) {
        let name = tool.name();
        // `IndexMap::insert` replaces the value while keeping the existing
        // slot position, and returns the previous value when the name was
        // already registered.
        if self.tools.insert(name.clone(), tool).is_some() {
            tracing::warn!(
                tool_name = %name,
                "a tool named {name:?} was already registered; replacing it with the new registration"
            );
        }
    }

    /// Remove a tool by name. Missing tools are ignored.
    pub fn delete_tool(&mut self, tool_name: &str) {
        // `shift_remove` preserves the order of the remaining tools;
        // `swap_remove` would not.
        self.tools.shift_remove(tool_name);
    }

    /// Merge another toolset into this one. Tools keep `toolset`'s
    /// registration order; names that already exist are replaced in place.
    pub fn add_tools(&mut self, toolset: ToolSet) {
        for (_, tool) in toolset.tools {
            self.insert(tool);
        }
    }

    pub(crate) fn get(&self, toolname: &str) -> Option<&ToolType> {
        self.tools.get(toolname)
    }

    /// Tool names in registration order.
    pub(crate) fn ordered_names(&self) -> impl Iterator<Item = &String> {
        self.tools.keys()
    }

    /// Tools in registration order.
    fn ordered_tools(&self) -> impl Iterator<Item = &ToolType> {
        self.tools.values()
    }

    /// Return definitions for all tools currently registered in the set, in
    /// registration order.
    pub async fn get_tool_definitions(&self) -> Result<Vec<ToolDefinition>, ToolSetError> {
        let mut defs = Vec::new();
        for tool in self.ordered_tools() {
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
                args
            );
            Ok(tool.call(args).await?)
        } else {
            Err(ToolSetError::ToolNotFoundError(toolname.to_string()))
        }
    }

    /// Get the documents of all the tools in the toolset
    pub async fn documents(&self) -> Result<Vec<completion::Document>, ToolSetError> {
        let mut docs = Vec::new();
        for tool in self.ordered_tools() {
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
        self.ordered_tools()
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
/// Builder for constructing a [`ToolSet`] with static and dynamic tools.
pub struct ToolSetBuilder {
    tools: Vec<ToolType>,
}

impl ToolSetBuilder {
    /// Add a regular tool that is always available when the set is used.
    pub fn static_tool(mut self, tool: impl ToolDyn + 'static) -> Self {
        self.tools.push(ToolType::Simple(Arc::new(tool)));
        self
    }

    /// Add a tool that can be represented as embeddings for dynamic retrieval.
    pub fn dynamic_tool(mut self, tool: impl ToolEmbeddingDyn + 'static) -> Self {
        self.tools.push(ToolType::Embedding(Arc::new(tool)));
        self
    }

    /// Build the tool set, keyed by each tool's name.
    pub fn build(self) -> ToolSet {
        let mut toolset = ToolSet::default();
        for tool in self.tools {
            toolset.insert(tool);
        }
        toolset
    }
}

#[cfg(test)]
mod tests {
    use crate::message::{DocumentSourceKind, ToolResultContent};
    use crate::test_utils::{
        MockExampleTool, MockImageOutputTool, MockObjectOutputTool, MockStringOutputTool,
        mock_math_toolset,
    };
    use serde_json::json;

    use super::*;

    fn get_test_toolset() -> ToolSet {
        mock_math_toolset()
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
        assert_eq!(
            toolset.ordered_names().cloned().collect::<Vec<_>>(),
            vec!["subtract".to_string()]
        );
    }

    #[test]
    fn deleting_a_middle_tool_preserves_order_of_survivors() {
        // Guards the `shift_remove` (not `swap_remove`) choice in `delete_tool`.
        // `swap_remove` would move the last tool into the deleted slot, so this
        // only catches a regression with 3+ tools and a non-last deletion: here
        // a `swap_remove("beta")` would yield [alpha, delta, gamma].
        let mut toolset = ToolSet::default();
        for name in ["alpha", "beta", "gamma", "delta"] {
            toolset.add_tool(named_tool(name, "test tool"));
        }

        toolset.delete_tool("beta");

        assert_eq!(
            toolset.ordered_names().cloned().collect::<Vec<_>>(),
            vec![
                "alpha".to_string(),
                "gamma".to_string(),
                "delta".to_string()
            ],
            "survivors must keep their registration order after a middle deletion"
        );
    }

    /// A tool whose name and definition are chosen at runtime, for ordering
    /// and duplicate-registration tests.
    struct NamedTool {
        name: String,
        description: String,
    }

    impl ToolDyn for NamedTool {
        fn name(&self) -> String {
            self.name.clone()
        }

        fn definition(&self, _prompt: String) -> WasmBoxedFuture<'_, ToolDefinition> {
            Box::pin(async move {
                ToolDefinition {
                    name: self.name.clone(),
                    description: self.description.clone(),
                    parameters: json!({ "type": "object", "properties": {} }),
                }
            })
        }

        fn call(&self, _args: String) -> WasmBoxedFuture<'_, Result<String, ToolError>> {
            let output = format!("called {}", self.description);
            Box::pin(async move { Ok(output) })
        }
    }

    fn named_tool(name: &str, description: &str) -> NamedTool {
        NamedTool {
            name: name.to_string(),
            description: description.to_string(),
        }
    }

    #[tokio::test]
    async fn tool_definitions_follow_registration_order() {
        // Enough names that any non-order-preserving storage would almost
        // surely surface a regression: its iteration order would differ from
        // insertion order.
        let names: Vec<String> = (0..32).map(|i| format!("tool_{i:02}")).collect();
        let mut toolset = ToolSet::default();
        for name in &names {
            toolset.add_tool(named_tool(name, "test tool"));
        }

        let defs = toolset.get_tool_definitions().await.unwrap();
        let def_names: Vec<String> = defs.into_iter().map(|def| def.name).collect();
        assert_eq!(def_names, names);

        let docs = toolset.documents().await.unwrap();
        let doc_ids: Vec<String> = docs.into_iter().map(|doc| doc.id).collect();
        assert_eq!(doc_ids, names);
    }

    #[tokio::test]
    async fn duplicate_registration_replaces_in_place() {
        let mut toolset = ToolSet::default();
        toolset.add_tool(named_tool("alpha", "first alpha"));
        toolset.add_tool(named_tool("beta", "beta"));
        toolset.add_tool(named_tool("alpha", "second alpha"));

        let defs = toolset.get_tool_definitions().await.unwrap();
        assert_eq!(
            defs.iter().map(|def| def.name.as_str()).collect::<Vec<_>>(),
            vec!["alpha", "beta"],
            "the duplicate should be deduped and keep its original position"
        );
        assert_eq!(
            defs[0].description, "second alpha",
            "the last registration should win"
        );

        let output = toolset.call("alpha", "{}".to_string()).await.unwrap();
        assert_eq!(output, "called second alpha");
    }

    #[tokio::test]
    async fn add_tools_merges_in_order_and_replaces_existing() {
        let mut base = ToolSet::default();
        base.add_tool(named_tool("alpha", "base alpha"));
        base.add_tool(named_tool("beta", "base beta"));

        let mut incoming = ToolSet::default();
        incoming.add_tool(named_tool("gamma", "incoming gamma"));
        incoming.add_tool(named_tool("alpha", "incoming alpha"));

        base.add_tools(incoming);

        let defs = base.get_tool_definitions().await.unwrap();
        assert_eq!(
            defs.iter().map(|def| def.name.as_str()).collect::<Vec<_>>(),
            vec!["alpha", "beta", "gamma"],
            "merged tools should follow registration order with replaced names keeping position"
        );
        assert_eq!(defs[0].description, "incoming alpha");
    }

    #[tokio::test]
    async fn string_tool_outputs_are_preserved_verbatim() {
        let mut toolset = ToolSet::default();
        toolset.add_tool(MockStringOutputTool);

        let output = toolset
            .call("string_output", "{}".to_string())
            .await
            .expect("tool should succeed");

        assert_eq!(output, "Hello\nWorld");
    }

    #[tokio::test]
    async fn structured_string_tool_outputs_remain_parseable() {
        let mut toolset = ToolSet::default();
        toolset.add_tool(MockImageOutputTool);

        let output = toolset
            .call("image_output", "{}".to_string())
            .await
            .expect("tool should succeed");
        let content = ToolResultContent::from_tool_output(output);

        assert_eq!(content.len(), 1);
        match content.first() {
            ToolResultContent::Image(image) => {
                assert!(matches!(image.data, DocumentSourceKind::Base64(_)));
                assert_eq!(image.media_type, Some(crate::message::ImageMediaType::PNG));
            }
            other => panic!("expected image tool result content, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn object_tool_outputs_still_serialize_as_json() {
        let mut toolset = ToolSet::default();
        toolset.add_tool(MockObjectOutputTool);

        let output = toolset
            .call("object_output", "{}".to_string())
            .await
            .expect("tool should succeed");

        assert!(output.starts_with('{'));
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&output).unwrap(),
            json!({
                "status": "ok",
                "count": 42
            })
        );
    }

    #[tokio::test]
    async fn null_args_are_preserved_for_unit_args() {
        let mut toolset = ToolSet::default();
        toolset.add_tool(MockExampleTool);

        let output = toolset
            .call("example_tool", "null".to_string())
            .await
            .expect("unit args should accept null without object fallback");

        assert_eq!(output, "Example answer");
    }

    // Struct-typed args with all-optional fields — serde rejects `null` for these
    // even though the fields are optional. The normalization in `ToolDyn::call`
    // falls back from `null` to `{}` so callers can omit the
    // wrapping `Option<Args>` workaround.
    #[tokio::test]
    async fn null_args_are_normalized_to_empty_object() {
        use crate::test_utils::MockToolError;

        #[derive(serde::Deserialize, serde::Serialize)]
        struct NoRequiredArgs {
            label: Option<String>,
        }

        struct NoArgTool;

        impl Tool for NoArgTool {
            const NAME: &'static str = "no_arg_tool";
            type Error = MockToolError;
            type Args = NoRequiredArgs;
            type Output = String;

            async fn definition(&self, _prompt: String) -> ToolDefinition {
                ToolDefinition {
                    name: Self::NAME.to_string(),
                    description: "Tool with no required arguments".to_string(),
                    parameters: json!({"type": "object", "properties": {}}),
                }
            }

            async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
                Ok(args.label.unwrap_or_else(|| "default".to_string()))
            }
        }

        let mut toolset = ToolSet::default();
        toolset.add_tool(NoArgTool);

        // `null` is what LLMs send when no arguments are provided; without the
        // normalization this would return `ToolError::JsonError`.
        let output = toolset
            .call("no_arg_tool", "null".to_string())
            .await
            .expect("null args should succeed after normalisation");

        assert_eq!(output, "default");
    }
}
