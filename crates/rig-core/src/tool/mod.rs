//! Tool authoring, registration, and canonical structured execution.
//!
//! A typed [`Tool`] implements one execution method. Rig erases it internally,
//! executes it through one structured path, and exposes a single [`ToolResult`]
//! view to hooks and runtime callers.

pub mod builtin;
pub(crate) mod extensions;
mod result;
pub mod server;

pub use extensions::{MissingToolContext, ToolContext};
pub use result::{ToolErrorKind, ToolExecutionError, ToolResult};

use std::{collections::HashMap, sync::Arc};

use futures::Future;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::{
    completion::{self, ToolDefinition},
    embeddings::{embed::EmbedError, tool::ToolSchema},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

/// A typed LLM tool.
///
/// Tool authors provide metadata and exactly one execution method. Runtime
/// context and host-only result metadata share the [`ToolContext`] path.
pub trait Tool: Sized + WasmCompatSend + WasmCompatSync {
    /// Unique registration and provider-facing name.
    const NAME: &'static str;
    /// Typed JSON arguments.
    type Args: for<'de> Deserialize<'de> + WasmCompatSend + WasmCompatSync;
    /// Serializable output.
    type Output: Serialize;

    /// Registration name. Defaults to [`Self::NAME`].
    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    /// Model-facing description.
    fn description(&self) -> String;

    /// JSON Schema for arguments.
    fn parameters(&self) -> serde_json::Value;

    /// Execute the tool.
    fn call(
        &self,
        context: &mut ToolContext,
        args: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, ToolExecutionError>> + WasmCompatSend;
}

/// A tool that can be stored in a vector store and reconstructed for RAG.
pub trait ToolEmbedding: Tool {
    /// Error returned while reconstructing the tool.
    type InitError: std::error::Error + WasmCompatSend + WasmCompatSync + 'static;
    /// Serializable static context.
    type Context: for<'de> Deserialize<'de> + Serialize;
    /// Runtime initialization state.
    type State: WasmCompatSend;

    /// Documents used to retrieve the tool.
    fn embedding_docs(&self) -> Vec<String>;
    /// Serializable tool context.
    fn context(&self) -> Self::Context;
    /// Reconstruct the tool.
    fn init(state: Self::State, context: Self::Context) -> Result<Self, Self::InitError>;
}

fn serialize_tool_output(output: impl Serialize) -> Result<String, ToolExecutionError> {
    let value = serde_json::to_value(output).map_err(|error| {
        ToolExecutionError::other(format!("failed to serialize tool output: {error}"))
            .with_source(error)
    })?;
    Ok(match value {
        serde_json::Value::String(text) => text,
        value => value.to_string(),
    })
}

fn parse_tool_args<A>(args: &str) -> Result<A, ToolExecutionError>
where
    A: for<'de> Deserialize<'de>,
{
    match serde_json::from_str(args) {
        Ok(parsed) => Ok(parsed),
        Err(original) if args.trim() == "null" => serde_json::from_str("{}").map_err(|_| {
            ToolExecutionError::invalid_args(format!("failed to parse tool arguments: {original}"))
                .with_source(original)
        }),
        Err(error) => Err(ToolExecutionError::invalid_args(format!(
            "failed to parse tool arguments: {error}"
        ))
        .with_source(error)),
    }
}

/// Crate-private, object-safe dispatch boundary.
pub(crate) trait ErasedTool: WasmCompatSend + WasmCompatSync {
    fn name(&self) -> String;
    fn description(&self) -> String;
    fn parameters(&self) -> serde_json::Value;
    fn execute<'a>(
        &'a self,
        args: String,
        context: &'a mut ToolContext,
    ) -> WasmBoxedFuture<'a, ToolResult>;
}

impl<T> ErasedTool for T
where
    T: Tool,
{
    fn name(&self) -> String {
        Tool::name(self)
    }

    fn description(&self) -> String {
        Tool::description(self)
    }

    fn parameters(&self) -> serde_json::Value {
        Tool::parameters(self)
    }

    fn execute<'a>(
        &'a self,
        args: String,
        context: &'a mut ToolContext,
    ) -> WasmBoxedFuture<'a, ToolResult> {
        Box::pin(async move {
            let args = match parse_tool_args::<T::Args>(&args) {
                Ok(args) => args,
                Err(error) => return ToolResult::failed(error),
            };
            match Tool::call(self, context, args).await {
                Ok(output) => match serialize_tool_output(output) {
                    Ok(output) => ToolResult::success(output),
                    Err(error) => ToolResult::failed(error),
                },
                Err(error) => ToolResult::failed(error),
            }
        })
    }
}

trait DynamicCallback:
    for<'a> Fn(
        &'a mut ToolContext,
        serde_json::Value,
    ) -> WasmBoxedFuture<'a, Result<serde_json::Value, ToolExecutionError>>
    + WasmCompatSend
    + WasmCompatSync
{
}

impl<F> DynamicCallback for F where
    F: for<'a> Fn(
            &'a mut ToolContext,
            serde_json::Value,
        ) -> WasmBoxedFuture<'a, Result<serde_json::Value, ToolExecutionError>>
        + WasmCompatSend
        + WasmCompatSync
{
}

/// A runtime-defined tool backed by one closure.
///
/// This is the only public dynamic execution surface; users never implement
/// Rig's object-safe dispatch mirror.
#[derive(Clone)]
pub struct DynamicTool {
    name: String,
    description: String,
    parameters: serde_json::Value,
    callback: Arc<dyn DynamicCallback>,
}

impl DynamicTool {
    /// Create a runtime-defined tool.
    pub fn new<F>(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
        callback: F,
    ) -> Self
    where
        F: for<'a> Fn(
                &'a mut ToolContext,
                serde_json::Value,
            )
                -> WasmBoxedFuture<'a, Result<serde_json::Value, ToolExecutionError>>
            + WasmCompatSend
            + WasmCompatSync
            + 'static,
    {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            callback: Arc::new(callback),
        }
    }

    /// Runtime name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Provider-facing definition.
    pub fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
        }
    }
}

impl ErasedTool for DynamicTool {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn parameters(&self) -> serde_json::Value {
        self.parameters.clone()
    }

    fn execute<'a>(
        &'a self,
        args: String,
        context: &'a mut ToolContext,
    ) -> WasmBoxedFuture<'a, ToolResult> {
        Box::pin(async move {
            let args = match serde_json::from_str(&args) {
                Ok(args) => args,
                Err(error) => {
                    return ToolResult::failed(
                        ToolExecutionError::invalid_args(format!(
                            "failed to parse tool arguments: {error}"
                        ))
                        .with_source(error),
                    );
                }
            };
            match (self.callback)(context, args).await {
                Ok(output) => match serialize_tool_output(output) {
                    Ok(output) => ToolResult::success(output),
                    Err(error) => ToolResult::failed(error),
                },
                Err(error) => ToolResult::failed(error),
            }
        })
    }
}

/// Generate the provider-facing definition for a typed tool.
pub fn tool_definition<T: Tool>(tool: &T) -> ToolDefinition {
    ToolDefinition {
        name: tool.name(),
        description: tool.description(),
        parameters: tool.parameters(),
    }
}

fn definition_with_name(name: impl Into<String>, tool: &dyn ErasedTool) -> ToolDefinition {
    ToolDefinition {
        name: name.into(),
        description: tool.description(),
        parameters: tool.parameters(),
    }
}

#[cfg(feature = "rmcp")]
#[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
pub mod rmcp;

pub(crate) trait ErasedEmbeddingTool: ErasedTool {
    fn serialized_context(&self) -> serde_json::Result<serde_json::Value>;
    fn embedding_docs(&self) -> Vec<String>;
}

impl<T> ErasedEmbeddingTool for T
where
    T: ToolEmbedding + 'static,
{
    fn serialized_context(&self) -> serde_json::Result<serde_json::Value> {
        serde_json::to_value(ToolEmbedding::context(self))
    }

    fn embedding_docs(&self) -> Vec<String> {
        ToolEmbedding::embedding_docs(self)
    }
}

#[derive(Clone)]
pub(crate) enum RegisteredTool {
    Static(Arc<dyn ErasedTool>),
    Embedding(Arc<dyn ErasedEmbeddingTool>),
}

impl RegisteredTool {
    fn erased(&self) -> &dyn ErasedTool {
        match self {
            Self::Static(tool) => &**tool,
            Self::Embedding(tool) => &**tool,
        }
    }

    pub(crate) fn name(&self) -> String {
        self.erased().name()
    }

    pub(crate) fn definition_with_name(&self, name: impl Into<String>) -> ToolDefinition {
        definition_with_name(name, self.erased())
    }

    pub(crate) async fn execute(&self, args: String, context: &mut ToolContext) -> ToolResult {
        self.erased().execute(args, context).await
    }
}

/// Errors while preparing a tool set's definitions/documents.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ToolSetError {
    /// JSON serialization failed while preparing tool data.
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),
    /// Tool processing was interrupted.
    #[error("Tool call interrupted")]
    Interrupted,
}

/// An ordered collection of tools.
#[derive(Default)]
pub struct ToolSet {
    pub(crate) tools: IndexMap<String, RegisteredTool>,
}

impl ToolSet {
    /// Build a set from homogeneous typed tools.
    pub fn from_tools<T>(tools: Vec<T>) -> Self
    where
        T: Tool + 'static,
    {
        let mut set = Self::default();
        for tool in tools {
            set.add_tool(tool);
        }
        set
    }

    /// Build a set from runtime-defined tools.
    pub fn from_dynamic_tools(tools: Vec<DynamicTool>) -> Self {
        let mut set = Self::default();
        for tool in tools {
            set.add_dynamic_tool(tool);
        }
        set
    }

    /// Create a builder.
    pub fn builder() -> ToolSetBuilder {
        ToolSetBuilder::default()
    }

    /// Whether the name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Register a typed tool.
    pub fn add_tool<T>(&mut self, tool: T) -> String
    where
        T: Tool + 'static,
    {
        self.insert(RegisteredTool::Static(Arc::new(tool)))
    }

    /// Register a runtime-defined tool.
    pub fn add_dynamic_tool(&mut self, tool: DynamicTool) -> String {
        self.insert(RegisteredTool::Static(Arc::new(tool)))
    }

    #[cfg(feature = "rmcp")]
    pub(crate) fn add_erased(&mut self, tool: Arc<dyn ErasedTool>) -> String {
        self.insert(RegisteredTool::Static(tool))
    }

    pub(crate) fn insert(&mut self, tool: RegisteredTool) -> String {
        let name = tool.name();
        if self.tools.insert(name.clone(), tool).is_some() {
            tracing::warn!(tool_name = %name, "replacing an existing tool registration");
        }
        name
    }

    /// Delete a tool by name.
    pub fn delete_tool(&mut self, name: &str) {
        self.tools.shift_remove(name);
    }

    /// Merge another set, preserving registration order and replacing duplicates.
    pub fn add_tools(&mut self, set: ToolSet) {
        for (name, tool) in set.tools {
            if self.tools.insert(name.clone(), tool).is_some() {
                tracing::warn!(tool_name = %name, "replacing an existing tool registration");
            }
        }
    }

    pub(crate) fn get(&self, name: &str) -> Option<&RegisteredTool> {
        self.tools.get(name)
    }

    pub(crate) fn ordered_names(&self) -> impl Iterator<Item = &String> {
        self.tools.keys()
    }

    /// Provider-facing definitions in registration order.
    pub fn get_tool_definitions(&self) -> Result<Vec<ToolDefinition>, ToolSetError> {
        Ok(self
            .tools
            .iter()
            .map(|(name, tool)| tool.definition_with_name(name.clone()))
            .collect())
    }

    /// Execute one registered tool through the canonical structured path.
    pub async fn execute(
        &self,
        name: &str,
        args: impl Into<String>,
        context: &mut ToolContext,
    ) -> ToolResult {
        let mut dispatch = context.for_dispatch();
        let result = match self.tools.get(name) {
            Some(tool) => tool.execute(args.into(), &mut dispatch).await,
            None => ToolResult::failed(
                ToolExecutionError::not_found(format!("no tool named `{name}` is registered"))
                    .with_model_feedback(format!("tool `{name}` not found")),
            ),
        };
        *context = dispatch;
        result
    }

    /// Documents describing all registered tools.
    pub async fn documents(&self) -> Result<Vec<completion::Document>, ToolSetError> {
        let mut docs = Vec::new();
        for (name, tool) in &self.tools {
            let definition = tool.definition_with_name(name.clone());
            docs.push(completion::Document {
                id: name.clone(),
                text: format!(
                    "Tool: {name}\nDefinition: \n{}",
                    serde_json::to_string_pretty(&definition)?
                ),
                additional_props: HashMap::new(),
            });
        }
        Ok(docs)
    }

    /// Convert embedding tools to vector-store schemas.
    pub fn schemas(&self) -> Result<Vec<ToolSchema>, EmbedError> {
        self.tools
            .iter()
            .filter_map(|(name, registered)| match registered {
                RegisteredTool::Embedding(tool) => {
                    Some(ToolSchema::from_tool(name.clone(), &**tool))
                }
                RegisteredTool::Static(_) => None,
            })
            .collect()
    }
}

/// Builder for static, runtime-defined, and embedding tools.
#[derive(Default)]
pub struct ToolSetBuilder {
    tools: Vec<RegisteredTool>,
}

impl ToolSetBuilder {
    /// Add a typed static tool.
    pub fn static_tool<T>(mut self, tool: T) -> Self
    where
        T: Tool + 'static,
    {
        self.tools.push(RegisteredTool::Static(Arc::new(tool)));
        self
    }

    /// Add a runtime-defined static tool.
    pub fn runtime_tool(mut self, tool: DynamicTool) -> Self {
        self.tools.push(RegisteredTool::Static(Arc::new(tool)));
        self
    }

    /// Add an embedding-backed tool.
    pub fn dynamic_tool<T>(mut self, tool: T) -> Self
    where
        T: ToolEmbedding + 'static,
    {
        self.tools.push(RegisteredTool::Embedding(Arc::new(tool)));
        self
    }

    /// Build the set.
    pub fn build(self) -> ToolSet {
        let mut set = ToolSet::default();
        for tool in self.tools {
            set.insert(tool);
        }
        set
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Echo;

    impl Tool for Echo {
        const NAME: &'static str = "echo";
        type Args = serde_json::Value;
        type Output = serde_json::Value;

        fn description(&self) -> String {
            "echo arguments".into()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        async fn call(
            &self,
            context: &mut ToolContext,
            args: Self::Args,
        ) -> Result<Self::Output, ToolExecutionError> {
            context.insert_result("result-metadata".to_string());
            Ok(args)
        }
    }

    #[tokio::test]
    async fn ordered_definitions_and_structured_execution_are_canonical() {
        let mut set = ToolSet::default();
        set.add_tool(Echo);
        let definitions = set.get_tool_definitions().unwrap();
        assert_eq!(definitions[0].name, "echo");

        let mut context = ToolContext::new();
        context.insert(7_u32);
        let result = set.execute("echo", r#"{"value":1}"#, &mut context).await;
        assert!(result.is_success());
        assert_eq!(result.model_output(), r#"{"value":1}"#);
        assert_eq!(context.get::<u32>(), Some(&7));
        assert_eq!(
            context.result::<String>().map(String::as_str),
            Some("result-metadata")
        );
    }

    #[tokio::test]
    async fn dynamic_tool_preserves_concrete_error() {
        #[derive(Debug, thiserror::Error)]
        #[error("boom")]
        struct Boom;

        let tool = DynamicTool::new(
            "dynamic",
            "fails",
            serde_json::json!({"type":"object"}),
            |_context, _args| {
                Box::pin(async { Err(ToolExecutionError::provider("upstream").with_source(Boom)) })
            },
        );
        let set = ToolSet::from_dynamic_tools(vec![tool]);
        let result = set.execute("dynamic", "{}", &mut ToolContext::new()).await;
        assert!(result.error().is_some_and(|error| error.is::<Boom>()));
    }
}

#[cfg(test)]
mod migrated_tests {
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

    #[test]
    fn test_get_tool_definitions() {
        let toolset = get_test_toolset();
        let tools = toolset.get_tool_definitions().unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(
            tools
                .iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>(),
            vec!["add", "subtract"],
            "provider definitions must use registered tool names in order"
        );
        assert!(tools.iter().all(|tool| !tool.description.is_empty()));
        assert!(tools.iter().all(|tool| tool.parameters.is_object()));
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
            toolset.add_dynamic_tool(named_tool(name, "test tool"));
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

    /// A runtime-defined tool used by ordering and duplicate-registration tests.
    fn named_tool(name: &str, description: &str) -> DynamicTool {
        let output = format!("called {description}");
        DynamicTool::new(
            name,
            description,
            json!({ "type": "object", "properties": {} }),
            move |_context, _args| {
                let output = output.clone();
                Box::pin(async move { Ok(serde_json::Value::String(output)) })
            },
        )
    }

    #[test]
    fn tool_definition_uses_flattened_dyn_metadata() {
        let tool = named_tool("alpha", "runtime description");
        let definition = tool.definition();

        assert_eq!(definition.name, "alpha");
        assert_eq!(definition.description, "runtime description");
        assert_eq!(definition.parameters["type"], "object");
    }

    #[tokio::test]
    async fn tool_definitions_follow_registration_order() {
        // Enough names that any non-order-preserving storage would almost
        // surely surface a regression: its iteration order would differ from
        // insertion order.
        let names: Vec<String> = (0..32).map(|i| format!("tool_{i:02}")).collect();
        let mut toolset = ToolSet::default();
        for name in &names {
            toolset.add_dynamic_tool(named_tool(name, "test tool"));
        }

        let defs = toolset.get_tool_definitions().unwrap();
        let def_names: Vec<String> = defs.into_iter().map(|def| def.name).collect();
        assert_eq!(def_names, names);

        let docs = toolset.documents().await.unwrap();
        let doc_ids: Vec<String> = docs.into_iter().map(|doc| doc.id).collect();
        assert_eq!(doc_ids, names);
    }

    #[tokio::test]
    async fn registered_name_is_definition_source_of_truth() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct ChangingNameTool {
            calls: AtomicUsize,
        }

        impl Tool for ChangingNameTool {
            const NAME: &'static str = "unused";
            type Args = serde_json::Value;
            type Output = String;

            fn name(&self) -> String {
                match self.calls.fetch_add(1, Ordering::SeqCst) {
                    0 => "registered".to_string(),
                    _ => "changed".to_string(),
                }
            }
            fn description(&self) -> String {
                "changes name after registration".to_string()
            }
            fn parameters(&self) -> serde_json::Value {
                json!({ "type": "object", "properties": {} })
            }
            async fn call(
                &self,
                _context: &mut ToolContext,
                _args: Self::Args,
            ) -> Result<Self::Output, ToolExecutionError> {
                Ok("ok".to_string())
            }
        }

        let mut toolset = ToolSet::default();
        toolset.add_tool(ChangingNameTool {
            calls: AtomicUsize::new(0),
        });

        let defs = toolset.get_tool_definitions().unwrap();
        assert_eq!(defs[0].name, "registered");

        let docs = toolset.documents().await.unwrap();
        assert_eq!(docs[0].id, "registered");
        assert!(docs[0].text.contains("registered"));
        assert!(!docs[0].text.contains("changed"));
    }

    #[test]
    fn dynamic_tool_schemas_use_registered_name() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        #[derive(Debug, thiserror::Error)]
        #[error("init error")]
        struct InitError;

        struct ChangingDynamicTool {
            calls: AtomicUsize,
        }

        impl Tool for ChangingDynamicTool {
            const NAME: &'static str = "unused";
            type Args = serde_json::Value;
            type Output = String;

            fn name(&self) -> String {
                match self.calls.fetch_add(1, Ordering::SeqCst) {
                    0 => "registered_dynamic".to_string(),
                    _ => "changed_dynamic".to_string(),
                }
            }

            fn description(&self) -> String {
                "dynamic tool".to_string()
            }

            fn parameters(&self) -> serde_json::Value {
                json!({ "type": "object", "properties": {} })
            }

            async fn call(
                &self,
                _context: &mut ToolContext,
                _args: Self::Args,
            ) -> Result<Self::Output, ToolExecutionError> {
                Ok("ok".to_string())
            }
        }

        impl ToolEmbedding for ChangingDynamicTool {
            type InitError = InitError;
            type Context = ();
            type State = ();

            fn embedding_docs(&self) -> Vec<String> {
                vec!["dynamic tool docs".to_string()]
            }

            fn context(&self) -> Self::Context {}

            fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
                Ok(Self {
                    calls: AtomicUsize::new(0),
                })
            }
        }

        let toolset = ToolSet::builder()
            .dynamic_tool(ChangingDynamicTool {
                calls: AtomicUsize::new(0),
            })
            .build();

        let schemas = toolset.schemas().unwrap();
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0].name, "registered_dynamic");
        assert_eq!(schemas[0].embedding_docs, vec!["dynamic tool docs"]);
    }

    #[tokio::test]
    async fn duplicate_registration_replaces_in_place() {
        let mut toolset = ToolSet::default();
        toolset.add_dynamic_tool(named_tool("alpha", "first alpha"));
        toolset.add_dynamic_tool(named_tool("beta", "beta"));
        toolset.add_dynamic_tool(named_tool("alpha", "second alpha"));

        let defs = toolset.get_tool_definitions().unwrap();
        assert_eq!(
            defs.iter().map(|def| def.name.as_str()).collect::<Vec<_>>(),
            vec!["alpha", "beta"],
            "the duplicate should be deduped and keep its original position"
        );
        assert_eq!(
            defs[0].description, "second alpha",
            "the last registration should win"
        );

        let output = toolset
            .execute("alpha", "{}", &mut ToolContext::new())
            .await
            .model_output()
            .to_string();
        assert_eq!(output, "called second alpha");
    }

    #[tokio::test]
    async fn add_tools_merges_in_order_and_replaces_existing() {
        let mut base = ToolSet::default();
        base.add_dynamic_tool(named_tool("alpha", "base alpha"));
        base.add_dynamic_tool(named_tool("beta", "base beta"));

        let mut incoming = ToolSet::default();
        incoming.add_dynamic_tool(named_tool("gamma", "incoming gamma"));
        incoming.add_dynamic_tool(named_tool("alpha", "incoming alpha"));

        base.add_tools(incoming);

        let defs = base.get_tool_definitions().unwrap();
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
            .execute("string_output", "{}", &mut ToolContext::new())
            .await
            .model_output()
            .to_string();

        assert_eq!(output, "Hello\nWorld");
    }

    #[tokio::test]
    async fn structured_string_tool_outputs_remain_parseable() {
        let mut toolset = ToolSet::default();
        toolset.add_tool(MockImageOutputTool);

        let output = toolset
            .execute("image_output", "{}", &mut ToolContext::new())
            .await
            .model_output()
            .to_string();
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
            .execute("object_output", "{}", &mut ToolContext::new())
            .await
            .model_output()
            .to_string();

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
            .execute("example_tool", "null", &mut ToolContext::new())
            .await
            .model_output()
            .to_string();

        assert_eq!(output, "Example answer");
    }

    // Struct-typed args with all-optional fields — serde rejects `null` for these
    // even though the fields are optional. The normalization in crate-private erased dispatch
    // falls back from `null` to `{}` so callers can omit the
    // wrapping `Option<Args>` workaround.
    #[tokio::test]
    async fn null_args_are_normalized_to_empty_object() {
        #[derive(serde::Deserialize, serde::Serialize)]
        struct NoRequiredArgs {
            label: Option<String>,
        }

        struct NoArgTool;

        impl Tool for NoArgTool {
            const NAME: &'static str = "no_arg_tool";
            type Args = NoRequiredArgs;
            type Output = String;

            fn description(&self) -> String {
                "Tool with no required arguments".to_string()
            }

            fn parameters(&self) -> serde_json::Value {
                json!({"type": "object", "properties": {}})
            }

            async fn call(
                &self,
                _context: &mut ToolContext,
                args: Self::Args,
            ) -> Result<Self::Output, ToolExecutionError> {
                Ok(args.label.unwrap_or_else(|| "default".to_string()))
            }
        }

        let mut toolset = ToolSet::default();
        toolset.add_tool(NoArgTool);

        // `null` is what LLMs send when no arguments are provided; without the
        // normalization this would return `ToolExecutionError::JsonError`.
        let output = toolset
            .execute("no_arg_tool", "null", &mut ToolContext::new())
            .await
            .model_output()
            .to_string();

        assert_eq!(output, "default");
    }
}
