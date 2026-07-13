//! Tool authoring, runtime dispatch, registration, and retrieval.
//!
//! Typed tools implement [`Tool`] with one execution method. [`DynamicTool`] is
//! the sole public runtime-defined tool type. Every path dispatches through an
//! ordinary `Result<_, ToolExecutionError>` while [`ToolContext`] carries both
//! inbound typed values and tool-authored result metadata.

pub mod builtin;
mod extensions;
mod result;
pub mod server;

pub use extensions::{MissingExtension, ToolContext};
pub use result::{ToolExecutionError, ToolExecutionErrorKind, ToolExecutionView};

use std::collections::HashMap;
use std::sync::Arc;

use futures::Future;
use indexmap::IndexMap;
use serde::{Serialize, de::DeserializeOwned};

use crate::{
    completion::{self, ToolDefinition},
    embeddings::{embed::EmbedError, tool::ToolSchema},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

pub(crate) use result::ToolDispatchResult;

/// A typed LLM tool with one canonical execution method.
pub trait Tool: Sized + WasmCompatSend + WasmCompatSync {
    /// Name used for registration, provider advertisement, and dispatch.
    const NAME: &'static str;

    /// JSON-deserializable arguments accepted by the tool.
    type Args: DeserializeOwned + WasmCompatSend + WasmCompatSync;
    /// Serializable successful output.
    type Output: Serialize;

    /// Runtime name. Override only when a typed tool intentionally derives its
    /// registration name from instance state.
    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    /// Model-facing description.
    fn description(&self) -> String;

    /// JSON Schema for arguments.
    fn parameters(&self) -> serde_json::Value;

    /// Execute the tool with its typed arguments and per-call context.
    fn call(
        &self,
        context: &mut ToolContext,
        args: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, ToolExecutionError>> + WasmCompatSend;
}

/// A tool that can be stored in a vector store and reconstructed at runtime.
pub trait ToolEmbedding: Tool {
    /// Error returned while reconstructing the tool.
    type InitError: std::error::Error + WasmCompatSend + WasmCompatSync + 'static;
    /// Serializable static configuration persisted in the vector store.
    type Context: DeserializeOwned + Serialize;
    /// Runtime state supplied during reconstruction.
    type State: WasmCompatSend;

    /// Text used to retrieve this tool.
    fn embedding_docs(&self) -> Vec<String>;
    /// Persisted static configuration.
    fn context(&self) -> Self::Context;
    /// Reconstruct the tool.
    fn init(state: Self::State, context: Self::Context) -> Result<Self, Self::InitError>;
}

fn serialize_tool_output(output: impl Serialize) -> Result<String, ToolExecutionError> {
    match serde_json::to_value(output) {
        Ok(serde_json::Value::String(text)) => Ok(text),
        Ok(value) => Ok(value.to_string()),
        Err(error) => Err(ToolExecutionError::other(format!(
            "failed to serialize tool output: {error}"
        ))
        .with_source(error)),
    }
}

fn parse_tool_args<A>(args: &str) -> Result<A, ToolExecutionError>
where
    A: DeserializeOwned,
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

pub(super) trait ErasedTool: WasmCompatSend + WasmCompatSync {
    fn name(&self) -> String;
    fn description(&self) -> String;
    fn parameters(&self) -> serde_json::Value;
    fn execute<'a>(
        &'a self,
        context: &'a mut ToolContext,
        args: &'a str,
    ) -> WasmBoxedFuture<'a, Result<String, ToolExecutionError>>;
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
        context: &'a mut ToolContext,
        args: &'a str,
    ) -> WasmBoxedFuture<'a, Result<String, ToolExecutionError>> {
        Box::pin(async move {
            let args = parse_tool_args::<T::Args>(args)?;
            let output = Tool::call(self, context, args).await?;
            serialize_tool_output(output)
        })
    }
}

/// Boxed callback returned by a runtime-defined [`DynamicTool`].
pub type DynamicToolFuture<'a> = WasmBoxedFuture<'a, Result<serde_json::Value, ToolExecutionError>>;

trait DynamicToolCallback: WasmCompatSend + WasmCompatSync {
    fn invoke<'a>(
        &'a self,
        context: &'a mut ToolContext,
        args: serde_json::Value,
    ) -> DynamicToolFuture<'a>;
}

impl<F> DynamicToolCallback for F
where
    F: for<'a> Fn(&'a mut ToolContext, serde_json::Value) -> DynamicToolFuture<'a>
        + WasmCompatSend
        + WasmCompatSync,
{
    fn invoke<'a>(
        &'a self,
        context: &'a mut ToolContext,
        args: serde_json::Value,
    ) -> DynamicToolFuture<'a> {
        self(context, args)
    }
}

struct CallbackTool {
    name: String,
    description: String,
    parameters: serde_json::Value,
    callback: Arc<dyn DynamicToolCallback>,
}

impl ErasedTool for CallbackTool {
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
        context: &'a mut ToolContext,
        args: &'a str,
    ) -> WasmBoxedFuture<'a, Result<String, ToolExecutionError>> {
        Box::pin(async move {
            let args = serde_json::from_str(args).map_err(|error| {
                ToolExecutionError::invalid_args(format!("failed to parse tool arguments: {error}"))
                    .with_source(error)
            })?;
            let output = self.callback.invoke(context, args).await?;
            serialize_tool_output(output)
        })
    }
}

/// Cloneable runtime-defined tool.
///
/// Typed tools normally register directly through [`ToolSet::add_tool`]. Use
/// this type when tool metadata or behavior is only known at runtime.
#[derive(Clone)]
pub struct DynamicTool {
    inner: Arc<dyn ErasedTool>,
}

impl DynamicTool {
    /// Erase a typed tool into a cloneable runtime value.
    pub fn new(tool: impl Tool + 'static) -> Self {
        Self {
            inner: Arc::new(tool),
        }
    }

    /// Construct a runtime tool from metadata and an async boxed callback.
    pub fn from_fn<F>(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
        callback: F,
    ) -> Self
    where
        F: for<'a> Fn(&'a mut ToolContext, serde_json::Value) -> DynamicToolFuture<'a>
            + WasmCompatSend
            + WasmCompatSync
            + 'static,
    {
        Self {
            inner: Arc::new(CallbackTool {
                name: name.into(),
                description: description.into(),
                parameters,
                callback: Arc::new(callback),
            }),
        }
    }

    #[cfg(feature = "rmcp")]
    pub(crate) fn from_erased(tool: impl ErasedTool + 'static) -> Self {
        Self {
            inner: Arc::new(tool),
        }
    }

    /// Runtime name.
    pub fn name(&self) -> String {
        self.inner.name()
    }

    /// Model-facing description.
    pub fn description(&self) -> String {
        self.inner.description()
    }

    /// JSON Schema for arguments.
    pub fn parameters(&self) -> serde_json::Value {
        self.inner.parameters()
    }

    /// Build the provider-facing definition for this runtime tool.
    pub fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name(),
            description: self.description(),
            parameters: self.parameters(),
        }
    }

    /// Execute through the canonical structured error path.
    pub async fn execute(
        &self,
        context: &mut ToolContext,
        args: &str,
    ) -> Result<String, ToolExecutionError> {
        context.clear_result_metadata();
        self.inner.execute(context, args).await
    }
}

impl std::fmt::Debug for DynamicTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicTool")
            .field("name", &self.name())
            .finish_non_exhaustive()
    }
}

/// Build a provider-facing definition from a typed tool.
pub fn tool_definition<T>(tool: &T) -> ToolDefinition
where
    T: Tool,
{
    ToolDefinition {
        name: tool.name(),
        description: tool.description(),
        parameters: tool.parameters(),
    }
}

pub(super) trait ErasedEmbeddingTool: ErasedTool {
    fn stored_context(&self) -> serde_json::Result<serde_json::Value>;
    fn embedding_docs(&self) -> Vec<String>;
}

impl<T> ErasedEmbeddingTool for T
where
    T: ToolEmbedding + 'static,
{
    fn stored_context(&self) -> serde_json::Result<serde_json::Value> {
        serde_json::to_value(ToolEmbedding::context(self))
    }

    fn embedding_docs(&self) -> Vec<String> {
        ToolEmbedding::embedding_docs(self)
    }
}

#[derive(Clone)]
pub(crate) enum RegisteredTool {
    Simple(DynamicTool),
    Embedding(Arc<dyn ErasedEmbeddingTool>),
}

impl RegisteredTool {
    fn name(&self) -> String {
        match self {
            Self::Simple(tool) => tool.name(),
            Self::Embedding(tool) => tool.name(),
        }
    }

    fn definition_with_name(&self, name: impl Into<String>) -> ToolDefinition {
        ToolDefinition {
            name: name.into(),
            description: match self {
                Self::Simple(tool) => tool.description(),
                Self::Embedding(tool) => tool.description(),
            },
            parameters: match self {
                Self::Simple(tool) => tool.parameters(),
                Self::Embedding(tool) => tool.parameters(),
            },
        }
    }

    async fn execute(
        &self,
        context: &mut ToolContext,
        args: &str,
    ) -> Result<String, ToolExecutionError> {
        context.clear_result_metadata();
        match self {
            Self::Simple(tool) => tool.inner.execute(context, args).await,
            Self::Embedding(tool) => tool.execute(context, args).await,
        }
    }
}

/// Errors produced while managing a tool set (not while executing a tool).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ToolSetError {
    /// JSON serialization failed while preparing tool metadata.
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),
    /// Tool call was interrupted.
    #[error("Tool call interrupted")]
    Interrupted,
}

/// Registration-ordered collection of tools.
#[derive(Default)]
pub struct ToolSet {
    pub(crate) tools: IndexMap<String, RegisteredTool>,
}

impl ToolSet {
    /// Create a set from typed tools of one concrete type.
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

    /// Create a set from heterogeneous runtime tools.
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

    /// Whether a tool is registered by name.
    pub fn contains(&self, tool_name: &str) -> bool {
        self.tools.contains_key(tool_name)
    }

    /// Register a typed tool and return its registered name.
    pub fn add_tool(&mut self, tool: impl Tool + 'static) -> String {
        self.insert(RegisteredTool::Simple(DynamicTool::new(tool)))
    }

    /// Register a runtime-defined tool and return its registered name.
    pub fn add_dynamic_tool(&mut self, tool: DynamicTool) -> String {
        self.insert(RegisteredTool::Simple(tool))
    }

    pub(crate) fn insert(&mut self, tool: RegisteredTool) -> String {
        let name = tool.name();
        if self.tools.insert(name.clone(), tool).is_some() {
            tracing::warn!(tool_name = %name, "replacing an already registered tool");
        }
        name
    }

    /// Remove a tool. Missing names are ignored.
    pub fn delete_tool(&mut self, tool_name: &str) {
        self.tools.shift_remove(tool_name);
    }

    /// Merge another set, replacing duplicates in place.
    pub fn add_tools(&mut self, toolset: ToolSet) {
        for (name, tool) in toolset.tools {
            if self.tools.insert(name.clone(), tool).is_some() {
                tracing::warn!(tool_name = %name, "replacing an already registered tool");
            }
        }
    }

    pub(crate) fn get(&self, tool_name: &str) -> Option<&RegisteredTool> {
        self.tools.get(tool_name)
    }

    pub(crate) fn ordered_names(&self) -> impl Iterator<Item = &String> {
        self.tools.keys()
    }

    fn ordered_entries(&self) -> impl Iterator<Item = (&String, &RegisteredTool)> {
        self.tools.iter()
    }

    /// Provider definitions in registration order.
    pub fn get_tool_definitions(&self) -> Result<Vec<ToolDefinition>, ToolSetError> {
        Ok(self
            .ordered_entries()
            .map(|(name, tool)| tool.definition_with_name(name.clone()))
            .collect())
    }

    /// Execute a registered tool through the canonical structured error path.
    pub async fn execute(
        &self,
        tool_name: &str,
        args: &str,
        context: &mut ToolContext,
    ) -> Result<String, ToolExecutionError> {
        context.clear_result_metadata();
        match self.tools.get(tool_name) {
            Some(tool) => {
                tracing::debug!(target: "rig", "Calling tool {tool_name} with args:\n{args}");
                tool.execute(context, args).await
            }
            None => Err(ToolExecutionError::not_found(format!(
                "no tool named `{tool_name}` is registered"
            ))
            .with_model_feedback(format!("tool `{tool_name}` not found"))),
        }
    }

    /// Embeddable documents for all tools in registration order.
    pub async fn documents(&self) -> Result<Vec<completion::Document>, ToolSetError> {
        let mut documents = Vec::new();
        for (name, tool) in self.ordered_entries() {
            let definition = tool.definition_with_name(name.clone());
            documents.push(completion::Document {
                id: name.clone(),
                text: format!(
                    "Tool: {name}\nDefinition: \n{}",
                    serde_json::to_string_pretty(&definition)?
                ),
                additional_props: HashMap::new(),
            });
        }
        Ok(documents)
    }

    /// Embedding schemas for retrievable tools.
    pub fn schemas(&self) -> Result<Vec<ToolSchema>, EmbedError> {
        self.ordered_entries()
            .filter_map(|(name, tool)| match tool {
                RegisteredTool::Embedding(tool) => Some(ToolSchema::from_parts(
                    name.clone(),
                    tool.stored_context(),
                    tool.embedding_docs(),
                )),
                RegisteredTool::Simple(_) => None,
            })
            .collect()
    }
}

/// Builder for static, runtime-defined, and retrievable tools.
#[derive(Default)]
pub struct ToolSetBuilder {
    tools: Vec<RegisteredTool>,
}

impl ToolSetBuilder {
    /// Add a typed static tool.
    pub fn static_tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools
            .push(RegisteredTool::Simple(DynamicTool::new(tool)));
        self
    }

    /// Add a runtime-defined static tool.
    pub fn runtime_tool(mut self, tool: DynamicTool) -> Self {
        self.tools.push(RegisteredTool::Simple(tool));
        self
    }

    /// Add a retrievable typed tool.
    pub fn dynamic_tool(mut self, tool: impl ToolEmbedding + 'static) -> Self {
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

#[cfg(feature = "rmcp")]
#[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
pub mod rmcp;

#[cfg(test)]
mod tests {
    use super::*;

    struct Echo;

    impl Tool for Echo {
        const NAME: &'static str = "echo";
        type Args = String;
        type Output = String;

        fn description(&self) -> String {
            "echo input".into()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "string"})
        }

        async fn call(
            &self,
            _context: &mut ToolContext,
            args: Self::Args,
        ) -> Result<Self::Output, ToolExecutionError> {
            Ok(args)
        }
    }

    #[tokio::test]
    async fn string_output_is_verbatim() {
        let mut set = ToolSet::default();
        set.add_tool(Echo);
        let mut context = ToolContext::new();
        assert_eq!(
            set.execute("echo", r#""hello\\nworld""#, &mut context)
                .await
                .unwrap(),
            "hello\\nworld"
        );
    }

    #[tokio::test]
    async fn unknown_tool_is_structured_not_found() {
        let error = ToolSet::default()
            .execute("missing", "{}", &mut ToolContext::new())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ToolExecutionErrorKind::NotFound);
    }

    struct MetadataTool;

    impl Tool for MetadataTool {
        const NAME: &'static str = "metadata";
        type Args = ();
        type Output = String;

        fn description(&self) -> String {
            "attaches metadata".to_string()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        async fn call(
            &self,
            context: &mut ToolContext,
            _args: Self::Args,
        ) -> Result<Self::Output, ToolExecutionError> {
            context.insert_result(42u32);
            Ok("ok".to_string())
        }
    }

    #[tokio::test]
    async fn missing_tool_clears_prior_result_metadata() {
        let mut set = ToolSet::default();
        set.add_tool(MetadataTool);
        let mut context = ToolContext::new();

        set.execute("metadata", "null", &mut context)
            .await
            .expect("metadata tool succeeds");
        assert_eq!(context.result::<u32>(), Some(&42));

        set.execute("missing", "{}", &mut context)
            .await
            .expect_err("missing tool fails");
        assert_eq!(context.result::<u32>(), None);
    }
}
