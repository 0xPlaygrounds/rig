//! Typed tools and Rig's canonical structured tool runtime.
//!
//! A typed [`Tool`] exposes one execution method. Runtime dispatch erases that
//! type internally and always returns one [`ToolExecution`]; there is no parallel
//! string-returning path. [`DynamicTool`] is the public adapter for tools whose
//! name, schema, or implementation is only known at runtime.

pub mod builtin;
mod context;
mod result;
pub mod server;

pub use context::{MissingContext, ToolContext};
pub use result::{ToolErrorKind, ToolExecution, ToolExecutionError, ToolExecutionStatus};

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use indexmap::IndexMap;
use serde::{Serialize, de::DeserializeOwned};

use crate::{
    completion::{self, ToolDefinition},
    embeddings::{embed::EmbedError, tool::ToolSchema},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

/// Typed authoring API for a tool.
///
/// Implementors describe their provider-facing schema and define exactly one
/// execution method. Inbound private values and outbound metadata both travel
/// through [`ToolContext`], while every failure uses
/// [`ToolExecutionError`]. The runtime handles argument decoding and canonical
/// output rendering.
///
/// # Example
///
/// ```
/// use rig_core::tool::{Tool, ToolContext, ToolExecutionError};
///
/// struct Add;
///
/// #[derive(serde::Deserialize)]
/// struct Args { x: i64, y: i64 }
///
/// impl Tool for Add {
///     const NAME: &'static str = "add";
///     type Args = Args;
///     type Output = i64;
///
///     fn description(&self) -> String { "Add two integers".into() }
///     fn parameters(&self) -> serde_json::Value {
///         serde_json::json!({
///             "type": "object",
///             "properties": { "x": {"type":"integer"}, "y": {"type":"integer"} },
///             "required": ["x", "y"]
///         })
///     }
///
///     async fn call(
///         &self,
///         _context: &mut ToolContext,
///         args: Self::Args,
///     ) -> Result<Self::Output, ToolExecutionError> {
///         Ok(args.x + args.y)
///     }
/// }
/// ```
pub trait Tool: Sized + WasmCompatSend + WasmCompatSync {
    /// Default registration and dispatch name.
    const NAME: &'static str;

    /// Typed arguments decoded from the model's JSON value.
    type Args: DeserializeOwned + WasmCompatSend + WasmCompatSync;

    /// Serializable output. Strings remain verbatim; other values become JSON.
    type Output: Serialize;

    /// Registration name. Runtime-backed adapters may override this.
    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    /// Provider-facing description.
    fn description(&self) -> String;

    /// Provider-facing JSON Schema for [`Args`](Self::Args).
    fn parameters(&self) -> serde_json::Value;

    /// Execute the tool with private per-call context.
    fn call(
        &self,
        context: &mut ToolContext,
        args: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, ToolExecutionError>> + WasmCompatSend;
}

/// A typed tool that can be reconstructed after vector-store retrieval.
pub trait ToolEmbedding: Tool {
    /// Error raised while reconstructing a retrieved tool.
    type InitError: std::error::Error + WasmCompatSend + WasmCompatSync + 'static;
    /// Serializable state stored with the embedding.
    type Context: DeserializeOwned + Serialize;
    /// External reconstruction state.
    type State: WasmCompatSend;

    /// Text embedded for retrieval.
    fn embedding_docs(&self) -> Vec<String>;

    /// Serializable reconstruction context.
    fn context(&self) -> Self::Context;

    /// Reconstruct the tool after retrieval.
    fn init(state: Self::State, context: Self::Context) -> Result<Self, Self::InitError>;
}

fn serialize_tool_output(output: impl Serialize) -> serde_json::Result<String> {
    match serde_json::to_value(output)? {
        serde_json::Value::String(text) => Ok(text),
        value => Ok(value.to_string()),
    }
}

fn parse_tool_args<A>(args: &str) -> serde_json::Result<A>
where
    A: DeserializeOwned,
{
    match serde_json::from_str(args) {
        Ok(parsed) => Ok(parsed),
        // Models and direct callers may omit arguments or emit `null` for a
        // schema with no required fields. Preserve `null` for types (such as
        // `()`) that accept it, otherwise retry both forms as the semantically
        // equivalent empty object. This also keeps argument-free MCP calls from
        // failing before the adapter can dispatch them.
        Err(error) if args.trim().is_empty() || args.trim() == "null" => {
            serde_json::from_str("{}").map_err(|_| error)
        }
        Err(error) => Err(error),
    }
}

/// Crate-private object-safe dispatch boundary. Public users choose [`Tool`] or
/// [`DynamicTool`] rather than implementing an erased mirror trait.
pub(crate) trait ErasedTool: WasmCompatSend + WasmCompatSync {
    fn name(&self) -> String;
    fn description(&self) -> String;
    fn parameters(&self) -> serde_json::Value;

    fn execute<'a>(
        &'a self,
        args: String,
        context: ToolContext,
    ) -> WasmBoxedFuture<'a, ToolExecution>;
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
        mut context: ToolContext,
    ) -> WasmBoxedFuture<'a, ToolExecution> {
        Box::pin(async move {
            let parsed = match parse_tool_args::<T::Args>(&args) {
                Ok(parsed) => parsed,
                Err(error) => {
                    let feedback = format!("failed to parse tool arguments: {error}");
                    return ToolExecution::failed(
                        ToolExecutionError::from_source(ToolErrorKind::InvalidArgs, error)
                            .with_model_feedback(feedback),
                        context,
                    );
                }
            };

            match Tool::call(self, &mut context, parsed).await {
                Ok(output) => match serialize_tool_output(output) {
                    Ok(output) => ToolExecution::success(output, context),
                    Err(error) => ToolExecution::failed(
                        ToolExecutionError::from_source(ToolErrorKind::Other, error)
                            .with_model_feedback("failed to serialize tool output"),
                        context,
                    ),
                },
                Err(error) => ToolExecution::failed(error, context),
            }
        })
    }
}

trait DynamicCall: WasmCompatSend + WasmCompatSync {
    fn invoke<'a>(
        &'a self,
        context: &'a mut ToolContext,
        args: serde_json::Value,
    ) -> WasmBoxedFuture<'a, Result<serde_json::Value, ToolExecutionError>>;
}

impl<F> DynamicCall for F
where
    F: for<'a> Fn(
            &'a mut ToolContext,
            serde_json::Value,
        ) -> WasmBoxedFuture<'a, Result<serde_json::Value, ToolExecutionError>>
        + WasmCompatSend
        + WasmCompatSync,
{
    fn invoke<'a>(
        &'a self,
        context: &'a mut ToolContext,
        args: serde_json::Value,
    ) -> WasmBoxedFuture<'a, Result<serde_json::Value, ToolExecutionError>> {
        self(context, args)
    }
}

/// A runtime-defined tool backed by an async callback.
///
/// This is the only public dynamic dispatch surface. Its callback receives the
/// same [`ToolContext`] as typed tools and returns a JSON value rendered with the
/// same canonical rules. The callback is boxed at construction, so callers never
/// implement or name Rig's object-safe erasure trait.
///
/// # Example
///
/// ```
/// use rig_core::tool::{DynamicTool, ToolExecutionError};
///
/// let echo = DynamicTool::new(
///     "echo",
///     "Echo arbitrary JSON",
///     serde_json::json!({"type": "object"}),
///     |_context, args| Box::pin(async move { Ok::<_, ToolExecutionError>(args) }),
/// );
/// assert_eq!(echo.name(), "echo");
/// ```
pub struct DynamicTool {
    name: String,
    description: String,
    parameters: serde_json::Value,
    call: Arc<dyn DynamicCall>,
}

impl Clone for DynamicTool {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
            call: Arc::clone(&self.call),
        }
    }
}

impl std::fmt::Debug for DynamicTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("parameters", &self.parameters)
            .finish_non_exhaustive()
    }
}

impl DynamicTool {
    /// Create a runtime-defined tool.
    pub fn new<F>(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
        call: F,
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
            call: Arc::new(call),
        }
    }

    /// Runtime registration name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Runtime provider-facing description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Runtime provider-facing JSON Schema.
    pub fn parameters(&self) -> &serde_json::Value {
        &self.parameters
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
        mut context: ToolContext,
    ) -> WasmBoxedFuture<'a, ToolExecution> {
        Box::pin(async move {
            let args = match serde_json::from_str(&args) {
                Ok(args) => args,
                Err(error) => {
                    return ToolExecution::failed(
                        ToolExecutionError::from_source(ToolErrorKind::InvalidArgs, error),
                        context,
                    );
                }
            };

            match self.call.invoke(&mut context, args).await {
                Ok(output) => match serialize_tool_output(output) {
                    Ok(output) => ToolExecution::success(output, context),
                    Err(error) => ToolExecution::failed(
                        ToolExecutionError::from_source(ToolErrorKind::Other, error)
                            .with_model_feedback("failed to serialize tool output"),
                        context,
                    ),
                },
                Err(error) => ToolExecution::failed(error, context),
            }
        })
    }
}

/// Build the provider-facing definition for a typed tool.
pub fn tool_definition<T>(tool: &T) -> ToolDefinition
where
    T: Tool,
{
    ToolDefinition {
        name: Tool::name(tool),
        description: Tool::description(tool),
        parameters: Tool::parameters(tool),
    }
}

fn tool_definition_with_name(name: impl Into<String>, tool: &dyn ErasedTool) -> ToolDefinition {
    ToolDefinition {
        name: name.into(),
        description: tool.description(),
        parameters: tool.parameters(),
    }
}

pub(crate) trait ErasedEmbeddingTool: ErasedTool {
    fn context(&self) -> serde_json::Result<serde_json::Value>;
    fn embedding_docs(&self) -> Vec<String>;
}

impl<T> ErasedEmbeddingTool for T
where
    T: ToolEmbedding + 'static,
{
    fn context(&self) -> serde_json::Result<serde_json::Value> {
        serde_json::to_value(ToolEmbedding::context(self))
    }

    fn embedding_docs(&self) -> Vec<String> {
        ToolEmbedding::embedding_docs(self)
    }
}

#[derive(Clone)]
pub(crate) enum RegisteredTool {
    Simple(Arc<dyn ErasedTool>),
    Embedding(Arc<dyn ErasedEmbeddingTool>),
}

impl RegisteredTool {
    fn name(&self) -> String {
        match self {
            Self::Simple(tool) => tool.name(),
            Self::Embedding(tool) => tool.name(),
        }
    }

    pub(crate) fn definition_with_name(&self, name: impl Into<String>) -> ToolDefinition {
        match self {
            Self::Simple(tool) => tool_definition_with_name(name, &**tool),
            Self::Embedding(tool) => tool_definition_with_name(name, &**tool),
        }
    }

    pub(crate) async fn execute(&self, args: String, context: ToolContext) -> ToolExecution {
        match self {
            Self::Simple(tool) => tool.execute(args, context).await,
            Self::Embedding(tool) => tool.execute(args, context).await,
        }
    }
}

/// Error produced while generating definitions or embedding documents for a
/// [`ToolSet`]. Execution failures are represented in [`ToolExecution`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ToolSetError {
    /// Tool metadata could not be encoded as JSON.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Registration-ordered collection of tools with one structured dispatch path.
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

    /// Start a [`ToolSetBuilder`].
    pub fn builder() -> ToolSetBuilder {
        ToolSetBuilder::default()
    }

    /// Whether a tool is registered by `name`.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Register a typed tool, returning its name.
    pub fn add_tool<T>(&mut self, tool: T) -> String
    where
        T: Tool + 'static,
    {
        self.insert(RegisteredTool::Simple(Arc::new(tool)))
    }

    /// Register a runtime-defined tool, returning its name.
    pub fn add_dynamic_tool(&mut self, tool: DynamicTool) -> String {
        self.insert(RegisteredTool::Simple(Arc::new(tool)))
    }

    pub(crate) fn insert(&mut self, tool: RegisteredTool) -> String {
        let name = tool.name();
        if self.tools.insert(name.clone(), tool).is_some() {
            tracing::warn!(tool_name = %name, "replacing an existing tool registration");
        }
        name
    }

    fn insert_with_name(&mut self, name: String, tool: RegisteredTool) {
        if self.tools.insert(name.clone(), tool).is_some() {
            tracing::warn!(tool_name = %name, "replacing an existing tool registration");
        }
    }

    /// Remove a registered tool. Surviving tools keep their order.
    pub fn delete_tool(&mut self, name: &str) {
        self.tools.shift_remove(name);
    }

    /// Merge another set in registration order. Existing names are replaced in
    /// place, so their original position remains stable.
    pub fn add_tools(&mut self, set: ToolSet) {
        for (name, tool) in set.tools {
            self.insert_with_name(name, tool);
        }
    }

    pub(crate) fn get(&self, name: &str) -> Option<&RegisteredTool> {
        self.tools.get(name)
    }

    pub(crate) fn ordered_names(&self) -> impl Iterator<Item = &String> {
        self.tools.keys()
    }

    fn ordered_entries(&self) -> impl Iterator<Item = (&String, &RegisteredTool)> {
        self.tools.iter()
    }

    /// Provider-facing definitions in registration order.
    pub fn get_tool_definitions(&self) -> Result<Vec<ToolDefinition>, ToolSetError> {
        Ok(self
            .ordered_entries()
            .map(|(name, tool)| tool.definition_with_name(name.clone()))
            .collect())
    }

    /// Execute one registered tool through the canonical structured path.
    ///
    /// A missing registration is returned as a `NotFound` status rather than a
    /// separate dispatch error.
    pub async fn execute(
        &self,
        name: &str,
        args: impl Into<String>,
        context: ToolContext,
    ) -> ToolExecution {
        match self.tools.get(name) {
            Some(tool) => tool.execute(args.into(), context).await,
            None => ToolExecution::failed(
                ToolExecutionError::not_found(format!("no tool named `{name}` is registered")),
                context,
            ),
        }
    }

    /// Embedding documents in registration order.
    pub async fn documents(&self) -> Result<Vec<completion::Document>, ToolSetError> {
        let mut docs = Vec::new();
        for (name, tool) in self.ordered_entries() {
            let definition = tool.definition_with_name(name.clone());
            docs.push(completion::Document {
                id: name.clone(),
                text: format!(
                    "Tool: {name}\nDefinition:\n{}",
                    serde_json::to_string_pretty(&definition)?
                ),
                additional_props: HashMap::new(),
            });
        }
        Ok(docs)
    }

    /// Vector-store schemas for embedding-backed tools.
    pub fn schemas(&self) -> Result<Vec<ToolSchema>, EmbedError> {
        self.ordered_entries()
            .filter_map(|(name, tool)| match tool {
                RegisteredTool::Embedding(tool) => {
                    Some(ToolSchema::from_erased(name.clone(), &**tool))
                }
                RegisteredTool::Simple(_) => None,
            })
            .collect()
    }
}

/// Builder for heterogeneous typed, runtime, and embedding-backed tools.
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
        self.tools.push(RegisteredTool::Simple(Arc::new(tool)));
        self
    }

    /// Add a runtime-defined tool.
    pub fn runtime_tool(mut self, tool: DynamicTool) -> Self {
        self.tools.push(RegisteredTool::Simple(Arc::new(tool)));
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

    /// Build the registration-ordered set.
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    use serde_json::json;

    use crate::message::{DocumentSourceKind, ToolResultContent};
    use crate::test_utils::{
        MockExampleTool, MockImageOutputTool, MockObjectOutputTool, MockStringOutputTool,
        MockToolError, mock_math_toolset,
    };

    use super::*;

    fn runtime_tool(name: &str, description: &str) -> DynamicTool {
        let description_for_call = description.to_string();
        DynamicTool::new(
            name,
            description,
            json!({"type": "object", "properties": {}}),
            move |_context, _args| {
                let output = format!("called {description_for_call}");
                Box::pin(async move { Ok(json!(output)) })
            },
        )
    }

    #[test]
    fn definitions_preserve_registration_order() {
        let set = mock_math_toolset();
        let definitions = set.get_tool_definitions().unwrap();
        assert_eq!(
            definitions
                .iter()
                .map(|definition| definition.name.as_str())
                .collect::<Vec<_>>(),
            ["add", "subtract"]
        );
        assert!(
            definitions
                .iter()
                .all(|definition| !definition.description.is_empty())
        );
    }

    #[test]
    fn deletion_preserves_survivor_order() {
        let mut set = ToolSet::default();
        for name in ["alpha", "beta", "gamma", "delta"] {
            set.add_dynamic_tool(runtime_tool(name, "test tool"));
        }
        set.delete_tool("beta");
        assert_eq!(
            set.ordered_names().map(String::as_str).collect::<Vec<_>>(),
            ["alpha", "gamma", "delta"]
        );
    }

    #[tokio::test]
    async fn duplicate_registration_replaces_in_place() {
        let mut set = ToolSet::default();
        set.add_dynamic_tool(runtime_tool("alpha", "first alpha"));
        set.add_dynamic_tool(runtime_tool("beta", "beta"));
        set.add_dynamic_tool(runtime_tool("alpha", "second alpha"));

        let definitions = set.get_tool_definitions().unwrap();
        assert_eq!(
            definitions
                .iter()
                .map(|definition| definition.name.as_str())
                .collect::<Vec<_>>(),
            ["alpha", "beta"]
        );
        assert_eq!(definitions[0].description, "second alpha");

        let execution = set.execute("alpha", "{}", ToolContext::new()).await;
        assert_eq!(execution.model_output(), "called second alpha");
    }

    #[tokio::test]
    async fn definitions_and_documents_use_registered_name() {
        let calls = Arc::new(AtomicUsize::new(0));
        let name_calls = Arc::clone(&calls);

        struct ChangingNameTool(Arc<AtomicUsize>);
        impl Tool for ChangingNameTool {
            const NAME: &'static str = "unused";
            type Args = serde_json::Value;
            type Output = String;

            fn name(&self) -> String {
                match self.0.fetch_add(1, Ordering::SeqCst) {
                    0 => "registered".into(),
                    _ => "changed".into(),
                }
            }

            fn description(&self) -> String {
                "changing name".into()
            }

            fn parameters(&self) -> serde_json::Value {
                json!({"type": "object"})
            }

            async fn call(
                &self,
                _context: &mut ToolContext,
                _args: Self::Args,
            ) -> Result<Self::Output, ToolExecutionError> {
                Ok("ok".into())
            }
        }

        let mut set = ToolSet::default();
        set.add_tool(ChangingNameTool(name_calls));
        assert_eq!(set.get_tool_definitions().unwrap()[0].name, "registered");
        let documents = set.documents().await.unwrap();
        assert_eq!(documents[0].id, "registered");
        assert!(!documents[0].text.contains("changed"));
    }

    #[tokio::test]
    async fn string_object_and_multimodal_output_rendering_is_unchanged() {
        let mut set = ToolSet::default();
        set.add_tool(MockStringOutputTool);
        set.add_tool(MockObjectOutputTool);
        set.add_tool(MockImageOutputTool);

        let string = set.execute("string_output", "{}", ToolContext::new()).await;
        assert_eq!(string.model_output(), "Hello\nWorld");

        let object = set.execute("object_output", "{}", ToolContext::new()).await;
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(object.model_output()).unwrap(),
            json!({"status": "ok", "count": 42})
        );

        let image = set.execute("image_output", "{}", ToolContext::new()).await;
        let content = ToolResultContent::from_tool_output(image.model_output().to_string());
        match content.first() {
            ToolResultContent::Image(image) => {
                assert!(matches!(image.data, DocumentSourceKind::Base64(_)));
                assert_eq!(image.media_type, Some(crate::message::ImageMediaType::PNG));
            }
            other => panic!("expected image output, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn empty_and_null_arguments_normalize_for_argument_free_tools() {
        let mut set = ToolSet::default();
        set.add_tool(MockExampleTool);
        let unit = set
            .execute("example_tool", "null", ToolContext::new())
            .await;
        assert_eq!(unit.model_output(), "Example answer");
        assert!(unit.status().is_success());

        #[derive(serde::Deserialize)]
        struct OptionalArgs {
            label: Option<String>,
        }
        struct OptionalTool;
        impl Tool for OptionalTool {
            const NAME: &'static str = "optional";
            type Args = OptionalArgs;
            type Output = String;

            fn description(&self) -> String {
                "optional args".into()
            }
            fn parameters(&self) -> serde_json::Value {
                json!({"type":"object", "properties":{}})
            }
            async fn call(
                &self,
                _context: &mut ToolContext,
                args: Self::Args,
            ) -> Result<Self::Output, ToolExecutionError> {
                Ok(args.label.unwrap_or_else(|| "default".into()))
            }
        }

        set.add_tool(OptionalTool);
        for args in ["", "null"] {
            let optional = set.execute("optional", args, ToolContext::new()).await;
            assert_eq!(optional.model_output(), "default");
            assert!(optional.status().is_success());
        }
    }

    #[tokio::test]
    async fn argument_and_serialization_errors_are_structured() {
        let mut set = ToolSet::default();
        set.add_tool(MockStringOutputTool);
        let invalid = set
            .execute("string_output", "{not-json", ToolContext::new())
            .await;
        assert!(invalid.status().is_error_kind(ToolErrorKind::InvalidArgs));
        assert!(
            invalid
                .status()
                .error()
                .unwrap()
                .downcast_source_ref::<serde_json::Error>()
                .is_some()
        );

        struct Unserializable;
        impl Serialize for Unserializable {
            fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                Err(serde::ser::Error::custom("cannot serialize"))
            }
        }
        struct BadOutput;
        impl Tool for BadOutput {
            const NAME: &'static str = "bad_output";
            type Args = ();
            type Output = Unserializable;
            fn description(&self) -> String {
                "bad output".into()
            }
            fn parameters(&self) -> serde_json::Value {
                json!({"type":"null"})
            }
            async fn call(
                &self,
                _context: &mut ToolContext,
                _args: Self::Args,
            ) -> Result<Self::Output, ToolExecutionError> {
                Ok(Unserializable)
            }
        }
        set.add_tool(BadOutput);
        let bad = set.execute("bad_output", "null", ToolContext::new()).await;
        assert!(bad.status().is_error_kind(ToolErrorKind::Other));
        assert_eq!(bad.model_output(), "failed to serialize tool output");
    }

    #[tokio::test]
    async fn context_carries_input_and_result_metadata() {
        #[derive(Clone, Debug, PartialEq)]
        struct Session(&'static str);
        #[derive(Clone, Debug, PartialEq)]
        struct RequestId(&'static str);

        struct ContextTool;
        impl Tool for ContextTool {
            const NAME: &'static str = "context";
            type Args = ();
            type Output = String;
            fn description(&self) -> String {
                "context".into()
            }
            fn parameters(&self) -> serde_json::Value {
                json!({"type":"null"})
            }
            async fn call(
                &self,
                context: &mut ToolContext,
                _args: Self::Args,
            ) -> Result<Self::Output, ToolExecutionError> {
                let session = context
                    .require::<Session>()
                    .map_err(|error| {
                        ToolExecutionError::permission_denied(error.to_string()).with_source(error)
                    })?
                    .0;
                context.insert_metadata(RequestId("request-1"));
                Ok(session.into())
            }
        }

        let mut set = ToolSet::default();
        set.add_tool(ContextTool);
        let mut context = ToolContext::new();
        context.insert(Session("session-1"));
        let execution = set.execute("context", "null", context).await;
        assert_eq!(execution.model_output(), "session-1");
        assert_eq!(
            execution.metadata::<RequestId>(),
            Some(&RequestId("request-1"))
        );
    }

    #[tokio::test]
    async fn missing_tool_is_a_not_found_execution() {
        let execution = ToolSet::default()
            .execute("missing", "{}", ToolContext::new())
            .await;
        assert!(execution.status().is_error_kind(ToolErrorKind::NotFound));
        assert!(execution.model_output().contains("missing"));
    }

    #[tokio::test]
    async fn dynamic_tool_uses_canonical_dispatch_and_context() {
        #[derive(Clone, Debug, PartialEq)]
        struct Marker(&'static str);

        let tool = DynamicTool::new(
            "dynamic",
            "dynamic tool",
            json!({"type":"object"}),
            |context, args| {
                Box::pin(async move {
                    let marker = context.require::<Marker>().map_err(|error| {
                        ToolExecutionError::permission_denied(error.to_string())
                    })?;
                    Ok(json!({"marker": marker.0, "args": args}))
                })
            },
        );
        let mut set = ToolSet::default();
        set.add_dynamic_tool(tool);
        let mut context = ToolContext::new();
        context.insert(Marker("private"));
        let execution = set.execute("dynamic", r#"{"x":1}"#, context).await;
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(execution.model_output()).unwrap(),
            json!({"marker":"private", "args":{"x":1}})
        );
    }

    #[test]
    fn embedding_schema_uses_registered_name() {
        #[derive(Debug, thiserror::Error)]
        #[error("init error")]
        struct InitError;

        struct EmbeddingTool;
        impl Tool for EmbeddingTool {
            const NAME: &'static str = "embedding";
            type Args = ();
            type Output = String;
            fn description(&self) -> String {
                "embedding tool".into()
            }
            fn parameters(&self) -> serde_json::Value {
                json!({"type":"null"})
            }
            async fn call(
                &self,
                _context: &mut ToolContext,
                _args: Self::Args,
            ) -> Result<Self::Output, ToolExecutionError> {
                Ok("ok".into())
            }
        }
        impl ToolEmbedding for EmbeddingTool {
            type InitError = InitError;
            type Context = ();
            type State = ();
            fn embedding_docs(&self) -> Vec<String> {
                vec!["embedding docs".into()]
            }
            fn context(&self) -> Self::Context {}
            fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
                Ok(Self)
            }
        }

        let set = ToolSet::builder().dynamic_tool(EmbeddingTool).build();
        let schemas = set.schemas().unwrap();
        assert_eq!(schemas[0].name, "embedding");
        assert_eq!(schemas[0].embedding_docs, ["embedding docs"]);
    }

    // Keep the shared test utility error reachable in this module; it ensures
    // test helpers have migrated to the unified execution envelope.
    const _: fn(MockToolError) = |_: MockToolError| {};
}
