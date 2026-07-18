//! Tool authoring, registration, and canonical structured execution.
//!
//! A typed [`Tool`] implements one [`Tool::call`] method. Rig erases it
//! internally, executes it through one structured path, and exposes a single
//! [`ToolResult`] view to hooks and runtime callers. [`ToolContext`] is the sole
//! path for typed inbound context and host-only result metadata.
//!
//! # Implementing a typed tool
//!
//! Ordinary serializable return values are converted to canonical model output
//! without first passing through a string.
//!
//! ```
//! use rig_core::tool::{Tool, ToolContext};
//! use serde::{Deserialize, Serialize};
//! use std::convert::Infallible;
//!
//! #[derive(Deserialize)]
//! struct AddArgs {
//!     left: i64,
//!     right: i64,
//! }
//!
//! #[derive(Serialize)]
//! struct Sum {
//!     value: i64,
//! }
//!
//! #[derive(Clone, Debug, PartialEq)]
//! struct AuditRecord(i64);
//!
//! struct Add;
//!
//! impl Tool for Add {
//!     const NAME: &'static str = "add";
//!     type Args = AddArgs;
//!     type Output = Sum;
//!     type Error = Infallible;
//!
//!     fn description(&self) -> String {
//!         "Add two integers".into()
//!     }
//!
//!     fn parameters(&self) -> serde_json::Value {
//!         serde_json::json!({
//!             "type": "object",
//!             "properties": {
//!                 "left": { "type": "integer" },
//!                 "right": { "type": "integer" }
//!             },
//!             "required": ["left", "right"]
//!         })
//!     }
//!
//!     async fn call(
//!         &self,
//!         context: &mut ToolContext,
//!         args: Self::Args,
//!     ) -> Result<Self::Output, Self::Error> {
//!         let value = args.left + args.right;
//!         context.insert_result(AuditRecord(value));
//!         Ok(Sum { value })
//!     }
//! }
//! ```
//!
//! Return [`ToolOutput`] for explicit JSON or multimodal presentation. A
//! [`ToolResultContent`](crate::message::ToolResultContent) or
//! [`OneOrMany`](crate::OneOrMany) of content blocks can also be used directly
//! as a typed tool output without being mistaken for ordinary JSON.
//!
//! ```
//! use rig_core::{
//!     message::{ImageMediaType, ToolResultContent},
//!     tool::ToolOutput,
//! };
//!
//! let output = ToolOutput::one(ToolResultContent::image_base64(
//!     "iVBORw0KGgo=",
//!     Some(ImageMediaType::PNG),
//!     None,
//! ));
//! assert!(matches!(
//!     output.as_content().first_ref(),
//!     ToolResultContent::Image(_)
//! ));
//! ```
//!
//! Explicit [`ToolExecutionError`] constructors keep their detailed message
//! model-visible so validation failures can tell the model how to recover. The
//! default [`Tool::map_error`] conversion preserves an arbitrary source error
//! for operators but exposes only safe kind-level feedback. Override
//! [`Tool::map_error`] or use [`ToolExecutionError::with_model_output`] when a
//! domain error has deliberate structured or actionable model feedback.
//!
//! # Migration from the parallel tool APIs
//!
//! | Removed concept | Canonical replacement |
//! | --- | --- |
//! | Multiple typed `call*` methods | One [`Tool::call`] method |
//! | Public dynamic dispatch traits | [`DynamicTool`] |
//! | Parallel error and failure types | [`ToolExecutionError`] and [`crate::tool::ToolErrorKind`] |
//! | Author-facing outcome enums | Ordinary `Result<T, Self::Error>` normalized at dispatch |
//! | Separate call/result extension maps | [`ToolContext`] |
//! | Parallel string/structured dispatch | [`ToolSet::execute`] and [`server::ToolServerHandle::execute`] |
//!
//! Model-visible output remains typed throughout dispatch. Rendering to text is
//! a terminal provider or telemetry concern; Rig does not reconstruct rich
//! content by parsing a returned string.

use std::{collections::HashMap, sync::Arc};

pub mod builtin;

use futures::Future;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::{
    completion::{self, ToolDefinition},
    embeddings::{embed::EmbedError, tool::ToolSchema},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

pub(crate) mod extensions;
#[cfg(feature = "rmcp")]
#[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
pub mod rmcp;
pub mod server;

pub use extensions::{MissingToolContext, ToolContext};
pub use rig_core::tool::{
    IntoToolOutput, ToolErrorKind, ToolExecutionError, ToolOutput, ToolResult,
};

/// A typed LLM tool.
///
/// Tool authors provide metadata and exactly one execution method. Runtime
/// context and host-only result metadata share the [`ToolContext`] path. Rig's
/// object-safe dispatch boundary is private; use [`DynamicTool`] when the tool
/// name or callback is only known at runtime.
pub trait Tool: Sized + WasmCompatSend + WasmCompatSync {
    /// Unique registration and provider-facing name.
    const NAME: &'static str;
    /// Typed JSON arguments.
    type Args: for<'de> Deserialize<'de> + WasmCompatSend + WasmCompatSync;
    /// Output convertible into Rig's canonical model presentation.
    ///
    /// Every owned serializable value implements [`IntoToolOutput`]
    /// automatically. [`ToolResultContent`](crate::message::ToolResultContent)
    /// and [`OneOrMany`](crate::OneOrMany) preserve rich content when returned
    /// directly; use [`ToolOutput`] when constructing the presentation
    /// explicitly.
    type Output: IntoToolOutput;
    /// Typed error returned by direct calls to this tool.
    ///
    /// Rig normalizes this error into [`ToolExecutionError`] only at the erased
    /// dispatch boundary. This keeps ordinary `?` propagation and typed unit
    /// tests available to tool authors without creating a second runtime error
    /// representation.
    type Error: std::error::Error + WasmCompatSend + WasmCompatSync + 'static;

    /// Model-facing description.
    fn description(&self) -> String;

    /// JSON Schema for arguments.
    fn parameters(&self) -> serde_json::Value;

    /// Normalize a typed author-facing error for runtime policy and telemetry.
    ///
    /// The default preserves the concrete source and classifies it as
    /// [`crate::tool::ToolErrorKind::Other`]. Override this method when the domain error can
    /// provide a more precise kind, retryability policy, or safe model output.
    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        ToolExecutionError::from_error(error)
    }

    /// Execute the tool.
    fn call(
        &self,
        context: &mut ToolContext,
        args: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + WasmCompatSend;
}

impl<T> Tool for T
where
    T: rig_core::tool::Tool,
{
    const NAME: &'static str = <T as rig_core::tool::Tool>::NAME;
    type Args = <T as rig_core::tool::Tool>::Args;
    type Output = <T as rig_core::tool::Tool>::Output;
    type Error = <T as rig_core::tool::Tool>::Error;

    fn description(&self) -> String {
        rig_core::tool::Tool::description(self)
    }

    fn parameters(&self) -> serde_json::Value {
        rig_core::tool::Tool::parameters(self)
    }

    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        rig_core::tool::Tool::map_error(self, error)
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        rig_core::tool::Tool::call(self, args).await
    }
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
    /// Whether the runtime backing this registration can still accept calls.
    ///
    /// In-process tools are always live. Remote adapters override this so the
    /// registry can retire disconnected owners without probing by execution.
    #[cfg(feature = "rmcp")]
    fn is_live(&self) -> bool {
        true
    }
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
        T::NAME.to_string()
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
                Ok(output) => match output.into_tool_output() {
                    Ok(output) => ToolResult::success(output),
                    Err(error) => ToolResult::failed(error),
                },
                Err(error) => ToolResult::failed(Tool::map_error(self, error)),
            }
        })
    }
}

trait DynamicCallback:
    for<'a> Fn(
        &'a mut ToolContext,
        serde_json::Value,
    ) -> WasmBoxedFuture<'a, Result<ToolOutput, ToolExecutionError>>
    + WasmCompatSend
    + WasmCompatSync
{
}

impl<F> DynamicCallback for F where
    F: for<'a> Fn(
            &'a mut ToolContext,
            serde_json::Value,
        ) -> WasmBoxedFuture<'a, Result<ToolOutput, ToolExecutionError>>
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
            ) -> WasmBoxedFuture<'a, Result<ToolOutput, ToolExecutionError>>
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
                Ok(output) => match output.into_tool_output() {
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
        name: T::NAME.to_string(),
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

    #[cfg(feature = "rmcp")]
    pub(crate) fn is_live(&self) -> bool {
        self.erased().is_live()
    }

    pub(crate) async fn execute(&self, args: String, context: &mut ToolContext) -> ToolResult {
        self.erased().execute(args, context).await
    }
}

/// One authoritative registry entry for execution and provider exposure.
#[derive(Clone)]
pub(crate) struct ToolRegistration {
    tool: RegisteredTool,
    always_exposed: bool,
}

impl ToolRegistration {
    fn new(tool: RegisteredTool, always_exposed: bool) -> Self {
        Self {
            tool,
            always_exposed,
        }
    }
}

/// The outcome of one isolated tool dispatch.
pub(crate) struct ToolDispatch {
    pub(crate) result: ToolResult,
    pub(crate) context: ToolContext,
}

/// Execute a resolved registry entry through the single dispatch boundary.
///
/// Every surface enters here with its caller-owned context. The helper clones
/// inbound values exactly once, clears prior result metadata, and returns the
/// per-dispatch context so callers can expose its metadata without publishing
/// mutations the tool made to its local inbound snapshot.
pub(crate) async fn dispatch_tool(
    name: &str,
    args: String,
    tool: Option<RegisteredTool>,
    context: &ToolContext,
) -> ToolDispatch {
    let mut dispatch_context = context.for_dispatch();
    let result = match tool {
        Some(tool) => {
            tracing::debug!(target: "rig", tool_name = name, "calling tool with args:\n{args}");
            tool.execute(args, &mut dispatch_context).await
        }
        None => ToolResult::failed(
            ToolExecutionError::not_found(format!("no tool named `{name}` is registered"))
                .with_model_feedback(format!("tool `{name}` not found")),
        ),
    };
    ToolDispatch {
        result,
        context: dispatch_context,
    }
}

/// An ordered collection of tools.
#[derive(Default)]
pub struct ToolSet {
    pub(crate) tools: IndexMap<String, ToolRegistration>,
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
        self.insert_registration(name.clone(), ToolRegistration::new(tool, true));
        name
    }

    fn insert_registration(&mut self, name: String, mut registration: ToolRegistration) {
        if let Some(current) = self.tools.get_mut(&name) {
            registration.always_exposed |= current.always_exposed;
            *current = registration;
            tracing::warn!(tool_name = %name, "replacing an existing tool registration");
        } else {
            self.tools.insert(name, registration);
        }
    }

    /// Delete a tool by name.
    pub fn delete_tool(&mut self, name: &str) {
        self.tools.shift_remove(name);
    }

    /// Merge another set, preserving registration order and replacing duplicates.
    pub fn add_tools(&mut self, set: ToolSet) {
        for (name, registration) in set.tools {
            self.insert_registration(name, registration);
        }
    }

    /// Merge tools that are advertised only when selected by a retrieval index.
    pub(crate) fn add_retrievable_tools(&mut self, set: ToolSet) {
        for (name, mut registration) in set.tools {
            registration.always_exposed = false;
            self.insert_registration(name, registration);
        }
    }

    pub(crate) fn get(&self, name: &str) -> Option<&RegisteredTool> {
        self.tools.get(name).map(|registration| &registration.tool)
    }

    pub(crate) fn always_exposed_names(&self) -> impl Iterator<Item = &String> {
        self.tools
            .iter()
            .filter_map(|(name, registration)| registration.always_exposed.then_some(name))
    }

    /// Provider-facing definitions in registration order.
    pub fn get_tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .map(|(name, registration)| registration.tool.definition_with_name(name.clone()))
            .collect()
    }

    /// Execute one registered tool through the canonical structured path.
    ///
    /// The tool receives a snapshot of inbound context. Result metadata is
    /// published back to `context`; mutations to inbound values are discarded.
    pub async fn execute(
        &self,
        name: &str,
        args: impl Into<String>,
        context: &mut ToolContext,
    ) -> ToolResult {
        context.clear_dispatch_result();
        let tool = self.get(name).cloned();
        let ToolDispatch {
            result,
            context: dispatch_context,
        } = dispatch_tool(name, args.into(), tool, context).await;
        context.accept_dispatch_result(dispatch_context);
        result
    }

    /// Documents describing all registered tools.
    pub fn documents(&self) -> Vec<completion::Document> {
        let mut docs = Vec::new();
        for (name, registration) in &self.tools {
            let definition = registration.tool.definition_with_name(name.clone());
            let serialized = serde_json::to_string_pretty(&definition).unwrap_or_else(|error| {
                tracing::warn!(
                    tool_name = %name,
                    %error,
                    "tool definition could not be pretty-printed; using a plain representation"
                );
                format!(
                    "name: {}\ndescription: {}\nparameters: {}",
                    definition.name, definition.description, definition.parameters
                )
            });
            docs.push(completion::Document {
                id: name.clone(),
                text: format!("Tool: {name}\nDefinition: \n{serialized}"),
                additional_props: HashMap::new(),
            });
        }
        docs
    }

    /// Convert embedding tools to vector-store schemas.
    pub fn schemas(&self) -> Result<Vec<ToolSchema>, EmbedError> {
        self.tools
            .iter()
            .filter_map(|(name, registration)| match &registration.tool {
                RegisteredTool::Embedding(tool) => Some(
                    tool.serialized_context()
                        .map_err(EmbedError::new)
                        .map(|context| ToolSchema {
                            name: name.clone(),
                            context,
                            embedding_docs: tool.embedding_docs(),
                        }),
                ),
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

    /// Add a runtime-defined tool.
    pub fn dynamic_tool(mut self, tool: DynamicTool) -> Self {
        self.tools.push(RegisteredTool::Static(Arc::new(tool)));
        self
    }

    /// Add a tool that is retrieved from an embedding index at prompt time.
    pub fn retrieved_tool<T>(mut self, tool: T) -> Self
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
    use std::{
        future::{Future, pending, poll_fn},
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
        task::Poll,
        time::Duration,
    };

    use crate::{
        OneOrMany,
        message::{ImageMediaType, ToolResultContent},
    };

    use super::*;

    fn rich_error_output(label: &str) -> ToolOutput {
        ToolOutput::content(
            OneOrMany::many([
                ToolResultContent::text(label),
                ToolResultContent::image_base64("base64data==", Some(ImageMediaType::PNG), None),
            ])
            .unwrap(),
        )
    }

    fn assert_rich_error_output(result: &ToolResult, label: &str) {
        let content = result.output().as_content();
        assert_eq!(content.len(), 2);
        assert!(matches!(
            content.first_ref(),
            ToolResultContent::Text(text) if text.text == label
        ));
        assert!(matches!(content.last_ref(), ToolResultContent::Image(_)));
    }

    struct CloneTracked(Arc<AtomicUsize>);

    impl Clone for CloneTracked {
        fn clone(&self) -> Self {
            self.0.fetch_add(1, Ordering::SeqCst);
            Self(self.0.clone())
        }
    }

    struct Echo;

    impl Tool for Echo {
        const NAME: &'static str = "echo";
        type Error = rig::tool::ToolExecutionError;
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
            if let Some(value) = context.get_mut::<u32>() {
                *value += 1;
            }
            context.insert_result("result-metadata".to_string());
            Ok(args)
        }
    }

    #[tokio::test]
    async fn toolset_dispatch_snapshot_is_canonical_and_returns_result_metadata() {
        let mut set = ToolSet::default();
        set.add_tool(Echo);
        let definitions = set.get_tool_definitions();
        assert_eq!(definitions[0].name, "echo");

        let mut context = ToolContext::new();
        context.insert(7_u32);
        let clones = Arc::new(AtomicUsize::new(0));
        context.insert(CloneTracked(clones.clone()));
        let result = set.execute("echo", r#"{"value":1}"#, &mut context).await;
        assert!(result.is_success());
        assert_eq!(
            result.output(),
            &ToolOutput::json(serde_json::json!({"value": 1}))
        );
        assert_eq!(context.get::<u32>(), Some(&7));
        assert_eq!(clones.load(Ordering::SeqCst), 1);
        assert_eq!(
            context.result::<String>().map(String::as_str),
            Some("result-metadata")
        );
    }

    struct PendingTool(Arc<AtomicBool>);

    impl Tool for PendingTool {
        const NAME: &'static str = "pending";
        type Error = rig::tool::ToolExecutionError;
        type Args = ();
        type Output = ();

        fn description(&self) -> String {
            "never completes".into()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        async fn call(
            &self,
            context: &mut ToolContext,
            _args: Self::Args,
        ) -> Result<Self::Output, ToolExecutionError> {
            context.insert_result("unpublished".to_string());
            self.0.store(true, Ordering::SeqCst);
            pending().await
        }
    }

    #[tokio::test]
    async fn cancelled_toolset_dispatch_does_not_retain_stale_result_metadata() {
        let mut set = ToolSet::default();
        let started = Arc::new(AtomicBool::new(false));
        set.add_tool(PendingTool(started.clone()));
        let mut context = ToolContext::new();
        context.insert_result("stale".to_string());

        let mut execution = Box::pin(set.execute(PendingTool::NAME, "null", &mut context));
        tokio::time::timeout(
            Duration::from_secs(1),
            poll_fn(|cx| {
                assert!(execution.as_mut().poll(cx).is_pending());
                started.load(Ordering::SeqCst).then_some(()).map_or_else(
                    || {
                        cx.waker().wake_by_ref();
                        Poll::Pending
                    },
                    Poll::Ready,
                )
            }),
        )
        .await
        .expect("pending tool did not start");
        drop(execution);

        assert!(context.result::<String>().is_none());
    }

    #[tokio::test]
    async fn framework_argument_errors_remain_actionable_to_the_model() {
        let mut set = ToolSet::default();
        set.add_tool(Echo);

        let result = set
            .execute("echo", "{not json", &mut ToolContext::new())
            .await;

        assert!(result.is_error_kind(ToolErrorKind::InvalidArgs));
        assert!(
            result
                .output()
                .as_text()
                .is_some_and(|message| message.starts_with("failed to parse tool arguments:"))
        );
        assert_eq!(
            result.output().as_text(),
            result.error().and_then(ToolExecutionError::model_feedback)
        );
    }

    struct ForeignErrorTool;

    impl Tool for ForeignErrorTool {
        const NAME: &'static str = "foreign_error";
        type Error = std::io::Error;
        type Args = ();
        type Output = ();

        fn description(&self) -> String {
            "returns a foreign error type".into()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        async fn call(
            &self,
            _context: &mut ToolContext,
            _args: Self::Args,
        ) -> Result<Self::Output, Self::Error> {
            Err(std::io::Error::other("operator-only detail"))
        }
    }

    #[tokio::test]
    async fn typed_foreign_errors_normalize_only_at_dispatch() {
        let direct: std::io::Error = ForeignErrorTool
            .call(&mut ToolContext::new(), ())
            .await
            .expect_err("direct call should retain its typed error");
        assert_eq!(direct.to_string(), "operator-only detail");

        let mut set = ToolSet::default();
        set.add_tool(ForeignErrorTool);
        let result = set
            .execute(ForeignErrorTool::NAME, "null", &mut ToolContext::new())
            .await;
        let error = result.error().expect("dispatch should normalize the error");
        assert_eq!(error.kind(), ToolErrorKind::Other);
        assert_eq!(error.message(), "operator-only detail");
        assert_eq!(error.model_feedback(), Some("the tool failed"));
        assert!(error.is::<std::io::Error>());
    }

    #[derive(Debug, thiserror::Error)]
    #[error("domain timeout")]
    struct DomainTimeout;

    struct ClassifiedErrorTool;

    impl Tool for ClassifiedErrorTool {
        const NAME: &'static str = "classified_error";
        type Error = DomainTimeout;
        type Args = ();
        type Output = ();

        fn description(&self) -> String {
            "classifies a domain error".into()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        fn map_error(&self, error: Self::Error) -> ToolExecutionError {
            ToolExecutionError::timeout("safe timeout feedback").with_source(error)
        }

        async fn call(
            &self,
            _context: &mut ToolContext,
            _args: Self::Args,
        ) -> Result<Self::Output, Self::Error> {
            Err(DomainTimeout)
        }
    }

    #[tokio::test]
    async fn tools_can_classify_typed_errors_at_the_erased_boundary() {
        let mut set = ToolSet::default();
        set.add_tool(ClassifiedErrorTool);
        let result = set
            .execute(ClassifiedErrorTool::NAME, "null", &mut ToolContext::new())
            .await;
        let error = result.error().expect("dispatch should normalize the error");
        assert_eq!(error.kind(), ToolErrorKind::Timeout);
        assert_eq!(error.retryable(), Some(true));
        assert_eq!(error.model_feedback(), Some("safe timeout feedback"));
        assert!(error.is::<DomainTimeout>());
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

    struct DirectRichOutput;

    impl Tool for DirectRichOutput {
        const NAME: &'static str = "direct_rich_output";
        type Error = rig::tool::ToolExecutionError;
        type Args = serde_json::Value;
        type Output = ToolResultContent;

        fn description(&self) -> String {
            "returns a direct rich-content value".into()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        async fn call(
            &self,
            _context: &mut ToolContext,
            _args: Self::Args,
        ) -> Result<Self::Output, ToolExecutionError> {
            Ok(ToolResultContent::image_base64(
                "base64data==",
                Some(ImageMediaType::PNG),
                None,
            ))
        }
    }

    #[tokio::test]
    async fn direct_rich_typed_output_is_not_serialized_as_json() {
        let mut set = ToolSet::default();
        set.add_tool(DirectRichOutput);

        let result = set
            .execute(DirectRichOutput::NAME, "{}", &mut ToolContext::new())
            .await;

        assert!(result.is_success());
        assert!(matches!(
            result.output().as_content().first_ref(),
            ToolResultContent::Image(_)
        ));
        assert_eq!(result.output().as_json(), None);
    }

    struct TypedRichError {
        refuse: bool,
    }

    impl Tool for TypedRichError {
        const NAME: &'static str = "typed_rich_error";
        type Error = rig::tool::ToolExecutionError;
        type Args = serde_json::Value;
        type Output = String;

        fn description(&self) -> String {
            "returns rich failure feedback".into()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        async fn call(
            &self,
            _context: &mut ToolContext,
            _args: Self::Args,
        ) -> Result<Self::Output, ToolExecutionError> {
            let error = if self.refuse {
                ToolExecutionError::refused("typed refusal")
            } else {
                ToolExecutionError::provider("typed failure")
            };
            Err(error.with_model_output(rich_error_output("typed feedback")))
        }
    }

    #[tokio::test]
    async fn typed_failures_and_refusals_preserve_rich_model_output() {
        for refuse in [false, true] {
            let mut set = ToolSet::default();
            set.add_tool(TypedRichError { refuse });

            let result = set
                .execute(TypedRichError::NAME, "{}", &mut ToolContext::new())
                .await;

            assert_eq!(result.is_refused(), refuse);
            assert_eq!(result.is_error(), !refuse);
            assert_rich_error_output(&result, "typed feedback");
        }
    }

    #[tokio::test]
    async fn dynamic_failures_and_refusals_preserve_rich_model_output() {
        for refuse in [false, true] {
            let tool = DynamicTool::new(
                "dynamic_rich_error",
                "returns rich failure feedback",
                serde_json::json!({"type": "object"}),
                move |_context, _args| {
                    Box::pin(async move {
                        let error = if refuse {
                            ToolExecutionError::refused("dynamic refusal")
                        } else {
                            ToolExecutionError::provider("dynamic failure")
                        };
                        Err(error.with_model_output(rich_error_output("dynamic feedback")))
                    })
                },
            );
            let set = ToolSet::from_dynamic_tools(vec![tool]);

            let result = set
                .execute("dynamic_rich_error", "{}", &mut ToolContext::new())
                .await;

            assert_eq!(result.is_refused(), refuse);
            assert_eq!(result.is_error(), !refuse);
            assert_rich_error_output(&result, "dynamic feedback");
        }
    }
}

#[cfg(test)]
mod migrated_tests {
    use crate::message::{DocumentSourceKind, ToolResultContent};
    use crate::test_utils::{
        MockExampleTool, MockImageOutputTool, MockObjectOutputTool, MockStringOutputTool,
        MockToolError, mock_math_toolset,
    };
    use serde_json::json;

    use super::*;

    fn get_test_toolset() -> ToolSet {
        mock_math_toolset()
    }

    #[test]
    fn test_get_tool_definitions() {
        let toolset = get_test_toolset();
        let tools = toolset.get_tool_definitions();
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
            toolset.tools.keys().cloned().collect::<Vec<_>>(),
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
            toolset.tools.keys().cloned().collect::<Vec<_>>(),
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
                Box::pin(async move { Ok(ToolOutput::text(output)) })
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

        let defs = toolset.get_tool_definitions();
        let def_names: Vec<String> = defs.into_iter().map(|def| def.name).collect();
        assert_eq!(def_names, names);

        let docs = toolset.documents();
        let doc_ids: Vec<String> = docs.into_iter().map(|doc| doc.id).collect();
        assert_eq!(doc_ids, names);
    }

    #[tokio::test]
    async fn typed_tool_name_is_definition_source_of_truth() {
        struct NamedTool;

        impl Tool for NamedTool {
            const NAME: &'static str = "canonical";
            type Error = rig::tool::ToolExecutionError;
            type Args = serde_json::Value;
            type Output = String;

            fn description(&self) -> String {
                "uses the canonical typed name".to_string()
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
        toolset.add_tool(NamedTool);

        let defs = toolset.get_tool_definitions();
        assert_eq!(defs[0].name, NamedTool::NAME);

        let docs = toolset.documents();
        assert_eq!(docs[0].id, NamedTool::NAME);
        assert!(docs[0].text.contains(NamedTool::NAME));
    }

    #[test]
    fn retrieved_tool_schemas_use_canonical_name() {
        #[derive(Debug, thiserror::Error)]
        #[error("init error")]
        struct InitError;

        struct RetrievedTool;

        impl Tool for RetrievedTool {
            const NAME: &'static str = "retrieved";
            type Error = rig::tool::ToolExecutionError;
            type Args = serde_json::Value;
            type Output = String;

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

        impl ToolEmbedding for RetrievedTool {
            type InitError = InitError;
            type Context = ();
            type State = ();

            fn embedding_docs(&self) -> Vec<String> {
                vec!["dynamic tool docs".to_string()]
            }

            fn context(&self) -> Self::Context {}

            fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
                Ok(Self)
            }
        }

        let toolset = ToolSet::builder().retrieved_tool(RetrievedTool).build();

        let schemas = toolset.schemas().unwrap();
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0].name, RetrievedTool::NAME);
        assert_eq!(schemas[0].embedding_docs, vec!["dynamic tool docs"]);
    }

    #[tokio::test]
    async fn duplicate_registration_replaces_in_place() {
        let mut toolset = ToolSet::default();
        toolset.add_dynamic_tool(named_tool("alpha", "first alpha"));
        toolset.add_dynamic_tool(named_tool("beta", "beta"));
        toolset.add_dynamic_tool(named_tool("alpha", "second alpha"));

        let defs = toolset.get_tool_definitions();
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
            .output()
            .render();
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

        let defs = base.get_tool_definitions();
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
            .await;

        assert_eq!(output.output(), &ToolOutput::text("Hello\nWorld"));
    }

    #[tokio::test]
    async fn json_shaped_string_output_stays_literal_text_through_dispatch() {
        struct JsonShapedStringTool;

        impl Tool for JsonShapedStringTool {
            const NAME: &'static str = "json_shaped_string";
            type Error = rig::tool::ToolExecutionError;
            type Args = serde_json::Value;
            type Output = String;

            fn description(&self) -> String {
                "Returns text that happens to look like a rich-content envelope".into()
            }

            fn parameters(&self) -> serde_json::Value {
                json!({"type": "object"})
            }

            async fn call(
                &self,
                _context: &mut ToolContext,
                _args: Self::Args,
            ) -> Result<Self::Output, ToolExecutionError> {
                Ok(r#"{"type":"image","data":"literal"}"#.to_string())
            }
        }

        let mut toolset = ToolSet::default();
        toolset.add_tool(JsonShapedStringTool);

        let result = toolset
            .execute(JsonShapedStringTool::NAME, "{}", &mut ToolContext::new())
            .await;

        assert_eq!(
            result.output(),
            &ToolOutput::text(r#"{"type":"image","data":"literal"}"#)
        );
    }

    #[tokio::test]
    async fn explicit_image_tool_outputs_remain_structured() {
        let mut toolset = ToolSet::default();
        toolset.add_tool(MockImageOutputTool);

        let result = toolset
            .execute("image_output", "{}", &mut ToolContext::new())
            .await;
        let content = result.output().clone().into_content();

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

        let result = toolset
            .execute("object_output", "{}", &mut ToolContext::new())
            .await;

        assert_eq!(
            result.output(),
            &ToolOutput::json(json!({
                "status": "ok",
                "count": 42
            }))
        );
    }

    #[tokio::test]
    async fn null_args_are_preserved_for_unit_args() {
        let mut toolset = ToolSet::default();
        toolset.add_tool(MockExampleTool);

        let output = toolset
            .execute("example_tool", "null", &mut ToolContext::new())
            .await;

        assert_eq!(output.output(), &ToolOutput::text("Example answer"));
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
            type Error = MockToolError;
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
            ) -> Result<Self::Output, Self::Error> {
                Ok(args.label.unwrap_or_else(|| "default".to_string()))
            }
        }

        let mut toolset = ToolSet::default();
        toolset.add_tool(NoArgTool);

        // `null` is what LLMs send when no arguments are provided; without the
        // normalization this would return an `InvalidArgs` execution error.
        let output = toolset
            .execute("no_arg_tool", "null", &mut ToolContext::new())
            .await;

        assert_eq!(output.output(), &ToolOutput::text("default"));
    }
}
