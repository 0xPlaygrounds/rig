//! Contextual tool authoring and the erased tool.
//!
//! A typed [`Tool`] implements one [`Tool::call`] method. Rig erases it
//! once ([`ErasedTool`]) into a tool-family handler and exposes a single
//! [`ToolResult`] view to hooks and runtime callers. [`ToolContext`] is the sole
//! path for typed inbound context and host-only result metadata.
//!
//! None of this is runtime-specific: a registry that holds tools by name,
//! pins them per turn and dispatches to them is a driver's business
//! (`rig_agent::tool::{ToolSet, ToolCatalog}` for the futures agent); this
//! module has only what a tool author implements and what a bus takes.
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
//! #[derive(Serialize, Deserialize)]
//! struct AuditRecord(i64);
//!
//! impl rig_core::tool::ContextValue for AuditRecord {
//!     const KEY: &'static str = "audit_record";
//! }
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
//!         let _ = context.insert_result(AuditRecord(value));
//!         Ok(Sum { value })
//!     }
//! }
//! ```
//!
//! Return [`ToolOutput`] for explicit JSON or multimodal presentation. A
//! [`ToolResultContent`](crate::message::ToolResultContent) or a `Vec` of
//! content blocks can also be used directly as a typed tool output without
//! being mistaken for ordinary JSON.
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
//!     output.as_content().first(),
//!     Some(ToolResultContent::Image(_))
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
//! | Parallel error and failure types | [`ToolExecutionError`] and [`ToolErrorKind`](super::ToolErrorKind) |
//! | Author-facing outcome enums | Ordinary `Result<T, Self::Error>` normalized at dispatch |
//! | Separate call/result extension maps | [`ToolContext`] |
//! | Parallel string/structured dispatch | `ToolSet::execute` and the registry handle's `execute`, both in `rig-agent` |
//!
//! Model-visible output remains typed throughout dispatch. Rendering to text is
//! a terminal provider or telemetry concern; Rig does not reconstruct rich
//! content by parsing a returned string.

use std::future::Future;

use serde::{Deserialize, Serialize};

use crate::{
    bus::{ErasedHandler, adapters::ToolCallback, adapters::ToolFn},
    completion::ToolDefinition,
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

use super::{
    IntoToolOutput, PortableDynamicTool, ToolContext, ToolExecutionError, ToolOutput, ToolResult,
    portable::LivenessFn,
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
    /// and `Vec<ToolResultContent>` preserve rich content when returned
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
    T: super::PortableTool,
{
    const NAME: &'static str = <T as super::PortableTool>::NAME;
    type Args = <T as super::PortableTool>::Args;
    type Output = <T as super::PortableTool>::Output;
    type Error = <T as super::PortableTool>::Error;

    fn description(&self) -> String {
        super::PortableTool::description(self)
    }

    fn parameters(&self) -> serde_json::Value {
        super::PortableTool::parameters(self)
    }

    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        super::PortableTool::map_error(self, error)
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        super::PortableTool::call(self, args).await
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

impl<T> ToolEmbedding for T
where
    T: super::PortableToolEmbedding,
{
    type InitError = <T as super::PortableToolEmbedding>::InitError;
    type Context = <T as super::PortableToolEmbedding>::Context;
    type State = <T as super::PortableToolEmbedding>::State;

    fn embedding_docs(&self) -> Vec<String> {
        super::PortableToolEmbedding::embedding_docs(self)
    }

    fn context(&self) -> Self::Context {
        super::PortableToolEmbedding::context(self)
    }

    fn init(state: Self::State, context: Self::Context) -> Result<Self, Self::InitError> {
        super::PortableToolEmbedding::init(state, context)
    }
}

fn parse_tool_args<A>(args: &str) -> Result<A, ToolExecutionError>
where
    A: serde::de::DeserializeOwned,
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

/// Normalize one erased invocation's outcome into the canonical [`ToolResult`].
///
/// Output conversion happens here rather than in each [`ErasedTool::execute`] so
/// a conversion failure and an execution failure reach the runtime as the same
/// kind of failed result.
/// Run a runtime-defined tool callback over raw JSON arguments: the parse
/// and result-shaping every tool shares, for the bus's `ToolFn` handler.
pub(crate) async fn execute_callback<F>(
    callback: &F,
    args: String,
    context: &mut ToolContext,
) -> ToolResult
where
    F: for<'a> Fn(
        &'a mut ToolContext,
        serde_json::Value,
    ) -> WasmBoxedFuture<'a, Result<ToolOutput, ToolExecutionError>>,
{
    let args = match parse_tool_args::<serde_json::Value>(&args) {
        Ok(args) => args,
        Err(error) => return ToolResult::failed(error),
    };
    tool_result_from(callback(context, args).await)
}

fn tool_result_from<O>(outcome: Result<O, ToolExecutionError>) -> ToolResult
where
    O: IntoToolOutput,
{
    match outcome.and_then(IntoToolOutput::into_tool_output) {
        Ok(output) => ToolResult::success(output),
        Err(error) => ToolResult::failed(error),
    }
}

/// The object-safe form of [`Tool`]: raw JSON arguments in, a
/// [`ToolResult`] out. This is the impl-side contract the bus's
/// `ToolAdapter` calls; nothing stores it behind a vtable.
pub trait ErasedTool: WasmCompatSend + WasmCompatSync {
    /// The tool's name.
    fn name(&self) -> String;
    /// The tool's description.
    fn description(&self) -> String;
    /// The JSON schema of the tool's arguments.
    fn parameters(&self) -> serde_json::Value;
    /// Run the tool on raw arguments, shaping the answer into a result.
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
            tool_result_from(
                Tool::call(self, context, args)
                    .await
                    .map_err(|error| Tool::map_error(self, error)),
            )
        })
    }
}

/// A tool defined at runtime: a name, a schema and a callback. The callback
/// is the handler ([`ToolFn`]); this struct is its definition plus the
/// erased handler a registry stages until a bus takes it.
#[derive(Clone)]
pub struct DynamicTool {
    definition: ToolDefinition,
    handler: ErasedHandler,
    liveness: Option<LivenessFn>,
}

impl DynamicTool {
    /// Define a tool from a callback over the dispatch-scoped context.
    pub fn new<F>(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
        callback: F,
    ) -> Self
    where
        F: ToolCallback + 'static,
    {
        let name = name.into();
        let description = description.into();
        let handler = ErasedHandler::new(ToolFn::new(
            name.clone(),
            description.clone(),
            parameters.clone(),
            callback,
        ));
        Self {
            definition: ToolDefinition {
                name,
                description,
                parameters,
            },
            handler,
            liveness: None,
        }
    }

    /// Adopt a portable tool, keeping its liveness probe.
    pub fn from_portable(tool: PortableDynamicTool) -> Self {
        let (definition, handler, liveness) = tool.into_parts();
        Self {
            definition,
            handler,
            liveness,
        }
    }

    /// The tool's name.
    pub fn name(&self) -> &str {
        &self.definition.name
    }

    /// The tool's definition.
    pub fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    /// The erased handler behind this definition.
    pub fn handler(&self) -> &ErasedHandler {
        &self.handler
    }

    /// The definition, the handler and the liveness probe, by value — what
    /// a registry stages from a runtime-defined tool.
    pub fn into_parts(self) -> (ToolDefinition, ErasedHandler, Option<LivenessFn>) {
        (self.definition, self.handler, self.liveness)
    }

    /// Whether the tool's owner still serves it (`true` without a probe).
    pub fn is_live(&self) -> bool {
        self.liveness.as_ref().is_none_or(|probe| probe())
    }
}

impl std::fmt::Debug for DynamicTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicTool")
            .field("name", &self.definition.name)
            .finish_non_exhaustive()
    }
}

impl From<PortableDynamicTool> for DynamicTool {
    fn from(tool: PortableDynamicTool) -> Self {
        Self::from_portable(tool)
    }
}

/// A tool's [`ToolDefinition`] from a typed tool.
pub fn tool_definition<T: Tool>(tool: &T) -> ToolDefinition {
    ToolDefinition {
        name: T::NAME.to_string(),
        description: tool.description(),
        parameters: tool.parameters(),
    }
}
