//! Contextual tool authoring and JSON-argument dispatch adapters.
//! [`ToolContext`] carries typed inbound values and host-only result metadata;
//! model-visible outputs retain their text, JSON, or multimodal representation.
//!
//! ```
//! use rig_core::{message::ToolName, tool::{DynamicTool, ToolOutput}};
//!
//! let tool = DynamicTool::new_with_context(ToolName::new("echo")?, "Echo JSON", serde_json::json!({}),
//!     |_context, args| Box::pin(async move { Ok(ToolOutput::json(args)) }));
//! assert_eq!(tool.name(), "echo");
//! # Ok::<(), rig_core::message::EmptyToolName>(())
//! ```

use std::{future::Future, sync::Arc};

use serde::{Deserialize, Serialize};

use crate::{
    completion::{ToolAnnotations, ToolDefinition, message::ToolName},
    serve::{ErasedHandler, adapters::ToolCallback},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

use super::{IntoToolOutput, ToolContext, ToolExecutionError, ToolOutput, ToolResult, Touches};

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

    /// What calling the tool does to the world, for the host's policies
    /// (approval, plan mode, which calls run side by side). Nothing is
    /// stated by default, which reads as the protocol's defaults
    /// ([`ToolAnnotations`]).
    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
    }

    /// What a call with `args` may read and change, so a host runs the
    /// calls of one reply that touch different things side by side. The
    /// default reads it from the [annotations](Self::annotations): a
    /// read-only tool reads anything, any other writes anything.
    fn footprint(&self, args: &Self::Args) -> Touches {
        let _ = args;
        Touches::from_annotations(&self.annotations())
    }

    /// Normalize a typed author-facing error for runtime policy and telemetry.
    ///
    /// The default preserves concrete sources for operators and exposes safe
    /// kind-level model feedback. An existing [`ToolExecutionError`] retains its
    /// classification and model output. Override to supply deliberate domain feedback.
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

pub(crate) fn parse_tool_args<A>(args: &str) -> Result<A, ToolExecutionError>
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

pub(crate) fn tool_result_from<O>(outcome: Result<O, ToolExecutionError>) -> ToolResult
where
    O: IntoToolOutput,
{
    match outcome.and_then(IntoToolOutput::into_tool_output) {
        Ok(output) => ToolResult::success(output),
        Err(error) => ToolResult::failed(error),
    }
}

/// Reports whether a tool's owner still serves it. Registries check lazily on
/// reads or reconciliation; calls before retirement may fail with a transport
/// error rather than `HandlerUnavailable`.
#[cfg(not(target_family = "wasm"))]
pub type LivenessFn = Arc<dyn Fn() -> bool + Send + Sync>;
/// A liveness probe (browser wasm: no `Send + Sync`, no threads).
#[cfg(target_family = "wasm")]
pub type LivenessFn = Arc<dyn Fn() -> bool>;

/// A tool defined at runtime: its definition and a callback over JSON
/// arguments. It is its own handler ([`Serve`](crate::serve::Serve)), keyed
/// by its name. The optional liveness probe supports registry retirement;
/// serving does not consult it.
#[derive(Clone)]
pub struct DynamicTool {
    definition: ToolDefinition,
    callback: Arc<dyn ToolCallback>,
    liveness: Option<LivenessFn>,
}

impl DynamicTool {
    /// Define a tool from a context-free callback over owned arguments.
    pub fn new<F>(
        name: ToolName,
        description: impl Into<String>,
        parameters: serde_json::Value,
        callback: F,
    ) -> Self
    where
        F: Fn(
                serde_json::Value,
            ) -> WasmBoxedFuture<'static, Result<ToolOutput, ToolExecutionError>>
            + WasmCompatSend
            + WasmCompatSync
            + 'static,
    {
        Self::new_with_context(
            name,
            description,
            parameters,
            move |_context: &mut ToolContext, arguments| callback(arguments),
        )
    }

    /// Define a tool from a callback over the dispatch-scoped context.
    pub fn new_with_context<F>(
        name: ToolName,
        description: impl Into<String>,
        parameters: serde_json::Value,
        callback: F,
    ) -> Self
    where
        F: ToolCallback + 'static,
    {
        Self {
            definition: ToolDefinition::new(name, description, parameters),
            callback: Arc::new(callback),
            liveness: None,
        }
    }

    /// The same tool, stating what calling it does to the world.
    pub fn with_annotations(mut self, annotations: ToolAnnotations) -> Self {
        self.definition.annotations = annotations;
        self
    }

    /// Attach a liveness probe.
    pub fn with_liveness<F>(mut self, is_live: F) -> Self
    where
        F: Fn() -> bool + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.liveness = Some(Arc::new(is_live));
        self
    }

    /// The tool's name.
    pub fn name(&self) -> &ToolName {
        &self.definition.name
    }

    /// The tool's definition.
    pub fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    /// Consumes the tool into its definition, its erased handler, and its
    /// optional liveness probe.
    pub fn into_parts(self) -> (ToolDefinition, ErasedHandler, Option<LivenessFn>) {
        let Self {
            definition,
            callback,
            liveness,
        } = self;
        let handler = ErasedHandler::new(Self {
            definition: definition.clone(),
            callback,
            liveness: None,
        });
        (definition, handler, liveness)
    }

    /// Runs the tool on JSON `args` with an empty [`ToolContext`], as a
    /// dispatch of its call runs it, and returns its output or its error
    /// (invalid arguments included). No registry or agent is involved.
    ///
    /// ```
    /// use rig_core::{message::ToolName, tool::{DynamicTool, ToolExecutionError, ToolOutput}};
    /// use serde_json::json;
    ///
    /// let add = DynamicTool::new(ToolName::new("add")?, "Adds a and b", json!({}), |args| {
    ///     Box::pin(async move {
    ///         let int = |key| args.get(key).and_then(serde_json::Value::as_i64);
    ///         let sum = int("a").zip(int("b")).map(|(a, b)| a + b);
    ///         sum.map(|sum| ToolOutput::json(json!(sum)))
    ///             .ok_or_else(|| ToolExecutionError::invalid_args("a and b must be integers"))
    ///     })
    /// });
    /// let output = futures::executor::block_on(add.run(json!({ "a": 2, "b": 3 })))?;
    /// assert_eq!(output.as_json(), Some(&json!(5)));
    /// assert!(futures::executor::block_on(add.run(json!({ "a": "two" }))).is_err());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub async fn run(&self, args: serde_json::Value) -> Result<ToolOutput, ToolExecutionError> {
        let dispatch = crate::serve::Dispatch::new(crate::effect::EffectId::from_raw(0), false);
        crate::serve::adapters::run_tool(&args.to_string(), &dispatch, async |context, args| {
            (self.callback)(context, args).await
        })
        .await
        .into_result()
    }

    /// Whether the tool's owner still serves it (`true` without a probe).
    pub fn is_live(&self) -> bool {
        self.liveness.as_ref().is_none_or(|probe| probe())
    }
}

impl crate::serve::Serve for DynamicTool {
    type Family = crate::effect::family::Tool;

    fn descriptor(&self) -> crate::effect::HandlerDescriptor {
        let ToolDefinition {
            name,
            description,
            parameters,
            ..
        } = &self.definition;
        crate::effect::HandlerDescriptor::tool(name, description, parameters.clone())
    }

    fn key(&self) -> crate::effect::HandlerKey {
        crate::effect::tool_key(&self.definition.name)
    }

    async fn serve(
        &self,
        kind: crate::effect::EffectKind,
        dispatch: crate::serve::Dispatch,
    ) -> crate::serve::Reply {
        crate::serve::adapters::call_tool(kind, dispatch, None, async |context, args| {
            (self.callback)(context, args).await
        })
        .await
    }
}

impl std::fmt::Debug for DynamicTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicTool")
            .field("name", &self.definition.name)
            .finish_non_exhaustive()
    }
}

/// A typed tool's [`Tool::NAME`] as a [`ToolName`].
///
/// An empty `NAME` fails the build where this is instantiated.
///
/// ```
/// use rig_core::{message::ToolChoice, tool::{Tool, ToolContext, tool_name}};
///
/// struct Ping;
///
/// impl Tool for Ping {
///     const NAME: &'static str = "ping";
///     type Args = serde_json::Value;
///     type Output = String;
///     type Error = std::convert::Infallible;
///
///     fn description(&self) -> String {
///         "Replies with pong.".to_string()
///     }
///
///     fn parameters(&self) -> serde_json::Value {
///         serde_json::json!({ "type": "object" })
///     }
///
///     async fn call(
///         &self,
///         _context: &mut ToolContext,
///         _args: Self::Args,
///     ) -> Result<String, Self::Error> {
///         Ok("pong".to_string())
///     }
/// }
///
/// let choice = ToolChoice::Specific { function_names: vec![tool_name::<Ping>()] };
/// assert_eq!(tool_name::<Ping>(), "ping");
/// # let _ = choice;
/// ```
pub fn tool_name<T: Tool>() -> ToolName {
    const { assert!(!T::NAME.is_empty(), "Tool::NAME cannot be empty") };
    ToolName::new_unchecked(T::NAME)
}

/// A tool's [`ToolDefinition`] from a typed tool, with its annotations.
pub fn tool_definition<T: Tool>(tool: &T) -> ToolDefinition {
    ToolDefinition::new(tool_name::<T>(), tool.description(), tool.parameters())
        .with_annotations(tool.annotations())
}
