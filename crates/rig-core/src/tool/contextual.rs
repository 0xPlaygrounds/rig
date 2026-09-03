//! Contextual tool authoring, the erased tool set, and canonical dispatch.
//!
//! A typed [`Tool`] implements one [`Tool::call`] method. Rig erases it
//! once ([`ErasedTool`]), stores it in an ordered [`ToolSet`], executes it
//! through one structured path ([`dispatch_tool`]), and exposes a single
//! [`ToolResult`] view to hooks and runtime callers. [`ToolContext`] is the sole
//! path for typed inbound context and host-only result metadata.
//!
//! None of this is runtime-specific: the futures agent driver (`rig-agent`)
//! layers its live registry, retrieval indexes, and managed tool sources over
//! these types, and a systems driver can hold a [`ToolSet`] or a
//! [`ToolCatalog`] in shared state and dispatch by name
//! without depending on either.
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
//! | Parallel string/structured dispatch | [`ToolSet::execute`] (and the registry handle's `execute` in `rig-agent`) |
//!
//! Model-visible output remains typed throughout dispatch. Rendering to text is
//! a terminal provider or telemetry concern; Rig does not reconstruct rich
//! content by parsing a returned string.

use std::{collections::HashMap, future::Future};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::{
    bus::{ErasedHandler, Key, adapters::ToolCallback, adapters::ToolFn},
    completion::{self, ToolDefinition},
    effect::{EffectKind, FamilyDescriptor, Outcome, ToolEmbeddingDescriptor, family},
    embeddings::{embed::EmbedError, tool::ToolSchema},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

use super::{
    IntoToolOutput, PortableDynamicTool, ToolContext, ToolExecutionError, ToolOutput, ToolResult,
    catalog::ToolCatalog,
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
    liveness: Option<super::portable::LivenessFn>,
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

/// One registration: the tool's serde description, the key it is (or will
/// be) served under, and the erased handler a bus takes. Cloning shares the
/// handler.
#[derive(Clone)]
pub struct RegisteredTool {
    definition: ToolDefinition,
    embedding: Option<ToolEmbeddingDescriptor>,
    key: Key<family::Tool>,
    handler: ErasedHandler,
    liveness: Option<super::portable::LivenessFn>,
}

impl std::fmt::Debug for RegisteredTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisteredTool")
            .field("name", &self.definition.name)
            .field("key", &self.key)
            .field("retrievable", &self.embedding.is_some())
            .finish_non_exhaustive()
    }
}

impl RegisteredTool {
    /// Register a typed tool.
    pub fn from_tool<T>(tool: T) -> Self
    where
        T: Tool + 'static,
    {
        let definition = tool_definition(&tool);
        Self::from_parts(
            definition,
            None,
            ErasedHandler::new(crate::bus::adapters::ToolAdapter::new(tool)),
            None,
        )
    }

    /// Register a retrievable tool: its embedding context rides on the
    /// descriptor.
    pub fn from_retrievable<T>(tool: T) -> Result<Self, serde_json::Error>
    where
        T: ToolEmbedding + 'static,
    {
        let definition = tool_definition(&tool);
        let embedding = ToolEmbeddingDescriptor {
            context: serde_json::to_value(tool.context())?,
            embedding_docs: tool.embedding_docs(),
        };
        let adapter = crate::bus::adapters::ToolAdapter::retrievable(tool)?;
        Ok(Self::from_parts(
            definition,
            Some(embedding),
            ErasedHandler::new(adapter),
            None,
        ))
    }

    /// Register a runtime-defined tool.
    pub fn from_dynamic(tool: DynamicTool) -> Self {
        Self::from_parts(tool.definition, None, tool.handler, tool.liveness)
    }

    /// Register any tool-family handler under the key its descriptor names
    /// — a replayer answering a recorded tool from the effect log, a host's
    /// own handler. Fails when the handler is not of the tool family.
    pub fn from_handler(
        handler: impl crate::bus::Serve + 'static,
    ) -> Result<Self, crate::error::ErrorReport> {
        let descriptor = handler.descriptor();
        let FamilyDescriptor::Tool {
            name,
            description,
            parameters,
            embedding,
        } = descriptor.family
        else {
            return Err(crate::error::ErrorReport::new(
                crate::error::ErrorKind::HandlerUnavailable,
                format!(
                    "handler `{}` serves the {} family, not tool_call",
                    descriptor.key,
                    descriptor.family.family()
                ),
            ));
        };
        Ok(Self {
            definition: ToolDefinition {
                name,
                description,
                parameters,
            },
            embedding,
            key: Key::new_unchecked(descriptor.key),
            handler: ErasedHandler::new(handler),
            liveness: None,
        })
    }

    /// Whether this registration is served under the default `tool:<name>`
    /// key (a registry that pins generations re-keys only those).
    pub fn has_default_key(&self) -> bool {
        *self.key.raw() == crate::bus::tool_key(&self.definition.name)
    }

    fn from_parts(
        definition: ToolDefinition,
        embedding: Option<ToolEmbeddingDescriptor>,
        handler: ErasedHandler,
        liveness: Option<super::portable::LivenessFn>,
    ) -> Self {
        let key = Key::new_unchecked(crate::bus::tool_key(&definition.name));
        Self {
            definition,
            embedding,
            key,
            handler,
            liveness,
        }
    }

    /// Serve this registration under `key` instead of the default
    /// `tool:<name>` (a registry that pins generations does this).
    pub fn with_key(mut self, key: Key<family::Tool>) -> Self {
        self.key = key;
        self
    }

    /// The tool's name.
    pub fn name(&self) -> String {
        self.definition.name.clone()
    }

    /// The key the handler is served under: a tool key, proven at
    /// construction (the descriptor was checked to be tool-family).
    pub fn key(&self) -> &Key<family::Tool> {
        &self.key
    }

    /// The erased handler.
    pub fn handler(&self) -> &ErasedHandler {
        &self.handler
    }

    /// The definition, advertised under `name`.
    pub fn definition_with_name(&self, name: impl Into<String>) -> ToolDefinition {
        ToolDefinition {
            name: name.into(),
            description: self.definition.description.clone(),
            parameters: self.definition.parameters.clone(),
        }
    }

    /// The family-keyed descriptor of this registration.
    pub fn descriptor(&self) -> FamilyDescriptor {
        FamilyDescriptor::Tool {
            name: self.definition.name.clone(),
            description: self.definition.description.clone(),
            parameters: self.definition.parameters.clone(),
            embedding: self.embedding.clone(),
        }
    }

    /// The embedding context, for retrievable tools.
    pub fn embedding(&self) -> Option<&ToolEmbeddingDescriptor> {
        self.embedding.as_ref()
    }

    /// Whether the tool's owner still serves it (`true` without a probe).
    pub fn is_live(&self) -> bool {
        self.liveness.as_ref().is_none_or(|probe| probe())
    }

    /// Run the tool here, without a bus, publishing its result metadata
    /// into `context` — the inline path of a standalone tool set.
    pub async fn execute(&self, args: String, context: &mut ToolContext) -> ToolResult {
        let name = self.definition.name.clone();
        let outcome = crate::bus::serve_inline(
            &self.handler,
            EffectKind::ToolCall {
                name,
                args,
                context: std::mem::take(context),
            },
        )
        .await;
        match outcome {
            Ok(Outcome::ToolResult {
                result,
                context: published,
            }) => {
                *context = published;
                result
            }
            Ok(other) => ToolResult::failed(ToolExecutionError::other(format!(
                "tool handler answered with a {} outcome",
                other.family()
            ))),
            Err(report) => ToolResult::failed(ToolExecutionError::other(report.message)),
        }
    }
}

/// One entry of a [`ToolSet`]: the registration plus whether it is
/// advertised on every request or only when retrieved.
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

/// A tool call's result together with the dispatch-scoped context it
/// published into.
pub struct ToolDispatch {
    /// The result.
    pub result: ToolResult,
    /// The context after the tool ran.
    pub context: ToolContext,
}

impl ToolDispatch {
    /// Publish the dispatch's result metadata into `context` and return the
    /// result.
    pub fn publish_to(self, context: &mut ToolContext) -> ToolResult {
        context.accept_dispatch_result(self.context);
        self.result
    }
}

/// Run `tool` (or answer `not found`) on a dispatch-scoped copy of
/// `context`.
pub async fn dispatch_tool(
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

/// A named set of tool registrations: the definition and advertisement
/// surface. Execution goes through the registrations' handlers — over a bus
/// once one has taken them, inline until then.
#[derive(Clone, Default)]
pub struct ToolSet {
    pub(crate) tools: IndexMap<String, ToolRegistration>,
}

impl ToolSet {
    /// A set from typed tools.
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

    /// A set from runtime-defined tools.
    pub fn from_dynamic_tools(tools: Vec<DynamicTool>) -> Self {
        let mut set = Self::default();
        for tool in tools {
            set.add_dynamic_tool(tool);
        }
        set
    }

    /// Whether a tool named `name` is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Register a typed tool; returns its name.
    pub fn add_tool<T>(&mut self, tool: T) -> String
    where
        T: Tool + 'static,
    {
        self.insert(RegisteredTool::from_tool(tool))
    }

    /// Register a runtime-defined tool; returns its name.
    pub fn add_dynamic_tool(&mut self, tool: DynamicTool) -> String {
        self.insert(RegisteredTool::from_dynamic(tool))
    }

    /// Register a portable tool; returns its name.
    pub fn add_portable_dynamic_tool(&mut self, tool: PortableDynamicTool) -> String {
        self.add_dynamic_tool(DynamicTool::from_portable(tool))
    }

    /// Register a retrievable tool; returns its name. A context that does
    /// not serialize is the error, and nothing is registered.
    pub fn add_retrieved_tool<T>(&mut self, tool: T) -> Result<String, serde_json::Error>
    where
        T: ToolEmbedding + 'static,
    {
        RegisteredTool::from_retrievable(tool).map(|registered| self.insert(registered))
    }

    /// Register an already-built registration; returns its name.
    pub fn add_registered(&mut self, tool: RegisteredTool) -> String {
        self.insert(tool)
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

    /// Remove the tool named `name`.
    pub fn delete_tool(&mut self, name: &str) {
        self.tools.shift_remove(name);
    }

    /// Merge `set`'s registrations, keeping their exposure.
    pub fn add_tools(&mut self, set: ToolSet) {
        for (name, registration) in set.tools {
            self.insert_registration(name, registration);
        }
    }

    /// Merge `set`'s registrations as retrieval-only (not always exposed).
    pub fn add_retrievable_tools(&mut self, set: ToolSet) {
        for (name, mut registration) in set.tools {
            registration.always_exposed = false;
            self.insert_registration(name, registration);
        }
    }

    /// The registration named `name`.
    pub fn get(&self, name: &str) -> Option<&RegisteredTool> {
        self.tools.get(name).map(|registration| &registration.tool)
    }

    /// Every registration, in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &RegisteredTool)> {
        self.tools
            .iter()
            .map(|(name, registration)| (name.as_str(), &registration.tool))
    }

    /// Every name, in insertion order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tools.keys().map(String::as_str)
    }

    /// Number of registrations.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Names advertised on every request.
    pub fn always_exposed_names(&self) -> impl Iterator<Item = &str> {
        self.tools
            .iter()
            .filter_map(|(name, registration)| registration.always_exposed.then_some(name.as_str()))
    }

    /// Move `name` to the end of the insertion order.
    pub fn move_to_end(&mut self, name: &str) -> bool {
        self.tools
            .shift_remove_entry(name)
            .is_some_and(|(name, registration)| {
                self.tools.insert(name, registration);
                true
            })
    }

    /// The always-exposed registrations as a catalog.
    pub fn catalog(&self) -> ToolCatalog {
        ToolCatalog::from_registered(
            self.tools
                .iter()
                .filter(|(_, registration)| registration.always_exposed)
                .map(|(name, registration)| (name.clone(), registration.tool.clone()))
                .collect(),
        )
    }

    /// Every definition, in insertion order.
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .map(|(name, registration)| registration.tool.definition_with_name(name.clone()))
            .collect()
    }

    /// Run the tool named `name` inline, publishing its result metadata
    /// into `context`.
    pub async fn execute(
        &self,
        name: &str,
        args: impl Into<String>,
        context: &mut ToolContext,
    ) -> ToolResult {
        context.clear_dispatch_result();
        let tool = self.get(name).cloned();
        let dispatch = dispatch_tool(name, args.into(), tool, context).await;
        dispatch.publish_to(context)
    }

    /// Every definition as a document, for embedding-based retrieval.
    pub fn documents(&self) -> Vec<completion::Document> {
        self.tools
            .iter()
            .map(|(name, registration)| {
                let definition = registration.tool.definition_with_name(name.clone());
                let serialized =
                    serde_json::to_string_pretty(&definition).unwrap_or_else(|error| {
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
                completion::Document {
                    id: name.clone(),
                    text: format!("Tool: {name}\nDefinition: \n{serialized}"),
                    additional_props: HashMap::new(),
                }
            })
            .collect()
    }

    /// The embedding schemas of the retrievable registrations.
    pub fn schemas(&self) -> Result<Vec<ToolSchema>, EmbedError> {
        Ok(self
            .tools
            .iter()
            .filter_map(|(name, registration)| {
                registration.tool.embedding().map(|embedding| ToolSchema {
                    name: name.clone(),
                    context: embedding.context.clone(),
                    embedding_docs: embedding.embedding_docs.clone(),
                })
            })
            .collect())
    }
}

#[cfg(test)]
mod tests;
