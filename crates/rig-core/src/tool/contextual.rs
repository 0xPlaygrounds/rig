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

use std::{collections::HashMap, future::Future, sync::Arc};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::{
    completion::{self, ToolDefinition},
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
fn tool_result_from<O>(outcome: Result<O, ToolExecutionError>) -> ToolResult
where
    O: IntoToolOutput,
{
    match outcome.and_then(IntoToolOutput::into_tool_output) {
        Ok(output) => ToolResult::success(output),
        Err(error) => ToolResult::failed(error),
    }
}

/// Object-safe dispatch boundary.
///
/// Every [`Tool`] erases into it; adapters for remote tool protocols (MCP,
/// for example) implement it directly.
pub trait ErasedTool: WasmCompatSend + WasmCompatSync {
    fn name(&self) -> String;
    fn description(&self) -> String;
    fn parameters(&self) -> serde_json::Value;
    /// Whether the runtime backing this registration can still accept calls.
    ///
    /// In-process tools are always live. Remote adapters override this so the
    /// registry can retire disconnected owners without probing by execution.
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
            tool_result_from(
                Tool::call(self, context, args)
                    .await
                    .map_err(|error| Tool::map_error(self, error)),
            )
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
    /// Liveness source inherited from a portable tool, if any.
    liveness: Option<PortableDynamicTool>,
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
            liveness: None,
        }
    }

    /// Adapt a portable dynamic tool for the classic contextual registry.
    ///
    /// The portable callback receives the same parsed JSON value **and the
    /// dispatch's [`ToolContext`]** (so context-aware portable tools see the
    /// agent's per-call values and their result inserts reach hooks); its
    /// [`ToolOutput`] or [`ToolExecutionError`] is forwarded unchanged.
    pub fn from_portable(tool: PortableDynamicTool) -> Self {
        let definition = tool.definition();
        let probe = tool.clone();
        let mut adapted = Self::new(
            definition.name,
            definition.description,
            definition.parameters,
            move |context, arguments| {
                let tool = tool.clone();
                Box::pin(async move { tool.execute_with(context, arguments).await })
            },
        );
        adapted.liveness = Some(probe);
        adapted
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

impl From<PortableDynamicTool> for DynamicTool {
    fn from(tool: PortableDynamicTool) -> Self {
        Self::from_portable(tool)
    }
}

impl ErasedTool for DynamicTool {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn is_live(&self) -> bool {
        self.liveness
            .as_ref()
            .is_none_or(PortableDynamicTool::is_live)
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
            let args = match parse_tool_args::<serde_json::Value>(&args) {
                Ok(args) => args,
                Err(error) => return ToolResult::failed(error),
            };
            tool_result_from((self.callback)(context, args).await)
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

/// The erased twin of [`ToolEmbedding`]: an [`ErasedTool`] that can also hand
/// its context and documents to a vector store.
pub trait ErasedEmbeddingTool: ErasedTool {
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

/// One erased tool as a [`ToolSet`] holds it: either a plain erased tool or
/// one that also carries embedding context for retrieval.
#[derive(Clone)]
pub enum RegisteredTool {
    /// A tool that is always dispatchable and carries no embedding context.
    Static(Arc<dyn ErasedTool>),
    /// A tool that can also be stored in, and retrieved from, a vector index.
    Embedding(Arc<dyn ErasedEmbeddingTool>),
}

impl RegisteredTool {
    fn erased(&self) -> &dyn ErasedTool {
        match self {
            Self::Static(tool) => &**tool,
            Self::Embedding(tool) => &**tool,
        }
    }

    /// The tool's own name.
    pub fn name(&self) -> String {
        self.erased().name()
    }

    /// The provider-facing definition, advertised under `name`.
    pub fn definition_with_name(&self, name: impl Into<String>) -> ToolDefinition {
        definition_with_name(name, self.erased())
    }

    /// Whether the backing runtime can still accept calls.
    pub fn is_live(&self) -> bool {
        self.erased().is_live()
    }

    /// Execute through the erased boundary.
    pub async fn execute(&self, args: String, context: &mut ToolContext) -> ToolResult {
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
pub struct ToolDispatch {
    /// The canonical result.
    pub result: ToolResult,
    /// The per-dispatch context the tool ran against (inbound snapshot plus
    /// whatever result metadata it inserted).
    pub context: ToolContext,
}

impl ToolDispatch {
    /// Publish the dispatch's result metadata back to the caller's context and
    /// surface the result. Mutations to the tool's inbound snapshot are
    /// discarded.
    pub fn publish_to(self, context: &mut ToolContext) -> ToolResult {
        context.accept_dispatch_result(self.context);
        self.result
    }
}

/// Execute a resolved registry entry through the single dispatch boundary.
///
/// Every surface enters here with its caller-owned context. The helper clones
/// inbound values exactly once, clears prior result metadata, and returns the
/// per-dispatch context so callers can expose its metadata without publishing
/// mutations the tool made to its local inbound snapshot.
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

/// An ordered collection of tools.
///
/// Cloning is cheap and shallow: the tool implementations are shared `Arc`s,
/// and names, ordering, and exposure flags are copied. A clone is a snapshot of
/// *what is registered* — adding to or removing from one set never affects the
/// other — not of any tool's internal state.
#[derive(Clone, Default)]
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

    /// Register a context-free dynamic tool without rewriting its callback.
    pub fn add_portable_dynamic_tool(&mut self, tool: PortableDynamicTool) -> String {
        self.add_dynamic_tool(DynamicTool::from_portable(tool))
    }

    /// Register a tool that is retrieved from an embedding index at prompt time.
    ///
    /// The registration keeps the tool's embedding context and documents, so
    /// [`ToolSet::schemas`] can hand them to a vector store.
    pub fn add_retrieved_tool<T>(&mut self, tool: T) -> String
    where
        T: ToolEmbedding + 'static,
    {
        self.insert(RegisteredTool::Embedding(Arc::new(tool)))
    }

    /// Register a pre-erased tool. The extension point for adapters that
    /// implement [`ErasedTool`] directly (remote tool protocols such as MCP).
    pub fn add_erased(&mut self, tool: Arc<dyn ErasedTool>) -> String {
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

    /// Merge tools that are advertised only when selected by a retrieval index
    /// (they stay dispatchable by name, but [`ToolSet::catalog`] and
    /// [`ToolSet::always_exposed_names`] skip them).
    pub fn add_retrievable_tools(&mut self, set: ToolSet) {
        for (name, mut registration) in set.tools {
            registration.always_exposed = false;
            self.insert_registration(name, registration);
        }
    }

    /// The registered implementation behind `name`.
    pub fn get(&self, name: &str) -> Option<&RegisteredTool> {
        self.tools.get(name).map(|registration| &registration.tool)
    }

    /// Registered names in registration order, including retrieval-only tools.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tools.keys().map(String::as_str)
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the set holds no tools.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Names advertised without retrieval, in registration order.
    pub fn always_exposed_names(&self) -> impl Iterator<Item = &str> {
        self.tools
            .iter()
            .filter_map(|(name, registration)| registration.always_exposed.then_some(name.as_str()))
    }

    /// Move a registration to the end of the order, keeping its tool and
    /// exposure flag. Returns `false` if no such tool is registered.
    pub fn move_to_end(&mut self, name: &str) -> bool {
        self.tools
            .shift_remove_entry(name)
            .is_some_and(|(name, registration)| {
                self.tools.insert(name, registration);
                true
            })
    }

    /// Pin the always-exposed registrations into a [`ToolCatalog`]: the
    /// provider definitions plus the exact implementations behind them, in
    /// registration order. Retrieval-only tools are left out.
    pub fn catalog(&self) -> ToolCatalog {
        ToolCatalog::from_registered(
            self.tools
                .iter()
                .filter(|(_, registration)| registration.always_exposed)
                .map(|(name, registration)| (name.clone(), registration.tool.clone()))
                .collect(),
        )
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
        let dispatch = dispatch_tool(name, args.into(), tool, context).await;
        dispatch.publish_to(context)
    }

    /// Documents describing all registered tools.
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

    use super::*;
    use crate::message::{ImageMediaType, ToolResultContent};
    use crate::tool::ToolErrorKind;

    fn rich_error_output(label: &str) -> ToolOutput {
        ToolOutput::content(vec![
            ToolResultContent::text(label),
            ToolResultContent::image_base64("base64data==", Some(ImageMediaType::PNG), None),
        ])
        .expect("fixture content is non-empty")
    }

    fn assert_rich_error_output(result: &ToolResult, label: &str) {
        let content = result.output().as_content();
        assert_eq!(content.len(), 2);
        assert!(matches!(
            content.first(),
            Some(ToolResultContent::Text(text)) if text.text == label
        ));
        assert!(matches!(content.last(), Some(ToolResultContent::Image(_))));
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
            result.output().as_content().first(),
            Some(ToolResultContent::Image(_))
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
