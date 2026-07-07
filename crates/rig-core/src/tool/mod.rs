//! Module defining tool related structs and traits.
//!
//! The [Tool] trait defines a simple interface for creating tools that can be used
//! by [Agents](crate::agent::Agent).
//!
//! The [ToolSet] struct is a collection of tools that can be used by an
//! [Agent](crate::agent::Agent).
//!
//! # Structured tool results
//!
//! A tool call resolves to a structured [`ToolExecutionResult`] — model-visible
//! output, a machine-readable [`ToolOutcome`] (success, a classified
//! [`ToolFailure`], skipped, or denied), and [`ToolResultExtensions`] metadata
//! that is never sent to the model. This is what flows to the
//! [`StepEvent::ToolResult`](crate::agent::StepEvent::ToolResult) hook so a
//! policy can steer on *why* a tool failed without parsing strings. A tool
//! classifies its own error type via [`Tool::classify_error`] and can return
//! richer outcomes/metadata via [`Tool::call_structured`] and [`ToolReturn`].

mod extensions;
mod result;
pub mod server;

pub use extensions::{MissingExtension, ToolCallExtensions, ToolResultExtensions};
pub use result::{
    ToolExecutionResult, ToolFailure, ToolFailureKind, ToolOutcome, ToolReturn, ToolReturnOutcome,
};
use std::fmt;
use std::sync::Arc;

use futures::Future;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::{
    completion::ToolDefinition,
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
///     async fn definition(&self) -> ToolDefinition {
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

    /// A method returning the tool definition (the provider-facing schema).
    fn definition(&self) -> impl Future<Output = ToolDefinition> + WasmCompatSend + WasmCompatSync;

    /// The tool execution method.
    /// Both the arguments and return value are a String since these values are meant to
    /// be the output and input of LLM models (respectively)
    fn call(
        &self,
        args: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + WasmCompatSend;

    /// Tool execution with per-call runtime extensions.
    ///
    /// Override this to access runtime values (auth, session IDs, etc.)
    /// injected by the caller via [`ToolCallExtensions`]. The default ignores
    /// the extensions and delegates to [`Tool::call`].
    ///
    /// **Override contract:** the default [`Tool::call_structured`] delegates
    /// here, so overriding this method is how you read extensions for the common
    /// case — the agent loop drives [`call_structured`](Self::call_structured),
    /// which reaches your override (with an empty [`ToolCallExtensions`] when no
    /// caller supplied one). Under dynamic dispatch this then becomes the single
    /// execution entry point (`call`'s body is unreachable that way; a direct
    /// `Tool::call` still runs it), so put your logic here and treat a missing
    /// value as the no-extensions case (e.g. [`ToolCallExtensions::get`]
    /// returning `None`). If you *also* override
    /// [`call_structured`](Self::call_structured), that override supersedes this
    /// method on the agent's structured path — put your logic there instead.
    fn call_with_extensions(
        &self,
        args: Self::Args,
        _extensions: &ToolCallExtensions,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + WasmCompatSend {
        self.call(args)
    }

    /// Classify an error returned by this tool into a structured [`ToolFailure`].
    ///
    /// This is how a tool's own error type reaches a hook, policy, or telemetry
    /// pipeline as a machine-readable [`ToolFailureKind`] — with no string
    /// parsing. The default classifies every error as
    /// [`ToolFailureKind::Other`] with the error's `Display` as the message;
    /// override it to map your error variants onto the standard kinds (timeout,
    /// not-found, rate-limited, …) and attach a `code` / `http_status` /
    /// `retryable` hint:
    ///
    /// ```rust,ignore
    /// fn classify_error(&self, error: &Self::Error) -> ToolFailure {
    ///     match error {
    ///         MyError::Timeout => ToolFailure::timeout(error.to_string()),
    ///         MyError::Http { status: 404, .. } => {
    ///             ToolFailure::not_found(error.to_string()).with_http_status(404)
    ///         }
    ///         other => ToolFailure::other(other.to_string()),
    ///     }
    /// }
    /// ```
    fn classify_error(&self, error: &Self::Error) -> ToolFailure {
        ToolFailure::other(error.to_string())
    }

    /// Execute the tool, returning a structured [`ToolReturn`] instead of a bare
    /// output.
    ///
    /// The richest tool-execution entry point. The default calls
    /// [`call_with_extensions`](Self::call_with_extensions) and wraps the output
    /// as a plain [`ToolReturn::success`] with no metadata, so a tool that only
    /// implements [`call`](Self::call) needs nothing extra. Override it to:
    ///
    /// - attach result metadata to a success
    ///   (`ToolReturn::success(out).with_extension(..)`);
    /// - report a handled failure that still shows output to the model
    ///   ([`ToolReturn::failed`]);
    /// - mark the call [`denied`](ToolReturn::denied) — the tool refused it (a
    ///   framework hook `Flow::Skip` is what yields a *skipped* outcome, not the tool).
    ///
    /// **Override contract:** this is the single entry point under *structured
    /// dynamic dispatch* — the agent loop routes every tool call here via the
    /// blanket [`ToolDyn`] impl. If you override it, the `call` /
    /// `call_with_extensions` bodies are unreachable on that structured path (a
    /// direct call still runs them), so put your logic here. A returned
    /// `Err(Self::Error)` is still classified via
    /// [`classify_error`](Self::classify_error).
    fn call_structured(
        &self,
        args: Self::Args,
        extensions: &ToolCallExtensions,
    ) -> impl Future<Output = Result<ToolReturn<Self::Output>, Self::Error>> + WasmCompatSend {
        async move {
            self.call_with_extensions(args, extensions)
                .await
                .map(ToolReturn::success)
        }
    }
}

/// Wrapper trait to allow for dynamic dispatch of simple tools
pub trait ToolDyn: WasmCompatSend + WasmCompatSync {
    /// Returns the tool name used for dispatch.
    fn name(&self) -> String;

    /// Returns the provider-facing tool schema.
    fn definition<'a>(&'a self) -> WasmBoxedFuture<'a, ToolDefinition>;

    /// Calls the tool with JSON-encoded arguments and returns model-facing text.
    fn call<'a>(&'a self, args: String) -> WasmBoxedFuture<'a, Result<String, ToolError>>;

    /// Dynamic dispatch variant of tool execution with per-call runtime extensions.
    ///
    /// The default ignores the extensions and delegates to [`ToolDyn::call`].
    /// The blanket impl for [`Tool`] types overrides this to thread the
    /// extensions through to [`Tool::call_with_extensions`].
    fn call_with_extensions<'a>(
        &'a self,
        args: String,
        _extensions: &'a ToolCallExtensions,
    ) -> WasmBoxedFuture<'a, Result<String, ToolError>> {
        self.call(args)
    }

    /// Execute the tool with per-call extensions, returning a structured
    /// [`ToolExecutionResult`] (model output + [`ToolOutcome`] + result
    /// extensions).
    ///
    /// This is the structured dynamic boundary the agent loop drives: the result
    /// flows through to the
    /// [`StepEvent::ToolResult`](crate::agent::StepEvent::ToolResult) hook event.
    /// Unlike [`call`](Self::call) it never returns a bare error — a failure is
    /// carried as [`ToolOutcome::Error`] inside the result, with the
    /// model-visible message on [`ToolExecutionResult::model_output`].
    ///
    /// The default wraps [`call_with_extensions`](Self::call_with_extensions): an
    /// `Ok` output becomes a [`ToolOutcome::Success`]; a [`ToolError`] is
    /// classified ([`ToolError::JsonError`] as
    /// [`ToolFailureKind::InvalidArgs`], otherwise [`ToolFailureKind::Other`]).
    /// The blanket impl for [`Tool`] types overrides this to route through
    /// [`Tool::call_structured`] and [`Tool::classify_error`]; a manual `ToolDyn`
    /// impl should override it to emit precise outcomes (e.g. a real timeout).
    fn call_structured<'a>(
        &'a self,
        args: String,
        extensions: &'a ToolCallExtensions,
    ) -> WasmBoxedFuture<'a, ToolExecutionResult> {
        Box::pin(async move {
            match self.call_with_extensions(args, extensions).await {
                Ok(model_output) => ToolExecutionResult::success(model_output),
                Err(err) => tool_error_to_execution_result(err),
            }
        })
    }
}

fn serialize_tool_output(output: impl Serialize) -> serde_json::Result<String> {
    match serde_json::to_value(output)? {
        serde_json::Value::String(text) => Ok(text),
        value => Ok(value.to_string()),
    }
}

/// Deserialize JSON tool arguments, normalizing a bare `null` (which LLMs
/// frequently send for tools whose arguments are all optional) to `{}`.
///
/// `serde_json::from_str::<T>("null")` fails for struct types even when every
/// field is `Option<_>`, because JSON null does not deserialize to an empty
/// object. Any args type that already accepts `null` (such as `()` or
/// `Option<T>`) is preserved; the fallback to `{}` only applies after the
/// original parse fails.
fn parse_tool_args<A>(args: &str) -> serde_json::Result<A>
where
    A: for<'de> Deserialize<'de>,
{
    match serde_json::from_str(args) {
        Ok(parsed) => Ok(parsed),
        Err(err) if args.trim() == "null" => serde_json::from_str("{}").map_err(|_| err),
        Err(err) => Err(err),
    }
}

/// Map a [`ToolError`] surfaced by a string-returning [`ToolDyn`] path into a
/// structured [`ToolExecutionResult`], classifying a JSON error as invalid
/// arguments. Used by the default [`ToolDyn::call_structured`] for manual
/// implementations that only provide the string [`ToolDyn::call`].
fn tool_error_to_execution_result(err: ToolError) -> ToolExecutionResult {
    let message = err.to_string();
    let failure = match err {
        ToolError::JsonError(_) => ToolFailure::invalid_args(message.clone()),
        ToolError::ToolCallError(_) => ToolFailure::other(message.clone()),
    };
    ToolExecutionResult::failed(message, failure)
}

impl<T: Tool> ToolDyn for T {
    fn name(&self) -> String {
        self.name()
    }

    fn definition<'a>(&'a self) -> WasmBoxedFuture<'a, ToolDefinition> {
        Box::pin(<Self as Tool>::definition(self))
    }

    fn call<'a>(&'a self, args: String) -> WasmBoxedFuture<'a, Result<String, ToolError>> {
        ToolDyn::call_with_extensions(self, args, &ToolCallExtensions::EMPTY)
    }

    fn call_with_extensions<'a>(
        &'a self,
        args: String,
        extensions: &'a ToolCallExtensions,
    ) -> WasmBoxedFuture<'a, Result<String, ToolError>> {
        Box::pin(async move {
            match parse_tool_args::<T::Args>(&args) {
                Ok(args) => <Self as Tool>::call_with_extensions(self, args, extensions)
                    .await
                    .map_err(|e| ToolError::ToolCallError(Box::new(e)))
                    .and_then(|output| serialize_tool_output(output).map_err(ToolError::JsonError)),
                Err(e) => Err(ToolError::JsonError(e)),
            }
        })
    }

    /// Routes through [`Tool::call_structured`] so rich returns
    /// ([`ToolReturn`]) and [`Tool::classify_error`] are honored: a JSON
    /// argument parse failure becomes an
    /// [`InvalidArgs`](ToolFailureKind::InvalidArgs) outcome, a returned
    /// `Err(Self::Error)` is classified, and a successful [`ToolReturn`] is
    /// serialized while preserving its outcome and extensions.
    fn call_structured<'a>(
        &'a self,
        args: String,
        extensions: &'a ToolCallExtensions,
    ) -> WasmBoxedFuture<'a, ToolExecutionResult> {
        Box::pin(async move {
            let parsed = match parse_tool_args::<T::Args>(&args) {
                Ok(parsed) => parsed,
                Err(err) => {
                    return ToolExecutionResult::failed(
                        format!("failed to parse tool arguments: {err}"),
                        ToolFailure::invalid_args(err.to_string()),
                    );
                }
            };
            match <Self as Tool>::call_structured(self, parsed, extensions).await {
                Ok(tool_return) => tool_return.into_execution_result(),
                Err(err) => {
                    let failure = self.classify_error(&err);
                    ToolExecutionResult::failed(err.to_string(), failure)
                }
            }
        })
    }
}

#[cfg(feature = "rmcp")]
#[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
pub mod rmcp;

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
/// (definitions) follows registration order and the tool
/// list sent to providers is deterministic across processes. Re-registering an
/// existing name replaces the implementation but keeps its original position.
#[derive(Default)]
pub struct ToolSet {
    pub(crate) tools: IndexMap<String, Arc<dyn ToolDyn>>,
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

    /// Add a tool to the toolset.
    pub fn add_tool(&mut self, tool: impl ToolDyn + 'static) {
        self.insert(Arc::new(tool));
    }

    /// Adds a boxed tool to the toolset. Useful for situations when dynamic dispatch is required.
    pub fn add_tool_boxed(&mut self, tool: Box<dyn ToolDyn>) {
        self.insert(Arc::from(tool));
    }

    pub(crate) fn insert(&mut self, tool: Arc<dyn ToolDyn>) {
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

    pub(crate) fn get(&self, toolname: &str) -> Option<&Arc<dyn ToolDyn>> {
        self.tools.get(toolname)
    }

    /// Tools in registration order.
    pub(crate) fn ordered_tools(&self) -> impl Iterator<Item = &Arc<dyn ToolDyn>> {
        self.tools.values()
    }

    /// Return definitions for all tools currently registered in the set, in
    /// registration order.
    pub async fn get_tool_definitions(&self) -> Result<Vec<ToolDefinition>, ToolSetError> {
        let mut defs = Vec::new();
        for tool in self.ordered_tools() {
            let def = tool.definition().await;
            defs.push(def);
        }
        Ok(defs)
    }

    /// Call a tool with the given name and arguments
    pub async fn call(&self, toolname: &str, args: String) -> Result<String, ToolSetError> {
        self.call_with_extensions(toolname, args, &ToolCallExtensions::EMPTY)
            .await
    }

    /// Call a tool with the given name, arguments, and per-call runtime extensions.
    ///
    /// The extensions are threaded through to [`Tool::call_with_extensions`],
    /// allowing tools to access caller-provided values (auth tokens, session
    /// IDs, etc.).
    pub async fn call_with_extensions(
        &self,
        toolname: &str,
        args: String,
        extensions: &ToolCallExtensions,
    ) -> Result<String, ToolSetError> {
        if let Some(tool) = self.tools.get(toolname) {
            tracing::debug!(target: "rig",
                "Calling tool {toolname} with args:\n{}",
                args
            );
            Ok(tool.call_with_extensions(args, extensions).await?)
        } else {
            Err(ToolSetError::ToolNotFoundError(toolname.to_string()))
        }
    }

    /// Call a tool by name, returning the structured [`ToolExecutionResult`].
    ///
    /// The structured counterpart of [`call`](Self::call): a failure is carried
    /// as [`ToolOutcome::Error`] inside the result rather than a `Result::Err`,
    /// and an unknown tool name resolves to a
    /// [`NotFound`](ToolFailureKind::NotFound) outcome. This is the path the
    /// agent loop drives so hooks and telemetry observe the structured outcome.
    pub async fn call_structured(
        &self,
        toolname: &str,
        args: String,
        extensions: &ToolCallExtensions,
    ) -> ToolExecutionResult {
        match self.tools.get(toolname) {
            Some(tool) => {
                tracing::debug!(target: "rig", "Calling tool {toolname} with args:\n{args}");
                tool.call_structured(args, extensions).await
            }
            None => ToolExecutionResult::failed(
                format!("tool `{toolname}` not found"),
                ToolFailure::not_found(format!("no tool named `{toolname}` is registered")),
            ),
        }
    }
}

#[derive(Default)]
/// Builder for constructing a [`ToolSet`].
pub struct ToolSetBuilder {
    tools: Vec<Arc<dyn ToolDyn>>,
}

impl ToolSetBuilder {
    /// Add a tool to the set.
    pub fn static_tool(mut self, tool: impl ToolDyn + 'static) -> Self {
        self.tools.push(Arc::new(tool));
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
            toolset.add_tool(named_tool(name, "test tool"));
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

        fn definition(&self) -> WasmBoxedFuture<'_, ToolDefinition> {
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

            async fn definition(&self) -> ToolDefinition {
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
