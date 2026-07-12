//! Runtime-neutral code-mode adapter contracts.
//!
//! Language engines (Boa, Python, Lua, WebAssembly, …) implement
//! [`CodeModeRuntime`] in companion crates. Core supplies resource limits,
//! cancellation context, explicit nested-tool allowlists, and a single wrapper
//! tool without selecting a scripting language.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    agent::RunContext,
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

use super::{Tool, ToolCallExtensions, ToolExecutionResult, ToolFailure};

/// Hard limits applied by a code-mode runtime.
#[derive(Debug, Clone)]
pub struct CodeModeLimits {
    /// Maximum source bytes.
    pub max_source_bytes: usize,
    /// Maximum nested tool calls.
    pub max_tool_calls: usize,
    /// Maximum returned bytes.
    pub max_output_bytes: usize,
}

impl Default for CodeModeLimits {
    fn default() -> Self {
        Self {
            max_source_bytes: 64 * 1024,
            max_tool_calls: 64,
            max_output_bytes: 256 * 1024,
        }
    }
}

/// One nested call requested by sandboxed code.
#[derive(Debug, Clone)]
pub struct CodeToolCall {
    /// Allowlisted tool name.
    pub name: String,
    /// Structured arguments.
    pub args: serde_json::Value,
}

/// Scoped dispatcher implemented by the host/agent runtime.
pub trait NestedToolDispatcher: WasmCompatSend + WasmCompatSync {
    /// Dispatch through the host's normal policy, hooks, tracing, and structured
    /// result path.
    fn dispatch<'a>(
        &'a self,
        call: CodeToolCall,
        context: &'a RunContext,
    ) -> WasmBoxedFuture<'a, ToolExecutionResult>;
}

/// Runtime-neutral execution input.
pub struct CodeModeRequest<'a> {
    /// Generated source code.
    pub source: &'a str,
    /// Tools available inside the runtime.
    pub allowed_tools: &'a BTreeSet<String>,
    /// Resource limits.
    pub limits: &'a CodeModeLimits,
    /// Run cancellation/deadline context.
    pub context: &'a RunContext,
    /// Only path for nested tool execution.
    pub dispatcher: &'a dyn NestedToolDispatcher,
}

/// Sandboxed language runtime adapter.
pub trait CodeModeRuntime: WasmCompatSend + WasmCompatSync {
    /// Language identifier shown to the model/host.
    fn language(&self) -> &'static str;

    /// Execute source under the supplied limits and host-function allowlist.
    fn execute<'a>(
        &'a self,
        request: CodeModeRequest<'a>,
    ) -> WasmBoxedFuture<'a, Result<String, CodeModeError>>;
}

/// Code-mode execution failure.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CodeModeError {
    /// Source or output limit was exceeded.
    #[error("code mode resource limit exceeded: {0}")]
    Limit(String),
    /// Run cancellation was observed.
    #[error("code mode cancelled: {0}")]
    Cancelled(String),
    /// Runtime rejected or failed to execute source.
    #[error("code mode runtime error: {0}")]
    Runtime(String),
    /// The run context extension was absent.
    #[error("code mode requires a run context")]
    MissingContext,
}

/// Arguments exposed by the wrapper tool.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CodeModeArgs {
    /// Source code in the configured runtime language.
    pub code: String,
}

/// Single model-facing tool wrapping a selected catalog.
pub struct CodeModeTool<R, D> {
    runtime: R,
    dispatcher: D,
    allowed_tools: BTreeSet<String>,
    limits: CodeModeLimits,
}

impl<R, D> CodeModeTool<R, D> {
    /// Build a wrapper. The wrapper's own `run_code` name is always removed from
    /// the nested allowlist to prevent direct recursion.
    pub fn new(runtime: R, dispatcher: D, allowed_tools: impl IntoIterator<Item = String>) -> Self {
        let mut allowed_tools = allowed_tools.into_iter().collect::<BTreeSet<_>>();
        allowed_tools.remove("run_code");
        Self {
            runtime,
            dispatcher,
            allowed_tools,
            limits: CodeModeLimits::default(),
        }
    }

    /// Override resource limits.
    pub fn limits(mut self, limits: CodeModeLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Selected tools that should be hidden from the outer model catalog and
    /// exposed only through this wrapper.
    pub fn wrapped_tools(&self) -> &BTreeSet<String> {
        &self.allowed_tools
    }
}

impl<R, D> Tool for CodeModeTool<R, D>
where
    R: CodeModeRuntime,
    D: NestedToolDispatcher,
{
    const NAME: &'static str = "run_code";

    type Error = CodeModeError;
    type Args = CodeModeArgs;
    type Output = String;

    fn description(&self) -> String {
        format!(
            "Execute {} code with an explicit allowlist of nested tools",
            self.runtime.language()
        )
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "code": { "type": "string" } },
            "required": ["code"]
        })
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Err(CodeModeError::MissingContext)
    }

    async fn call_with_extensions(
        &self,
        args: Self::Args,
        extensions: &ToolCallExtensions,
    ) -> Result<Self::Output, Self::Error> {
        if args.code.len() > self.limits.max_source_bytes {
            return Err(CodeModeError::Limit("source is too large".into()));
        }
        let context = extensions
            .get::<RunContext>()
            .ok_or(CodeModeError::MissingContext)?;
        if let Some(reason) = context.control().cancellation_reason() {
            return Err(CodeModeError::Cancelled(reason.into()));
        }
        let output = self
            .runtime
            .execute(CodeModeRequest {
                source: &args.code,
                allowed_tools: &self.allowed_tools,
                limits: &self.limits,
                context,
                dispatcher: &self.dispatcher,
            })
            .await?;
        if output.len() > self.limits.max_output_bytes {
            return Err(CodeModeError::Limit("output is too large".into()));
        }
        Ok(output)
    }

    fn classify_error(&self, error: &Self::Error) -> ToolFailure {
        match error {
            CodeModeError::Cancelled(_) => ToolFailure::cancelled(error.to_string()),
            CodeModeError::Limit(_) => {
                ToolFailure::invalid_args(error.to_string()).with_retryable(true)
            }
            CodeModeError::Runtime(_) => ToolFailure::other(error.to_string()).with_retryable(true),
            CodeModeError::MissingContext => ToolFailure::other(error.to_string()),
        }
    }
}
