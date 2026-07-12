//! Runtime-agnostic code-mode adapter.
//!
//! Core defines resource/cancellation/catalog boundaries; JavaScript, Python,
//! Lua, shell, and WebAssembly engines remain optional adapters.

use std::sync::Arc;

use serde::Deserialize;

use crate::{
    completion::ToolDefinition,
    tool::{ToolCallExtensions, ToolDyn, ToolError, ToolExecutionResult, server::ToolServerHandle},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

/// Limits enforced by a code runtime adapter.
#[derive(Clone, Debug)]
pub struct CodeModeLimits {
    /// Maximum nested tool calls made by one program.
    pub max_tool_calls: usize,
    /// Maximum nested call depth.
    pub max_depth: usize,
    /// Maximum program source bytes.
    pub max_source_bytes: usize,
}

impl Default for CodeModeLimits {
    fn default() -> Self {
        Self {
            max_tool_calls: 64,
            max_depth: 8,
            max_source_bytes: 256 * 1024,
        }
    }
}

/// Host capabilities passed to an installed runtime adapter.
#[derive(Clone)]
pub struct CodeExecutionContext {
    /// Catalog selected for exposure inside the runtime.
    pub catalog: Vec<ToolDefinition>,
    /// Context-aware native/MCP dispatch handle.
    pub tool_server: ToolServerHandle,
    /// Host-only call/run extensions, including [`RunContext`](crate::agent::RunContext).
    pub extensions: ToolCallExtensions,
    /// Resource and recursion limits.
    pub limits: CodeModeLimits,
}

/// Pluggable language/sandbox runtime for [`CodeModeTool`].
pub trait CodeRuntime: WasmCompatSend + WasmCompatSync {
    /// Execute source with an explicit allowlisted host context.
    fn execute<'a>(
        &'a self,
        code: &'a str,
        context: CodeExecutionContext,
    ) -> WasmBoxedFuture<'a, ToolExecutionResult>;
}

#[derive(Deserialize)]
struct CodeModeArgs {
    code: String,
    #[allow(dead_code)]
    description: Option<String>,
}

/// One model-facing `run_code` tool backed by an optional runtime adapter.
///
/// The wrapped catalog is not automatically registered alongside this tool,
/// preventing accidental exposure of both wrapped and unwrapped tools.
#[derive(Clone)]
pub struct CodeModeTool<R> {
    runtime: Arc<R>,
    tool_server: ToolServerHandle,
    catalog: Vec<ToolDefinition>,
    limits: CodeModeLimits,
    name: String,
}

impl<R> CodeModeTool<R>
where
    R: CodeRuntime,
{
    /// Build a code-mode tool over an explicitly selected catalog.
    pub fn new(runtime: R, tool_server: ToolServerHandle, catalog: Vec<ToolDefinition>) -> Self {
        Self {
            runtime: Arc::new(runtime),
            tool_server,
            catalog,
            limits: CodeModeLimits::default(),
            name: "run_code".to_owned(),
        }
    }

    /// Override resource limits.
    pub fn with_limits(mut self, limits: CodeModeLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Override the model-facing wrapper name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

impl<R> ToolDyn for CodeModeTool<R>
where
    R: CodeRuntime + 'static,
{
    fn name(&self) -> String {
        self.name.clone()
    }

    fn description(&self) -> String {
        "Execute a bounded program that can call the selected tool catalog.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "required": ["code"],
            "properties": {
                "code": {"type": "string"},
                "description": {"type": "string"}
            }
        })
    }

    fn metadata(&self) -> crate::completion::ToolMetadata {
        crate::completion::ToolMetadata {
            kind: crate::completion::ToolKind::Composite,
            execution: crate::completion::ToolExecutionPolicy::Sequential,
            source: Some("code_mode".to_owned()),
            attributes: Default::default(),
        }
    }

    fn call<'a>(&'a self, args: String) -> WasmBoxedFuture<'a, Result<String, ToolError>> {
        Box::pin(async move {
            let result = self.call_structured(args, &ToolCallExtensions::EMPTY).await;
            Ok(result.model_output().to_owned())
        })
    }

    fn call_structured<'a>(
        &'a self,
        args: String,
        extensions: &'a ToolCallExtensions,
    ) -> WasmBoxedFuture<'a, ToolExecutionResult> {
        Box::pin(async move {
            let parsed: CodeModeArgs = match serde_json::from_str(&args) {
                Ok(parsed) => parsed,
                Err(error) => {
                    return ToolExecutionResult::failed(
                        format!("invalid code-mode arguments: {error}"),
                        crate::tool::ToolFailure::invalid_args(error.to_string()),
                    );
                }
            };
            if parsed.code.len() > self.limits.max_source_bytes {
                return ToolExecutionResult::failed(
                    "code exceeds configured source limit",
                    crate::tool::ToolFailure::invalid_args("source too large"),
                );
            }
            if extensions
                .get::<crate::agent::RunContext>()
                .is_some_and(crate::agent::RunContext::is_cancelled)
            {
                return ToolExecutionResult::failed(
                    "run cancelled",
                    crate::tool::ToolFailure::cancelled("run cancelled"),
                );
            }
            self.runtime
                .execute(
                    &parsed.code,
                    CodeExecutionContext {
                        catalog: self.catalog.clone(),
                        tool_server: self.tool_server.clone(),
                        extensions: extensions.clone(),
                        limits: self.limits.clone(),
                    },
                )
                .await
        })
    }
}
