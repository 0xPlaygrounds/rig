//! Portable, context-free tool authoring contracts and canonical values.
//!
//! Runtime-owned context, registries, dispatch, servers, policy, and
//! concurrency live in runtime crates. A portable tool receives only owned,
//! typed arguments, which allows independent runtimes to dispatch it through
//! their own effect boundaries.

pub mod builtin;
mod output;
mod result;

pub use output::{IntoToolOutput, ToolOutput};
pub use result::{ToolErrorKind, ToolExecutionError, ToolResult};

use std::{future::Future, sync::Arc};

use serde::Deserialize;

use crate::{
    completion::ToolDefinition,
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

/// A portable typed LLM tool.
///
/// The contract is deliberately context-free. Runtime-specific inbound state
/// belongs in a runtime-owned contextual-tool adapter, not in `rig-core`.
pub trait Tool: Sized + WasmCompatSend + WasmCompatSync {
    /// Unique registration and provider-facing name.
    const NAME: &'static str;
    /// Typed JSON arguments.
    type Args: for<'de> Deserialize<'de> + WasmCompatSend + WasmCompatSync;
    /// Canonical model-visible output.
    type Output: IntoToolOutput;
    /// Typed author-facing error.
    type Error: std::error::Error + WasmCompatSend + WasmCompatSync + 'static;

    /// Model-facing description.
    fn description(&self) -> String;

    /// JSON Schema for arguments.
    fn parameters(&self) -> serde_json::Value;

    /// Normalize an author-facing error at a runtime dispatch boundary.
    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        ToolExecutionError::from_error(error)
    }

    /// Execute using only owned portable arguments.
    fn call(
        &self,
        args: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + WasmCompatSend;
}

trait DynamicCallback:
    Fn(serde_json::Value) -> WasmBoxedFuture<'static, Result<ToolOutput, ToolExecutionError>>
    + WasmCompatSend
    + WasmCompatSync
{
}

impl<F> DynamicCallback for F where
    F: Fn(serde_json::Value) -> WasmBoxedFuture<'static, Result<ToolOutput, ToolExecutionError>>
        + WasmCompatSend
        + WasmCompatSync
{
}

/// A portable runtime-defined tool backed by an owned-argument callback.
#[derive(Clone)]
pub struct DynamicTool {
    name: String,
    description: String,
    parameters: serde_json::Value,
    callback: Arc<dyn DynamicCallback>,
}

impl DynamicTool {
    /// Create a portable dynamic tool.
    pub fn new<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
        callback: F,
    ) -> Self
    where
        F: Fn(serde_json::Value) -> Fut + WasmCompatSend + WasmCompatSync + 'static,
        Fut: Future<Output = Result<ToolOutput, ToolExecutionError>> + WasmCompatSend + 'static,
    {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            callback: Arc::new(move |args| Box::pin(callback(args))),
        }
    }

    /// Runtime-defined name.
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

    /// Execute with owned JSON arguments.
    pub async fn call(&self, args: serde_json::Value) -> Result<ToolOutput, ToolExecutionError> {
        (self.callback)(args).await
    }
}

/// Generate the provider-facing definition for a portable typed tool.
pub fn tool_definition<T>(tool: &T) -> ToolDefinition
where
    T: Tool,
{
    ToolDefinition {
        name: T::NAME.to_string(),
        description: tool.description(),
        parameters: tool.parameters(),
    }
}
