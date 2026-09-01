//! Context-free tool authoring contracts.
//!
//! Portable tools receive owned, deserialized arguments only. Runtime identity,
//! authorization, mutable context, capability state, and lifecycle metadata
//! remain outside this module.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{
    completion::ToolDefinition,
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

use super::{IntoToolOutput, ToolContext, ToolExecutionError, ToolOutput};

/// A context-free typed tool that can be executed by any Rig runtime.
pub trait PortableTool: Sized + WasmCompatSend + WasmCompatSync {
    /// Unique registration and provider-facing name.
    const NAME: &'static str;
    /// Owned JSON arguments.
    type Args: for<'de> Deserialize<'de> + WasmCompatSend + WasmCompatSync;
    /// Canonical model-visible output.
    type Output: IntoToolOutput + WasmCompatSend;
    /// Concrete author-facing failure.
    type Error: std::error::Error + WasmCompatSend + WasmCompatSync + 'static;

    /// Model-facing description.
    fn description(&self) -> String;

    /// JSON Schema for arguments.
    fn parameters(&self) -> serde_json::Value;

    /// Normalize a concrete failure at the runtime effect boundary.
    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        ToolExecutionError::from_error(error)
    }

    /// Execute one owned invocation without runtime access.
    fn call(
        &self,
        arguments: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + WasmCompatSend;
}

/// A portable tool that can be embedded and reconstructed for discovery.
pub trait PortableToolEmbedding: PortableTool {
    /// Failure returned while reconstructing the typed implementation.
    type InitError: std::error::Error + WasmCompatSend + WasmCompatSync + 'static;
    /// Serializable reconstruction data.
    type Context: for<'de> Deserialize<'de> + Serialize;
    /// Runtime initialization state supplied by the authoring integration.
    type State: WasmCompatSend;

    /// Documents used by a discovery implementation.
    fn embedding_docs(&self) -> Vec<String>;
    /// Serializable reconstruction data.
    fn context(&self) -> Self::Context;
    /// Reconstruct the typed implementation.
    fn init(state: Self::State, context: Self::Context) -> Result<Self, Self::InitError>;
}

trait PortableDynamicCallback:
    for<'a> Fn(
        &'a mut ToolContext,
        serde_json::Value,
    ) -> WasmBoxedFuture<'a, Result<ToolOutput, ToolExecutionError>>
    + WasmCompatSend
    + WasmCompatSync
{
}

impl<F> PortableDynamicCallback for F where
    F: for<'a> Fn(
            &'a mut ToolContext,
            serde_json::Value,
        ) -> WasmBoxedFuture<'a, Result<ToolOutput, ToolExecutionError>>
        + WasmCompatSend
        + WasmCompatSync
{
}

/// A runtime-authored context-free tool implementation.
#[derive(Clone)]
pub struct PortableDynamicTool {
    name: String,
    description: String,
    parameters: serde_json::Value,
    callback: Arc<dyn PortableDynamicCallback>,
    /// Optional liveness probe for tools backed by a remote transport; `None`
    /// means always live (the in-process default).
    liveness: Option<Arc<dyn LivenessProbe>>,
}

/// Object-safe liveness probe (a `Fn() -> bool` behind the crate's wasm-aware
/// `Send`/`Sync` markers).
trait LivenessProbe: WasmCompatSend + WasmCompatSync {
    fn is_live(&self) -> bool;
}

impl<F> LivenessProbe for F
where
    F: Fn() -> bool + WasmCompatSend + WasmCompatSync,
{
    fn is_live(&self) -> bool {
        self()
    }
}

impl std::fmt::Debug for PortableDynamicTool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PortableDynamicTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("parameters", &self.parameters)
            .finish_non_exhaustive()
    }
}

impl PortableDynamicTool {
    /// Create a context-free dynamic tool from an owned async callback.
    ///
    /// The callback never sees the per-call [`ToolContext`]; use
    /// [`Self::new_with_context`] when it should.
    pub fn new<F>(
        name: impl Into<String>,
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

    /// Create a dynamic tool whose callback receives the per-call
    /// [`ToolContext`]: typed inbound values the runtime supplies (the model
    /// never sees them) and a result map the tool can publish host-only
    /// metadata into.
    pub fn new_with_context<F>(
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

    /// Attach a liveness probe. Registries use it to retire tools whose remote
    /// backing (an MCP connection, for example) can no longer accept calls,
    /// without probing by execution. In-process tools never need one.
    pub fn with_liveness<F>(mut self, is_live: F) -> Self
    where
        F: Fn() -> bool + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.liveness = Some(Arc::new(is_live));
        self
    }

    /// Whether the tool's backing can still accept calls (`true` unless a
    /// liveness probe says otherwise).
    pub fn is_live(&self) -> bool {
        self.liveness.as_ref().is_none_or(|probe| probe.is_live())
    }

    /// Provider-facing name.
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

    /// Execute the callback with owned arguments and a fresh, empty
    /// [`ToolContext`] (any result metadata the tool publishes is discarded).
    pub async fn execute(
        &self,
        arguments: serde_json::Value,
    ) -> Result<ToolOutput, ToolExecutionError> {
        let mut context = ToolContext::new();
        self.execute_with(&mut context, arguments).await
    }

    /// Execute the callback against the caller's [`ToolContext`]: inbound
    /// values are visible to the tool and its `insert_result`s land on
    /// `context`.
    pub async fn execute_with(
        &self,
        context: &mut ToolContext,
        arguments: serde_json::Value,
    ) -> Result<ToolOutput, ToolExecutionError> {
        (self.callback)(context, arguments).await
    }
}

/// Generate provider-facing metadata for a portable typed tool.
pub fn portable_tool_definition<T>(tool: &T) -> ToolDefinition
where
    T: PortableTool,
{
    ToolDefinition {
        name: T::NAME.to_owned(),
        description: tool.description(),
        parameters: tool.parameters(),
    }
}

#[cfg(test)]
mod tests;
