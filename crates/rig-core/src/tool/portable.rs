//! Context-free tool authoring contracts.
//!
//! Portable tools receive owned, deserialized arguments only. Runtime identity,
//! authorization, mutable context, capability state, and lifecycle metadata
//! remain outside this module.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{
    completion::ToolDefinition,
    effect::{EffectKind, Outcome},
    serve::{ErasedHandler, adapters::ToolCallback, adapters::ToolFn},
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

/// A liveness probe: whether the tool's owner still serves it. A plain
/// function, not a behaviour trait — it is exempt from the one-erasure
/// rule the way `dyn Fn` is everywhere else.
///
/// Deregistration is lazy: a registry consults the probe on its next read
/// (a snapshot for a request, a managed reconcile), not when the owner's
/// transport closes. A dispatch that reaches the tool in that window fails
/// at the transport, as the tool's own error, not as `HandlerUnavailable`.
#[cfg(not(target_family = "wasm"))]
pub type LivenessFn = Arc<dyn Fn() -> bool + Send + Sync>;
/// A liveness probe (browser wasm: no `Send + Sync`, no threads).
#[cfg(target_family = "wasm")]
pub type LivenessFn = Arc<dyn Fn() -> bool>;

/// A tool defined at runtime by a callback, portable across hosts: the
/// definition plus the erased handler (the callback is the handler). An
/// optional liveness probe lets a registry retire it when its owner (an MCP
/// transport) goes away — the probe is deregistration, not a second
/// execution path.
#[derive(Clone)]
pub struct PortableDynamicTool {
    definition: ToolDefinition,
    handler: ErasedHandler,
    liveness: Option<LivenessFn>,
}

impl std::fmt::Debug for PortableDynamicTool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PortableDynamicTool")
            .field("name", &self.definition.name)
            .field("description", &self.definition.description)
            .field("parameters", &self.definition.parameters)
            .finish_non_exhaustive()
    }
}

impl PortableDynamicTool {
    /// Define a tool from a context-free callback.
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

    /// Define a tool from a callback over the dispatch-scoped context.
    pub fn new_with_context<F>(
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

    /// Attach a liveness probe.
    pub fn with_liveness<F>(mut self, is_live: F) -> Self
    where
        F: Fn() -> bool + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.liveness = Some(Arc::new(is_live));
        self
    }

    /// Whether the tool's owner still serves it (`true` without a probe).
    pub fn is_live(&self) -> bool {
        self.liveness.as_ref().is_none_or(|probe| probe())
    }

    /// The tool's name.
    pub fn name(&self) -> &str {
        &self.definition.name
    }

    /// The tool's definition.
    pub fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    /// The erased handler.
    pub fn handler(&self) -> &ErasedHandler {
        &self.handler
    }

    /// The definition, the handler and the liveness probe, by value — what
    /// a registry stages from a portable tool.
    pub fn into_parts(self) -> (ToolDefinition, ErasedHandler, Option<LivenessFn>) {
        (self.definition, self.handler, self.liveness)
    }

    /// Run the tool inline with an empty context.
    pub async fn execute(
        &self,
        arguments: serde_json::Value,
    ) -> Result<ToolOutput, ToolExecutionError> {
        let mut context = ToolContext::new();
        self.execute_with(&mut context, arguments).await
    }

    /// Run the tool inline, publishing into `context`.
    pub async fn execute_with(
        &self,
        context: &mut ToolContext,
        arguments: serde_json::Value,
    ) -> Result<ToolOutput, ToolExecutionError> {
        let published = crate::tool::PublishedContext::new();
        let outcome = crate::serve::serve_inline_with(
            &self.handler,
            EffectKind::ToolCall {
                name: self.definition.name.clone(),
                args: arguments.to_string(),
            },
            vec![
                std::sync::Arc::new(std::mem::take(context)),
                published.clone() as std::sync::Arc<dyn std::any::Any + Send + Sync>,
            ],
        )
        .await;
        match outcome {
            Ok(Outcome::ToolResult { result }) => {
                *context = published.take().unwrap_or_default();
                result.into_result()
            }
            Ok(other) => Err(ToolExecutionError::other(format!(
                "tool handler answered with a {} outcome",
                other.family()
            ))),
            Err(report) => Err(ToolExecutionError::other(report.message)),
        }
    }
}

/// A tool's [`ToolDefinition`] from a typed portable tool.
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
