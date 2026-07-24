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

use super::{IntoToolOutput, ToolExecutionError, ToolOutput};

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
    Fn(serde_json::Value) -> WasmBoxedFuture<'static, Result<ToolOutput, ToolExecutionError>>
    + WasmCompatSend
    + WasmCompatSync
{
}

impl<F> PortableDynamicCallback for F where
    F: Fn(serde_json::Value) -> WasmBoxedFuture<'static, Result<ToolOutput, ToolExecutionError>>
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
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            callback: Arc::new(callback),
        }
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

    /// Execute the callback with owned arguments.
    pub async fn execute(
        &self,
        arguments: serde_json::Value,
    ) -> Result<ToolOutput, ToolExecutionError> {
        (self.callback)(arguments).await
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
mod tests {
    use std::convert::Infallible;

    use serde::{Deserialize, Serialize};

    use super::*;

    #[derive(Deserialize)]
    struct AddArgs {
        left: i64,
        right: i64,
    }

    #[derive(Serialize)]
    struct Sum {
        value: i64,
    }

    struct Add;

    impl PortableTool for Add {
        const NAME: &'static str = "add";
        type Args = AddArgs;
        type Output = Sum;
        type Error = Infallible;

        fn description(&self) -> String {
            "Add two integers".to_string()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        async fn call(&self, arguments: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok(Sum {
                value: arguments.left + arguments.right,
            })
        }
    }

    #[tokio::test]
    async fn portable_tools_execute_without_runtime_context() {
        let output = Add.call(AddArgs { left: 2, right: 3 }).await;
        let Ok(output) = output;
        assert_eq!(output.value, 5);
        assert_eq!(portable_tool_definition(&Add).name, "add");
    }

    #[tokio::test]
    async fn portable_dynamic_tools_receive_owned_arguments() {
        let tool = PortableDynamicTool::new(
            "echo",
            "Echo a JSON value",
            serde_json::json!({"type": "object"}),
            |arguments| Box::pin(async move { Ok(ToolOutput::json(arguments)) }),
        );

        let arguments = serde_json::json!({"value": "hello"});
        let output = tool.execute(arguments.clone()).await;
        assert!(output.is_ok());
        let Ok(output) = output else {
            return;
        };
        assert_eq!(output.as_json(), Some(&arguments));
    }
}
