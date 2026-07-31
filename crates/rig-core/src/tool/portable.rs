//! Context-free tool authoring contracts.
//!
//! Portable tools receive owned, deserialized arguments only. Runtime identity,
//! authorization, mutable context, capability state, and lifecycle metadata
//! remain outside this module.

use std::sync::Arc;

use serde::Deserialize;

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

    /// Erase a typed [`PortableTool`] into a dynamic record.
    ///
    /// The record mirrors the typed dispatch semantics exactly: arguments are
    /// deserialized into `T::Args` (with a `null` → `{}` fallback for
    /// argument structs whose fields are all optional — what models send when
    /// no arguments are provided), output is normalized through the concrete
    /// [`IntoToolOutput`] implementation, and typed errors are normalized
    /// through [`PortableTool::map_error`].
    pub fn from_portable<T>(tool: T) -> Self
    where
        T: PortableTool + 'static,
    {
        let definition = portable_tool_definition(&tool);
        let tool = Arc::new(tool);
        Self::new(
            definition.name,
            definition.description,
            definition.parameters,
            move |arguments| {
                let tool = Arc::clone(&tool);
                Box::pin(async move {
                    let args = parse_portable_args::<T::Args>(arguments)?;
                    match tool.call(args).await {
                        Ok(output) => output.into_tool_output(),
                        Err(error) => Err(tool.map_error(error)),
                    }
                })
            },
        )
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

/// Parse model-emitted JSON arguments for a typed portable tool, with the
/// classic `null` → `{}` fallback for all-optional argument structs.
fn parse_portable_args<A>(arguments: serde_json::Value) -> Result<A, ToolExecutionError>
where
    A: for<'de> Deserialize<'de>,
{
    let was_null = arguments.is_null();
    // Parse from the serialized text (not `from_value`) so parse failures
    // carry the classic `at line N column M` positions — the error text is
    // model-visible and recorded in replay cassettes.
    let raw = arguments.to_string();
    match serde_json::from_str(&raw) {
        Ok(parsed) => Ok(parsed),
        Err(original) if was_null => serde_json::from_str("{}").map_err(|_| {
            ToolExecutionError::invalid_args(format!("failed to parse tool arguments: {original}"))
                .with_source(original)
        }),
        Err(error) => Err(ToolExecutionError::invalid_args(format!(
            "failed to parse tool arguments: {error}"
        ))
        .with_source(error)),
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

    impl IntoToolOutput for Sum {
        fn into_tool_output(self) -> Result<ToolOutput, ToolExecutionError> {
            crate::tool::serialize_to_tool_output(&self)
        }
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
