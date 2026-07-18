//! Portable tool authoring contracts and canonical tool values.
//!
//! Runtime-owned context, registries, dispatch, and servers intentionally live
//! outside this crate. A portable tool receives only its typed arguments, which
//! lets classic and ECS runtimes execute it through owned effect inputs.

pub mod builtin;
mod output;
mod result;

pub use output::{IntoToolOutput, ToolOutput};
pub use result::{ToolErrorKind, ToolExecutionError, ToolResult};

use crate::{
    completion::ToolDefinition,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use futures::Future;
use serde::{Deserialize, Serialize};

/// A portable typed tool that can be executed by any Rig runtime.
pub trait Tool: Sized + WasmCompatSend + WasmCompatSync {
    /// Unique provider-facing name.
    const NAME: &'static str;
    /// Typed JSON arguments.
    type Args: for<'de> Deserialize<'de> + WasmCompatSend + WasmCompatSync;
    /// Canonical model-facing output.
    type Output: IntoToolOutput;
    /// Typed author-facing error.
    type Error: std::error::Error + WasmCompatSend + WasmCompatSync + 'static;

    /// Model-facing description.
    fn description(&self) -> String;

    /// JSON Schema for arguments.
    fn parameters(&self) -> serde_json::Value;

    /// Normalize a tool error for runtime policy and telemetry.
    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        ToolExecutionError::from_error(error)
    }

    /// Execute from owned, canonical arguments without runtime state.
    fn call(
        &self,
        args: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + WasmCompatSend;
}

/// A portable tool that can be stored in a vector store and reconstructed.
pub trait ToolEmbedding: Tool {
    /// Reconstruction error.
    type InitError: std::error::Error + WasmCompatSend + WasmCompatSync + 'static;
    /// Serializable static context.
    type Context: for<'de> Deserialize<'de> + Serialize;
    /// Runtime initialization state.
    type State: WasmCompatSend;

    /// Documents used to retrieve this tool.
    fn embedding_docs(&self) -> Vec<String>;
    /// Serializable tool context.
    fn context(&self) -> Self::Context;
    /// Reconstruct the tool.
    fn init(state: Self::State, context: Self::Context) -> Result<Self, Self::InitError>;
}

/// Generate the provider-facing definition for a portable tool.
pub fn tool_definition<T: Tool>(tool: &T) -> ToolDefinition {
    ToolDefinition {
        name: T::NAME.to_string(),
        description: tool.description(),
        parameters: tool.parameters(),
    }
}
