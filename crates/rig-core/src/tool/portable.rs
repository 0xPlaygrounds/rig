//! Context-free tool authoring contracts.
//!
//! Portable tools receive owned, deserialized arguments only. Runtime identity,
//! authorization, mutable context, capability state, and lifecycle metadata
//! remain outside the typed portable call boundary.
//!
//! ```
//! use rig_core::tool::{PortableTool, tool_definition};
//!
//! struct Echo;
//!
//! impl PortableTool for Echo {
//!     const NAME: &'static str = "echo";
//!     type Args = serde_json::Value;
//!     type Output = serde_json::Value;
//!     type Error = std::convert::Infallible;
//!
//!     fn description(&self) -> String {
//!         "Echo JSON".to_string()
//!     }
//!
//!     fn parameters(&self) -> serde_json::Value {
//!         serde_json::json!({"type": "object"})
//!     }
//!
//!     async fn call(&self, arguments: Self::Args) -> Result<Self::Output, Self::Error> {
//!         Ok(arguments)
//!     }
//! }
//!
//! assert_eq!(tool_definition(&Echo).name, "echo");
//! ```

use serde::{Deserialize, Serialize};

use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

use super::{IntoToolOutput, ToolExecutionError};

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

#[cfg(test)]
mod tests;
