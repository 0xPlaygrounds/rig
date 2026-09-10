//! Tool authoring, registration, and canonical structured execution.
//!
//! A typed [`Tool`] implements one [`Tool::call`] method. Rig erases it
//! internally, executes it through one structured path, and exposes a single
//! [`ToolResult`] view to hooks and runtime callers. [`ToolContext`] is the sole
//! path for typed inbound context and host-only result metadata.
//!
//! # Implementing a typed tool
//!
//! Ordinary serializable return values are converted to canonical model output
//! without first passing through a string.
//!
//! ```
//! use rig_agent::tool::{Tool, ToolContext};
//! use serde::{Deserialize, Serialize};
//! use std::convert::Infallible;
//!
//! #[derive(Deserialize)]
//! struct AddArgs {
//!     left: i64,
//!     right: i64,
//! }
//!
//! #[derive(Serialize)]
//! struct Sum {
//!     value: i64,
//! }
//!
//! #[derive(Clone, Debug, PartialEq)]
//! struct AuditRecord(i64);
//!
//! struct Add;
//!
//! impl Tool for Add {
//!     const NAME: &'static str = "add";
//!     type Args = AddArgs;
//!     type Output = Sum;
//!     type Error = Infallible;
//!
//!     fn description(&self) -> String {
//!         "Add two integers".into()
//!     }
//!
//!     fn parameters(&self) -> serde_json::Value {
//!         serde_json::json!({
//!             "type": "object",
//!             "properties": {
//!                 "left": { "type": "integer" },
//!                 "right": { "type": "integer" }
//!             },
//!             "required": ["left", "right"]
//!         })
//!     }
//!
//!     async fn call(
//!         &self,
//!         context: &mut ToolContext,
//!         args: Self::Args,
//!     ) -> Result<Self::Output, Self::Error> {
//!         let value = args.left + args.right;
//!         context.insert_result(AuditRecord(value));
//!         Ok(Sum { value })
//!     }
//! }
//! ```
//!
//! Return [`ToolOutput`] for explicit JSON or multimodal presentation. A
//! [`ToolResultContent`](rig_core::message::ToolResultContent) or a `Vec` of
//! content blocks can also be used directly as a typed tool output without
//! being mistaken for ordinary JSON.
//!
//! ```
//! use rig_core::{
//!     message::{ImageMediaType, ToolResultContent},
//!     tool::ToolOutput,
//! };
//!
//! let output = ToolOutput::one(ToolResultContent::image_base64(
//!     "iVBORw0KGgo=",
//!     Some(ImageMediaType::PNG),
//!     None,
//! ));
//! assert!(matches!(
//!     output.as_content().first(),
//!     Some(ToolResultContent::Image(_))
//! ));
//! ```
//!
//! Explicit [`ToolExecutionError`] constructors keep their detailed message
//! model-visible so validation failures can tell the model how to recover. The
//! default [`Tool::map_error`] conversion preserves an arbitrary source error
//! for operators but exposes only safe kind-level feedback. Override
//! [`Tool::map_error`] or use [`ToolExecutionError::with_model_output`] when a
//! domain error has deliberate structured or actionable model feedback.
//!
//! # Migration from the parallel tool APIs
//!
//! | Removed concept | Canonical replacement |
//! | --- | --- |
//! | Multiple typed `call*` methods | One [`Tool::call`] method |
//! | Public dynamic dispatch traits | [`DynamicTool`] |
//! | Parallel error and failure types | [`ToolExecutionError`] and [`crate::tool::ToolErrorKind`] |
//! | Author-facing outcome enums | Ordinary `Result<T, Self::Error>` normalized at dispatch |
//! | Separate call/result extension maps | [`ToolContext`] |
//! | Parallel string/structured dispatch | [`ToolSet::execute`] and [`server::ToolServerHandle::execute`] |
//!
//! Model-visible output remains typed throughout dispatch. Rendering to text is
//! a terminal provider or telemetry concern; Rig does not reconstruct rich

pub mod builtin;
pub mod server;

pub use rig_core::tool::{
    DynamicTool, ErasedEmbeddingTool, ErasedTool, IntoToolOutput, PortableDynamicTool,
    RegisteredTool, Tool, ToolCatalog, ToolDispatch, ToolEmbedding, ToolErrorKind,
    ToolExecutionError, ToolOutput, ToolResult, ToolSet, dispatch_tool, tool_definition,
};
pub use rig_core::tool::{MissingToolContext, ToolContext};

#[cfg(test)]
mod toolset_clone_tests;

#[cfg(test)]
mod migrated_tests;
