//! Portable tool contracts and canonical execution values.
//!
//! The context-free [`PortableTool`] boundary can be adapted by any runtime
//! without importing a registry, mutable context, lifecycle state, or executor.

pub mod builtin;
mod output;
pub mod portable;
mod result;
pub use output::{IntoToolOutput, ToolOutput, serialize_to_tool_output};
pub use portable::{PortableDynamicTool, PortableTool, portable_tool_definition};
pub use result::{ToolErrorKind, ToolExecutionError, ToolResult};
