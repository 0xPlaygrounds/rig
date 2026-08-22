//! Portable tool contracts and canonical execution values.
//!
//! The context-free [`PortableTool`] boundary can be adapted by any runtime
//! without importing a registry, mutable context, lifecycle state, or executor.

pub mod builtin;
pub mod context;
pub mod managed;
mod output;
pub mod portable;
mod result;
pub use context::{MissingToolContext, ToolContext};
pub use managed::{ManagedToolSink, ManagedToolToken};
pub use output::{IntoToolOutput, ToolOutput};
pub use portable::{
    PortableDynamicTool, PortableTool, PortableToolEmbedding, portable_tool_definition,
};
pub use result::{ToolErrorKind, ToolExecutionError, ToolResult};
