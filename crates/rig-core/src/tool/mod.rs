//! Portable tool contracts and canonical execution values.
//!
//! The context-free [`Tool`] boundary can be adapted by any runtime without
//! importing a registry, mutable context, lifecycle state, or executor.

pub mod builtin;
mod output;
pub mod portable;
mod result;
pub use output::{IntoToolOutput, ToolOutput};
pub use portable::{
    PortableDynamicTool, PortableDynamicTool as DynamicTool, PortableTool, PortableTool as Tool,
    PortableToolEmbedding, PortableToolEmbedding as ToolEmbedding, portable_tool_definition,
    portable_tool_definition as tool_definition,
};
pub use result::{ToolErrorKind, ToolExecutionError, ToolResult};
