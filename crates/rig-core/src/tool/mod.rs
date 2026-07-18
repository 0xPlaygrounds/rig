//! Portable tool contracts and canonical execution values.
//!
//! The context-free [`portable::PortableTool`] boundary can be adapted by any
//! runtime without importing a registry, mutable context, lifecycle state, or
//! executor. During the classic-runtime extraction, the existing context,
//! registry, server, and dispatch surface remains re-exported from this module
//! through a temporary same-crate bridge. That bridge moves to `rig-agent` as a
//! unit and is not part of the target `rig-core` API.

pub mod builtin;
mod classic;
pub(crate) mod extensions;
mod output;
pub mod portable;
mod result;
pub mod server;

#[cfg(feature = "rmcp")]
#[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
pub mod rmcp;

pub use classic::{DynamicTool, Tool, ToolEmbedding, ToolSet, ToolSetBuilder, tool_definition};
pub(crate) use classic::{ErasedEmbeddingTool, RegisteredTool, ToolDispatch, dispatch_tool};
pub use extensions::{MissingToolContext, ToolContext};
pub use output::{IntoToolOutput, ToolOutput};
pub use portable::{
    PortableDynamicTool, PortableTool, PortableToolEmbedding, portable_tool_definition,
};
pub use result::{ToolErrorKind, ToolExecutionError, ToolResult};
