//! Tool contracts, the erased tool set, and canonical execution values.
//!
//! Two authoring surfaces share one execution path:
//!
//! * the context-free [`PortableTool`] boundary, adaptable by any runtime
//!   without importing a registry, mutable context, lifecycle state, or
//!   executor; and
//! * the contextual [`Tool`] trait (every portable tool is one), whose
//!   [`Tool::call`] also receives the dispatch's [`ToolContext`].
//!
//! Both erase into [`ErasedTool`], collect into an ordered [`ToolSet`], and pin
//! into a [`ToolCatalog`] — definitions plus dispatch by name — that a driver
//! holds for one model turn. Retrieval, managed remote tool sources, and the
//! live registry handle live in `rig-agent`, layered over these types.

pub mod builtin;
pub mod catalog;
pub mod context;
pub mod contextual;
pub mod managed;
mod output;
pub mod portable;
mod result;
pub use catalog::ToolCatalog;
pub use context::{ToolContext, ToolContextError};
pub use contextual::{
    DynamicTool, ErasedTool, RegisteredTool, Tool, ToolDispatch, ToolEmbedding, ToolSet,
    dispatch_tool, tool_definition,
};
pub use managed::{ManagedToolSink, ManagedToolToken};
pub use output::{IntoToolOutput, ToolOutput};
pub use portable::{
    PortableDynamicTool, PortableTool, PortableToolEmbedding, portable_tool_definition,
};
pub use result::{ToolErrorKind, ToolExecutionError, ToolResult};
