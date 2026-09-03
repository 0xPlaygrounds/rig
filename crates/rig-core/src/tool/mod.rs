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
//! Both erase into [`ErasedTool`], a tool-family handler a bus takes. The
//! registry that collects tools by name, pins them per model turn and
//! dispatches to them (`ToolSet`, `ToolCatalog`), retrieval, managed remote
//! tool sources and the live registry handle live in `rig-agent`, layered
//! over these types.

pub mod builtin;
pub mod context;
pub mod contextual;
pub mod managed;
mod output;
pub mod portable;
mod result;
pub use context::{ContextValue, ToolContext, ToolContextError};
pub use contextual::{DynamicTool, ErasedTool, Tool, ToolEmbedding, tool_definition};
pub use managed::{ManagedToolSink, ManagedToolToken};
pub use output::{IntoToolOutput, ToolOutput};
pub use portable::{
    LivenessFn, PortableDynamicTool, PortableTool, PortableToolEmbedding, portable_tool_definition,
};
pub use result::{ToolErrorKind, ToolExecutionError, ToolResult};
