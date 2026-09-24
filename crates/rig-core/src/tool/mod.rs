//! Tool authoring contracts, dispatch context, and canonical execution values.
//! [`PortableTool`] provides context-free calls; [`Tool`] adds [`ToolContext`].
//! Both can be adapted into [`ErasedTool`] handlers.
//!
//! ```
//! use rig_core::tool::ToolOutput;
//!
//! let output = ToolOutput::json(serde_json::json!({"count": 3}));
//! assert!(output.as_json().is_some());
//! ```

pub mod builtin;
pub mod context;
pub mod contextual;
pub mod managed;
mod output;
pub mod portable;
mod result;
pub use context::{
    ContextValue, PublishedContext, ToolContext, ToolContextError, ToolResultContext,
};
pub use contextual::{DynamicTool, ErasedTool, LivenessFn, Tool, ToolEmbedding, tool_definition};
pub use managed::{ManagedToolSink, ManagedToolToken};
pub use output::{IntoToolOutput, ToolOutput};
pub use portable::{PortableTool, PortableToolEmbedding};
pub use result::{ToolErrorKind, ToolExecutionError, ToolResult};
