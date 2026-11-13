//! Tool authoring contracts, dispatch context, and canonical execution values.
//! [`Tool`] is the tool trait; each call gets a [`ToolContext`].
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
mod decode;
mod facts;
pub mod feedback;
mod footprint;
pub mod managed;
mod output;
mod result;
pub mod schema;
pub mod validate;
pub use context::{
    ContextValue, PublishedContext, ToolContext, ToolContextError, ToolResultContext,
};
pub use contextual::{DynamicTool, LivenessFn, Tool, ToolEmbedding, tool_definition, tool_name};
pub use decode::decode_args;
pub use facts::{
    ChangeKind, ContentHash, ExitStatus, FileChange, FileRead, FileState, FilesChanged, FilesRead,
    Hunk, HunkLine, KeptEnd, Lines, LinesShown, OutputCut, content_hash,
};
pub use footprint::{Scope, Touches};
pub use managed::{ManagedToolSink, ManagedToolToken};
pub use output::{IntoToolOutput, ToolOutput};
pub use result::{ToolErrorKind, ToolExecutionError, ToolResult};
pub use schema::args_schema;
