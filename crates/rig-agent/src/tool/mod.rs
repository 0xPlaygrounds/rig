//! Tool authoring and canonical structured execution, as plain records.
//!
//! Tools are authored against the portable, context-free contracts owned by
//! [`rig_core::tool`]: implement [`PortableTool`] (one typed `call(args)`
//! method), or build a [`PortableDynamicTool`] record from a closure at
//! runtime. Both erase to the same executable record; the runtime holds those
//! records in a [`ToolExecutor`](crate::executor::ToolExecutor) and advertises
//! their definitions through a
//! [`ToolCatalog`](crate::agent::prepare::ToolCatalog).
//!
//! ```
//! use rig_agent::tool::{PortableTool, ToolExecutionError};
//! use serde::Deserialize;
//!
//! #[derive(Deserialize)]
//! struct AddArgs {
//!     left: i64,
//!     right: i64,
//! }
//!
//! struct Add;
//!
//! impl PortableTool for Add {
//!     const NAME: &'static str = "add";
//!     type Args = AddArgs;
//!     type Output = i64;
//!     type Error = std::convert::Infallible;
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
//!     async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
//!         Ok(args.left + args.right)
//!     }
//! }
//! ```
//!
//! The classic contextual `Tool` trait, `ToolContext`, `ToolSet`, and the
//! `ToolServer` actor were removed: close over your state in a
//! [`PortableDynamicTool`] callback (or your `PortableTool` struct's fields)
//! instead of threading a runtime context. Stored callbacks, state, and tool
//! values must be `Send + Sync` on every target; browser-local state may still
//! be created inside the returned future or retrieved through `thread_local!`.
//! `PortableDynamicTool::new` accepts an ordinary async closure and boxes its
//! future only inside the record's private `Arc<dyn Fn + Send + Sync>` callback
//! field. The record itself is concrete and non-generic.
//! MCP tools live in the `rig-rmcp` crate.

pub mod builtin;
pub mod router_support;

pub use rig_core::tool::{
    IntoToolOutput, PortableDynamicTool, PortableTool, ToolErrorKind, ToolExecutionError,
    ToolOutput, ToolResult, portable_tool_definition, serialize_to_tool_output,
};
