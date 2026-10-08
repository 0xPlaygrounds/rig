//! Tools as entities, and the built-in coding tools.
//!
//! A tool is an entity holding a [`ToolDef`]: the definition the model reads
//! and the rig handler that runs it. Plugins add tools with
//! [`RigCodeAppExt::add_tool`](crate::RigCodeAppExt::add_tool); the built-ins
//! below are registered the same way by [`ToolsPlugin`].

mod edit;
mod read;
mod search;
mod shell;
mod write;

use bevy_app::{App, Plugin};
use bevy_ecs::prelude::*;
use rig_core::{completion::ToolDefinition, serve::ErasedHandler, tool::ToolExecutionError};

pub use edit::Edit;
pub use read::Read;
pub use search::Search;
pub use shell::Shell;
pub use write::Write;

use crate::RigCodeAppExt as _;

/// A tool the model may call.
#[derive(Component, Debug)]
pub struct ToolDef {
    /// The name, description and argument schema the model reads.
    pub definition: ToolDefinition,
    /// Runs the tool for one call.
    pub handler: ErasedHandler,
}

/// Registers the built-in tools: `read`, `edit`, `write`, `shell`, `search`.
pub struct ToolsPlugin;

impl Plugin for ToolsPlugin {
    fn build(&self, app: &mut App) {
        app.add_tool(Read)
            .add_tool(Edit)
            .add_tool(Write)
            .add_tool(Shell)
            .add_tool(Search);
    }
}

/// A failure the model reads verbatim, so it can correct its next call.
fn fail(message: impl Into<String>) -> ToolExecutionError {
    ToolExecutionError::other(message)
}

/// A failed file operation on `path`, worded for the model.
fn io_fail(action: &str, path: &str, error: &std::io::Error) -> ToolExecutionError {
    fail(format!("cannot {action} {path}: {error}"))
}
