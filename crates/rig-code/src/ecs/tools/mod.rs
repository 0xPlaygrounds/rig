//! Tools are entities. A plugin adds one with
//! [`RigAppExt::add_tool`](super::RigAppExt::add_tool), which spawns a
//! [`RegisteredTool`] holding the tool's definition and its handler.
//! [`ToolsPlugin`] adds the built-in `read`, `write`, `edit`, `bash` and
//! `search` that way.

use std::path::{Path, PathBuf};

use bevy::prelude::*;
use rig_core::{
    completion::ToolDefinition,
    serve::{ErasedHandler, adapters::ToolAdapter},
    tool::{ToolContext, ToolExecutionError, tool_definition},
};

use super::{RigAppExt, agent::Workdir};

mod bash;
mod files;
mod search;

/// A tool agents can call: what the model is told, and the handler that
/// runs it.
#[derive(Component, Clone, Debug)]
pub struct RegisteredTool {
    /// The definition sent to the model.
    pub definition: ToolDefinition,
    /// Runs a call through the one dispatch path.
    pub handler: ErasedHandler,
}

/// Spawn the entity of `tool`.
pub(super) fn register<T: rig_core::tool::Tool + 'static>(app: &mut App, tool: T) {
    let definition = tool_definition(&tool);
    let handler = ErasedHandler::new(ToolAdapter::new(tool));
    app.world_mut().spawn((
        Name::new(format!("tool {}", definition.name.as_str())),
        RegisteredTool {
            definition,
            handler,
        },
    ));
}

/// The built-in tools.
pub struct ToolsPlugin;

impl Plugin for ToolsPlugin {
    fn build(&self, app: &mut App) {
        app.add_tool(files::Read)
            .add_tool(files::Write)
            .add_tool(files::Edit)
            .add_tool(bash::Bash)
            .add_tool(search::Search);
    }
}

/// `path` resolved against the calling agent's working directory, which the
/// dispatch carries as a scope.
fn resolve(context: &ToolContext, path: &str) -> PathBuf {
    let path = Path::new(path);
    match context.scope::<Workdir>() {
        Some(workdir) => workdir.0.join(path),
        None => path.to_path_buf(),
    }
}

/// An IO failure on `path` as the model sees it.
fn io_error(path: &Path, error: std::io::Error) -> ToolExecutionError {
    let message = format!("{}: {error}", path.display());
    match error.kind() {
        std::io::ErrorKind::NotFound => ToolExecutionError::not_found(message),
        std::io::ErrorKind::PermissionDenied => ToolExecutionError::permission_denied(message),
        _ => ToolExecutionError::other(message),
    }
}

/// The last `limit` bytes of `text`, cut at a character boundary.
fn tail(text: &str, limit: usize) -> &str {
    let mut start = text.len().saturating_sub(limit);
    while !text.is_char_boundary(start) {
        start += 1;
    }
    text.get(start..).unwrap_or_default()
}
