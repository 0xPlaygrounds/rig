//! The built-in tools: read, edit, write, shell and search. Each runs its
//! blocking work on a thread of its own, so no task pool thread blocks.

mod blocking;
mod edit;
mod read;
mod search;
mod shell;
mod write;

use bevy_app::prelude::*;
use rig_core::tool::ToolExecutionError;

use crate::core::tools::AppToolsExt;

pub use edit::Edit;
pub use read::Read;
pub use search::Search;
pub use shell::Shell;
pub use write::Write;

/// Most lines a tool returns.
const MAX_LINES: usize = 2000;
/// Most bytes a tool returns.
const MAX_BYTES: usize = 50 * 1024;

/// Registers the built-in tools with [`AppToolsExt::add_tool`].
pub struct BuiltinToolsPlugin;

impl Plugin for BuiltinToolsPlugin {
    fn build(&self, app: &mut App) {
        app.add_tool(Read)
            .add_tool(Edit)
            .add_tool(Write)
            .add_tool(Shell)
            .add_tool(Search);
    }
}

/// `line` cut to its first `limit` characters.
fn clip(line: &str, limit: usize) -> &str {
    match line.char_indices().nth(limit) {
        Some((end, _)) => line.get(..end).unwrap_or(line),
        None => line,
    }
}

/// A model-visible error for an I/O failure on `path`.
fn io_error(path: &str, error: std::io::Error) -> ToolExecutionError {
    let message = format!("{path}: {error}");
    match error.kind() {
        std::io::ErrorKind::NotFound => ToolExecutionError::not_found(message),
        std::io::ErrorKind::PermissionDenied => ToolExecutionError::permission_denied(message),
        _ => ToolExecutionError::other(message),
    }
}
