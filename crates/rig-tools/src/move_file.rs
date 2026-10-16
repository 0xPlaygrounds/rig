//! The `move` tool.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rig_core::completion::ToolAnnotations;
use rig_core::tool::{
    FileChange, FileState, FilesChanged, Scope, Tool, ToolContext, ToolExecutionError, Touches,
    args_schema,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::fs::{Previous, io_error};
use crate::workspace::{Local, Stat, Workspace};
use crate::{MAX_FILE_BYTES, blocking};

/// Moves or renames a file, creating the new path's directory; a directory
/// is refused, and so is an existing file at the new path unless the call
/// says to overwrite it. Publishes the files it changed ([`FilesChanged`]),
/// by their [resolved](Workspace) paths: the old one deleted, and
/// the new one created (or, overwritten, changed) with its hunks.
#[derive(Clone, Debug)]
pub struct Move {
    /// Where the file is moved.
    pub workspace: Arc<dyn Workspace>,
    /// The directory relative paths are resolved from; `None` is the
    /// workspace's working directory.
    pub base: Option<PathBuf>,
    /// Whether a path that resolves outside `base` is refused, before
    /// anything is moved.
    pub confine: bool,
}

impl Default for Move {
    /// Moves this machine's files, from this process's working directory.
    fn default() -> Self {
        Self {
            workspace: Arc::new(Local),
            base: None,
            confine: false,
        }
    }
}

/// Arguments of [`Move`].
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MoveArgs {
    /// The file to move.
    pub from: String,
    /// Its new path.
    pub to: String,
    /// Replace a file already at `to`.
    #[serde(default)]
    pub overwrite: bool,
}

impl MoveArgs {
    /// Move `from` to `to`, refusing to replace a file there.
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            overwrite: false,
        }
    }

    /// The same move, replacing a file at `to`.
    pub fn overwrite(mut self) -> Self {
        self.overwrite = true;
        self
    }
}

impl Tool for Move {
    const NAME: &'static str = "move";
    type Args = MoveArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Move or rename a file, not with `mv` in `shell`, creating the new path's directory. \
         Refuses directories, and an existing file at `to` unless `overwrite` is true."
            .to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<MoveArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
            .with_destructive(true)
            .with_idempotent(false)
            .with_open_world(false)
    }

    fn footprint(&self, args: &MoveArgs) -> Touches {
        let base = self.base.as_deref();
        Touches::Writes(Scope::Paths(vec![
            self.workspace.resolve(base, &args.from),
            self.workspace.resolve(base, &args.to),
        ]))
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: MoveArgs,
    ) -> Result<String, ToolExecutionError> {
        let base = self.base.as_deref();
        let from = self.workspace.place(base, self.confine, &args.from)?;
        let to = self.workspace.place(base, self.confine, &args.to)?;
        let workspace = self.workspace.clone();
        let overwrite = args.overwrite;
        let changes = blocking(move || move_file(&*workspace, &from, &to, overwrite)).await?;
        context.insert_result(FilesChanged(changes))?;
        Ok(format!("Moved {} to {}.", args.from, args.to))
    }
}

/// Moves the file `from` to `to` in `workspace`, and says what that did to
/// both.
fn move_file(
    workspace: &dyn Workspace,
    from: &Path,
    to: &Path,
    overwrite: bool,
) -> Result<Vec<FileChange>, ToolExecutionError> {
    let (shown_from, shown_to) = (from.to_string_lossy(), to.to_string_lossy());
    let size = match workspace
        .stat(from)
        .map_err(|error| io_error(&shown_from, error))?
    {
        Stat::File { size, .. } => size,
        Stat::Missing => {
            return Err(ToolExecutionError::not_found(format!(
                "{shown_from}: no such file"
            )));
        }
        Stat::Dir | Stat::Other => {
            return Err(ToolExecutionError::invalid_args(format!(
                "{shown_from} is not a regular file; `move` moves files only"
            )));
        }
    };
    if from == to {
        return Err(ToolExecutionError::invalid_args(format!(
            "{shown_from} and {shown_to} are the same file"
        )));
    }
    match workspace
        .stat(to)
        .map_err(|error| io_error(&shown_to, error))?
    {
        Stat::Missing => {}
        Stat::File { .. } if overwrite => {}
        Stat::File { .. } => {
            return Err(ToolExecutionError::invalid_args(format!(
                "{shown_to} already exists; set `overwrite` to replace it"
            )));
        }
        Stat::Dir | Stat::Other => {
            return Err(ToolExecutionError::invalid_args(format!(
                "{shown_to} is not a regular file; give the file's new path"
            )));
        }
    }
    // The moved bytes, when the file is small enough to diff and hash.
    let moved = (size <= MAX_FILE_BYTES)
        .then(|| workspace.read(from, MAX_FILE_BYTES).ok())
        .flatten()
        .map(|contents| contents.bytes);
    let before = workspace.previous(to);
    workspace
        .rename(from, to)
        .map_err(|error| io_error(&shown_from, error))?;
    // The moved file's content is in the created half's diff, so the
    // deleted half has none.
    let bytes = moved.as_deref().unwrap_or_default();
    let mut created = Previous::change(before, &shown_to, Some(bytes));
    let read = created.after.take().filter(|_| moved.is_some());
    let state = read.unwrap_or_else(|| FileState::unread(size));
    created.after = Some(state.clone());
    let deleted =
        FileChange::deleted(shown_from, state.lines.unwrap_or_default()).with_before(state);
    Ok(vec![deleted, created])
}

#[cfg(test)]
mod tests;
