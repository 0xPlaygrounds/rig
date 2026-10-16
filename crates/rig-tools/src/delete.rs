//! The `delete` tool.

use std::path::PathBuf;
use std::sync::Arc;

use rig_core::completion::ToolAnnotations;
use rig_core::tool::{
    FilesChanged, Scope, Tool, ToolContext, ToolExecutionError, Touches, args_schema,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::blocking;
use crate::fs::{Previous, io_error};
use crate::workspace::{Local, Stat, Workspace};

/// Deletes a file; a directory is refused. Publishes the file it deleted
/// ([`FilesChanged`]): its [resolved](Workspace) path, and how many
/// lines it had.
#[derive(Clone, Debug)]
pub struct Delete {
    /// Where the file is deleted.
    pub workspace: Arc<dyn Workspace>,
    /// The directory a relative path is deleted in; `None` is the
    /// workspace's working directory.
    pub base: Option<PathBuf>,
    /// Whether a path that resolves outside `base` is refused, before
    /// anything is deleted.
    pub confine: bool,
}

impl Default for Delete {
    /// Deletes this machine's files, from this process's working directory.
    fn default() -> Self {
        Self {
            workspace: Arc::new(Local),
            base: None,
            confine: false,
        }
    }
}

/// Arguments of [`Delete`].
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DeleteArgs {
    /// The file to delete.
    pub path: String,
}

impl DeleteArgs {
    /// Delete `path`.
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }
}

impl Tool for Delete {
    const NAME: &'static str = "delete";
    type Args = DeleteArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Delete a file, not with `rm` in `shell`. Directories are refused.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<DeleteArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
            .with_destructive(true)
            .with_idempotent(false)
            .with_open_world(false)
    }

    fn footprint(&self, args: &DeleteArgs) -> Touches {
        Touches::Writes(Scope::Paths(vec![
            self.workspace.resolve(self.base.as_deref(), &args.path),
        ]))
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: DeleteArgs,
    ) -> Result<String, ToolExecutionError> {
        let resolved = self
            .workspace
            .place(self.base.as_deref(), self.confine, &args.path)?;
        let workspace = self.workspace.clone();
        let change = blocking(move || {
            let path = resolved.to_string_lossy();
            match workspace
                .stat(&resolved)
                .map_err(|error| io_error(&path, error))?
            {
                Stat::File { .. } => {}
                Stat::Missing => {
                    return Err(ToolExecutionError::not_found(format!(
                        "{path}: no such file"
                    )));
                }
                Stat::Dir | Stat::Other => {
                    return Err(ToolExecutionError::invalid_args(format!(
                        "{path} is not a regular file; `delete` removes files only"
                    )));
                }
            }
            let previous = workspace.previous(&resolved).unwrap_or_default();
            workspace
                .remove(&resolved)
                .map_err(|error| io_error(&path, error))?;
            Ok(Previous::change(Some(previous), &path, None))
        })
        .await?;
        context.insert_result(FilesChanged(vec![change]))?;
        Ok(format!("Deleted {}.", args.path))
    }
}

#[cfg(test)]
mod tests;
