//! The `write` tool.

use std::path::PathBuf;
use std::sync::Arc;

use rig_core::completion::ToolAnnotations;
use rig_core::tool::{
    ChangeKind, FileChange, FilesChanged, Scope, Tool, ToolContext, ToolExecutionError, Touches,
    args_schema,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::blocking;
use crate::fs::{Previous, io_error};
use crate::workspace::{Local, Workspace};

/// Writes a whole file, creating its parent directories. Publishes the file
/// it changed ([`FilesChanged`]): its [resolved](Workspace) path,
/// and the diff.
#[derive(Clone, Debug)]
pub struct Write {
    /// Where the file is written.
    pub workspace: Arc<dyn Workspace>,
    /// The directory a relative path is written in; `None` is the
    /// workspace's working directory.
    pub base: Option<PathBuf>,
    /// Whether a path that resolves outside `base` is refused, before
    /// anything is written.
    pub confine: bool,
}

impl Default for Write {
    /// Writes this machine's files, from this process's working directory.
    fn default() -> Self {
        Self {
            workspace: Arc::new(Local),
            base: None,
            confine: false,
        }
    }
}

/// Arguments of [`Write`].
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WriteArgs {
    /// The file to write.
    pub path: String,
    /// The whole new content.
    pub content: String,
}

impl WriteArgs {
    /// Write `content` to `path`.
    pub fn new(path: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            content: content.into(),
        }
    }
}

impl Tool for Write {
    const NAME: &'static str = "write";
    type Args = WriteArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Write a file with the given content, replacing it if it exists and creating parent \
         directories. Use it for new files and complete rewrites only; change part of a file \
         with `edit`."
            .to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<WriteArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
            .with_destructive(true)
            .with_idempotent(true)
            .with_open_world(false)
    }

    fn footprint(&self, args: &WriteArgs) -> Touches {
        Touches::Writes(Scope::Paths(vec![
            self.workspace.resolve(self.base.as_deref(), &args.path),
        ]))
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: WriteArgs,
    ) -> Result<String, ToolExecutionError> {
        let resolved = self
            .workspace
            .place(self.base.as_deref(), self.confine, &args.path)?;
        let workspace = self.workspace.clone();
        let (output, change) = blocking(move || {
            let path = resolved.to_string_lossy();
            let before = workspace.previous(&resolved);
            workspace
                .write(&resolved, args.content.as_bytes())
                .map_err(|error| io_error(&path, error))?;
            let change = Previous::change(before, &path, Some(args.content.as_bytes()));
            Ok::<_, ToolExecutionError>((wrote(&args.path, &change), change))
        })
        .await?;
        context.insert_result(FilesChanged(vec![change]))?;
        Ok(output)
    }
}

/// The reply to a write of `path` that made `change`: that it created the
/// file, or replaced it and, when it was read, how many lines it had.
fn wrote(path: &str, change: &FileChange) -> String {
    let had = change.before.as_ref().and_then(|before| before.lines);
    match (change.kind, had) {
        (ChangeKind::Created, _) => format!("Created {path}."),
        (_, Some(1)) => format!("Replaced {path}, which had 1 line."),
        (_, Some(lines)) => format!("Replaced {path}, which had {lines} lines."),
        (_, None) => format!("Replaced {path}."),
    }
}

#[cfg(test)]
mod tests;
