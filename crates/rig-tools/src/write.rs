//! The `write` tool.

use std::fs;
use std::path::{Path, PathBuf};

use rig_core::completion::ToolAnnotations;
use rig_core::tool::{
    FilesChanged, Scope, Tool, ToolContext, ToolExecutionError, Touches, args_schema, content_hash,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::fs::{file_change, io_error, place, resolve, write_atomic};
use crate::{MAX_FILE_BYTES, blocking};

/// Writes a whole file, creating its parent directories. Publishes the file
/// it changed ([`FilesChanged`]): its [resolved](crate::fs::resolve) path,
/// and the diff.
#[derive(Clone, Debug, Default)]
pub struct Write {
    /// The directory a relative path is written in; `None` is this
    /// process's working directory.
    pub base: Option<PathBuf>,
    /// Whether a path that resolves outside `base` is refused, before
    /// anything is written.
    pub confine: bool,
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

impl Write {
    /// When to pick this tool, for the system prompt.
    pub const RULES: &'static [&'static str] =
        &["Use `write` for new files and complete rewrites only."];
}

impl Tool for Write {
    const NAME: &'static str = "write";
    type Args = WriteArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Write a file with the given content, replacing it if it exists and creating parent \
         directories. Prefer `edit` for changes to an existing file."
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
        Touches::Writes(Scope::Paths(vec![resolve(
            self.base.as_deref(),
            &args.path,
        )]))
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: WriteArgs,
    ) -> Result<String, ToolExecutionError> {
        let resolved = place(self.base.as_deref(), self.confine, &args.path)?;
        let (output, change) = blocking(move || {
            let path = resolved.to_string_lossy();
            let before = previous(&resolved);
            write_atomic(&resolved, args.content.as_bytes())
                .map_err(|error| io_error(&path, error))?;
            let change = file_change(&path, before.as_deref(), &args.content)
                .with_after(content_hash(args.content.as_bytes()));
            let output = format!("Wrote {} bytes to {}.", args.content.len(), args.path);
            Ok::<_, ToolExecutionError>((output, change))
        })
        .await?;
        context.insert_result(FilesChanged(vec![change]))?;
        Ok(output)
    }
}

/// The text `path` held before it is written: `None` when there was no
/// file; empty when it was too large or not a file to diff.
fn previous(path: &Path) -> Option<String> {
    let meta = fs::metadata(path).ok()?;
    if !meta.is_file() || meta.len() > MAX_FILE_BYTES {
        return Some(String::new());
    }
    let bytes = fs::read(path).unwrap_or_default();
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

#[cfg(test)]
mod tests;
