//! The `write` tool.

use std::fs;
use std::path::Path;

use rig_core::completion::ToolAnnotations;
use rig_core::tool::{FilesChanged, Tool, ToolContext, ToolExecutionError, args_schema};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::fs::{absolute, file_change, io_error, write_atomic};
use crate::{MAX_FILE_BYTES, blocking};

/// Writes a whole file, creating its parent directories. Publishes the file
/// it changed ([`FilesChanged`]): its absolute path, and the diff.
pub struct Write;

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

    async fn call(
        &self,
        context: &mut ToolContext,
        args: WriteArgs,
    ) -> Result<String, ToolExecutionError> {
        let (output, change) = blocking(move || {
            let path = args.path.as_str();
            let before = previous(Path::new(path));
            write_atomic(Path::new(path), args.content.as_bytes())
                .map_err(|error| io_error(path, error))?;
            let change = file_change(&absolute(path), before.as_deref(), &args.content);
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
