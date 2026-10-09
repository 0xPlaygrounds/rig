//! The `write` tool.

use std::path::Path;

use rig_core::tool::{PortableTool, ToolExecutionError, args_schema};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::blocking;
use crate::fs::{io_error, write_atomic};

/// Writes a whole file, creating its parent directories.
pub struct Write;

/// Arguments of [`Write`].
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WriteArgs {
    /// The file to write.
    path: String,
    /// The whole new content.
    content: String,
}

impl Write {
    /// When to pick this tool, for the system prompt.
    pub const RULES: &'static [&'static str] =
        &["Use `write` for new files and complete rewrites only."];
}

impl PortableTool for Write {
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

    async fn call(&self, args: WriteArgs) -> Result<String, ToolExecutionError> {
        blocking(move || {
            let path = args.path.as_str();
            write_atomic(Path::new(path), args.content.as_bytes())
                .map_err(|error| io_error(path, error))?;
            Ok(format!(
                "Wrote {} bytes to {}.",
                args.content.len(),
                args.path
            ))
        })
        .await
    }
}
