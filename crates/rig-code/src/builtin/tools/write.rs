//! The `write` tool.

use std::path::Path;

use rig_core::tool::{PortableTool, ToolExecutionError};
use serde::Deserialize;
use serde_json::json;

use super::{io_error, write_atomic};
use crate::core::blocking::blocking;

/// Writes a whole file, creating its parent directories.
pub struct Write;

/// Arguments of [`Write`].
#[derive(Deserialize)]
pub struct WriteArgs {
    path: String,
    content: String,
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
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file to write."},
                "content": {"type": "string", "description": "The whole new content."}
            },
            "required": ["path", "content"]
        })
    }

    async fn call(&self, args: WriteArgs) -> Result<String, ToolExecutionError> {
        blocking(move || {
            if let Some(parent) = Path::new(&args.path).parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent).map_err(|error| io_error(&args.path, error))?;
            }
            write_atomic(&args.path, args.content.as_bytes())?;
            Ok(format!(
                "Wrote {} bytes to {}.",
                args.content.len(),
                args.path
            ))
        })
        .await
    }
}
