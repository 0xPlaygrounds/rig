use std::path::Path;

use rig_core::tool::{PortableTool, ToolExecutionError};
use serde::Deserialize;

use super::io_fail;

/// Writes a whole file, creating parent directories.
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
        "Write content to a file, replacing it if it exists and creating missing \
         parent directories."
            .to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path, relative to the working directory."},
                "content": {"type": "string", "description": "The full file content."}
            },
            "required": ["path", "content"]
        })
    }

    async fn call(&self, args: WriteArgs) -> Result<String, ToolExecutionError> {
        if let Some(parent) = Path::new(&args.path).parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .map_err(|error| io_fail("create the directory for", &args.path, &error))?;
        }
        std::fs::write(&args.path, &args.content)
            .map_err(|error| io_fail("write", &args.path, &error))?;
        Ok(format!(
            "Wrote {} bytes to {}.",
            args.content.len(),
            args.path
        ))
    }
}
