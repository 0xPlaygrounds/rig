//! The `edit` tool.

use rig_core::tool::{PortableTool, ToolExecutionError};
use serde::Deserialize;
use serde_json::json;

use super::blocking::blocking;
use super::io_error;

/// Replaces exact text in a file.
pub struct Edit;

/// Arguments of [`Edit`].
#[derive(Deserialize)]
pub struct EditArgs {
    path: String,
    old_text: String,
    new_text: String,
    #[serde(default)]
    replace_all: bool,
}

impl PortableTool for Edit {
    const NAME: &'static str = "edit";
    type Args = EditArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Replace exact text in a file. old_text must match the file exactly, whitespace \
         included, and only once unless replace_all is true."
            .to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file to edit."},
                "old_text": {"type": "string", "description": "The exact text to replace."},
                "new_text": {"type": "string", "description": "The replacement."},
                "replace_all": {"type": "boolean", "description": "Replace every match."}
            },
            "required": ["path", "old_text", "new_text"]
        })
    }

    async fn call(&self, args: EditArgs) -> Result<String, ToolExecutionError> {
        blocking(move || edit(args)).await
    }
}

fn edit(args: EditArgs) -> Result<String, ToolExecutionError> {
    if args.old_text.is_empty() {
        return Err(ToolExecutionError::invalid_args("old_text is empty"));
    }
    let text = std::fs::read_to_string(&args.path).map_err(|error| io_error(&args.path, error))?;
    let matches = text.matches(&args.old_text).count();
    if matches == 0 {
        return Err(ToolExecutionError::invalid_args(format!(
            "old_text was not found in {}",
            args.path
        )));
    }
    if matches > 1 && !args.replace_all {
        return Err(ToolExecutionError::invalid_args(format!(
            "old_text matches {matches} times in {}; add context to make it unique or set \
             replace_all",
            args.path
        )));
    }
    let edited = if args.replace_all {
        text.replace(&args.old_text, &args.new_text)
    } else {
        text.replacen(&args.old_text, &args.new_text, 1)
    };
    std::fs::write(&args.path, edited).map_err(|error| io_error(&args.path, error))?;
    Ok(format!(
        "Edited {} ({matches} replacement{}).",
        args.path,
        if matches == 1 { "" } else { "s" }
    ))
}
