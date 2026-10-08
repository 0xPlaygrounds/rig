use rig_core::tool::{PortableTool, ToolExecutionError};
use serde::Deserialize;

use super::{fail, io_fail};

/// Replaces one exact, unique piece of a file.
pub struct Edit;

/// Arguments of [`Edit`].
#[derive(Deserialize)]
pub struct EditArgs {
    path: String,
    old_text: String,
    new_text: String,
}

impl PortableTool for Edit {
    const NAME: &'static str = "edit";
    type Args = EditArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Replace old_text with new_text in a file. old_text must match the file \
         exactly, whitespace included, and occur exactly once; include surrounding \
         lines to make it unique."
            .to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path, relative to the working directory."},
                "old_text": {"type": "string", "description": "The exact text to replace."},
                "new_text": {"type": "string", "description": "The replacement text."}
            },
            "required": ["path", "old_text", "new_text"]
        })
    }

    async fn call(&self, args: EditArgs) -> Result<String, ToolExecutionError> {
        if args.old_text.is_empty() {
            return Err(fail("old_text is empty; use write to create a file."));
        }
        let text = std::fs::read_to_string(&args.path)
            .map_err(|error| io_fail("read", &args.path, &error))?;
        match text.matches(&args.old_text).count() {
            0 => Err(fail(format!(
                "old_text was not found in {}. Read the file and copy the text exactly.",
                args.path
            ))),
            1 => {
                let edited = text.replacen(&args.old_text, &args.new_text, 1);
                std::fs::write(&args.path, edited)
                    .map_err(|error| io_fail("write", &args.path, &error))?;
                Ok(format!("Edited {}.", args.path))
            }
            count => Err(fail(format!(
                "old_text occurs {count} times in {}. Include more surrounding lines \
                 so it occurs once.",
                args.path
            ))),
        }
    }
}
