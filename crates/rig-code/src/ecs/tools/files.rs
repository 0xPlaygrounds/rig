//! The `read`, `write` and `edit` tools.

use serde::Deserialize;
use serde_json::json;

use rig_core::tool::{Tool, ToolContext, ToolExecutionError};

use super::{io_error, resolve};

/// Most lines `read` returns at once.
const READ_LINES: usize = 2000;
/// Most bytes `read` returns at once.
const READ_BYTES: usize = 50 * 1024;

/// Reads a text file with line numbers.
pub struct Read;

#[derive(Deserialize)]
pub struct ReadArgs {
    path: String,
    offset: Option<usize>,
    limit: Option<usize>,
}

impl Tool for Read {
    const NAME: &'static str = "read";
    type Args = ReadArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        format!(
            "Read a text file. Lines are numbered from 1. Returns at most {READ_LINES} lines or \
             {} KB; use offset and limit to read more.",
            READ_BYTES / 1024
        )
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path, relative to the working directory or absolute"},
                "offset": {"type": "integer", "description": "First line to read, from 1"},
                "limit": {"type": "integer", "description": "Most lines to read"}
            },
            "required": ["path"]
        })
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: ReadArgs,
    ) -> Result<String, ToolExecutionError> {
        let path = resolve(context, &args.path);
        let text = std::fs::read_to_string(&path).map_err(|error| io_error(&path, error))?;
        let first = args.offset.unwrap_or(1).max(1);
        let limit = args.limit.unwrap_or(READ_LINES).min(READ_LINES);
        let total = text.lines().count();
        let mut out = String::new();
        let mut last = first.saturating_sub(1);
        for (number, line) in text.lines().enumerate().skip(first - 1).take(limit) {
            let numbered = format!("{:>6}\t{line}\n", number + 1);
            if out.len() + numbered.len() > READ_BYTES {
                break;
            }
            out.push_str(&numbered);
            last = number + 1;
        }
        if last < total {
            out.push_str(&format!(
                "[showing lines {first}-{last} of {total}; continue with offset {}]\n",
                last + 1
            ));
        }
        Ok(out)
    }
}

/// Writes a whole file.
pub struct Write;

#[derive(Deserialize)]
pub struct WriteArgs {
    path: String,
    content: String,
}

impl Tool for Write {
    const NAME: &'static str = "write";
    type Args = WriteArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Create or overwrite a file with the given content, creating parent directories.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path, relative to the working directory or absolute"},
                "content": {"type": "string", "description": "The whole new content"}
            },
            "required": ["path", "content"]
        })
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: WriteArgs,
    ) -> Result<String, ToolExecutionError> {
        let path = resolve(context, &args.path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| io_error(parent, error))?;
        }
        std::fs::write(&path, &args.content).map_err(|error| io_error(&path, error))?;
        Ok(format!(
            "Wrote {} bytes to {}",
            args.content.len(),
            path.display()
        ))
    }
}

/// Replaces one exact occurrence of a text in a file.
pub struct Edit;

#[derive(Deserialize)]
pub struct EditArgs {
    path: String,
    old_text: String,
    new_text: String,
}

impl Tool for Edit {
    const NAME: &'static str = "edit";
    type Args = EditArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Replace old_text with new_text in a file. old_text must match exactly, whitespace \
         included, and occur exactly once; include surrounding lines to make it unique."
            .to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path, relative to the working directory or absolute"},
                "old_text": {"type": "string", "description": "The exact text to replace"},
                "new_text": {"type": "string", "description": "The replacement"}
            },
            "required": ["path", "old_text", "new_text"]
        })
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: EditArgs,
    ) -> Result<String, ToolExecutionError> {
        let path = resolve(context, &args.path);
        let text = std::fs::read_to_string(&path).map_err(|error| io_error(&path, error))?;
        if args.old_text.is_empty() {
            return Err(ToolExecutionError::invalid_args("old_text is empty"));
        }
        match text.matches(&args.old_text).count() {
            1 => {}
            0 => {
                return Err(ToolExecutionError::invalid_args(format!(
                    "old_text was not found in {}",
                    path.display()
                )));
            }
            count => {
                return Err(ToolExecutionError::invalid_args(format!(
                    "old_text occurs {count} times in {}; include more context to make it unique",
                    path.display()
                )));
            }
        }
        let edited = text.replacen(&args.old_text, &args.new_text, 1);
        std::fs::write(&path, edited).map_err(|error| io_error(&path, error))?;
        Ok(format!("Edited {}", path.display()))
    }
}
