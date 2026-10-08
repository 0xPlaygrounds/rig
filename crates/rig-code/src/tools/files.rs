//! Reading, writing and editing files.

use std::path::{Path, PathBuf};

use rig_core::tool::{PortableTool, ToolExecutionError, ToolOutput};
use serde::Deserialize;
use serde_json::json;

use super::truncate;

/// Lines `read` returns when the call sets no limit.
const DEFAULT_LINES: usize = 2000;
/// Files larger than this are not read or edited.
const MAX_FILE: u64 = 16 * 1024 * 1024;

fn io_error(path: &Path, error: std::io::Error) -> ToolExecutionError {
    match error.kind() {
        std::io::ErrorKind::NotFound => {
            ToolExecutionError::not_found(format!("{}: no such file", path.display()))
        }
        _ => ToolExecutionError::other(format!("{}: {error}", path.display())),
    }
}

/// The bytes of the regular file `path`, refusing anything else (a device
/// or a pipe could be read forever) and files over [`MAX_FILE`].
fn read_file(path: &Path) -> Result<Vec<u8>, ToolExecutionError> {
    let metadata = std::fs::metadata(path).map_err(|error| io_error(path, error))?;
    if !metadata.is_file() {
        return Err(ToolExecutionError::invalid_args(format!(
            "{} is not a regular file",
            path.display()
        )));
    }
    if metadata.len() > MAX_FILE {
        return Err(ToolExecutionError::invalid_args(format!(
            "{} is {} bytes, over the {MAX_FILE}-byte limit; use the shell tool",
            path.display(),
            metadata.len()
        )));
    }
    std::fs::read(path).map_err(|error| io_error(path, error))
}

/// Reads a text file with line numbers.
pub(super) struct Read;

#[derive(Deserialize)]
pub(super) struct ReadArgs {
    path: PathBuf,
    offset: Option<usize>,
    limit: Option<usize>,
}

impl PortableTool for Read {
    const NAME: &'static str = "read";
    type Args = ReadArgs;
    type Output = ToolOutput;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        format!(
            "Read a text file. Lines are numbered from 1. Returns at most {DEFAULT_LINES} lines \
             unless `limit` says otherwise; use `offset` to continue."
        )
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File to read." },
                "offset": { "type": "integer", "description": "First line to return, from 1." },
                "limit": { "type": "integer", "description": "Most lines to return." }
            },
            "required": ["path"]
        })
    }

    async fn call(&self, args: ReadArgs) -> Result<ToolOutput, ToolExecutionError> {
        let bytes = read_file(&args.path)?;
        let text = String::from_utf8_lossy(&bytes);
        let first = args.offset.unwrap_or(1).max(1);
        let limit = args.limit.unwrap_or(DEFAULT_LINES);
        let total = text.lines().count();
        let mut out = String::new();
        for (number, line) in text.lines().enumerate().skip(first - 1).take(limit) {
            out.push_str(&format!("{:>6}\t{line}\n", number + 1));
        }
        let last = (first - 1).saturating_add(limit);
        if last < total {
            out.push_str(&format!(
                "[{} more lines; continue with offset {}]\n",
                total - last,
                last + 1
            ));
        }
        if out.is_empty() {
            out = format!("[the file has {total} lines]");
        }
        Ok(ToolOutput::text(truncate(&out, false)))
    }
}

/// Writes a whole file, creating its directory.
pub(super) struct Write;

#[derive(Deserialize)]
pub(super) struct WriteArgs {
    path: PathBuf,
    content: String,
}

impl PortableTool for Write {
    const NAME: &'static str = "write";
    type Args = WriteArgs;
    type Output = ToolOutput;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Write a file with the given content, replacing it if it exists.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File to write." },
                "content": { "type": "string", "description": "The whole new content." }
            },
            "required": ["path", "content"]
        })
    }

    async fn call(&self, args: WriteArgs) -> Result<ToolOutput, ToolExecutionError> {
        if let Some(parent) = args
            .path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).map_err(|error| io_error(&args.path, error))?;
        }
        std::fs::write(&args.path, &args.content).map_err(|error| io_error(&args.path, error))?;
        Ok(ToolOutput::text(format!(
            "wrote {} bytes to {}",
            args.content.len(),
            args.path.display()
        )))
    }
}

/// Replaces exact text in a file.
pub(super) struct Edit;

#[derive(Deserialize)]
pub(super) struct EditArgs {
    path: PathBuf,
    old_text: String,
    new_text: String,
    #[serde(default)]
    replace_all: bool,
}

impl PortableTool for Edit {
    const NAME: &'static str = "edit";
    type Args = EditArgs;
    type Output = ToolOutput;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Replace `old_text` with `new_text` in a file. `old_text` must match exactly, \
         whitespace included, and occur once unless `replace_all` is true."
            .to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File to edit." },
                "old_text": { "type": "string", "description": "Exact text to replace." },
                "new_text": { "type": "string", "description": "Replacement text." },
                "replace_all": { "type": "boolean", "description": "Replace every occurrence." }
            },
            "required": ["path", "old_text", "new_text"]
        })
    }

    async fn call(&self, args: EditArgs) -> Result<ToolOutput, ToolExecutionError> {
        if args.old_text.is_empty() || args.old_text == args.new_text {
            return Err(ToolExecutionError::invalid_args(
                "old_text must be non-empty and differ from new_text",
            ));
        }
        let text = String::from_utf8(read_file(&args.path)?).map_err(|_| {
            ToolExecutionError::invalid_args(format!("{} is not UTF-8 text", args.path.display()))
        })?;
        let count = text.matches(&args.old_text).count();
        if count == 0 {
            return Err(ToolExecutionError::invalid_args(format!(
                "old_text was not found in {}; read the file and copy the text exactly",
                args.path.display()
            )));
        }
        if count > 1 && !args.replace_all {
            return Err(ToolExecutionError::invalid_args(format!(
                "old_text occurs {count} times in {}; add context to make it unique or set \
                 replace_all",
                args.path.display()
            )));
        }
        let edited = if args.replace_all {
            text.replace(&args.old_text, &args.new_text)
        } else {
            text.replacen(&args.old_text, &args.new_text, 1)
        };
        std::fs::write(&args.path, edited).map_err(|error| io_error(&args.path, error))?;
        Ok(ToolOutput::text(format!(
            "replaced {count} occurrence{} in {}",
            if count == 1 { "" } else { "s" },
            args.path.display()
        )))
    }
}
