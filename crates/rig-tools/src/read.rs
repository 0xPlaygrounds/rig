//! The `read` tool.

use rig_core::tool::{PortableTool, ToolExecutionError};
use serde::Deserialize;
use serde_json::json;

use crate::fs::read_text;
use crate::{MAX_BYTES, MAX_FILE_BYTES, MAX_LINES, blocking, clip};

/// Characters of one line shown; the rest of a longer line is cut.
const MAX_LINE_CHARS: usize = 2000;

/// Reads a text file as numbered lines.
pub struct Read;

/// Arguments of [`Read`].
#[derive(Deserialize)]
pub struct ReadArgs {
    path: String,
    offset: Option<usize>,
    limit: Option<usize>,
}

impl Read {
    /// When to pick this tool, for the system prompt.
    pub const RULES: &'static [&'static str] = &[
        "Use `read` to look at a file, not `cat`, `head` or `sed` in `shell`.",
        "Read several files at once by calling `read` several times in one reply.",
    ];
}

impl PortableTool for Read {
    const NAME: &'static str = "read";
    type Args = ReadArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        format!(
            "Read a regular text file of at most {} MB. Returns numbered lines, at most \
             {MAX_LINES} lines or {} KB; use offset and limit for longer files.",
            MAX_FILE_BYTES / (1024 * 1024),
            MAX_BYTES / 1024
        )
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file to read."},
                "offset": {"type": "integer", "description": "The first line to read, from 1."},
                "limit": {"type": "integer", "description": "How many lines to read."}
            },
            "required": ["path"]
        })
    }

    async fn call(&self, args: ReadArgs) -> Result<String, ToolExecutionError> {
        blocking(move || read(args)).await
    }
}

fn read(args: ReadArgs) -> Result<String, ToolExecutionError> {
    let text = read_text(&args.path)?;
    let first = args.offset.unwrap_or(1);
    let limit = args.limit.unwrap_or(MAX_LINES).min(MAX_LINES);
    Ok(numbered(&text, first, limit, MAX_BYTES))
}

/// `text`'s lines from line `first` (counted from 1) as [`Read`] shows
/// them: numbered, at most `limit` lines and about `max_bytes` bytes, each
/// line cut at 2000 characters, and a last line saying how many lines
/// were left out and the offset to read on from.
pub fn numbered(text: &str, first: usize, limit: usize, max_bytes: usize) -> String {
    let first = first.max(1);
    let total = text.lines().count();
    let mut out = String::new();
    let mut last = first - 1;
    for (number, line) in text.lines().enumerate().skip(first - 1).take(limit) {
        // A clipped line always fits, so a file of very long lines (minified
        // code, lockfiles) still makes progress.
        let shown = clip(line, MAX_LINE_CHARS);
        if out.len() + shown.len() > max_bytes {
            break;
        }
        out.push_str(&format!("{:>6}\t{shown}", number + 1));
        if shown.len() < line.len() {
            out.push_str(&format!(" [line cut at {MAX_LINE_CHARS} characters]"));
        }
        out.push('\n');
        last = number + 1;
    }
    if last < total {
        out.push_str(&format!(
            "[{} more lines; read on with offset {}]\n",
            total - last,
            last + 1
        ));
    }
    if total == 0 {
        out.push_str("[empty file]\n");
    }
    out
}
