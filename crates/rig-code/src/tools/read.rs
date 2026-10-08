use rig_core::tool::{PortableTool, ToolExecutionError};
use serde::Deserialize;

use super::{fail, io_fail};

/// The most lines one read returns.
const MAX_LINES: usize = 2000;
/// The most bytes of file text one read returns.
const MAX_BYTES: usize = 50 * 1024;
/// The largest file `read` opens; it reads the whole file to page through it.
const MAX_FILE_BYTES: u64 = 32 * 1024 * 1024;

/// Reads a text file as numbered lines, a window at a time.
pub struct Read;

/// Arguments of [`Read`].
#[derive(Deserialize)]
pub struct ReadArgs {
    path: String,
    offset: Option<usize>,
    limit: Option<usize>,
}

impl PortableTool for Read {
    const NAME: &'static str = "read";
    type Args = ReadArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        format!(
            "Read a text file. Returns numbered lines, at most {MAX_LINES} lines or \
             {} KB per call; use offset (1-based line) and limit to page through \
             longer files.",
            MAX_BYTES / 1024
        )
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path, relative to the working directory."},
                "offset": {"type": "integer", "description": "First line to return, 1-based."},
                "limit": {"type": "integer", "description": "Most lines to return."}
            },
            "required": ["path"]
        })
    }

    async fn call(&self, args: ReadArgs) -> Result<String, ToolExecutionError> {
        // Checked before opening: opening a FIFO or a device can block forever.
        let meta =
            std::fs::metadata(&args.path).map_err(|error| io_fail("read", &args.path, &error))?;
        if !meta.is_file() {
            return Err(fail(format!("{} is not a regular file", args.path)));
        }
        if meta.len() > MAX_FILE_BYTES {
            return Err(fail(format!(
                "{} is {} MB; read opens files up to {} MB",
                args.path,
                meta.len() / (1024 * 1024),
                MAX_FILE_BYTES / (1024 * 1024)
            )));
        }
        let text = std::fs::read_to_string(&args.path)
            .map_err(|error| io_fail("read", &args.path, &error))?;
        let start = args.offset.unwrap_or(1).max(1);
        let limit = args.limit.unwrap_or(MAX_LINES).clamp(1, MAX_LINES);
        let mut out = String::new();
        let mut next = None;
        let mut total = 0;
        for (index, line) in text.lines().enumerate() {
            total = index + 1;
            if total < start || next.is_some() {
                continue;
            }
            if total >= start.saturating_add(limit)
                || (!out.is_empty() && out.len() + line.len() > MAX_BYTES)
            {
                next = Some(total);
                continue;
            }
            // A single line longer than the cap is cut rather than skipped.
            let line: String = line.chars().take(MAX_BYTES).collect();
            out.push_str(&format!("{total:>6}\t{line}\n"));
        }
        if text.is_empty() {
            return Ok(format!("{} is empty.", args.path));
        }
        if start > total {
            return Ok(format!(
                "{} has {total} lines; offset {start} is past the end.",
                args.path
            ));
        }
        if let Some(next) = next {
            out.push_str(&format!(
                "[{} more lines; continue with offset={next}]\n",
                total + 1 - next
            ));
        }
        Ok(out)
    }
}
