//! The `read` tool.

use std::path::PathBuf;

use rig_core::tool::{
    FileRead, FilesRead, Tool, ToolContext, ToolExecutionError, args_schema, content_hash,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::fs::{place, read_text};
use crate::{MAX_BYTES, MAX_FILE_BYTES, MAX_LINES, blocking, clip};

/// Characters of one line shown; the rest of a longer line is cut.
const MAX_LINE_CHARS: usize = 2000;

/// Reads a text file as numbered lines. Publishes the file it read
/// ([`FilesRead`]), by its [resolved](crate::fs::resolve) path, with the
/// hash of the bytes it read.
#[derive(Clone, Debug, Default)]
pub struct Read {
    /// The directory a relative path is read from; `None` is this
    /// process's working directory.
    pub base: Option<PathBuf>,
    /// Whether a path that resolves outside `base` is refused, before
    /// anything is read.
    pub confine: bool,
}

/// Arguments of [`Read`].
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadArgs {
    /// The file to read.
    pub path: String,
    /// The first line to read, from 1.
    pub offset: Option<usize>,
    /// How many lines to read.
    pub limit: Option<usize>,
}

impl ReadArgs {
    /// Read `path` from its start.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            offset: None,
            limit: None,
        }
    }

    /// The same read, of at most `limit` lines from line `offset` (from 1).
    pub fn lines(mut self, offset: usize, limit: usize) -> Self {
        self.offset = Some(offset);
        self.limit = Some(limit);
        self
    }

    /// What [`Read`] shows of `text`, the file's whole text, for this read:
    /// its lines from `offset` (default 1), at most `limit` of them (default
    /// and most [`MAX_LINES`]) and [`MAX_BYTES`], [numbered](numbered).
    ///
    /// ```
    /// use rig_tools::ReadArgs;
    ///
    /// let shown = ReadArgs::new("a.txt").lines(2, 1).show("one\ntwo\nthree\n");
    /// assert_eq!(shown, "     2\ttwo\n[1 more lines; read on with offset 3]\n");
    /// ```
    pub fn show(&self, text: &str) -> String {
        let first = self.offset.unwrap_or(1);
        let limit = self.limit.unwrap_or(MAX_LINES).min(MAX_LINES);
        numbered(text, first, limit, MAX_BYTES)
    }
}

impl Read {
    /// When to pick this tool, for the system prompt.
    pub const RULES: &'static [&'static str] = &[
        "Use `read` to look at a file, not `cat`, `head` or `sed` in `shell`.",
        "Read several files at once by calling `read` several times in one reply.",
    ];
}

impl Tool for Read {
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
        args_schema::<ReadArgs>()
    }

    fn annotations(&self) -> rig_core::completion::ToolAnnotations {
        rig_core::completion::ToolAnnotations::default()
            .with_read_only(true)
            .with_open_world(false)
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: ReadArgs,
    ) -> Result<String, ToolExecutionError> {
        let path = place(self.base.as_deref(), self.confine, &args.path)?
            .to_string_lossy()
            .into_owned();
        let (output, hash) = blocking({
            let path = path.clone();
            move || {
                let text = read_text(&path)?;
                Ok::<_, ToolExecutionError>((args.show(&text), content_hash(text.as_bytes())))
            }
        })
        .await?;
        context.insert_result(FilesRead(vec![FileRead::new(path, hash)]))?;
        Ok(output)
    }
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

#[cfg(test)]
mod tests;
