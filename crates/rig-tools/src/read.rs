//! The `read` tool.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;

use rig_core::tool::{
    FileRead, FilesRead, KeptEnd, LinesShown, OutputCut, Scope, Tool, ToolContext,
    ToolExecutionError, Touches, args_schema, content_hash,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::workspace::{Local, Workspace};
use crate::{MAX_BYTES, MAX_FILE_BYTES, MAX_LINES, blocking, clip};

/// Characters of one line shown; the rest of a longer line is cut.
const MAX_LINE_CHARS: usize = 2000;

/// Reads a text file's lines as they are. Publishes the file it read
/// ([`FilesRead`]), by its [resolved](Workspace) path, with the
/// hash of the bytes it read and the lines it showed, and, when lines after
/// those were left out, [`OutputCut`].
#[derive(Clone, Debug)]
pub struct Read {
    /// Where the file is read.
    pub workspace: Arc<dyn Workspace>,
    /// The directory a relative path is read from; `None` is the
    /// workspace's working directory.
    pub base: Option<PathBuf>,
    /// Whether a path that resolves outside `base` is refused, before
    /// anything is read.
    pub confine: bool,
}

impl Default for Read {
    /// Reads this machine's files, from this process's working directory.
    fn default() -> Self {
        Self {
            workspace: Arc::new(Local),
            base: None,
            confine: false,
        }
    }
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
    /// and most [`MAX_LINES`]) and [`MAX_BYTES`], as [`show_lines`] shows them.
    ///
    /// ```
    /// use rig_tools::ReadArgs;
    ///
    /// let shown = ReadArgs::new("a.txt").lines(2, 1).show("one\ntwo\nthree\n");
    /// assert_eq!(shown, "two\n[1 more lines; read on with offset 3]\n");
    /// ```
    pub fn show(&self, text: &str) -> String {
        self.shown(text).0
    }

    /// [`Self::show`], and which lines it shows: what [`Read`] publishes
    /// in its [`FileRead`], for code that shows a file as `read` would and
    /// records what it showed.
    ///
    /// ```
    /// use rig_tools::ReadArgs;
    ///
    /// let (_, lines) = ReadArgs::new("a.txt").lines(2, 1).shown("one\ntwo\nthree\n");
    /// assert_eq!((lines.first, lines.count, lines.total), (2, 1, 3));
    /// ```
    pub fn shown(&self, text: &str) -> (String, LinesShown) {
        let first = self.offset.unwrap_or(1);
        let limit = self.limit.unwrap_or(MAX_LINES).min(MAX_LINES);
        lines_from(text, first, limit, MAX_BYTES)
    }
}

impl Tool for Read {
    const NAME: &'static str = "read";
    type Args = ReadArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        format!(
            "Read a regular text file of at most {} MB. Returns its lines as they are, at most \
             {MAX_LINES} lines or {} KB, then the offset to read on from if it stopped early; \
             use offset (from 1) and limit for part of a file, and `rg -n` in `shell` to find a \
             line. Use `read` to look at a file, not `cat`, `head` or `sed` in `shell`, and read \
             several files at once by calling `read` several times in one reply.",
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

    fn footprint(&self, args: &ReadArgs) -> Touches {
        Touches::Reads(Scope::Paths(vec![
            self.workspace.resolve(self.base.as_deref(), &args.path),
        ]))
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: ReadArgs,
    ) -> Result<String, ToolExecutionError> {
        let path = self
            .workspace
            .place(self.base.as_deref(), self.confine, &args.path)?;
        let shown = path.to_string_lossy().into_owned();
        let workspace = self.workspace.clone();
        let (output, read) = blocking(move || {
            let text = workspace.read_text(&path)?;
            let (output, lines) = args.shown(&text);
            let read = FileRead::new(shown, content_hash(text.as_bytes())).with_lines(lines);
            Ok::<_, ToolExecutionError>((output, read))
        })
        .await?;
        let cut = read.lines.and_then(cut_of);
        context.insert_result(FilesRead(vec![read]))?;
        if let Some(cut) = cut {
            context.insert_result(cut)?;
        }
        Ok(output)
    }
}

/// How a read that showed `lines` was cut: from its first line on, the
/// lines after those it showed, which its last line names.
fn cut_of(lines: LinesShown) -> Option<OutputCut> {
    let from_first = lines.total.checked_sub(lines.first.saturating_sub(1))?;
    (lines.count < from_first).then_some(OutputCut {
        kept: KeptEnd::Start,
        kept_lines: lines.count,
        total_lines: from_first,
        spill: None,
    })
}

/// `text`'s lines from line `first` (counted from 1) as [`Read`] shows
/// them: as they are, at most `limit` lines and about `max_bytes` bytes,
/// each line cut at 2000 characters, and a last line saying how many lines
/// were left out and the offset to read on from, or that `first` is past
/// the end.
pub fn show_lines(text: &str, first: usize, limit: usize, max_bytes: usize) -> String {
    lines_from(text, first, limit, max_bytes).0
}

/// [`show_lines`], and which lines it shows.
fn lines_from(text: &str, first: usize, limit: usize, max_bytes: usize) -> (String, LinesShown) {
    let first = first.max(1);
    let total = text.lines().count();
    let mut out = String::new();
    let mut last = first - 1;
    for (index, line) in text.lines().enumerate().skip(first - 1).take(limit) {
        // A clipped line always fits, so a file of very long lines (minified
        // code, lockfiles) still makes progress.
        let shown = clip(line, MAX_LINE_CHARS);
        if out.len() + shown.len() > max_bytes {
            break;
        }
        out.push_str(shown);
        if shown.len() < line.len() {
            let _ = write!(out, " [line cut at {MAX_LINE_CHARS} characters]");
        }
        out.push('\n');
        last = index + 1;
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
    } else if first > total {
        out.push_str(&format!(
            "[offset {first} is past the end: the file has {total} lines]\n"
        ));
    }
    let count = last.saturating_sub(first - 1);
    (
        out,
        LinesShown {
            first,
            count,
            total,
        },
    )
}

#[cfg(test)]
mod tests;
