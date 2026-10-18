//! Facts a tool call publishes about what it did, beside the text the model
//! reads: the files it read ([`FilesRead`]), the files it changed
//! ([`FilesChanged`]), how a process it ran ended ([`ExitStatus`]) and
//! whether its output was cut to fit ([`OutputCut`]). A
//! file's [`ContentHash`] says which content the tool saw or left, so a
//! host can tell whether a file changed since without reading it again; a
//! change's [`FileState`]s also give each side's size and lines.
//! They are [`ContextValue`]s a tool publishes with
//! [`ToolContext::insert_result`](super::ToolContext::insert_result), so
//! they reach the host in the call's [`ToolResultContext`](super::ToolResultContext)
//! and the effect log, whatever the tool: a built-in one, a plugin's, or an
//! MCP server's adapter. A host reads them back with
//! [`ToolResultContext::get`](super::ToolResultContext::get).
//!
//! ```
//! use rig_core::tool::{
//!     ExitStatus, FileChange, FileRead, FilesChanged, FilesRead, ToolContext, content_hash,
//! };
//!
//! let mut context = ToolContext::new();
//! let readme = FileRead::new("/work/README.md", content_hash(b"# Work\n"));
//! context.insert_result(FilesRead(vec![readme]))?;
//! context.insert_result(FilesChanged(vec![FileChange::modified("/work/src/lib.rs", 3, 1)]))?;
//! context.insert_result(ExitStatus::Exited(0))?;
//! let published = context.result_context();
//! assert_eq!(published.get::<ExitStatus>()?.map(|status| status.success()), Some(true));
//! let changed = published.get::<FilesChanged>()?.unwrap_or_default();
//! assert_eq!(changed.paths().collect::<Vec<_>>(), ["/work/src/lib.rs"]);
//! let read = published.get::<FilesRead>()?.unwrap_or_default();
//! assert_eq!(read.paths().collect::<Vec<_>>(), ["/work/README.md"]);
//! assert_eq!(read.0.first().map(|file| &file.hash), Some(&content_hash(b"# Work\n")));
//! # Ok::<(), rig_core::tool::ToolContextError>(())
//! ```

use std::fmt::{self, Write as _};

use serde::{Deserialize, Serialize};

use super::ContextValue;

/// The SHA-256 of a file's bytes, as lowercase hex: which content a tool
/// read or left ([`content_hash`]).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ContentHash(pub String);

/// The [`ContentHash`] of `bytes`.
///
/// ```
/// use rig_core::tool::content_hash;
///
/// assert_eq!(
///     content_hash(b"").0,
///     "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
/// );
/// ```
pub fn content_hash(bytes: &[u8]) -> ContentHash {
    use sha2::Digest as _;
    let digest = sha2::Sha256::digest(bytes);
    ContentHash(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

/// The files a tool call read, in the order it read them, each once: a
/// file it showed the model, or one it found matches in.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FilesRead(pub Vec<FileRead>);

impl ContextValue for FilesRead {
    const KEY: &'static str = "files_read";
}

impl FilesRead {
    /// The read files' paths, in order.
    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|file| file.path.as_str())
    }
}

/// One file a tool call read.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileRead {
    /// The file, as [`FileChange::path`] spells it.
    pub path: String,
    /// The hash of the bytes the tool read.
    pub hash: ContentHash,
    /// Which of its lines the tool showed, when it showed some of them
    /// rather than matches in it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lines: Option<LinesShown>,
}

impl FileRead {
    /// `path` was read, and held the content `hash` names.
    pub fn new(path: impl Into<String>, hash: ContentHash) -> Self {
        Self {
            path: path.into(),
            hash,
            lines: None,
        }
    }

    /// This read, having shown `lines` of the file.
    #[must_use]
    pub fn with_lines(mut self, lines: LinesShown) -> Self {
        self.lines = Some(lines);
        self
    }
}

/// The lines of a file a tool showed: `count` lines from line `first`, of
/// the `total` it has.
///
/// ```
/// use rig_core::tool::LinesShown;
///
/// assert!(LinesShown { first: 1, count: 3, total: 3 }.whole());
/// assert!(!LinesShown { first: 2, count: 2, total: 3 }.whole());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LinesShown {
    /// The first line shown, counted from 1.
    pub first: usize,
    /// How many lines were shown.
    pub count: usize,
    /// How many lines the file has.
    pub total: usize,
}

impl LinesShown {
    /// Whether every line of the file was shown.
    pub fn whole(&self) -> bool {
        self.first <= 1 && self.count >= self.total
    }
}

/// The files a tool call changed, in the order it changed them.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FilesChanged(pub Vec<FileChange>);

impl ContextValue for FilesChanged {
    const KEY: &'static str = "files_changed";
}

impl FilesChanged {
    /// The changed files' paths, in order.
    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|change| change.path.as_str())
    }
}

/// One file a tool call changed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileChange {
    /// The file: an absolute path when the tool knows the directory it ran
    /// in, as rig-tools' file tools do, else as the call named it.
    pub path: String,
    /// What happened to it.
    pub kind: ChangeKind,
    /// Lines added.
    pub added: usize,
    /// Lines removed.
    pub removed: usize,
    /// The change as hunks, each line with its ending, with three lines of
    /// context; none when the tool did not diff the file (such as one too
    /// large to read). [`unified`](Self::unified) renders them as a diff.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hunks: Vec<Hunk>,
    /// The file's text before the change, when the tool read it whole as
    /// UTF-8 within its read limit; `None` for a created file.
    /// [`after_text`](Self::after_text) is it with the hunks applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before_text: Option<String>,
    /// The file before the change, when the tool or a checkpoint around it
    /// knows it; `None` for a created file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before: Option<FileState>,
    /// The file the change left, when the tool knows it; `None` for a
    /// deleted file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<FileState>,
}

/// A file on one side of a [`FileChange`]: as it was before the change, or
/// as the change left it.
///
/// ```
/// use rig_core::tool::{FileState, content_hash};
///
/// let state = FileState::of(b"one\ntwo");
/// assert_eq!((state.bytes, state.lines), (7, Some(2)));
/// assert_eq!(state.hash, Some(content_hash(b"one\ntwo")));
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FileState {
    /// Its size in bytes.
    pub bytes: u64,
    /// Its lines, as [`str::lines`] counts them, when it was read (not
    /// when it was too large to read).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lines: Option<usize>,
    /// The hash of its bytes, when it was read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash: Option<ContentHash>,
}

impl FileState {
    /// A file that held `bytes`, read whole.
    pub fn of(bytes: &[u8]) -> Self {
        let breaks = bytes.iter().filter(|byte| **byte == b'\n').count();
        let unended = !bytes.is_empty() && !bytes.ends_with(b"\n");
        Self {
            bytes: bytes.len() as u64,
            lines: Some(breaks + usize::from(unended)),
            hash: Some(content_hash(bytes)),
        }
    }

    /// A file of `bytes` bytes that was not read, such as one too large to.
    pub fn unread(bytes: u64) -> Self {
        Self {
            bytes,
            lines: None,
            hash: None,
        }
    }
}

impl FileChange {
    /// `path` was created with `lines` lines.
    pub fn created(path: impl Into<String>, lines: usize) -> Self {
        Self {
            path: path.into(),
            kind: ChangeKind::Created,
            added: lines,
            removed: 0,
            hunks: Vec::new(),
            before_text: None,
            before: None,
            after: None,
        }
    }

    /// `path` was changed: `added` lines added and `removed` removed.
    pub fn modified(path: impl Into<String>, added: usize, removed: usize) -> Self {
        Self {
            path: path.into(),
            kind: ChangeKind::Modified,
            added,
            removed,
            hunks: Vec::new(),
            before_text: None,
            before: None,
            after: None,
        }
    }

    /// `path`, which had `lines` lines, was deleted.
    pub fn deleted(path: impl Into<String>, lines: usize) -> Self {
        Self {
            path: path.into(),
            kind: ChangeKind::Deleted,
            added: 0,
            removed: lines,
            hunks: Vec::new(),
            before_text: None,
            before: None,
            after: None,
        }
    }

    /// This change, made by `hunks`.
    #[must_use]
    pub fn with_hunks(mut self, hunks: Vec<Hunk>) -> Self {
        self.hunks = hunks;
        self
    }

    /// This change, made to a file that held `text`.
    #[must_use]
    pub fn with_before_text(mut self, text: impl Into<String>) -> Self {
        self.before_text = Some(text.into());
        self
    }

    /// The change as a unified diff, its hunks under a `---`/`+++` header
    /// naming the path (`/dev/null` on the side with no file); empty when
    /// it has no hunks.
    ///
    /// ```
    /// use rig_core::tool::{FileChange, Hunk, HunkLine, Lines};
    ///
    /// let change = FileChange::modified("a.rs", 1, 1).with_hunks(vec![Hunk {
    ///     old: Lines { start: 1, count: 2 },
    ///     new: Lines { start: 1, count: 2 },
    ///     lines: vec![
    ///         HunkLine::Same("one\n".into()),
    ///         HunkLine::Removed("two\n".into()),
    ///         HunkLine::Added("2".into()),
    ///     ],
    /// }]);
    /// assert_eq!(
    ///     change.unified(),
    ///     "--- a.rs\n+++ a.rs\n@@ -1,2 +1,2 @@\n one\n-two\n+2\n\\ No newline at end of file\n"
    /// );
    /// ```
    pub fn unified(&self) -> String {
        if self.hunks.is_empty() {
            return String::new();
        }
        let side = |gone| {
            if self.kind == gone {
                "/dev/null"
            } else {
                self.path.as_str()
            }
        };
        let mut out = format!(
            "--- {}\n+++ {}\n",
            side(ChangeKind::Created),
            side(ChangeKind::Deleted)
        );
        for hunk in &self.hunks {
            let _ = writeln!(out, "@@ -{} +{} @@", hunk.old, hunk.new);
            for line in &hunk.lines {
                out.push(line.sign());
                out.push_str(line.text());
                if !line.text().ends_with(['\r', '\n']) {
                    out.push_str("\n\\ No newline at end of file\n");
                }
            }
        }
        out
    }

    /// The file's text after the change: [`before_text`](Self::before_text)
    /// (empty for a created file) with the hunks applied. `None` for a
    /// deleted file, when the text before is not known, or when the hunks
    /// do not make the whole change (they do not count the lines
    /// [`added`](Self::added) and [`removed`](Self::removed), or do not fit
    /// the text before).
    ///
    /// ```
    /// use rig_core::tool::{FileChange, Hunk, HunkLine, Lines};
    ///
    /// let change = FileChange::modified("a.rs", 1, 1)
    ///     .with_before_text("one\ntwo\n")
    ///     .with_hunks(vec![Hunk {
    ///         old: Lines { start: 2, count: 1 },
    ///         new: Lines { start: 2, count: 1 },
    ///         lines: vec![HunkLine::Removed("two\n".into()), HunkLine::Added("2\n".into())],
    ///     }]);
    /// assert_eq!(change.after_text().as_deref(), Some("one\n2\n"));
    /// assert_eq!(change.changed_lines().collect::<Vec<_>>(), [2]);
    /// ```
    pub fn after_text(&self) -> Option<String> {
        let before = match self.kind {
            ChangeKind::Deleted => return None,
            ChangeKind::Created => "",
            ChangeKind::Modified => self.before_text.as_deref()?,
        };
        let lines = self.hunks.iter().flat_map(|hunk| &hunk.lines);
        let counted = |sign| lines.clone().filter(|line| line.sign() == sign).count();
        if (counted('+'), counted('-')) != (self.added, self.removed) {
            return None;
        }
        let mut old = text_lines(before);
        let mut next = 1;
        let mut out = String::with_capacity(before.len());
        for hunk in &self.hunks {
            while next < hunk.old.start {
                out.push_str(old.next()?);
                next += 1;
            }
            for line in &hunk.lines {
                if let HunkLine::Added(text) = line {
                    out.push_str(text);
                    continue;
                }
                if old.next()? != line.text() {
                    return None;
                }
                next += 1;
                if let HunkLine::Same(text) = line {
                    out.push_str(text);
                }
            }
        }
        out.extend(old);
        Some(out)
    }

    /// The lines the change added or rewrote, counted from 1 in the file
    /// after it.
    pub fn changed_lines(&self) -> impl Iterator<Item = usize> + '_ {
        self.hunks.iter().flat_map(|hunk| {
            let new = hunk.lines.iter().filter(|line| line.sign() != '-');
            new.zip(hunk.new.start..)
                .filter_map(|(line, at)| matches!(line, HunkLine::Added(_)).then_some(at))
        })
    }

    /// This change, made to the file `before`.
    #[must_use]
    pub fn with_before(mut self, before: FileState) -> Self {
        self.before = Some(before);
        self
    }

    /// This change, leaving the file `after`.
    #[must_use]
    pub fn with_after(mut self, after: FileState) -> Self {
        self.after = Some(after);
        self
    }
}

/// One run of a [`FileChange`]: the lines it removed and added, between
/// the lines around them it kept.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Hunk {
    /// The lines of the file before the change that it spans.
    pub old: Lines,
    /// The lines of the file after the change that it spans.
    pub new: Lines,
    /// Its lines in order, each with its line ending (`\n`, `\r\n` or a
    /// lone `\r`; none on a last line that has none).
    pub lines: Vec<HunkLine>,
}

/// Lines of a file that a [`Hunk`] spans: `count` lines from line `start`,
/// counted from 1. An empty span starts at the line it comes before. Shown
/// as a unified diff's range (`3,2`; `3`; `2,0` for an empty one before
/// line 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Lines {
    /// The first line.
    pub start: usize,
    /// How many lines.
    pub count: usize,
}

impl fmt::Display for Lines {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.count {
            1 => write!(f, "{}", self.start),
            0 => write!(f, "{},0", self.start.saturating_sub(1)),
            count => write!(f, "{},{count}", self.start),
        }
    }
}

/// A line of a [`Hunk`], with its ending. Its wire form is the line as a
/// unified diff shows it: its sign (` `, `-` or `+`), then its text.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(into = "String", try_from = "String")]
pub enum HunkLine {
    /// A line the change kept, around those it changed.
    Same(String),
    /// A line of the file before that the change removed.
    Removed(String),
    /// A line of the file after that the change added.
    Added(String),
}

impl HunkLine {
    /// The line's text, with its ending.
    pub fn text(&self) -> &str {
        match self {
            Self::Same(text) | Self::Removed(text) | Self::Added(text) => text,
        }
    }

    /// Its sign in a unified diff: ` `, `-` or `+`.
    pub fn sign(&self) -> char {
        match self {
            Self::Same(_) => ' ',
            Self::Removed(_) => '-',
            Self::Added(_) => '+',
        }
    }
}

impl From<HunkLine> for String {
    fn from(line: HunkLine) -> Self {
        let sign = line.sign();
        let (HunkLine::Same(mut text) | HunkLine::Removed(mut text) | HunkLine::Added(mut text)) =
            line;
        text.insert(0, sign);
        text
    }
}

impl TryFrom<String> for HunkLine {
    type Error = String;

    fn try_from(mut line: String) -> Result<Self, Self::Error> {
        let make = match line.chars().next() {
            Some(' ') => Self::Same,
            Some('-') => Self::Removed,
            Some('+') => Self::Added,
            _ => return Err(format!("a hunk line starts with ' ', '-' or '+': {line:?}")),
        };
        line.remove(0);
        Ok(make(line))
    }
}

/// `text`'s lines, each with its ending: `\n`, `\r\n` or a lone `\r`, as
/// hunks count them.
fn text_lines(mut text: &str) -> impl Iterator<Item = &str> {
    std::iter::from_fn(move || {
        let end = match text.find(['\r', '\n']) {
            Some(at) if text.get(at..).is_some_and(|rest| rest.starts_with("\r\n")) => at + 2,
            Some(at) => at + 1,
            None => text.len(),
        };
        let (line, rest) = text.split_at_checked(end)?;
        text = rest;
        (!line.is_empty()).then_some(line)
    })
}

/// What a tool call did to a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeKind {
    /// It did not exist before.
    Created,
    /// It existed and was changed.
    Modified,
    /// It existed and is gone.
    Deleted,
}

/// A tool's output was cut to fit: the model's text keeps one end, or both
/// ends and some lines from between, and says so in a line of its own. A
/// front reads this fact, not that line's wording, to draw and count the
/// cut.
///
/// ```
/// use rig_core::tool::{KeptEnd, OutputCut};
///
/// let cut = OutputCut { kept: KeptEnd::Ends, kept_lines: 2000, total_lines: 5120, spill: None };
/// assert_eq!(cut.cut_lines(), 3120);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputCut {
    /// Which end of the output the text keeps.
    pub kept: KeptEnd,
    /// How many lines of the output the text keeps, not counting the lines
    /// that say what was cut.
    pub kept_lines: usize,
    /// How many lines the whole output had.
    pub total_lines: usize,
    /// The file that holds all of the output, when the tool kept it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spill: Option<String>,
}

impl ContextValue for OutputCut {
    const KEY: &'static str = "output_cut";
}

impl OutputCut {
    /// How many lines of the output the text leaves out.
    pub fn cut_lines(&self) -> usize {
        self.total_lines.saturating_sub(self.kept_lines)
    }
}

/// Which part of a cut output a tool kept ([`OutputCut::kept`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KeptEnd {
    /// Its first lines, such as a long file read from the top. The line
    /// that says so comes last.
    Start,
    /// Its first and last lines, and lines from between, such as a
    /// command's output with the errors from its middle. The line that says
    /// so comes first.
    Ends,
}

/// How a process a tool call ran ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExitStatus {
    /// It exited with this code.
    Exited(i32),
    /// A signal it did not expect ended it.
    Signaled,
    /// It ran past its time limit and was killed.
    TimedOut,
    /// The tool killed it for another reason, such as printing too much or
    /// the call being cancelled.
    Killed,
}

impl ContextValue for ExitStatus {
    const KEY: &'static str = "exit_status";
}

impl ExitStatus {
    /// Whether it exited with code 0.
    pub fn success(&self) -> bool {
        matches!(self, Self::Exited(0))
    }

    /// The code it exited with, if it exited.
    pub fn code(&self) -> Option<i32> {
        match self {
            Self::Exited(code) => Some(*code),
            Self::Signaled | Self::TimedOut | Self::Killed => None,
        }
    }
}

#[cfg(test)]
mod tests;
