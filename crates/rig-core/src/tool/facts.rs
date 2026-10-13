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
    /// The change as a unified diff, when the tool made one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diff: Option<String>,
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
            diff: None,
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
            diff: None,
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
            diff: None,
            before: None,
            after: None,
        }
    }

    /// This change with `diff`, its unified diff.
    #[must_use]
    pub fn with_diff(mut self, diff: impl Into<String>) -> Self {
        self.diff = Some(diff.into());
        self
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

/// A tool call's output was cut to fit: the text the model reads keeps
/// one end of it, and says so in a line of its own on the side it cut,
/// before what it keeps when it kept the end and after it when it kept the
/// start. A front draws that line whole, and leaves it out of what it
/// counts as output, by this fact rather than by the line's wording.
///
/// ```
/// use rig_core::tool::{KeptEnd, OutputCut};
///
/// let cut = OutputCut { kept: KeptEnd::End, kept_lines: 2000, total_lines: 5120, spill: None };
/// assert_eq!(cut.cut_lines(), 3120);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputCut {
    /// Which end of the output the text keeps.
    pub kept: KeptEnd,
    /// How many lines of the output the text keeps, not counting the line
    /// that says it was cut.
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

/// Which end of a cut output a tool kept ([`OutputCut::kept`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KeptEnd {
    /// Its first lines, such as a long file read from the top.
    Start,
    /// Its last lines, such as the end of a command's output.
    End,
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
