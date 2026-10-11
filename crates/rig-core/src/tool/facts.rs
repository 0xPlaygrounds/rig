//! Facts a tool call publishes about what it did, beside the text the model
//! reads: the files it changed ([`FilesChanged`]) and how a process it ran
//! ended ([`ExitStatus`]). They are [`ContextValue`]s a tool publishes with
//! [`ToolContext::insert_result`](super::ToolContext::insert_result), so
//! they reach the host in the call's [`ToolResultContext`](super::ToolResultContext)
//! and the effect log, whatever the tool: a built-in one, a plugin's, or an
//! MCP server's adapter. A host reads them back with
//! [`ToolResultContext::get`](super::ToolResultContext::get).
//!
//! ```
//! use rig_core::tool::{ExitStatus, FileChange, FilesChanged, ToolContext};
//!
//! let mut context = ToolContext::new();
//! context.insert_result(FilesChanged(vec![FileChange::modified("src/lib.rs", 3, 1)]))?;
//! context.insert_result(ExitStatus::Exited(0))?;
//! let published = context.result_context();
//! assert_eq!(published.get::<ExitStatus>()?.map(|status| status.success()), Some(true));
//! let changed = published.get::<FilesChanged>()?.unwrap_or_default();
//! assert_eq!(changed.paths().collect::<Vec<_>>(), ["src/lib.rs"]);
//! # Ok::<(), rig_core::tool::ToolContextError>(())
//! ```

use serde::{Deserialize, Serialize};

use super::ContextValue;

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
    /// The file, as the call named it: relative paths are relative to the
    /// directory the tool ran in.
    pub path: String,
    /// What happened to it.
    pub kind: ChangeKind,
    /// Lines added.
    pub added: usize,
    /// Lines removed.
    pub removed: usize,
}

impl FileChange {
    /// `path` was created with `lines` lines.
    pub fn created(path: impl Into<String>, lines: usize) -> Self {
        Self {
            path: path.into(),
            kind: ChangeKind::Created,
            added: lines,
            removed: 0,
        }
    }

    /// `path` was changed: `added` lines added and `removed` removed.
    pub fn modified(path: impl Into<String>, added: usize, removed: usize) -> Self {
        Self {
            path: path.into(),
            kind: ChangeKind::Modified,
            added,
            removed,
        }
    }

    /// `path`, which had `lines` lines, was deleted.
    pub fn deleted(path: impl Into<String>, lines: usize) -> Self {
        Self {
            path: path.into(),
            kind: ChangeKind::Deleted,
            added: 0,
            removed: lines,
        }
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
