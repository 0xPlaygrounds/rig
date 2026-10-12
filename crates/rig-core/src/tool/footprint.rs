//! What a tool call may read and change ([`Touches`]), so a host runs the
//! calls of one reply that touch different things side by side, and holds
//! back only those that conflict.
//!
//! ```
//! use std::path::PathBuf;
//! use rig_core::tool::{Scope, Touches};
//!
//! let paths = |path: &str| Scope::Paths(vec![PathBuf::from(path)]);
//! let edit = Touches::Writes(paths("/work/src/a.rs"));
//! assert!(!edit.conflicts_with(&Touches::Reads(paths("/work/src/b.rs"))));
//! assert!(edit.conflicts_with(&Touches::Reads(paths("/work/src"))));
//! assert!(edit.conflicts_with(&Touches::Independent));
//! assert!(!Touches::Reads(Scope::Anything).conflicts_with(&Touches::Independent));
//! ```

use std::path::{Component, PathBuf};

use serde::{Deserialize, Serialize};

use crate::completion::ToolAnnotations;

/// What a tool call may read and change, so calls that touch different
/// things run side by side.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Touches {
    /// Reads what the scope names and changes nothing: run again after a
    /// restart.
    Reads(Scope),
    /// Reads and may change what the scope names.
    Writes(Scope),
    /// Changes nothing the other calls of its reply see, such as handing
    /// work to another agent. It may read anything, so it waits behind an
    /// earlier writer; it is never run again after a restart.
    Independent,
}

/// What a footprint names: these paths (absolute; a directory covers what
/// is under it), or anything at all. A guard compares paths as they are,
/// so a host makes a footprint [`normalized`](Touches::normalized) before
/// any guard sees it; the rig kernel does so for every tool call.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    /// These files and directories.
    Paths(Vec<PathBuf>),
    /// Anything at all.
    Anything,
}

impl Scope {
    /// Whether the two name something in common: a path is, or is under, a
    /// path of the other, compared component by component; `Anything`
    /// meets everything.
    pub fn overlaps(&self, other: &Scope) -> bool {
        match (self, other) {
            (Self::Paths(ours), Self::Paths(theirs)) => ours.iter().any(|ours| {
                theirs
                    .iter()
                    .any(|theirs| ours.starts_with(theirs) || theirs.starts_with(ours))
            }),
            (Self::Anything, _) | (_, Self::Anything) => true,
        }
    }

    /// The scope with each path's `.` and `..` folded away, by the path's
    /// text alone, so `/work/src/../../etc` names `/etc`, as a guard
    /// comparing paths must see it. A relative path names no place a guard
    /// can check, so a scope with one is [`Anything`](Self::Anything).
    pub fn normalized(self) -> Scope {
        match self {
            Self::Paths(paths) => paths
                .into_iter()
                .map(normal)
                .collect::<Option<Vec<_>>>()
                .map_or(Self::Anything, Self::Paths),
            Self::Anything => Self::Anything,
        }
    }
}

/// `path` with its `.` and `..` folded away, or `None` when it is relative.
/// `..` at the root stays at the root.
fn normal(path: PathBuf) -> Option<PathBuf> {
    if !path.is_absolute() {
        return None;
    }
    let mut normal = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                normal.pop();
            }
            Component::CurDir => {}
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {
                normal.push(component);
            }
        }
    }
    Some(normal)
}

impl Touches {
    /// Whether a call touching `self` must not run beside one touching
    /// `other`: one of them writes what the other reads or writes.
    /// [`Independent`](Self::Independent) counts as `Reads(Anything)` here.
    pub fn conflicts_with(&self, other: &Touches) -> bool {
        let (ours, we_write) = self.parts();
        let (theirs, they_write) = other.parts();
        (we_write || they_write) && ours.overlaps(theirs)
    }

    /// Whether it only reads: [`Reads`](Self::Reads), and nothing else.
    pub fn reads_only(&self) -> bool {
        matches!(self, Self::Reads(_))
    }

    /// The footprint with its scope [`normalized`](Scope::normalized).
    pub fn normalized(self) -> Touches {
        match self {
            Self::Reads(scope) => Self::Reads(scope.normalized()),
            Self::Writes(scope) => Self::Writes(scope.normalized()),
            Self::Independent => Self::Independent,
        }
    }

    /// From a tool's annotations: `Reads(Anything)` when it says it only
    /// reads, else `Writes(Anything)`.
    pub fn from_annotations(annotations: &ToolAnnotations) -> Self {
        match annotations.read_only() {
            true => Self::Reads(Scope::Anything),
            false => Self::Writes(Scope::Anything),
        }
    }

    /// What it names, and whether it writes there.
    fn parts(&self) -> (&Scope, bool) {
        match self {
            Self::Reads(scope) => (scope, false),
            Self::Writes(scope) => (scope, true),
            Self::Independent => (&Scope::Anything, false),
        }
    }
}

#[cfg(test)]
mod tests;
