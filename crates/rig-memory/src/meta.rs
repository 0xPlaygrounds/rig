//! What a session says about itself beside its logs, kept as files of its
//! store: what a listing shows of it ([`SessionMeta`], in [`META`]) and
//! the session it was forked from ([`ForkedFrom`], in [`PARENT`]).
//!
//! These files are versioned implicitly: they change only by fields that
//! a reader not knowing them skips and a reader missing them takes as
//! empty. A change that breaks that adds a `"v"` field and a step that
//! reads the older form, as the logs' format does.

use std::collections::BTreeMap;
use std::io;

use rig_core::conversation::MessageId;
use serde::{Deserialize, Serialize};

use crate::Files;

/// The file of a session's store that holds its [`SessionMeta`].
pub const META: &str = "meta.json";

/// The file of a forked session's store that holds its [`ForkedFrom`].
pub const PARENT: &str = "parent.json";

/// Most characters of a title made by [`title_of`].
const TITLE_CHARS: usize = 60;

/// What a listing shows of a session, kept in its store: what it is
/// about, the labels plugins give it, and when it last changed. Its logs
/// hold the same facts; this is the cache a listing reads without folding
/// them. Fields a reader does not know are skipped, and missing ones are
/// empty.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionMeta {
    /// What the session is about: the start of the first message the user
    /// typed ([`title_of`]); empty until a turn ended.
    pub title: String,
    /// Plugins' labels by tag, such as a goal or what it cost.
    pub labels: BTreeMap<String, String>,
    /// When it last changed, in milliseconds since the Unix epoch.
    pub updated: u64,
    /// The main agent's model, `vendor/model`; empty when unknown.
    pub model: String,
    /// How many messages the main agent's active line holds.
    pub messages: usize,
}

impl SessionMeta {
    /// The listing kept in `files`, if it has one that reads. Generic, so
    /// it is passed as it is where a store is any `Files`, such as
    /// `store.as_deref().and_then(SessionMeta::read)`.
    pub fn read(files: &(impl Files + ?Sized)) -> Option<Self> {
        serde_json::from_slice(&files.read_file(META).ok()?).ok()
    }

    /// Keeps it as the listing in `files`.
    ///
    /// # Errors
    ///
    /// When `files` cannot write it.
    pub fn write(&self, files: &(impl Files + ?Sized)) -> io::Result<()> {
        files.write_file(META, &serde_json::to_vec_pretty(self)?)
    }
}

/// The session a session was forked or split from, and where the copy
/// stopped, with the text of the message it stopped at: kept in the copy's
/// store, so a view of a session's family reads no other session's log.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForkedFrom {
    /// The id of the session it was copied from.
    pub session: String,
    /// For a fork, where the copy stopped: an agent's id (its log's name)
    /// and the id of the message it stopped before; `None` for a copy of
    /// everything, and for a split.
    pub before: Option<(String, MessageId)>,
    /// For a split, the agent whose line it copied and the last message it
    /// copied; `None` for a fork.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub through: Option<(String, MessageId)>,
    /// For a split, the title of the line it copied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<String>,
    /// What the user wrote in the message the copy stopped at (before it,
    /// for a fork; through it, for a split), its text without attached
    /// files; `None` for a copy of everything, or for a message with no
    /// text of the user's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

impl ForkedFrom {
    /// The session the session in `files` was forked from, if it was.
    /// Generic, as [`SessionMeta::read`] is.
    pub fn read(files: &(impl Files + ?Sized)) -> Option<Self> {
        serde_json::from_slice(&files.read_file(PARENT).ok()?).ok()
    }

    /// Writes it into the forked session's `files`.
    ///
    /// # Errors
    ///
    /// When `files` cannot write it.
    pub fn write(&self, files: &(impl Files + ?Sized)) -> io::Result<()> {
        files.write_file(PARENT, &serde_json::to_vec(self)?)
    }
}

/// A title or label of `text`, as a listing shows it: on one line, each
/// run of whitespace one space, cut to 60 characters (the last an
/// ellipsis when cut).
///
/// ```
/// use rig_memory::title_of;
///
/// assert_eq!(title_of("  fix\n the   parser "), "fix the parser");
/// assert_eq!(title_of(&"a".repeat(80)).chars().count(), 60);
/// ```
pub fn title_of(text: &str) -> String {
    let line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    match line.char_indices().nth(TITLE_CHARS) {
        None => line,
        Some(_) => {
            let kept = line.chars().take(TITLE_CHARS - 1);
            kept.chain(['…']).collect()
        }
    }
}

#[cfg(test)]
mod tests;
