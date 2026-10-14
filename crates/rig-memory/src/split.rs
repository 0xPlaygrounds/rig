//! Splits: one agent's line of a session copied into a new session
//! ([`split`]), with ids and made-times kept, the pointer the source keeps
//! to it ([`SplitPointer`]), and every line of a conversation as a tree
//! ([`lines`]).

use std::io;

use rig_core::conversation::{AsideId, Conversation, MessageId};
use serde::{Deserialize, Serialize};

use crate::{LoadError, Store};

/// The kind of the aside a line keeps after the last entry a split of it
/// copied: a [`SplitPointer`].
pub const SPLIT: &str = "rig.split";

/// A pointer in a line to a split of it kept in another session.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitPointer {
    /// The session the split is kept in.
    pub session: String,
    /// The agent of that session whose line it is.
    pub agent: String,
    /// The split's title.
    pub title: String,
    /// Entries moved out of this line into it; 0 for a copy.
    #[serde(default)]
    pub moved: usize,
    /// Whether it was merged back.
    #[serde(default)]
    pub merged: bool,
}

/// What [`split`] copies: one agent's line through message `through` (the
/// whole line with `None`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitOf<'a> {
    /// The agent, its log's name.
    pub agent: &'a str,
    /// Its line: a kept branch's aside, the active line with `None`.
    pub line: Option<AsideId>,
    /// The last message copied; the line's last with `None`.
    pub through: Option<MessageId>,
}

/// Writes into `to`, an empty store, a session holding only `of`'s line of
/// the session in `from` (read strictly, older formats through the chain):
/// one log for the agent, the same id, its line as one reset with ids and
/// made-times kept, its saved values as they are now, the session log's
/// values, the blobs its images name, and the session's plugin files. The
/// asides that follow the copied entries come along, nested branches
/// among them, but not the branches on the path to the line copied. The
/// copy as written.
///
/// # Errors
///
/// Writing nothing: as [`Session::read`](crate::Session::read), when
/// `through` is not on the line or the point parts a call from its
/// result, or when a write fails.
pub fn split(
    from: &dyn Store,
    to: &dyn Store,
    of: &SplitOf<'_>,
) -> Result<Conversation, LoadError> {
    let _ = (from, to, of);
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "splitting a session is not supported yet",
    )
    .into())
}

/// A line of a conversation, as [`lines`] lists it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Line {
    /// Which line it is.
    pub id: LineId,
    /// Its title.
    pub title: String,
    /// The position in the list of the line it is kept in; `None` for the
    /// active line.
    pub inside: Option<usize>,
    /// How many entries it shares with the line it is kept in.
    pub at: usize,
    /// How many entries it has of its own.
    pub entries: usize,
    /// Whether its entries were moved out of the line it is kept in.
    pub moved: bool,
    /// Whether it was merged back.
    pub merged: bool,
}

/// Which line a [`Line`] is.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LineId {
    /// The active line.
    Active,
    /// A kept branch, by its aside.
    Branch(AsideId),
    /// A split kept in another session: that session and its agent.
    Session {
        /// The session's id.
        session: String,
        /// The agent's id.
        agent: String,
    },
}

/// The lines of `conversation`: the active one, each kept branch (nested
/// ones inside theirs), and each split kept in another session (its
/// pointer), as a tree in display order, each after the line it is kept
/// in.
pub fn lines(conversation: &Conversation) -> Vec<Line> {
    let _ = conversation;
    Vec::new()
}
