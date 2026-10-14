//! What a listing of saved sessions reads: a session's [`Listing`] from its
//! store's files alone, and one agent's line read without the rest
//! ([`read_line`]).

use std::io;

use rig_core::conversation::{AsideId, Conversation};

use crate::{Files, ForkedFrom, LoadError, SessionMeta, Store};

/// What a listing shows of a saved session, from its store's files alone
/// (no log is folded).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Listing {
    /// What it says about itself; empty when it has no listing file.
    pub meta: SessionMeta,
    /// The session it was forked or split from, if it was.
    pub parent: Option<ForkedFrom>,
}

impl Listing {
    /// The listing of the session whose store's files are `files`.
    pub fn read(files: &(impl Files + ?Sized)) -> Self {
        Self {
            meta: SessionMeta::read(files).unwrap_or_default(),
            parent: ForkedFrom::read(files),
        }
    }
}

/// One agent's line of the session in `store`: `agent` (the main agent when
/// `None`, found from the logs' headers alone), its `line` (active when
/// `None`), folded from that agent's log only, older formats through the
/// chain, writing nothing. The agent's id and the line as a conversation,
/// with ids and made-times kept.
///
/// # Errors
///
/// As [`load`](crate::load), or when there is no such agent or line.
pub fn read_line(
    store: &dyn Store,
    agent: Option<&str>,
    line: Option<AsideId>,
) -> Result<(String, Conversation), LoadError> {
    let _ = (store, agent, line);
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "reading one line is not supported yet",
    )
    .into())
}
