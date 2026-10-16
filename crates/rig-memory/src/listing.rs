//! What a listing of saved sessions reads: a session's [`Listing`] from its
//! store's files alone, and one agent's line read without the rest
//! ([`read_line`]).

use std::io;

use rig_core::conversation::{AsideId, Author, Conversation};

use crate::format::SESSION;
use crate::session::{Fold, Folded, fold, log_lines, read_header};
use crate::{Files, ForkedFrom, SessionError, SessionMeta, Store, main_order};

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
/// chain, writing nothing. Returns the agent's id and the line as a
/// conversation, with ids and made-times kept.
///
/// # Errors
///
/// As [`load`](crate::load), or when there is no such agent or line.
pub fn read_line(
    store: &dyn Store,
    agent: Option<&str>,
    line: Option<AsideId>,
) -> Result<(String, Conversation), SessionError> {
    let how = Fold::WHOLE;
    let folded = match agent {
        Some(agent) => fold(store, agent, how)?,
        None => main_log(store, how)?,
    };
    let Some(folded) = folded.filter(|folded| folded.name != SESSION) else {
        let why = format!("the session has no agent {}", agent.unwrap_or("to talk to"));
        return Err(io::Error::new(io::ErrorKind::NotFound, why).into());
    };
    let mut conversation = folded.conversation;
    if let Some(line) = line {
        conversation
            .switch_branch(line, Author::rig())
            .map_err(|why| io::Error::new(io::ErrorKind::NotFound, why.to_string()))?;
    }
    Ok((folded.name, conversation))
}

/// The main agent's log, folded: of the logs whose header names no parent,
/// the first by [`main_order`] that is not retired, as
/// [`Session::main_agent`](crate::Session::main_agent) finds it; only its
/// candidates are folded.
fn main_log(store: &dyn Store, how: Fold) -> Result<Option<Folded>, SessionError> {
    let mut candidates = Vec::new();
    for name in store.logs()? {
        if name == SESSION {
            continue;
        }
        let Some(lines) = log_lines(store, &name)? else {
            continue;
        };
        if let Some((_, _, first)) = lines.iter().next() {
            let (_, header) = read_header(&name, first)?;
            if header.parent.is_none() {
                candidates.push(name);
            }
        }
    }
    candidates.sort_by(|a, b| main_order(false, a).cmp(&main_order(false, b)));
    for name in candidates {
        if let Some(folded) = fold(store, &name, how)?.filter(|folded| !folded.retired) {
            return Ok(Some(folded));
        }
    }
    Ok(None)
}

/// The first session `id` was forked or split from, through every parent
/// in `parents` (each session's parent, by id); `id` itself when it has
/// none.
pub fn family_root<'a>(parents: &'a [(String, String)], id: &'a str) -> &'a str {
    let parent = |id: &str| -> Option<&'a str> {
        let found = parents.iter().find(|(child, _)| child == id);
        found.map(|(_, parent)| parent.as_str())
    };
    // A loop cannot happen, but a bound costs nothing.
    let up = std::iter::successors(Some(id), |id| parent(id));
    up.take(parents.len() + 1).last().unwrap_or(id)
}

#[cfg(test)]
mod tests;
