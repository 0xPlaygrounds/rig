//! One agent's conversation of a saved session, read without the rest
//! ([`read_conversation`]), as a session picker previews it.

use std::io;

use rig_core::conversation::Conversation;

use crate::format::SESSION;
use crate::session::{Fold, Folded, fold, log_lines, read_header};
use crate::{CURRENT, SessionError, Store};

/// The conversation of `agent` (the main agent when `None`, found from the
/// logs' headers alone) of the session in `store`, folded from that
/// agent's log only, writing nothing. Returns the agent's id and its
/// conversation, with ids and made-times kept.
///
/// # Errors
///
/// As [`load`](crate::load), or when there is no such agent.
pub fn read_conversation(
    store: &dyn Store,
    agent: Option<&str>,
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
    Ok((folded.name, folded.conversation))
}

/// The main agent's log, folded: of the logs whose header names no parent
/// and that are not retired, the one the session's [`CURRENT`] names, else
/// the first by id, as [`Session::main_agent`](crate::Session::main_agent)
/// finds it; only its candidates are folded.
fn main_log(store: &dyn Store, how: Fold) -> Result<Option<Folded>, SessionError> {
    let own = Fold {
        images: false,
        ..how
    };
    let current = fold(store, SESSION, own)?
        .and_then(|session| session.state.get(CURRENT)?.as_str().map(str::to_owned));
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
                candidates.push((Some(&name) != current.as_ref(), name));
            }
        }
    }
    candidates.sort();
    for (.., name) in candidates {
        if let Some(folded) = fold(store, &name, how)?.filter(|folded| !folded.retired) {
            return Ok(Some(folded));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests;
