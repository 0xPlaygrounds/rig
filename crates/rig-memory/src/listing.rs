//! One agent's conversation of a saved session, read without the rest
//! ([`read_conversation`]), as a session picker previews it.

use std::io;

use rig_core::conversation::Conversation;

use crate::format::SESSION;
use crate::session::{Fold, Folded, fold, log_lines, read_header};
use crate::{SessionError, Store, main_order};

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

#[cfg(test)]
mod tests;
