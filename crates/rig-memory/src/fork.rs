//! Forking an agent's log: [`Writer::fork_log`](crate::Writer::fork_log)
//! starts a log as a copy of another, up to a turn boundary ([`turn_cut`]).
//! The copied records keep their stamps, so [`as_of`](crate::as_of) and
//! [`history`](crate::history) read the copy as they read its source, and a
//! compaction made before the cut can be undone in the copy.

use std::io;

use rig_core::conversation::{Conversation, MessageId};

use crate::SessionError;
use crate::format::{Body, LogLines, parse};
use crate::session::read_header;

/// Whether `conversation` is at a turn boundary: no turn runs, no tool
/// call waits for its result, and the model owes no answer.
fn at_rest(conversation: &Conversation) -> bool {
    let ended = conversation.turns().next_back();
    !conversation.awaits_model()
        && conversation.pending_calls().is_empty()
        && ended.is_none_or(|(_, turn)| turn.end.is_some())
}

/// How many bytes of `lines`, the log `name`, a copy of it up to a turn
/// boundary at or before message `through` keeps: through the last record
/// after which the conversation is at rest and holds nothing after
/// `through` (every record with `None`); the header alone when there is
/// none. A user message that asks for a turn is at no boundary, so a cut
/// through it is just before it.
///
/// # Errors
///
/// When the log does not read, or never held `through`.
pub(crate) fn turn_cut(
    name: &str,
    lines: &LogLines,
    through: Option<MessageId>,
) -> Result<u64, SessionError> {
    let mut records = lines.iter();
    let Some((_, mut cut, first)) = records.next() else {
        return Ok(0);
    };
    read_header(name, first)?;
    let refuse = |line, why| SessionError::Record {
        log: name.to_owned(),
        line,
        why,
    };
    let mut conversation = Conversation::new();
    let mut held = false;
    for (line, end, bytes) in records {
        if let (_, Body::Change(recorded)) = parse(bytes).map_err(|why| refuse(line, why))? {
            conversation
                .apply(&recorded)
                .map_err(|error| refuse(line, error.to_string()))?;
        }
        if let Some(through) = through {
            let last = conversation.entries().next_back().map(|entry| entry.id);
            if last != Some(through) && (held || conversation.index_of(through).is_some()) {
                break;
            }
            held |= last == Some(through);
        }
        if at_rest(&conversation) {
            cut = end;
        }
    }
    match through {
        Some(through) if !held && conversation.index_of(through).is_none() => {
            let why = format!("the log `{name}` holds no message {through:?}");
            Err(io::Error::new(io::ErrorKind::NotFound, why).into())
        }
        _ => Ok(cut),
    }
}

#[cfg(test)]
mod tests;
