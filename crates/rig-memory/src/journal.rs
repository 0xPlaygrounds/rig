//! A session's event journal: `sessions/<id>/events.jsonl`, append-only,
//! written only by the agent server. It starts with a
//! `{"format":"rig-events/1","session":…}` header line, fsyncs per appended
//! batch, and truncates a torn tail to its last complete line on open.

use std::io;
use std::path::Path;

use rig_protocol::{EventRecord, Seq, SessionId};

/// An open journal.
pub struct Journal;

impl Journal {
    /// Opens or creates the journal of `session` in its directory `dir`.
    pub fn open(dir: &Path, session: &SessionId) -> io::Result<Journal> {
        let _ = (dir, session);
        Err(io::ErrorKind::Unsupported.into())
    }

    /// Appends `recs`, whose seqs continue from [`last`](Journal::last), and
    /// fsyncs them.
    pub fn append(&mut self, recs: &[EventRecord]) -> io::Result<()> {
        let _ = recs;
        Err(io::ErrorKind::Unsupported.into())
    }

    /// At most `limit` records after `after`, in order.
    pub fn read_after(&self, after: Seq, limit: usize) -> io::Result<Vec<EventRecord>> {
        let _ = (after, limit);
        Err(io::ErrorKind::Unsupported.into())
    }

    /// The last appended seq, `Seq(0)` when empty.
    pub fn last(&self) -> Seq {
        Seq(0)
    }
}
