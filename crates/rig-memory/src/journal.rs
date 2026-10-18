//! A session's event journal: `sessions/<id>/events.jsonl`, append-only,
//! written only by the agent server. It starts with a
//! `{"format":"rig-events/1","session":…}` header line, fsyncs per appended
//! batch, and truncates a torn tail to its last complete line on open.
//!
//! ```
//! use rig_memory::journal::Journal;
//! use rig_protocol::{EventRecord, Seq, SessionId};
//!
//! let dir = std::env::temp_dir().join(format!("rig-journal-doc-{}", std::process::id()));
//! let session = SessionId("s-1".into());
//! let mut journal = Journal::open(&dir, &session)?;
//! let record = |seq| EventRecord {
//!     session: session.clone(),
//!     seq: Seq(seq),
//!     at_ms: 0,
//!     kind: "notice".into(),
//!     data: serde_json::json!({"level": "info", "text": "hi"}),
//! };
//! journal.append(&[record(1), record(2)])?;
//! drop(journal);
//!
//! let journal = Journal::open(&dir, &session)?;
//! assert_eq!(journal.last(), Seq(2));
//! assert_eq!(journal.read_after(Seq(1), 10)?.len(), 1);
//! # std::fs::remove_dir_all(&dir)?;
//! # Ok::<(), std::io::Error>(())
//! ```

use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use rig_protocol::{EventRecord, Seq, SessionId};
use serde::{Deserialize, Serialize};

/// The format this journal writes and reads.
pub const FORMAT: &str = "rig-events/1";
/// The journal's file name in its session directory.
pub const FILE: &str = "events.jsonl";

/// The first line of a journal.
#[derive(Debug, Serialize, Deserialize)]
struct Header {
    format: String,
    session: SessionId,
}

/// An open journal.
pub struct Journal {
    path: PathBuf,
    file: File,
    session: SessionId,
    /// The byte offset of each record's line: `offsets[n]` holds seq `n + 1`.
    offsets: Vec<u64>,
    /// Where the next line goes.
    end: u64,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

impl Journal {
    /// Opens or creates the journal of `session` in its directory `dir`.
    /// A journal of another format or session is refused; a torn last line
    /// (no newline, or not a record) is cut off.
    pub fn open(dir: &Path, session: &SessionId) -> io::Result<Journal> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join(FILE);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        let mut journal = Journal {
            path,
            file,
            session: session.clone(),
            offsets: Vec::new(),
            end: 0,
        };
        if bytes.is_empty() {
            let mut line = serde_json::to_vec(&Header {
                format: FORMAT.to_owned(),
                session: session.clone(),
            })
            .map_err(io::Error::other)?;
            line.push(b'\n');
            journal.file.write_all(&line)?;
            journal.file.sync_data()?;
            journal.end = line.len() as u64;
            return Ok(journal);
        }
        let mut lines = bytes.split_inclusive(|b| *b == b'\n');
        let header = lines.next().unwrap_or_default();
        let parsed: Header = match header.strip_suffix(b"\n") {
            Some(line) => serde_json::from_slice(line).map_err(|e| {
                invalid(format!(
                    "{}: not a rig journal ({e})",
                    journal.path.display()
                ))
            })?,
            None => return Err(invalid(format!("{}: torn header", journal.path.display()))),
        };
        if parsed.format != FORMAT {
            return Err(invalid(format!(
                "{} is format {}; this rig reads {FORMAT}: a newer rig wrote it, update rig",
                journal.path.display(),
                parsed.format
            )));
        }
        if &parsed.session != session {
            return Err(invalid(format!(
                "{} belongs to session {}",
                journal.path.display(),
                parsed.session.0
            )));
        }
        let mut end = header.len() as u64;
        for line in lines {
            let whole = line
                .strip_suffix(b"\n")
                .and_then(|l| serde_json::from_slice::<EventRecord>(l).ok())
                .filter(|r| r.seq.0 == journal.offsets.len() as u64 + 1);
            if whole.is_none() {
                break;
            }
            journal.offsets.push(end);
            end += line.len() as u64;
        }
        if end < bytes.len() as u64 {
            journal.file.set_len(end)?;
            journal.file.sync_data()?;
        }
        journal.end = end;
        Ok(journal)
    }

    /// The session this journal belongs to.
    pub fn session(&self) -> &SessionId {
        &self.session
    }

    /// Appends `recs`, whose seqs continue from [`last`](Journal::last), and
    /// fsyncs them. A gap or repeat is refused before anything is written.
    pub fn append(&mut self, recs: &[EventRecord]) -> io::Result<()> {
        let mut buf = Vec::new();
        let mut offsets = Vec::with_capacity(recs.len());
        for (next, rec) in (self.last().0 + 1..).zip(recs) {
            if rec.seq.0 != next {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("seq {} does not follow {}", rec.seq.0, next - 1),
                ));
            }
            offsets.push(self.end + buf.len() as u64);
            serde_json::to_writer(&mut buf, rec).map_err(io::Error::other)?;
            buf.push(b'\n');
        }
        if buf.is_empty() {
            return Ok(());
        }
        self.file.seek(SeekFrom::Start(self.end))?;
        if let Err(e) = self
            .file
            .write_all(&buf)
            .and_then(|()| self.file.sync_data())
        {
            // Leave no partial batch for the next append to follow.
            self.file.set_len(self.end).ok();
            return Err(e);
        }
        self.end += buf.len() as u64;
        self.offsets.extend(offsets);
        Ok(())
    }

    /// At most `limit` records after `after`, in order.
    pub fn read_after(&self, after: Seq, limit: usize) -> io::Result<Vec<EventRecord>> {
        let Some(&start) = usize::try_from(after.0)
            .ok()
            .and_then(|i| self.offsets.get(i))
        else {
            return Ok(Vec::new());
        };
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(start))?;
        let mut reader = BufReader::new(file.take(self.end.saturating_sub(start)));
        let mut out = Vec::new();
        let mut line = Vec::new();
        while out.len() < limit {
            line.clear();
            if reader.read_until(b'\n', &mut line)? == 0 {
                break;
            }
            let body = line.strip_suffix(b"\n").unwrap_or(&line);
            out.push(serde_json::from_slice(body).map_err(|e| invalid(e.to_string()))?);
        }
        Ok(out)
    }

    /// The last appended seq, `Seq(0)` when empty.
    pub fn last(&self) -> Seq {
        Seq(self.offsets.len() as u64)
    }
}

#[cfg(test)]
mod tests;
