//! The saved format, [`FORMAT`]: one JSON-lines log per agent, named
//! by the agent's id, plus the session's own ([`SESSION`]). Each line is a
//! [`Stamp`] and one record: a `header` first, then `state` (a saved value
//! by key; the latest wins, `null` removes it), `retired`, or one
//! [`Change`](rig_core::conversation::Change) of the agent's conversation
//! (`push`, `splice`, `halt`, `aside`, `reset`) with who made it (`by`).
//! The header's `lineage` is the log's own
//! [`Lineage`](rig_core::conversation::Lineage), chosen when it was
//! started, so a conversation saved to it can be told from one that only
//! has as many changes. Its `rewrites` say who rewrote the log in place
//! ([`rewrite`](crate::rewrite)) and when; each rewrite gives the log a
//! fresh lineage.
//!
//! A record keeps when each entry was made: a new entry was made when its
//! record says (`t`), an entry put back as it was says so itself (`made`),
//! and a reset's state carries each entry's.
//!
//! ```json
//! {"seq":0,"n":0,"t":1791779409181,"type":"header","format":"rig-session/4","log":"cd9a…","lineage":8125…}
//! {"seq":1,"n":1,"t":1791779409181,"type":"state","key":"rig_ecs::agent::ToolAccess","value":"All"}
//! {"seq":2,"n":2,"t":1791779409182,"type":"push","entries":[{"message":{…},"id":0}]}
//! ```

use std::fmt;
use std::io;

use rig_core::conversation::{
    Aside, AsideId, Author, Change, ChangeRecord, Conversation, Entry, Halt, Lineage, MessageId,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The saved format a log's header names: the one format this rig reads.
pub const FORMAT: &str = "rig-session/4";

/// The log of the session's own saved values, beside the agents' logs.
pub const SESSION: &str = "session";

/// Where a record is in a session: its place among the session's records
/// in the order they were made (`n`, what [`as_of`](crate::as_of) cuts
/// by), its place in its log (`seq`, dense from
/// the header's 0), and when it was written (`t`, milliseconds since the
/// Unix epoch).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stamp {
    /// Its place among the session's records: it increases along every
    /// log, and across the logs one [`Writer`](crate::Writer) or
    /// [`Session::save`](crate::Session::save) wrote.
    pub n: u64,
    /// Its place in its log.
    pub seq: u64,
    /// When it was written.
    pub t: u64,
}

/// Why a session cannot be read or saved. Reading is strict: a session
/// with any record it cannot read is refused whole, and nothing is written.
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    /// The store failed.
    #[error(transparent)]
    Io(#[from] io::Error),
    /// The directory holds the single-file layout older than `rig-session/2` (a
    /// top-level `*.jsonl`); it is left as it is.
    #[error("this session was made by another version of rig; start a new session")]
    OldLayout,
    /// A log's header names a format other than [`FORMAT`], or none; it is
    /// left as it is.
    #[error(
        "this session was made by another version of rig; start a new session (the log `{log}` \
         is in format {}, not {FORMAT})",
        .found.as_deref().unwrap_or("(none)")
    )]
    Format {
        /// The log.
        log: String,
        /// The format its header names.
        found: Option<String>,
    },
    /// A record cannot be read, or its change does not apply.
    #[error("the log `{log}` cannot be read at line {line}: {why}")]
    Record {
        /// The log.
        log: String,
        /// The line, from 1.
        line: usize,
        /// Why.
        why: String,
    },
    /// The log moved on since the conversation saved to it was read: saving
    /// would lose the changes made there.
    #[error("the log `{log}` changed since it was read")]
    Changed {
        /// The log.
        log: String,
    },
}

impl From<SessionError> for io::Error {
    /// The store's own error back; any other as
    /// [`io::ErrorKind::InvalidData`], so `?` works in code that returns
    /// [`io::Result`].
    fn from(error: SessionError) -> Self {
        match error {
            SessionError::Io(error) => error,
            error => io::Error::new(io::ErrorKind::InvalidData, error),
        }
    }
}

/// What reading a session repaired, or found it must work around.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Repair {
    /// The log ended in a line a crash cut short. The log is read up to
    /// `kept` bytes; the `cut` bytes after were set aside in the named file
    /// `set_aside` when the log was opened for writing (`None` when it was
    /// only read).
    TornTail {
        /// The log.
        log: String,
        /// The bytes read.
        kept: u64,
        /// The bytes cut.
        cut: u64,
        /// The file they were set aside in.
        set_aside: Option<String>,
    },
    /// An image's blob cannot be read, so the image has no data.
    LostImage {
        /// The log.
        log: String,
        /// Why.
        why: String,
    },
}

impl fmt::Display for Repair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TornTail {
                log,
                cut,
                set_aside,
                ..
            } => {
                write!(f, "the log `{log}` ended in a line cut short ({cut} bytes)")?;
                match set_aside {
                    Some(file) => write!(f, ", set aside in {file}"),
                    None => Ok(()),
                }
            }
            Self::LostImage { log, why } => write!(f, "an image of the log `{log}` is lost: {why}"),
        }
    }
}

/// The first record of a log.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct Header {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(default)]
    pub log: String,
    /// The log of the agent that spawned this one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    /// The log's own lineage, which every conversation loaded from it or
    /// saved to it carries.
    #[serde(default)]
    pub lineage: Lineage,
    /// Who rewrote the log in place, and when, oldest first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rewrites: Vec<LogRewrite>,
}

/// A rewrite of a log in place ([`rewrite`](crate::rewrite)): when, and by
/// whom, such as the plugin that redacted it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogRewrite {
    /// When, in milliseconds since the Unix epoch.
    pub t: u64,
    /// Who.
    pub by: Author,
}

impl Header {
    /// The header of a log just started, with a fresh lineage.
    pub fn new(log: &str, parent: Option<&str>) -> Self {
        Self {
            format: Some(FORMAT.to_owned()),
            log: log.to_owned(),
            parent: parent.map(str::to_owned),
            lineage: Lineage::fresh(),
            rewrites: Vec::new(),
        }
    }
}

/// A record that is no change of the conversation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum Meta {
    Header(Header),
    State {
        key: String,
        #[serde(default)]
        value: Option<Value>,
    },
    Retired,
}

/// A record as read.
pub(crate) enum Body {
    Meta(Meta),
    /// Its `at` is the stamp's `t`.
    Change(ChangeRecord),
}

/// The fields every line starts with.
#[derive(Deserialize)]
struct Head<'a> {
    seq: u64,
    n: u64,
    t: u64,
    #[serde(rename = "type", borrow)]
    kind: std::borrow::Cow<'a, str>,
}

/// The record types that are changes of the conversation.
const CHANGES: [&str; 5] = ["push", "splice", "halt", "aside", "reset"];

/// A line's stamp, and whether it is a change of the conversation, read
/// without the rest of it.
pub(crate) fn stamp_of(line: &[u8]) -> Result<(Stamp, bool), String> {
    let head: Head<'_> = serde_json::from_slice(line).map_err(|error| error.to_string())?;
    Ok((head.stamp(), CHANGES.contains(&&*head.kind)))
}

impl Head<'_> {
    fn stamp(&self) -> Stamp {
        Stamp {
            n: self.n,
            seq: self.seq,
            t: self.t,
        }
    }
}

/// A line read in one pass: its stamp and type, and, for a change, who
/// made it and every field a change has, of which the type says which it
/// uses. A change read whole without its stamp would be buffered twice:
/// once for the fields around it, once for its type.
#[derive(Deserialize)]
struct Read<'a> {
    seq: u64,
    n: u64,
    t: u64,
    #[serde(rename = "type", borrow)]
    kind: std::borrow::Cow<'a, str>,
    #[serde(default)]
    by: Option<Author>,
    entries: Option<Vec<Entry>>,
    from: Option<usize>,
    to: Option<usize>,
    removed: Option<Vec<MessageId>>,
    with: Option<Vec<Entry>>,
    #[serde(default)]
    halt: Option<Halt>,
    id: Option<AsideId>,
    #[serde(default)]
    aside: Option<Aside>,
    state: Option<Box<Conversation>>,
}

impl Read<'_> {
    /// The change the line holds.
    fn change(self) -> Result<Change, String> {
        fn given<T>(field: Option<T>, name: &str) -> Result<T, String> {
            field.ok_or_else(|| format!("missing field `{name}`"))
        }
        Ok(match &*self.kind {
            "push" => Change::Push {
                entries: given(self.entries, "entries")?,
            },
            "splice" => Change::Splice {
                from: given(self.from, "from")?,
                to: given(self.to, "to")?,
                removed: given(self.removed, "removed")?,
                with: given(self.with, "with")?,
            },
            "halt" => Change::Halt { halt: self.halt },
            "aside" => Change::Aside {
                id: given(self.id, "id")?,
                aside: self.aside,
            },
            "reset" => Change::Reset {
                state: given(self.state, "state")?,
            },
            other => return Err(format!("unknown record type {other:?}")),
        })
    }
}

/// Reads one line.
pub(crate) fn parse(line: &[u8]) -> Result<(Stamp, Body), String> {
    let mut read: Read<'_> = serde_json::from_slice(line).map_err(|error| error.to_string())?;
    let stamp = Stamp {
        n: read.n,
        seq: read.seq,
        t: read.t,
    };
    let body = match &*read.kind {
        "header" | "state" | "retired" => {
            Body::Meta(serde_json::from_slice(line).map_err(|error| error.to_string())?)
        }
        _ => {
            let by = read.by.take();
            Body::Change(ChangeRecord {
                change: read.change()?,
                by,
                at: Some(stamp.t),
            })
        }
    };
    Ok((stamp, body))
}

/// A line as written.
#[derive(Serialize)]
struct Line<'a> {
    seq: u64,
    n: u64,
    t: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    by: Option<&'a Author>,
    #[serde(flatten)]
    body: Out<'a>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Out<'a> {
    Meta(&'a Meta),
    Change(&'a Change),
}

/// Writes a meta record line, newline included, to `out`.
pub(crate) fn write_meta(out: &mut Vec<u8>, stamp: Stamp, meta: &Meta) -> serde_json::Result<()> {
    write(out, stamp, None, Out::Meta(meta))
}

/// Writes a change line, newline included, to `out`.
pub(crate) fn write_change(
    out: &mut Vec<u8>,
    stamp: Stamp,
    by: Option<&Author>,
    change: &Change,
) -> serde_json::Result<()> {
    write(out, stamp, by, Out::Change(change))
}

/// Writes a line, newline included, to `out`.
fn write(
    out: &mut Vec<u8>,
    stamp: Stamp,
    by: Option<&Author>,
    body: Out<'_>,
) -> serde_json::Result<()> {
    let line = Line {
        seq: stamp.seq,
        n: stamp.n,
        t: stamp.t,
        by,
        body,
    };
    serde_json::to_writer(&mut *out, &line)?;
    out.push(b'\n');
    Ok(())
}

/// The complete lines of a log as stored. A torn tail a crash left (bytes
/// after the last newline, or a last line that is not JSON) is not one.
#[derive(Clone, Debug, Default)]
pub(crate) struct LogLines {
    bytes: Vec<u8>,
    /// Where each complete line, newline included, ends.
    ends: Vec<usize>,
}

impl LogLines {
    pub fn parse(bytes: Vec<u8>) -> Self {
        let mut ends: Vec<usize> = memchr::memchr_iter(b'\n', &bytes)
            .map(|at| at + 1)
            .collect();
        let last_start = ends
            .len()
            .checked_sub(2)
            .and_then(|at| ends.get(at))
            .copied();
        let last = ends
            .last()
            .and_then(|end| bytes.get(last_start.unwrap_or(0)..end.saturating_sub(1)));
        if last.is_some_and(|line| {
            !line.is_empty() && serde_json::from_slice::<serde::de::IgnoredAny>(line).is_err()
        }) {
            ends.pop();
        }
        Self { bytes, ends }
    }

    /// Each complete line that is not blank, without its newline, with its
    /// number from 1 and the byte offset just past it.
    pub fn iter(&self) -> impl Iterator<Item = (usize, u64, &[u8])> {
        let starts = std::iter::once(0).chain(self.ends.iter().copied());
        starts
            .zip(self.ends.iter().copied())
            .enumerate()
            .filter_map(|(at, (start, end))| {
                let line = self.bytes.get(start..end.saturating_sub(1))?;
                (!line.is_empty()).then_some((at + 1, end as u64, line))
            })
    }

    /// The bytes the complete lines cover.
    pub fn len(&self) -> u64 {
        self.ends.last().map_or(0, |end| *end as u64)
    }

    /// The bytes after the complete lines: a torn tail.
    pub fn tail(&self) -> &[u8] {
        let len = usize::try_from(self.len()).unwrap_or(usize::MAX);
        self.bytes.get(len..).unwrap_or_default()
    }

    /// The log's first `len` bytes, no further than its complete lines.
    pub fn prefix(&self, len: u64) -> &[u8] {
        let len = usize::try_from(len.min(self.len())).unwrap_or(0);
        self.bytes.get(..len).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests;
