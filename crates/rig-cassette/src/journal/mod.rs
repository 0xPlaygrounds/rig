//! An agent session's journal: each agent's [`Conversation`], and the
//! session log that keeps it, one append-only log per agent and
//! content-addressed blobs such as images, behind a [`JournalStore`], which
//! also holds the session's other files by name. A
//! [`Writer`] logs a running session's changes, [`read_session`] reads a
//! session back, [`write_session`] writes one from data, and
//! [`fork_session`] copies one as of a message. [`MemoryStore`] keeps a
//! session in memory; with feature `jsonl`, [`JsonlDirStore`] keeps it as
//! JSON-lines files in a directory. With feature `reflect`, the types a
//! Bevy app keeps in its components are reflected.
//!
//! ```
//! use rig_cassette::journal::{MemoryStore, Origin, SavedAgent, SavedSession};
//! use rig_cassette::journal::{read_session, write_session};
//! use rig_core::completion::Message;
//!
//! let mut agent = SavedAgent { id: "imported".to_owned(), ..SavedAgent::default() };
//! agent.conversation.append(Message::user("Fix the parser"), None);
//! agent.conversation.append(Message::user("From the CI bot"), Some(Origin::plugin("ci")));
//! let session = SavedSession { agents: vec![agent], ..SavedSession::default() };
//! let store = MemoryStore::default();
//! write_session(&store, &session)?;
//! let read = read_session(&store)?;
//! assert_eq!(read.agents[0].conversation, session.agents[0].conversation);
//! # Ok::<(), std::io::Error>(())
//! ```

mod conversation;
#[cfg(feature = "jsonl")]
mod jsonl;
mod session;

pub use conversation::{
    Conversation, Entry, Halt, MessageId, Origin, OriginKind, RequestId, STOPPED, Spliced, short_id,
};
#[cfg(feature = "jsonl")]
pub use jsonl::{EFFECT_LOG, JsonlDirStore};
pub use session::{
    SESSION, SavedAgent, SavedSession, Stamp, Writer, fork_session, now_ms, read_session,
    write_session,
};

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::io;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use rig_core::completion::Message;
use rig_core::message::DocumentSourceKind::{Base64, Raw, Unknown, Url};
use rig_core::message::{Image, ImageMediaType};
use sha2::{Digest, Sha256};

/// Storage for a session's files: each agent's log, the blobs its messages
/// name, and the session's other files by name. Every method takes
/// `&self`, so one store is shared by every agent's log.
pub trait JournalStore: Send + Sync + 'static {
    /// The ids of the agents that have a log, in no particular order.
    fn agents(&self) -> io::Result<Vec<String>>;

    /// The whole log of `agent`.
    fn read(&self, agent: &str) -> io::Result<Vec<u8>>;

    /// Cuts the log of `agent` to its first `len` bytes, such as a torn
    /// last line a crash left.
    fn truncate(&self, agent: &str, len: u64) -> io::Result<()>;

    /// Appends `bytes` to the log of `agent`, which it starts when there is
    /// none.
    fn append(&self, agent: &str, bytes: &[u8]) -> io::Result<()>;

    /// Stores the blob `name`, unless it is stored already: a name is a
    /// hash of the content.
    fn put_blob(&self, name: &str, bytes: &[u8]) -> io::Result<()>;

    /// The blob `name`.
    fn blob(&self, name: &str) -> io::Result<Vec<u8>>;

    /// The names of the blobs stored, in no particular order. A store that
    /// cannot list them answers [`io::ErrorKind::Unsupported`], the
    /// default; a store that wraps another forwards it.
    fn blobs(&self) -> io::Result<Vec<String>> {
        Err(io::ErrorKind::Unsupported.into())
    }

    /// Removes the log of `agent`, if there is one, such as for an agent
    /// gone for good. A store that cannot answers
    /// [`io::ErrorKind::Unsupported`], the default; a store that wraps
    /// another forwards it.
    fn remove(&self, agent: &str) -> io::Result<()> {
        let _ = agent;
        Err(io::ErrorKind::Unsupported.into())
    }

    /// Replaces the session's file `name` with `bytes`, atomically: a
    /// reader sees the old content or the new. A name is the file's place
    /// in the session, such as `meta.json` or `tui/notices.jsonl`; a store
    /// refuses one [`check_file_name`] refuses, and may refuse names its
    /// own layout uses, with [`io::ErrorKind::InvalidInput`].
    fn write_file(&self, name: &str, bytes: &[u8]) -> io::Result<()>;

    /// Adds `bytes` at the end of the file `name`, which it creates when
    /// there is none.
    fn append_file(&self, name: &str, bytes: &[u8]) -> io::Result<()>;

    /// The file `name`, [`io::ErrorKind::NotFound`] when there is none.
    fn read_file(&self, name: &str) -> io::Result<Vec<u8>>;

    /// Removes the file `name`; removing one that is not there succeeds.
    fn remove_file(&self, name: &str) -> io::Result<()>;

    /// The names of the session's files, as [`Self::write_file`] takes
    /// them, in no particular order, such as to copy a session into
    /// another store. A store that cannot list them answers
    /// [`io::ErrorKind::Unsupported`], the default; a store that wraps
    /// another forwards it.
    fn files(&self) -> io::Result<Vec<String>> {
        Err(io::ErrorKind::Unsupported.into())
    }
}

/// Refuses `name` as a [`JournalStore`] file name unless it is relative and
/// `/`-separated, with no empty, `.` or `..` part and no `\`, so it names
/// a place inside the session wherever the store keeps it.
///
/// ```
/// use rig_cassette::journal::check_file_name;
///
/// assert!(check_file_name("tui/notices.jsonl").is_ok());
/// assert!(check_file_name("../elsewhere").is_err());
/// assert!(check_file_name("/etc/passwd").is_err());
/// ```
///
/// # Errors
///
/// [`io::ErrorKind::InvalidInput`], naming the file.
pub fn check_file_name(name: &str) -> io::Result<()> {
    let bad = |part: &str| matches!(part, "" | "." | "..") || part.contains('\\');
    if name.split('/').any(bad) {
        let why = format!("{name:?} is no file name in a session");
        return Err(io::Error::new(io::ErrorKind::InvalidInput, why));
    }
    Ok(())
}

/// The complete lines of a JSON-lines log, read from a store without
/// writing to it: a host's journal records, whatever their shape. A torn
/// tail a crash left (bytes after the last newline, or a last line that is
/// not JSON) is left out; a writer that goes on appending
/// [`truncate`](JournalStore::truncate)s the log to [`Self::len`] first.
///
/// ```
/// use rig_cassette::journal::{JournalStore, LogLines, MemoryStore};
///
/// let store = MemoryStore::default();
/// store.append("agent", b"{\"seq\":1}\n{\"seq\":2}\n{\"seq\":3,")?;
/// let log = LogLines::read(&store, "agent")?;
/// let lines: Vec<_> = log.iter().map(|(end, line)| (end, line.len())).collect();
/// assert_eq!(lines, [(10, 9), (20, 9)]);
/// assert!(log.torn() && log.len() == 20);
/// # Ok::<(), std::io::Error>(())
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LogLines {
    /// The log as stored, the torn tail included.
    bytes: Vec<u8>,
    /// Where each complete line, newline included, ends.
    ends: Vec<usize>,
}

impl LogLines {
    /// The log of `agent` in `store`.
    ///
    /// # Errors
    ///
    /// When the store cannot read it, such as [`io::ErrorKind::NotFound`]
    /// for an agent with no log.
    pub fn read(store: &dyn JournalStore, agent: &str) -> io::Result<Self> {
        Ok(Self::parse(store.read(agent)?))
    }

    /// The complete lines of `bytes`, a log as stored.
    pub fn parse(bytes: Vec<u8>) -> Self {
        let mut ends: Vec<usize> = bytes
            .iter()
            .enumerate()
            .filter(|(_, byte)| **byte == b'\n')
            .map(|(at, _)| at + 1)
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

    /// Each complete line that is not blank, without its newline, after the
    /// byte offset just past it: the length to keep the log up to and
    /// including that line.
    pub fn iter(&self) -> impl Iterator<Item = (u64, &[u8])> {
        let starts = std::iter::once(0).chain(self.ends.iter().copied());
        starts
            .zip(self.ends.iter().copied())
            .filter_map(|(start, end)| {
                let line = self.bytes.get(start..end.saturating_sub(1))?;
                (!line.is_empty()).then_some((end as u64, line))
            })
    }

    /// The bytes the complete lines cover.
    pub fn len(&self) -> u64 {
        self.ends.last().map_or(0, |end| *end as u64)
    }

    /// Whether the log has no complete line.
    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    /// Whether the stored log goes on past [`Self::len`]: a torn tail.
    pub fn torn(&self) -> bool {
        self.len() < self.bytes.len() as u64
    }

    /// The log's first `len` bytes, no further than its complete lines.
    pub fn prefix(&self, len: u64) -> &[u8] {
        let len = usize::try_from(len.min(self.len())).unwrap_or(0);
        self.bytes.get(..len).unwrap_or_default()
    }
}

/// Copies the session in `from` into `to`, an empty store, such as to fork
/// it: every blob, and every agent's log as far as `keep` says, given the
/// agent and its lines: the length to copy (a line's end from
/// [`LogLines::iter`], or [`LogLines::len`] for all of it), or `None` to
/// leave the agent out. A torn tail is never copied.
///
/// # Errors
///
/// When a store fails, including a `from` that cannot list its blobs.
pub fn fork(
    from: &dyn JournalStore,
    to: &dyn JournalStore,
    mut keep: impl FnMut(&str, &LogLines) -> Option<u64>,
) -> io::Result<()> {
    for agent in from.agents()? {
        let log = LogLines::read(from, &agent)?;
        if let Some(len) = keep(&agent, &log) {
            to.append(&agent, log.prefix(len))?;
        }
    }
    for name in from.blobs()? {
        to.put_blob(&name, &from.blob(&name)?)?;
    }
    Ok(())
}

/// How a logged message names an image stored as a blob, in place of its
/// data: `blob:<sha256>.<ext>`.
const BLOB: &str = "blob:";

/// `message` as it is logged: the data of each image in it stored in
/// `blobs` once per content, and named there instead. Borrowed when it has
/// no image data to store; data that is not valid base64 stays inline.
pub fn store_images<'m>(
    message: &'m Message,
    blobs: &dyn JournalStore,
) -> io::Result<Cow<'m, Message>> {
    let inline = |image: &Image| matches!(image.data, Base64(_) | Raw(_));
    if !message.images().any(inline) {
        return Ok(Cow::Borrowed(message));
    }
    let mut message = message.clone();
    for image in message.images_mut() {
        let bytes = match &image.data {
            Base64(data) => STANDARD.decode(data).ok(),
            Raw(bytes) => Some(bytes.clone()),
            _ => None,
        };
        if let Some(bytes) = bytes {
            let extension = image.media_type.as_ref();
            let extension = extension.map_or("bin", ImageMediaType::extension);
            let name = format!("{:x}.{extension}", Sha256::digest(&bytes));
            blobs.put_blob(&name, &bytes)?;
            image.data = Url(format!("{BLOB}{name}"));
        }
    }
    Ok(Cow::Owned(message))
}

/// Puts the data of each image `message` names in `blobs` back in place.
/// An image whose blob cannot be read has no data any more ([`Unknown`]),
/// which a request sends as a placeholder; the first such failure is
/// returned.
pub fn load_images(message: &mut Message, blobs: &dyn JournalStore) -> io::Result<()> {
    let mut gone = Ok(());
    for image in message.images_mut() {
        if let Url(url) = &image.data
            && let Some(name) = url.strip_prefix(BLOB)
        {
            image.data = match blobs.blob(name) {
                Ok(bytes) => Base64(STANDARD.encode(bytes)),
                Err(failure) => {
                    gone = gone.and(Err(failure));
                    Unknown
                }
            };
        }
    }
    gone
}

/// A session kept in memory, such as for tests or where there is no file
/// system. Cloning shares the contents.
#[derive(Clone, Default)]
pub struct MemoryStore(Arc<Mutex<Memory>>);

#[derive(Default)]
struct Memory {
    logs: BTreeMap<String, Vec<u8>>,
    blobs: BTreeMap<String, Vec<u8>>,
    files: BTreeMap<String, Vec<u8>>,
}

impl MemoryStore {
    fn memory(&self) -> MutexGuard<'_, Memory> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl JournalStore for MemoryStore {
    fn agents(&self) -> io::Result<Vec<String>> {
        Ok(self.memory().logs.keys().cloned().collect())
    }

    fn read(&self, agent: &str) -> io::Result<Vec<u8>> {
        self.memory()
            .logs
            .get(agent)
            .cloned()
            .ok_or_else(|| io::ErrorKind::NotFound.into())
    }

    fn truncate(&self, agent: &str, len: u64) -> io::Result<()> {
        let len = usize::try_from(len).map_err(io::Error::other)?;
        if let Some(log) = self.memory().logs.get_mut(agent) {
            log.truncate(len);
        }
        Ok(())
    }

    fn append(&self, agent: &str, bytes: &[u8]) -> io::Result<()> {
        self.memory()
            .logs
            .entry(agent.to_owned())
            .or_default()
            .extend_from_slice(bytes);
        Ok(())
    }

    fn put_blob(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        self.memory()
            .blobs
            .entry(name.to_owned())
            .or_insert_with(|| bytes.to_vec());
        Ok(())
    }

    fn blob(&self, name: &str) -> io::Result<Vec<u8>> {
        self.memory()
            .blobs
            .get(name)
            .cloned()
            .ok_or_else(|| io::ErrorKind::NotFound.into())
    }

    fn blobs(&self) -> io::Result<Vec<String>> {
        Ok(self.memory().blobs.keys().cloned().collect())
    }

    fn remove(&self, agent: &str) -> io::Result<()> {
        self.memory().logs.remove(agent);
        Ok(())
    }

    fn write_file(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        check_file_name(name)?;
        self.memory().files.insert(name.to_owned(), bytes.to_vec());
        Ok(())
    }

    fn append_file(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        check_file_name(name)?;
        let mut memory = self.memory();
        let file = memory.files.entry(name.to_owned()).or_default();
        file.extend_from_slice(bytes);
        Ok(())
    }

    fn read_file(&self, name: &str) -> io::Result<Vec<u8>> {
        check_file_name(name)?;
        self.memory()
            .files
            .get(name)
            .cloned()
            .ok_or_else(|| io::ErrorKind::NotFound.into())
    }

    fn remove_file(&self, name: &str) -> io::Result<()> {
        check_file_name(name)?;
        self.memory().files.remove(name);
        Ok(())
    }

    fn files(&self) -> io::Result<Vec<String>> {
        Ok(self.memory().files.keys().cloned().collect())
    }
}

#[cfg(test)]
mod tests;
