//! Reading a session back, saving to it from data, and its history:
//! [`Session::read`] folds every log, strictly, without writing anything,
//! and [`Session::read_lenient`] every log that reads; [`Session::save`],
//! [`load`] and [`save`] are for programs that keep a conversation over
//! time without a running [`Writer`](crate::Writer); [`history`] and
//! [`as_of`] read one log's records and its conversation as of a record.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io;
use std::ops::ControlFlow;

use rig_core::conversation::{AsideId, Author, Change, Conversation, MessageId, Recorded};
use serde_json::Value;

use crate::blobs::{load_images, load_images_with, messages_mut};
use crate::format::{Body, Header, LogLines, Meta, SESSION, parse, stamp_of};
use crate::writer::Book;
use crate::{FORMAT, LoadError, Repair, Rewrote, Stamp, Store};

/// A session as its logs left it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Session {
    /// Its agents, in the order their logs were started.
    pub agents: Vec<SavedAgent>,
    /// The session's own saved values by key: the [`SESSION`] log's.
    pub state: BTreeMap<String, Value>,
    /// What reading it repaired or worked around.
    pub repairs: Vec<Repair>,
}

/// An agent as its log left it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SavedAgent {
    /// Its id, the name of its log.
    pub id: String,
    /// The id of the agent that spawned it, if any.
    pub parent: Option<String>,
    /// When its log was started, in milliseconds since the Unix epoch.
    pub started_ms: u64,
    /// Its conversation, continuing its log: its lineage the log's, its
    /// revision the number of changes the log holds.
    pub conversation: Conversation,
    /// The newest value of each of its saved values, by key.
    pub state: BTreeMap<String, Value>,
    /// Whether it is gone for good: a restore skips it, and the agents it
    /// spawned.
    pub retired: bool,
    /// Who rewrote its log in place, and when, oldest first.
    pub rewrites: Vec<Rewrote>,
}

/// A log [`Session::read_lenient`] could not read.
#[derive(Debug)]
pub struct Unreadable {
    /// Its name: an agent's id, or [`SESSION`].
    pub log: String,
    /// Why it could not be read.
    pub error: LoadError,
}

/// One record of a log, as [`history`] reads it.
#[derive(Clone, Debug, PartialEq)]
pub enum Record {
    /// A change of the conversation, its `at` the time it was written.
    Change(Recorded),
    /// A saved value set, or with `None` removed.
    State {
        /// Its key.
        key: String,
        /// Its value.
        value: Option<Value>,
    },
    /// The agent is gone for good.
    Retired,
}

/// A log read and folded.
pub(crate) struct Folded {
    pub name: String,
    pub header: Header,
    pub started_ms: u64,
    pub conversation: Conversation,
    /// The record that made each message.
    pub made: HashMap<MessageId, Stamp>,
    /// The last record that put, replaced or removed each aside (a reset
    /// counts for every aside before and after it), and who made it.
    pub asides_set: HashMap<AsideId, (Stamp, Option<Author>)>,
    pub state: BTreeMap<String, Value>,
    pub retired: bool,
    /// The `seq` after the last record's.
    pub next_seq: u64,
    /// The session's `n` after the last record's.
    pub next_n: u64,
    /// The bytes read and the bytes of a torn tail after them.
    pub torn: Option<(u64, u64)>,
    /// Why images of it are lost.
    pub lost: Vec<String>,
}

/// A log read strictly: its header, and where it ends.
struct Read {
    header: Header,
    started: Stamp,
    last: Stamp,
}

/// The log `name` of `store` as stored; `None` when there is none.
pub(crate) fn log_lines(store: &dyn Store, name: &str) -> Result<Option<LogLines>, LoadError> {
    match store.read_log(name) {
        Err(failure) if failure.kind() == io::ErrorKind::NotFound => Ok(None),
        bytes => Ok(Some(LogLines::parse(bytes?))),
    }
}

/// Reads the log `name` of `store` strictly, as [`read_lines`] does.
fn read(
    store: &dyn Store,
    name: &str,
    each: impl FnMut(Stamp, Body) -> Result<ControlFlow<()>, String>,
) -> Result<Option<Read>, LoadError> {
    match log_lines(store, name)? {
        Some(lines) => read_lines(name, &lines, each),
        None => Ok(None),
    }
}

/// Reads `lines`, the log `name`, strictly, giving each record after the
/// header to `each`, which may stop the reading early. `None` when it
/// holds no complete line.
fn read_lines(
    name: &str,
    lines: &LogLines,
    mut each: impl FnMut(Stamp, Body) -> Result<ControlFlow<()>, String>,
) -> Result<Option<Read>, LoadError> {
    let mut records = lines.iter();
    let Some((_, _, first)) = records.next() else {
        return Ok(None);
    };
    let refuse = |line, why| LoadError::Record {
        log: name.to_owned(),
        line,
        why,
    };
    let (started, header) = read_header(name, first)?;
    let mut last = started;
    for (line, _, bytes) in records {
        let (stamp, body) = parse(bytes).map_err(|why| refuse(line, why))?;
        if matches!(body, Body::Meta(Meta::Header(_))) {
            return Err(refuse(line, "a second header".to_owned()));
        }
        if each(stamp, body)
            .map_err(|why| refuse(line, why))?
            .is_break()
        {
            break;
        }
        last = stamp;
    }
    Ok(Some(Read {
        header,
        started,
        last,
    }))
}

/// The header `first`, the first line of the log `name`, read strictly:
/// of this format, with the log's lineage.
pub(crate) fn read_header(name: &str, first: &[u8]) -> Result<(Stamp, Header), LoadError> {
    let (started, header) = match parse(first) {
        Ok((stamp, Body::Meta(Meta::Header(header)))) => (stamp, header),
        _ => {
            let found = serde_json::from_slice::<Header>(first)
                .ok()
                .and_then(|header| header.format);
            return Err(LoadError::Format {
                log: name.to_owned(),
                found,
            });
        }
    };
    if header.format.as_deref() != Some(FORMAT) {
        let found = header.format;
        let log = name.to_owned();
        return Err(LoadError::Format { log, found });
    }
    if !header.lineage.is_log() {
        return Err(LoadError::Record {
            log: name.to_owned(),
            line: 1,
            why: "a header without the log's lineage".to_owned(),
        });
    }
    Ok((started, header))
}

/// How a log is folded.
#[derive(Clone, Copy, Default)]
pub(crate) struct Fold {
    /// Only the records made before the session's record `n`.
    pub until: Option<u64>,
    /// Whether images are put back from their blobs.
    pub images: bool,
}

/// Folds the log `name` of `store`: every change applied in order, the
/// newest of each saved value, whether it was retired. `None` when there
/// is no log.
pub(crate) fn fold(store: &dyn Store, name: &str, how: Fold) -> Result<Option<Folded>, LoadError> {
    match log_lines(store, name)? {
        Some(lines) => fold_lines(name, &lines, how, &|blob| store.blob(blob)),
        None => Ok(None),
    }
}

/// Folds `lines`, the log `name`, as [`fold`] does, putting images back
/// from the blobs `blob` reads. `None` when it holds no complete line.
pub(crate) fn fold_lines(
    name: &str,
    lines: &LogLines,
    how: Fold,
    blob: &dyn Fn(&str) -> io::Result<Vec<u8>>,
) -> Result<Option<Folded>, LoadError> {
    let mut conversation = Conversation::new();
    let mut made = HashMap::new();
    let mut asides_set = HashMap::new();
    let mut state = BTreeMap::new();
    let mut retired = false;
    let mut lost = Vec::new();
    let read = read_lines(name, lines, |stamp, body| {
        if how.until.is_some_and(|until| stamp.n >= until) {
            return Ok(ControlFlow::Break(()));
        }
        match body {
            Body::Meta(Meta::State { key, value }) => match value.filter(|v| !v.is_null()) {
                Some(value) => state.insert(key, value),
                None => state.remove(&key),
            },
            Body::Meta(Meta::Retired) => {
                retired = true;
                None
            }
            Body::Meta(Meta::Header(_)) => None,
            Body::Change(mut recorded) => {
                if how.images {
                    let loaded = messages_mut(&mut recorded.change, &mut |message| {
                        if let Err(failure) = load_images_with(message, blob) {
                            lost.push(failure.to_string());
                        }
                        Ok(())
                    });
                    if let Err(failure) = loaded {
                        lost.push(failure.to_string());
                    }
                }
                let set = |id| (id, (stamp, recorded.by.clone()));
                match &recorded.change {
                    Change::Aside { id, .. } => asides_set.extend([set(*id)]),
                    Change::Reset { state } => {
                        let ids = conversation.asides().chain(state.asides());
                        asides_set.extend(ids.map(|(id, _)| set(id)));
                    }
                    _ => {}
                }
                conversation
                    .apply(&recorded)
                    .map_err(|error| error.to_string())?;
                for id in made_by(&recorded.change, &conversation) {
                    made.entry(id).or_insert(stamp);
                }
                None
            }
        };
        Ok(ControlFlow::Continue(()))
    })?;
    let Some(read) = read else {
        return Ok(None);
    };
    let changes = conversation.revision();
    conversation.continue_log(read.header.lineage, changes);
    let tail = lines.tail().len() as u64;
    Ok(Some(Folded {
        name: name.to_owned(),
        started_ms: read.started.t,
        header: read.header,
        conversation,
        made,
        asides_set,
        state,
        retired,
        next_seq: read.last.seq.saturating_add(1),
        next_n: read.last.n.saturating_add(1),
        torn: (tail > 0).then(|| (lines.len(), tail)),
        lost,
    }))
}

/// The ids a change put in, of the entries it may have made.
fn made_by(change: &Change, conversation: &Conversation) -> Vec<MessageId> {
    match change {
        Change::Push { entries } | Change::Splice { with: entries, .. } => {
            entries.iter().filter_map(|entry| entry.id).collect()
        }
        Change::Reset { .. } => conversation.entries().map(|entry| entry.id).collect(),
        Change::Halt { .. } | Change::Aside { .. } => Vec::new(),
    }
}

/// Every log of `store`, folded, with images.
pub(crate) fn fold_all(store: &dyn Store, images: bool) -> Result<Vec<Folded>, LoadError> {
    let how = Fold {
        until: None,
        images,
    };
    let mut folded = Vec::new();
    for name in store.logs()? {
        folded.extend(fold(store, &name, how)?);
    }
    Ok(folded)
}

/// What reading `folded`, the logs of `store` that hold a record, repaired,
/// torn tails not set aside. A log that holds no complete line but a torn
/// one, such as one whose first write was cut short, is torn whole.
pub(crate) fn repairs(store: &dyn Store, folded: &[Folded]) -> Result<Vec<Repair>, LoadError> {
    let mut repairs = Vec::new();
    for log in store.logs()? {
        if folded.iter().any(|folded| folded.name == log) {
            continue;
        }
        // A log that cannot be read is refused or set aside by its fold.
        let lines = log_lines(store, &log).ok().flatten();
        let Some(lines) = lines.filter(|lines| lines.len() == 0) else {
            continue;
        };
        let cut = lines.tail().len() as u64;
        if cut > 0 {
            repairs.push(Repair::TornTail {
                log,
                kept: 0,
                cut,
                set_aside: None,
            });
        }
    }
    for log in folded {
        if let Some((kept, cut)) = log.torn {
            repairs.push(Repair::TornTail {
                log: log.name.clone(),
                kept,
                cut,
                set_aside: None,
            });
        }
        repairs.extend(log.lost.iter().map(|why| Repair::LostImage {
            log: log.name.clone(),
            why: why.clone(),
        }));
    }
    Ok(repairs)
}

/// The names of the logs a restore skips: those retired, and the agents
/// they spawned, all the way down.
pub(crate) fn retired(logs: &[Folded]) -> HashSet<String> {
    let logs = logs.iter().map(|log| {
        let parent = log.header.parent.as_deref();
        (log.name.as_str(), parent, log.retired)
    });
    skipped(logs).into_iter().map(str::to_owned).collect()
}

/// Of `logs`, each a log's name, its agent's parent and whether it was
/// retired: those retired, and the agents they spawned, all the way down.
pub(crate) fn skipped<'a>(
    logs: impl Iterator<Item = (&'a str, Option<&'a str>, bool)>,
) -> HashSet<&'a str> {
    let logs: Vec<_> = logs.collect();
    let parents: HashMap<&str, Option<&str>> = logs
        .iter()
        .map(|&(name, parent, _)| (name, parent))
        .collect();
    let gone: HashSet<&str> = logs
        .iter()
        .filter(|&&(_, _, retired)| retired)
        .map(|&(name, ..)| name)
        .collect();
    let mut skipped = HashSet::new();
    for &(name, ..) in &logs {
        // A loop cannot happen, but a bound costs nothing.
        let up = std::iter::successors(Some(name), |id| parents.get(id).copied().flatten());
        if up.take(logs.len() + 1).any(|id| gone.contains(id)) {
            skipped.insert(name);
        }
    }
    skipped
}

/// The session of `folded`, with `repairs`.
pub(crate) fn session_of(folded: Vec<Folded>, repairs: Vec<Repair>) -> Session {
    let mut session = Session {
        repairs,
        ..Session::default()
    };
    for log in folded {
        if log.name == SESSION {
            session.state = log.state;
            continue;
        }
        session.agents.push(SavedAgent {
            id: log.name,
            parent: log.header.parent,
            started_ms: log.started_ms,
            conversation: log.conversation,
            state: log.state,
            retired: log.retired,
            rewrites: log.header.rewrites,
        });
    }
    session
        .agents
        .sort_by(|a, b| (a.started_ms, &a.id).cmp(&(b.started_ms, &b.id)));
    session
}

impl Session {
    /// Reads the session kept in `store`, without writing to it: a torn
    /// last line is reported and left. Retired agents are read too.
    ///
    /// # Errors
    ///
    /// When the store fails, or any log is in another format or holds a
    /// record that cannot be read or does not apply: the whole session is
    /// refused.
    pub fn read(store: &dyn Store) -> Result<Self, LoadError> {
        let folded = fold_all(store, true)?;
        let repairs = repairs(store, &folded)?;
        Ok(session_of(folded, repairs))
    }

    /// Reads each log of the session kept in `store` that reads, as
    /// [`Self::read`] does, and sets aside each that does not, with why:
    /// for a reader that shows what it can, such as an export or a search.
    /// A restore reads strictly, so it never goes on from part of a
    /// session.
    ///
    /// # Errors
    ///
    /// When the store cannot list its logs.
    pub fn read_lenient(store: &dyn Store) -> Result<(Self, Vec<Unreadable>), LoadError> {
        let how = Fold {
            until: None,
            images: true,
        };
        let mut folded = Vec::new();
        let mut unreadable = Vec::new();
        for log in store.logs()? {
            match fold(store, &log, how) {
                Ok(read) => folded.extend(read),
                Err(error) => unreadable.push(Unreadable { log, error }),
            }
        }
        let repairs = repairs(store, &folded)?;
        Ok((session_of(folded, repairs), unreadable))
    }

    /// The agent `id`.
    pub fn agent(&self, id: &str) -> Option<&SavedAgent> {
        self.agents.iter().find(|agent| agent.id == id)
    }

    /// The agent `id`, to change it.
    pub fn agent_mut(&mut self, id: &str) -> Option<&mut SavedAgent> {
        self.agents.iter_mut().find(|agent| agent.id == id)
    }

    /// Saves the session to `store`, the one it was read from (or an empty
    /// one): appends each agent's unsaved changes (an agent new to the
    /// store gets a header), its changed and removed saved values, and its
    /// retirement, and the session's changed values. Each agent's
    /// conversation is left with nothing unsaved.
    ///
    /// # Errors
    ///
    /// [`LoadError::Changed`] when a log that holds changes moved on since
    /// the agent's conversation was read, or holds another conversation,
    /// before anything is written; otherwise as
    /// [`Self::read`], or when a write fails.
    pub fn save(&mut self, store: &dyn Store) -> Result<(), LoadError> {
        let folded = fold_all(store, false)?;
        let had = |id: &str| folded.iter().find(|log| log.name == id);
        for agent in &self.agents {
            if agent.id == SESSION {
                let why = format!("{SESSION:?} is the session's own log, not an agent's");
                return Err(io::Error::new(io::ErrorKind::InvalidInput, why).into());
            }
            check_continues(had(&agent.id), &agent.id, &agent.conversation)?;
        }
        let mut book = Book::after(&folded);
        book.start();
        for agent in &mut self.agents {
            book.open_log(&agent.id, agent.parent.as_deref());
            book.save(store, &agent.id, &mut agent.conversation);
            let known = had(&agent.id).map(|log| &log.state);
            book.states(&agent.id, known, &agent.state);
            if agent.retired && !had(&agent.id).is_some_and(|log| log.retired) {
                book.retire(&agent.id);
            }
        }
        let known = had(SESSION).map(|log| &log.state);
        book.states(SESSION, known, &self.state);
        book.flush(store)
    }
}

/// Refuses to save `conversation` to the log `name`, now `log`, unless
/// its unsaved changes continue that log or the log holds no change, so
/// a reset to it loses nothing.
fn check_continues(
    log: Option<&Folded>,
    name: &str,
    conversation: &Conversation,
) -> Result<(), LoadError> {
    let Some(log) = log else {
        return Ok(());
    };
    let changes = log.conversation.revision();
    if changes > 0 && !conversation.continues(log.header.lineage, changes) {
        return Err(LoadError::Changed {
            log: name.to_owned(),
        });
    }
    Ok(())
}

/// The conversation the log `log` of `store` holds; an empty one when
/// there is none. It continues the log (its lineage and its number of
/// changes), so [`save`] can tell whether the log moved on or the
/// conversation is another.
///
/// # Errors
///
/// As [`Session::read`], for that log.
pub fn load(store: &dyn Store, log: &str) -> Result<Conversation, LoadError> {
    let how = Fold {
        until: None,
        images: true,
    };
    Ok(fold(store, log, how)?.map_or_else(Conversation::new, |log| log.conversation))
}

/// Appends the unsaved changes of `conversation` to the log `log` of
/// `store`, starting it when there is none, and leaves the conversation
/// with nothing unsaved.
///
/// ```
/// use rig_core::message::Message;
/// use rig_memory::{MemoryStore, load, save};
///
/// let store = MemoryStore::default();
/// let mut conversation = load(&store, "user-123")?;
/// conversation.push(Message::user("My name is Alice."));
/// save(&store, "user-123", &mut conversation)?;
/// assert_eq!(load(&store, "user-123")?, conversation);
/// # Ok::<(), rig_memory::LoadError>(())
/// ```
///
/// # Errors
///
/// [`LoadError::Changed`] when `conversation` does not continue the log
/// (the log moved on since it was loaded, or it was loaded from another
/// log or built elsewhere) and the log holds changes, writing nothing, so
/// no change made there is lost; otherwise as [`load`], or when a write
/// fails. Onto an empty log any conversation is saved, whole when it was
/// not begun from nothing.
pub fn save(
    store: &dyn Store,
    log: &str,
    conversation: &mut Conversation,
) -> Result<(), LoadError> {
    let folded = fold(store, log, Fold::default())?;
    check_continues(folded.as_ref(), log, conversation)?;
    let mut book = Book::after(folded.as_slice());
    book.raise_n(next_n(store)?);
    book.start();
    book.open_log(log, None);
    book.save(store, log, conversation);
    book.flush(store)
}

/// The session's `n` after the last record of every log of `store`.
fn next_n(store: &dyn Store) -> Result<u64, LoadError> {
    let mut next = 0;
    for name in store.logs()? {
        let lines = LogLines::parse(store.read_log(&name)?);
        if let Some((_, _, last)) = lines.iter().last()
            && let Ok((stamp, _)) = stamp_of(last)
        {
            next = next.max(stamp.n.saturating_add(1));
        }
    }
    Ok(next)
}

/// Every record of the log `log` of `store` after its header, in order,
/// with its stamp; images are put back from their blobs. Empty when there
/// is no log.
///
/// # Errors
///
/// As [`load`].
pub fn history(store: &dyn Store, log: &str) -> Result<Vec<(Stamp, Record)>, LoadError> {
    let mut records = Vec::new();
    read(store, log, |stamp, body| {
        let record = match body {
            Body::Change(mut recorded) => {
                // A lost image is read without data, as by a restore.
                let _ = messages_mut(&mut recorded.change, &mut |message| {
                    let _ = load_images(message, store);
                    Ok(())
                });
                Record::Change(recorded)
            }
            Body::Meta(Meta::State { key, value }) => Record::State {
                key,
                value: value.filter(|value| !value.is_null()),
            },
            Body::Meta(Meta::Retired) => Record::Retired,
            Body::Meta(Meta::Header(_)) => return Ok(ControlFlow::Continue(())),
        };
        records.push((stamp, record));
        Ok(ControlFlow::Continue(()))
    })?;
    Ok(records)
}

/// The conversation of the log `log` of `store` as the records made before
/// the session's record `n` left it ([`Stamp::n`]); an empty one when
/// there is no log.
///
/// # Errors
///
/// As [`load`].
pub fn as_of(store: &dyn Store, log: &str, n: u64) -> Result<Conversation, LoadError> {
    let how = Fold {
        until: Some(n),
        images: true,
    };
    Ok(fold(store, log, how)?.map_or_else(Conversation::new, |log| log.conversation))
}
#[cfg(test)]
mod tests;
