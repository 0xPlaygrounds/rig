//! Reading a session back ([`Session`]), saving to it without a running
//! [`Writer`](crate::Writer) ([`load`], [`save`]), and one log's history
//! ([`history`], [`as_of`]).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io;
use std::ops::ControlFlow;

use rig_core::conversation::{ChangeRecord, Conversation};
use serde_json::Value;

use crate::blobs::{
    BLOBS, load_images, load_images_with, messages_mut, named_blobs, put_blob, read_blob,
};
use crate::format::{Body, Header, LogLines, Meta, SESSION, parse, stamp_of};
use crate::store::now;
use crate::writer::Book;
use crate::{FORMAT, LogRewrite, Repair, SessionError, Stamp, Store, check_log_name};

/// A session as its logs left it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Session {
    /// Its agents, in the order their logs were started.
    pub agents: Vec<SavedAgent>,
    /// The session's own saved values by key: the [`SESSION`] log's.
    pub saved: BTreeMap<String, Value>,
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
    pub saved: BTreeMap<String, Value>,
    /// Whether it is gone for good: a restore skips it, and the agents it
    /// spawned.
    pub retired: bool,
    /// Who rewrote its log in place, and when, oldest first.
    pub rewrites: Vec<LogRewrite>,
}

/// A log [`Session::read_lenient`] could not read.
#[derive(Debug)]
pub struct Unreadable {
    /// Its name: an agent's id, or [`SESSION`].
    pub log: String,
    /// Why it could not be read.
    pub error: SessionError,
}

/// One record of a log, as [`history`] reads it.
#[derive(Clone, Debug, PartialEq)]
pub enum LogRecord {
    /// A change of the conversation, its `at` the time it was written.
    Change(ChangeRecord),
    /// A saved value set, or with `None` removed.
    Saved {
        /// Its key.
        key: String,
        /// Its value.
        value: Option<Value>,
    },
    /// The agent is gone for good.
    Retired,
}

/// A log read and folded.
#[derive(Default)]
pub(crate) struct Folded {
    pub name: String,
    pub header: Header,
    pub started_ms: u64,
    pub conversation: Conversation,
    pub state: BTreeMap<String, Value>,
    pub retired: bool,
    /// The `seq` after the last record's.
    pub next_seq: u64,
    /// The session's `n` after the last record's.
    pub next_n: u64,
    /// The bytes of its complete lines.
    pub len: u64,
    /// The bytes read and the bytes of a torn tail after them.
    pub torn: Option<(u64, u64)>,
    /// Why images of it are lost.
    pub lost: Vec<io::Error>,
}

/// A log read strictly: its header, and where it ends.
struct Read {
    header: Header,
    started: Stamp,
    last: Stamp,
}

/// The log `name` of `store` as stored; `None` when there is none.
pub(crate) fn log_lines(store: &dyn Store, name: &str) -> Result<Option<LogLines>, SessionError> {
    match store.read_log(name) {
        Err(failure) if failure.kind() == io::ErrorKind::NotFound => Ok(None),
        bytes => Ok(Some(LogLines::parse(bytes?))),
    }
}

/// Every log of `store` as stored, in the store's order.
pub(crate) fn read_logs(store: &dyn Store) -> Result<Vec<(String, LogLines)>, SessionError> {
    let mut logs = Vec::new();
    for name in store.logs()? {
        logs.extend(log_lines(store, &name)?.map(|lines| (name, lines)));
    }
    Ok(logs)
}

/// Reads the log `name` of `store` strictly, as [`read_lines`] does.
fn read(
    store: &dyn Store,
    name: &str,
    each: impl FnMut(Stamp, Body) -> Result<ControlFlow<()>, String>,
) -> Result<Option<Read>, SessionError> {
    match log_lines(store, name)? {
        Some(lines) => read_lines(name, &lines, each),
        None => Ok(None),
    }
}

/// Reads `lines`, the log `name`, strictly, giving each record after the
/// header, in this format, to `each`, which may stop the reading early.
/// `None` when it holds no complete line.
fn read_lines(
    name: &str,
    lines: &LogLines,
    mut each: impl FnMut(Stamp, Body) -> Result<ControlFlow<()>, String>,
) -> Result<Option<Read>, SessionError> {
    let Some((_, _, first)) = lines.iter().next() else {
        return Ok(None);
    };
    let refuse = |line, why| SessionError::Record {
        log: name.to_owned(),
        line,
        why,
    };
    let (started, header) = read_header(name, first)?;
    let mut last = started;
    for (line, _, bytes) in lines.iter().skip(1) {
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

/// The header `first`, the first line of the log `name`, read strictly: of
/// [`FORMAT`], with the log's lineage.
pub(crate) fn read_header(name: &str, first: &[u8]) -> Result<(Stamp, Header), SessionError> {
    let (started, header) = match parse(first) {
        Ok((stamp, Body::Meta(Meta::Header(header)))) => (stamp, header),
        _ => {
            let found = serde_json::from_slice::<Header>(first)
                .ok()
                .and_then(|header| header.format);
            return Err(SessionError::Format {
                log: name.to_owned(),
                found,
            });
        }
    };
    if header.format.as_deref() != Some(FORMAT) {
        let found = header.format;
        let log = name.to_owned();
        return Err(SessionError::Format { log, found });
    }
    if !header.lineage.is_log() {
        return Err(SessionError::Record {
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

impl Fold {
    /// Every record, images put back.
    pub const WHOLE: Self = Self {
        until: None,
        images: true,
    };
}

/// Folds the log `name` of `store`: every change applied in order, the
/// newest of each saved value, whether it was retired. `None` when there
/// is no log.
pub(crate) fn fold(
    store: &dyn Store,
    name: &str,
    how: Fold,
) -> Result<Option<Folded>, SessionError> {
    match log_lines(store, name)? {
        Some(lines) => fold_lines(name, &lines, how, &|blob| read_blob(store, blob)),
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
) -> Result<Option<Folded>, SessionError> {
    fold_lines_with(name, lines, (how, blob), &mut |_, _| {})
}

/// [`fold_lines`], giving `each` every change with the conversation the
/// changes before it left.
fn fold_lines_with(
    name: &str,
    lines: &LogLines,
    (how, blob): (Fold, &dyn Fn(&str) -> io::Result<Vec<u8>>),
    each: &mut dyn FnMut(&Conversation, &ChangeRecord),
) -> Result<Option<Folded>, SessionError> {
    let mut conversation = Conversation::new();
    let mut state = BTreeMap::new();
    let mut retired = false;
    let mut lost = Vec::new();
    let read = read_lines(name, lines, |stamp, body| {
        if how.until.is_some_and(|until| stamp.n >= until) {
            return Ok(ControlFlow::Break(()));
        }
        match body {
            Body::Meta(meta) => note_meta(meta, &mut state, &mut retired),
            Body::Change(mut recorded) => {
                if how.images {
                    let _ = messages_mut(&mut recorded.change, &mut |message| {
                        if let Err(failure) = load_images_with(message, blob) {
                            lost.push(failure);
                        }
                        Ok(())
                    });
                }
                each(&conversation, &recorded);
                conversation
                    .apply(&recorded)
                    .map_err(|error| error.to_string())?;
            }
        }
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
        state,
        retired,
        next_seq: read.last.seq.saturating_add(1),
        next_n: read.last.n.saturating_add(1),
        len: lines.len(),
        torn: (tail > 0).then(|| (lines.len(), tail)),
        lost,
    }))
}

/// A saved value set or removed, or the log retired, noted into `state`
/// and `retired`.
pub(crate) fn note_meta(meta: Meta, state: &mut BTreeMap<String, Value>, retired: &mut bool) {
    match meta {
        Meta::State { key, value } => {
            match value.filter(|value| !value.is_null()) {
                Some(value) => state.insert(key, value),
                None => state.remove(&key),
            };
        }
        Meta::Retired => *retired = true,
        Meta::Header(_) => {}
    }
}

/// `lines`, the log `name`, read for saving to it as [`fold_lines`] reads
/// it, but its changes only counted by their stamps, not read or applied:
/// its conversation is empty, continuing the log; its saved values and
/// retirement are read. `None` when it holds no complete line.
fn tally(name: &str, lines: &LogLines) -> Result<Option<Folded>, SessionError> {
    let Some((_, _, first)) = lines.iter().next() else {
        return Ok(None);
    };
    let (started, header) = read_header(name, first)?;
    let (mut last, mut changes) = (started, 0);
    let (mut state, mut retired) = (BTreeMap::new(), false);
    for (line, _, bytes) in lines.iter().skip(1) {
        let refuse = |why| SessionError::Record {
            log: name.to_owned(),
            line,
            why,
        };
        let (stamp, change) = stamp_of(bytes).map_err(refuse)?;
        if !change && let (_, Body::Meta(meta)) = parse(bytes).map_err(refuse)? {
            note_meta(meta, &mut state, &mut retired);
        }
        last = stamp;
        changes += u64::from(change);
    }
    let mut conversation = Conversation::new();
    conversation.continue_log(header.lineage, changes);
    let tail = lines.tail().len() as u64;
    Ok(Some(Folded {
        name: name.to_owned(),
        started_ms: started.t,
        header,
        conversation,
        state,
        retired,
        next_seq: last.seq.saturating_add(1),
        next_n: last.n.saturating_add(1),
        len: lines.len(),
        torn: (tail > 0).then(|| (lines.len(), tail)),
        ..Folded::default()
    }))
}

/// Every log of `logs`, the session of `store`, folded, with images put
/// back from its blobs.
pub(crate) fn fold_all(
    store: &dyn Store,
    logs: &[(String, LogLines)],
) -> Result<Vec<Folded>, SessionError> {
    let mut folded = Vec::new();
    for (name, lines) in logs {
        folded.extend(fold_lines(name, lines, Fold::WHOLE, &|blob| {
            read_blob(store, blob)
        })?);
    }
    Ok(folded)
}

/// What reading `folded`, the logs of `store` that hold a record, repaired,
/// torn tails not set aside.
pub(crate) fn repairs(store: &dyn Store, folded: &[Folded]) -> Result<Vec<Repair>, SessionError> {
    let torn = torn_tails(store, folded)?.into_iter();
    let mut repairs: Vec<Repair> = torn
        .map(|(log, kept, cut)| Repair::TornTail {
            log,
            kept,
            cut,
            set_aside: None,
        })
        .collect();
    for log in folded {
        repairs.extend(log.lost.iter().map(|why| Repair::LostImage {
            log: log.name.clone(),
            why: why.to_string(),
        }));
    }
    Ok(repairs)
}

/// Each log of `store` that ends in a torn line, with the bytes before the
/// line and its bytes: each of `folded`, the logs that hold a record, that
/// does, and each log that holds no complete line but a torn one, such as
/// one whose first write was cut short.
pub(crate) fn torn_tails(
    store: &dyn Store,
    folded: &[Folded],
) -> Result<Vec<(String, u64, u64)>, SessionError> {
    let mut torn = Vec::new();
    for log in store.logs()? {
        match folded.iter().find(|folded| folded.name == log) {
            Some(folded) => torn.extend(folded.torn.map(|(kept, cut)| (log, kept, cut))),
            None => torn.extend(torn_whole(store, &log).map(|cut| (log, 0, cut))),
        }
    }
    Ok(torn)
}

/// The bytes of the torn line the log `log` of `store` holds, when it holds
/// no complete line but a torn one.
fn torn_whole(store: &dyn Store, log: &str) -> Option<u64> {
    // A log that cannot be read is refused or set aside by its fold.
    let lines = log_lines(store, log).ok().flatten();
    let lines = lines.filter(|lines| lines.len() == 0)?;
    let cut = lines.tail().len() as u64;
    (cut > 0).then_some(cut)
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
        // Parents come from the logs' headers; the bound stops a corrupt store's loop.
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
            session.saved = log.state;
            continue;
        }
        session.agents.push(SavedAgent {
            id: log.name,
            parent: log.header.parent,
            started_ms: log.started_ms,
            conversation: log.conversation,
            saved: log.state,
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
    pub fn read(store: &dyn Store) -> Result<Self, SessionError> {
        let folded = fold_all(store, &read_logs(store)?)?;
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
    pub fn read_lenient(store: &dyn Store) -> Result<(Self, Vec<Unreadable>), SessionError> {
        let mut folded = Vec::new();
        let mut unreadable = Vec::new();
        for log in store.logs()? {
            match fold(store, &log, Fold::WHOLE) {
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
    /// conversation is left with nothing unsaved after its log is written.
    /// If a write fails, uncommitted conversations keep their pending changes
    /// for a retry; logs already written are not appended again. A torn last line of a log
    /// written to is first set aside, as [`Writer::open`](crate::Writer::open)
    /// does.
    ///
    /// # Errors
    ///
    /// [`SessionError::Changed`] when a log that holds changes moved on since
    /// the agent's conversation was read, or holds another conversation,
    /// before anything is written; when the store fails, a log is in another
    /// format, or a line has no stamp; or when a write fails. It reads the
    /// stamps of the logs' changes, not the changes, so a change that does
    /// not apply is refused by [`Self::read`], not here.
    pub fn save(&mut self, store: &dyn Store) -> Result<(), SessionError> {
        let mut folded = Vec::new();
        for (name, lines) in &read_logs(store)? {
            folded.extend(tally(name, lines)?);
        }
        let had = |id: &str| folded.iter().find(|log| log.name == id);
        for agent in &self.agents {
            if agent.id == SESSION {
                let why = format!("{SESSION:?} is the session's own log, not an agent's");
                return Err(io::Error::new(io::ErrorKind::InvalidInput, why).into());
            }
            check_continues(had(&agent.id), &agent.id, &agent.conversation)?;
        }
        let mut book = Book::after(&folded);
        let torn = torn_tails(store, &folded)?.into_iter();
        book.set_aside(torn.map(|(log, _, cut)| (log, cut)));
        book.start();
        let mut staged = self.agents.clone();
        for agent in &mut staged {
            let _ = book.opened(&agent.id, agent.parent.as_deref());
            let put = &mut |name: String, bytes: Vec<u8>| put_blob(store, &name, &bytes);
            book.save(put, &agent.id, &mut agent.conversation);
            let known = had(&agent.id).map(|log| &log.state);
            book.states(&agent.id, known, &agent.saved);
            if agent.retired && !had(&agent.id).is_some_and(|log| log.retired) {
                book.retire(&agent.id);
            }
        }
        let known = had(SESSION).map(|log| &log.state);
        book.states(SESSION, known, &self.saved);
        let result = book.flush(store);
        for (agent, saved) in self.agents.iter_mut().zip(staged) {
            if result.is_ok() || book.flushed(&agent.id) {
                agent.conversation = saved.conversation;
            }
        }
        result
    }
}

/// Refuses to save `conversation` to the log `name`, now `log`, unless
/// its unsaved changes continue that log or the log holds no change, so
/// a reset to it loses nothing.
fn check_continues(
    log: Option<&Folded>,
    name: &str,
    conversation: &Conversation,
) -> Result<(), SessionError> {
    let Some(log) = log else {
        return Ok(());
    };
    let changes = log.conversation.revision();
    if changes > 0 && !conversation.continues(log.header.lineage, changes) {
        return Err(SessionError::Changed {
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
/// An image whose blob is gone loads with no data, as a restore reads it;
/// any other failure to read a blob fails the load, so a store that is
/// down for a moment never loses an image from what a run sends.
///
/// # Errors
///
/// As [`Session::read`], for that log; when `log` is no log name; or when
/// a blob it names cannot be read for a reason other than that it is gone.
pub fn load(store: &dyn Store, log: &str) -> Result<Conversation, SessionError> {
    check_log_name(log)?;
    conversation_of(fold(store, log, Fold::WHOLE)?)
}

/// The conversation of `folded`, an empty one for no log; refused when an
/// image of it could not be read for a reason other than that its blob is
/// gone.
fn conversation_of(folded: Option<Folded>) -> Result<Conversation, SessionError> {
    let Some(folded) = folded else {
        return Ok(Conversation::new());
    };
    let mut lost = folded.lost.into_iter();
    match lost.find(|failure| failure.kind() != io::ErrorKind::NotFound) {
        Some(failure) => Err(failure.into()),
        None => Ok(folded.conversation),
    }
}

/// Appends the unsaved changes of `conversation` to the log `log` of
/// `store`, starting it when there is none, and leaves the conversation
/// with nothing unsaved. A torn last line of the log is first set aside, as
/// [`Writer::open`](crate::Writer::open) does.
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
/// # Ok::<(), rig_memory::SessionError>(())
/// ```
///
/// # Errors
///
/// [`SessionError::Changed`] when `conversation` does not continue the log
/// (the log moved on since it was loaded, or it was loaded from another
/// log or built elsewhere) and the log holds changes, or when another
/// writer changed the log between this save's read and its append,
/// writing nothing to it, so no change made there is lost; when the store
/// fails, `log` is in another format, or a line of it has no stamp; or
/// when a write fails. It
/// reads the stamps of `log`'s records, not the records, so a change that
/// does not apply is refused by [`load`], not here. Onto an empty log any
/// conversation is saved, whole when it was not begun from nothing.
pub fn save(
    store: &dyn Store,
    log: &str,
    conversation: &mut Conversation,
) -> Result<(), SessionError> {
    now(save_async(store, log, conversation))
}

/// [`load`] from an [`AsyncStore`](crate::AsyncStore), such as a database
/// or IndexedDB: the log read once, and the blobs its records name.
///
/// # Errors
///
/// As [`load`].
pub async fn load_async<S: crate::AsyncStore + ?Sized>(
    store: &S,
    log: &str,
) -> Result<Conversation, SessionError> {
    check_log_name(log)?;
    let bytes = match store.read_log_async(log).await {
        Err(failure) if failure.kind() == io::ErrorKind::NotFound => {
            return Ok(Conversation::new());
        }
        bytes => bytes?,
    };
    let mut named = std::collections::BTreeSet::new();
    named_blobs(&bytes, &mut named);
    let mut blobs = BTreeMap::new();
    for name in named {
        let read = store.read_file_async(&format!("{BLOBS}{name}")).await;
        blobs.insert(name, read);
    }
    let blob = |name: &str| match blobs.get(name) {
        Some(Ok(bytes)) => Ok(bytes.clone()),
        Some(Err(failure)) => Err(io::Error::new(failure.kind(), failure.to_string())),
        None => Err(io::ErrorKind::NotFound.into()),
    };
    conversation_of(fold_lines(
        log,
        &LogLines::parse(bytes),
        Fold::WHOLE,
        &blob,
    )?)
}

/// [`save`] to an [`AsyncStore`](crate::AsyncStore): the log read once,
/// then the blobs of new images written, then the changes appended with
/// one write, which the store makes only if the log is still as it was
/// read. A save reads only its own log, so one store holds many
/// conversations, a log each. A save cut short, such as by a dropped
/// future, has appended nothing, a cut last line (set aside by the next
/// save), or, for a remote store that already applied it, all of it.
///
/// ```
/// # async fn run() -> Result<(), rig_memory::SessionError> {
/// use rig_core::message::Message;
/// use rig_memory::{MemoryStore, load_async, save_async};
///
/// let store = MemoryStore::default();
/// let mut conversation = load_async(&store, "user-123").await?;
/// conversation.push(Message::user("My name is Alice."));
/// save_async(&store, "user-123", &mut conversation).await?;
/// assert_eq!(load_async(&store, "user-123").await?, conversation);
/// # Ok(()) }
/// ```
///
/// # Errors
///
/// As [`save`]. Of two saves that read the log at the same length, even
/// at once, the store takes one and the other fails with
/// [`SessionError::Changed`]. A save that fails leaves `conversation`
/// with its unsaved changes, so it can be tried again.
pub async fn save_async<S: crate::AsyncStore + ?Sized>(
    store: &S,
    log: &str,
    conversation: &mut Conversation,
) -> Result<(), SessionError> {
    let lines = match store.read_log_async(log).await {
        Err(failure) if failure.kind() == io::ErrorKind::NotFound => LogLines::parse(Vec::new()),
        bytes => LogLines::parse(bytes?),
    };
    let folded = tally(log, &lines)?;
    check_continues(folded.as_ref(), log, conversation)?;
    let mut book = Book::after(folded.as_slice());
    // This also covers a log of no complete line but a torn one, such as one whose first write
    // was cut.
    let cut = lines.tail().len() as u64;
    book.set_aside((cut > 0).then(|| (log.to_owned(), cut)));
    let mut staged = conversation.clone();
    book.start();
    let _ = book.opened(log, None);
    let mut blobs = Vec::new();
    let put = &mut |name, bytes| {
        blobs.push((name, bytes));
        Ok(())
    };
    book.save(put, log, &mut staged);
    for (name, bytes) in blobs {
        store
            .write_file_async(&format!("{BLOBS}{name}"), &bytes)
            .await?;
    }
    book.flush_async(store).await?;
    *conversation = staged;
    Ok(())
}

/// Every record of the log `log` of `store` after its header, in order,
/// with its stamp; images are put back from their blobs. Empty when there
/// is no log.
///
/// # Errors
///
/// As [`load`].
pub fn history(store: &dyn Store, log: &str) -> Result<Vec<(Stamp, LogRecord)>, SessionError> {
    let mut records = Vec::new();
    read(store, log, |stamp, body| {
        let record = match body {
            Body::Change(mut recorded) => {
                // A lost image is read without data, as by a restore.
                let _ = messages_mut(&mut recorded.change, &mut |message| {
                    let _ = load_images(message, store);
                    Ok(())
                });
                LogRecord::Change(recorded)
            }
            Body::Meta(Meta::State { key, value }) => LogRecord::Saved {
                key,
                value: value.filter(|value| !value.is_null()),
            },
            Body::Meta(Meta::Retired) => LogRecord::Retired,
            Body::Meta(Meta::Header(_)) => return Ok(ControlFlow::Continue(())),
        };
        records.push((stamp, record));
        Ok(ControlFlow::Continue(()))
    })?;
    Ok(records)
}

/// Reads the log `log` of `store` once, as [`load`] does, giving `each`
/// every change with the conversation the changes before it left. Nothing
/// when there is no log.
///
/// # Errors
///
/// As [`load`].
pub(crate) fn replay(
    store: &dyn Store,
    log: &str,
    mut each: impl FnMut(&Conversation, &ChangeRecord),
) -> Result<(), SessionError> {
    if let Some(lines) = log_lines(store, log)? {
        let blob = |blob: &str| read_blob(store, blob);
        fold_lines_with(log, &lines, (Fold::WHOLE, &blob), &mut each)?;
    }
    Ok(())
}

/// The conversation of the log `log` of `store` as the records made before
/// the session's record `n` left it ([`Stamp::n`]); an empty one when
/// there is no log.
///
/// # Errors
///
/// As [`load`].
pub fn as_of(store: &dyn Store, log: &str, n: u64) -> Result<Conversation, SessionError> {
    let how = Fold {
        until: Some(n),
        ..Fold::WHOLE
    };
    conversation_of(fold(store, log, how)?)
}
#[cfg(test)]
mod tests;
