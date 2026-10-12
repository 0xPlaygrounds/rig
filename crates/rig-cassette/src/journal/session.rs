//! The session log format: one JSON-lines log per agent in a
//! [`JournalStore`], a header, then one record per change of the agent's
//! [`Conversation`] (a message, a splice, a halt, the agent retired) and
//! per change of a saved component, by type path. The session's own saved
//! values go in the [`SESSION`] log. Nothing is ever rewritten.
//!
//! A [`Writer`] queues records as a host makes changes and writes them on
//! [`flush`](Writer::flush), a subagent's before its parent's.
//! [`read_session`] folds every log back without writing anything,
//! [`write_session`] makes a store hold a session built from data, and
//! [`fork_session`] copies a session as it was when one of its messages was
//! made.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use rig_core::completion::Message;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use web_time::{SystemTime, UNIX_EPOCH};

use super::conversation::{Conversation, Entry, Halt, MessageId, Origin};
use super::{JournalStore, LogLines, fork, load_images, store_images};

/// The log of the session's saved values, beside the agents' logs.
pub const SESSION: &str = "session";

/// The first record of an agent log.
#[derive(Serialize, Deserialize, Clone, Debug)]
struct Header {
    /// The agent's id.
    agent: String,
    /// The id of the agent that spawned it, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parent: Option<String>,
}

/// One record of an agent log, tagged by `type`.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Record<'a> {
    /// The first record.
    Header(Header),
    /// A message added to the conversation; a user message after a user
    /// message goes into it.
    Message {
        message: Cow<'a, Message>,
        /// Where it came from, when it was not the user's plain text, the
        /// model's reply or tool results.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        origin: Option<Origin>,
    },
    /// Messages `from..to` of the conversation as the log folded it so far
    /// are now `with`.
    Splice {
        from: usize,
        to: usize,
        with: Vec<Entry>,
    },
    /// The user's last message was left unanswered, which a restore keeps,
    /// or, with no reason, asked for again.
    Halt { reason: Option<Halt> },
    /// A saved component, by type path; `value: null` when it was
    /// removed. The latest per type wins.
    Component {
        component: String,
        value: Option<Value>,
    },
    /// The agent is gone for good: a restore skips it, and the agents it
    /// spawned.
    Retired,
}

/// A line of an agent log.
#[derive(Serialize, Deserialize, Clone, Debug)]
struct Line<'a> {
    /// Dense from 0, the header's, in each log.
    seq: u64,
    /// Its place among the records of every log of the session, in the
    /// order they were made: what a fork cuts the other logs by.
    n: u64,
    /// When it was written, in milliseconds since the Unix epoch.
    t: u64,
    #[serde(flatten)]
    record: Record<'a>,
}

/// Milliseconds since the Unix epoch, also on the web.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or_default()
}

/// One agent's log: what is queued for it and the newest value of each
/// saved component, so an unchanged value is not logged again.
#[derive(Default)]
struct AgentLog {
    /// 0 for an agent the user started, 1 for its subagents, and so on.
    depth: usize,
    next_seq: u64,
    /// Whether the log holds a message or a saved change, so it is
    /// written.
    started: bool,
    pending: Vec<u8>,
    components: HashMap<String, Value>,
}

/// The records of every log, before and after they are written.
#[derive(Default)]
struct Book {
    store: Option<Arc<dyn JournalStore>>,
    /// The next record's [`Line::n`].
    next_n: u64,
    /// Records are queued only once started.
    live: bool,
    /// Why the log stopped: every record after a failed write is dropped.
    failure: Option<String>,
    agents: HashMap<String, AgentLog>,
}

impl Book {
    /// Goes on after the logs `folded`, each as it ends, the session's
    /// next record numbered `next_n` or later.
    fn after(&mut self, folded: &[Folded], next_n: u64) {
        let parents: HashMap<&str, Option<&str>> = folded
            .iter()
            .map(|log| (log.header.agent.as_str(), log.header.parent.as_deref()))
            .collect();
        for log in folded {
            // A loop cannot happen, but a bound costs nothing.
            let ancestors = std::iter::successors(log.header.parent.as_deref(), |parent| {
                parents.get(parent).copied().flatten()
            });
            let restored = ancestors.take_while(|parent| parents.contains_key(parent));
            let agent = AgentLog {
                depth: restored.take(folded.len()).count(),
                next_seq: log.next_seq,
                started: true,
                components: log.components.clone(),
                ..AgentLog::default()
            };
            self.agents.insert(log.header.agent.clone(), agent);
        }
        self.next_n = self.next_n.max(next_n);
    }

    /// The log of `agent`, started when it has none with a header naming
    /// `parent`, the agent that spawned it.
    fn agent(&mut self, agent: &str, parent: Option<&str>) -> Option<&mut AgentLog> {
        if !self.live || self.failure.is_some() {
            return None;
        }
        if !self.agents.contains_key(agent) {
            let parent_depth = |parent| self.agents.get(parent).map_or(0, |log| log.depth);
            let mut log = AgentLog {
                depth: parent.map_or(0, |parent| parent_depth(parent) + 1),
                ..AgentLog::default()
            };
            let header = Header {
                agent: agent.to_owned(),
                parent: parent.map(str::to_owned),
            };
            if let Err(failure) = enqueue(&mut log, self.next_n, Record::Header(header)) {
                self.failure = Some(failure.to_string());
                return None;
            }
            self.next_n += 1;
            self.agents.insert(agent.to_owned(), log);
        }
        self.agents.get_mut(agent)
    }

    /// Queues `record` for `agent`; its `seq`, or `None` when nothing is
    /// logged.
    fn record(&mut self, agent: &str, record: Record<'_>) -> Option<u64> {
        self.agent(agent, None)?;
        let n = self.next_n;
        let queued = enqueue(self.agents.get_mut(agent)?, n, record);
        match queued {
            Ok(seq) => {
                self.next_n += 1;
                Some(seq)
            }
            Err(failure) => {
                self.failure = Some(failure.to_string());
                None
            }
        }
    }

    /// Queues the record `make` builds of `messages`, their images stored
    /// as blobs in `store`, for `agent`, which starts its log.
    fn messages<'m>(
        &mut self,
        store: &dyn JournalStore,
        agent: &str,
        messages: impl IntoIterator<Item = &'m Message>,
        make: impl FnOnce(Vec<Cow<'m, Message>>) -> Record<'m>,
    ) {
        if !self.live || self.failure.is_some() {
            return;
        }
        let stored: Result<Vec<_>, _> = messages
            .into_iter()
            .map(|message| store_images(message, store))
            .collect();
        match stored {
            Ok(stored) => {
                if self.record(agent, make(stored)).is_some()
                    && let Some(log) = self.agents.get_mut(agent)
                {
                    log.started = true;
                }
            }
            Err(failure) => self.failure = Some(format!("storing an image failed: {failure}")),
        }
    }

    /// Queues the saved component `component` of the log `agent` when its
    /// value changed; `None` or `null` when it was removed. With `starts`,
    /// the change starts the agent's log.
    fn component(&mut self, agent: &str, component: &str, value: Option<Value>, starts: bool) {
        let value = value.filter(|value| !value.is_null());
        let known = self
            .agents
            .get(agent)
            .and_then(|log| log.components.get(component));
        if known == value.as_ref() {
            return;
        }
        let record = Record::Component {
            component: component.to_owned(),
            value: value.clone(),
        };
        if self.record(agent, record).is_some()
            && let Some(log) = self.agents.get_mut(agent)
        {
            log.started |= starts;
            match value {
                Some(value) => log.components.insert(component.to_owned(), value),
                None => log.components.remove(component),
            };
        }
    }

    /// Writes every queued record to `store`: each started log's records
    /// with one write, a subagent's log before its parent's, and the
    /// [`SESSION`] log once any is started. A failed write stops the
    /// logging.
    fn flush(&mut self, store: &dyn JournalStore) {
        if self.failure.is_some() {
            return;
        }
        let any = self.agents.values().any(|log| log.started);
        let mut due: Vec<(usize, String)> = self
            .agents
            .iter()
            .filter(|(agent, log)| log.started || any && agent.as_str() == SESSION)
            .filter(|(_, log)| !log.pending.is_empty())
            .map(|(agent, log)| (log.depth, agent.clone()))
            .collect();
        due.sort_by(|a, b| b.cmp(a));
        for (_, agent) in due {
            let Some(log) = self.agents.get_mut(&agent) else {
                continue;
            };
            if let Err(failure) = store.append(&agent, &log.pending) {
                self.failure = Some(format!("writing the log of {agent} failed: {failure}"));
                return;
            }
            log.pending.clear();
        }
    }
}

/// Queues `record` as the next line of `log`, the session's record `n`;
/// its `seq`.
fn enqueue(log: &mut AgentLog, n: u64, record: Record<'_>) -> serde_json::Result<u64> {
    let seq = log.next_seq;
    let line = Line {
        seq,
        n,
        t: now_ms(),
        record,
    };
    serde_json::to_writer(&mut log.pending, &line)?;
    log.pending.push(b'\n');
    log.next_seq += 1;
    Ok(seq)
}

/// A running session's logs: the records queued for them and each agent's
/// latest-wins state. Every method takes `&self`, so a clone shares them.
/// Nothing is logged before [`Self::start`], nor once a write failed. A
/// host logs each change as it makes it to the agent's [`Conversation`],
/// so the log folds back to what it has.
#[derive(Clone, Default)]
pub struct Writer(Arc<Mutex<Book>>);

impl Writer {
    fn book(&self) -> MutexGuard<'_, Book> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Reads the session in `store` to restore it, as [`read_session`]
    /// does, but first cuts off a torn last line a crash left, and leaves
    /// out a retired agent and the agents it spawned. The writer then logs
    /// to `store`, after its logs, once started.
    ///
    /// # Errors
    ///
    /// When the store cannot list its logs; the writer logs to it anyway.
    pub fn restore(&self, store: Arc<dyn JournalStore>) -> io::Result<SavedSession> {
        let read = read_logs(&*store, true);
        let mut book = self.book();
        book.store = Some(store);
        let mut logs = read?;
        let next_n = next_n(&logs.folded);
        let skipped = retired(&logs.folded);
        logs.folded
            .retain(|log| !skipped.contains(&log.header.agent));
        book.after(&logs.folded, next_n);
        drop(book);
        Ok(session_of(logs))
    }

    /// Starts logging, once the host restored the session
    /// ([`Self::restore`]): what it did to restore is not logged.
    pub fn start(&self) {
        let mut book = self.book();
        book.live = book.store.is_some();
    }

    /// Whether records are logged: the writer was started, and no write
    /// failed.
    pub fn is_live(&self) -> bool {
        let book = self.book();
        book.live && book.failure.is_none()
    }

    /// Starts the log of `agent`, spawned by the agent `parent`, unless it
    /// has one.
    pub fn open(&self, agent: &str, parent: Option<&str>) {
        self.book().agent(agent, parent);
    }

    /// Logs `message`, added to the conversation of `agent`, its images
    /// stored as blobs, with where it came from.
    pub fn message(&self, agent: &str, message: &Message, origin: Option<&Origin>) {
        self.messages(agent, [message], |mut stored| Record::Message {
            message: stored.pop().unwrap_or(Cow::Borrowed(message)),
            origin: origin.cloned(),
        });
    }

    /// Logs the splice of messages `from..to` of the conversation of
    /// `agent` with `with` ([`Conversation::splice`]).
    pub fn splice(&self, agent: &str, from: usize, to: usize, with: &[Entry]) {
        let messages = with.iter().map(|entry| &entry.message);
        self.messages(agent, messages, |stored| {
            let with = stored.into_iter().zip(with);
            let with = with.map(|(message, entry)| Entry {
                message: message.into_owned(),
                ..entry.clone()
            });
            Record::Splice {
                from,
                to,
                with: with.collect(),
            }
        });
    }

    fn messages<'m>(
        &self,
        agent: &str,
        messages: impl IntoIterator<Item = &'m Message>,
        make: impl FnOnce(Vec<Cow<'m, Message>>) -> Record<'m>,
    ) {
        let mut book = self.book();
        if let Some(store) = book.store.clone() {
            book.messages(&*store, agent, messages, make);
        }
    }

    /// Logs that the user's last message in the conversation of `agent` was
    /// left unanswered for `reason`, or, with `None`, asked for again.
    pub fn halt(&self, agent: &str, reason: Option<Halt>) {
        self.book().record(agent, Record::Halt { reason });
    }

    /// Logs that `agent` is gone for good.
    pub fn retire(&self, agent: &str) {
        self.book().record(agent, Record::Retired);
    }

    /// Logs the saved component `component` in the log `agent`, an agent's
    /// id or [`SESSION`], when its value changed; `None` or `null` when it
    /// was removed. With `starts`, the change starts the agent's log.
    pub fn component(&self, agent: &str, component: &str, value: Option<Value>, starts: bool) {
        self.book().component(agent, component, value, starts);
    }

    /// Writes every queued record: each started log's records with one
    /// write, a subagent's log before its parent's, so a parent never
    /// refers to a subagent message that is not stored, and the [`SESSION`]
    /// log once any is started. No fsync. A failed write stops the
    /// logging.
    pub fn flush(&self) {
        let mut book = self.book();
        if let Some(store) = book.store.clone() {
            book.flush(&*store);
        }
    }

    /// Why the logging stopped, if it did.
    pub fn failure(&self) -> Option<String> {
        self.book().failure.clone()
    }
}

/// The first fields of a line, read from every line to find the newest
/// record of each saved component, and where a fork cuts.
#[derive(Deserialize)]
struct Envelope {
    seq: u64,
    n: u64,
    t: u64,
    #[serde(rename = "type")]
    kind: String,
    component: Option<String>,
}

/// An agent log folded into the agent's state.
struct Folded {
    header: Header,
    /// When the log was started, in milliseconds since the Unix epoch.
    started_ms: u64,
    /// The conversation.
    conversation: Conversation,
    /// The session's `n` of the record that made each message.
    made: HashMap<MessageId, u64>,
    /// The newest value of each saved component, by type path.
    components: HashMap<String, Value>,
    /// The `seq` after the last line's.
    next_seq: u64,
    /// The session's `n` after the log's last line's.
    next_n: u64,
    /// Whether the agent was retired.
    retired: bool,
}

/// Folds `log`, an agent's complete lines: every message, splice and halt
/// in order, as they were made live, so the ids are the live ones. Records
/// of unknown types are skipped. Images are left in the store.
fn fold(log: &LogLines) -> Result<Folded, String> {
    let lines: Vec<&[u8]> = log.iter().map(|(_, line)| line).collect();
    let header = match lines
        .first()
        .and_then(|first| serde_json::from_slice::<Line>(first).ok())
    {
        Some(Line {
            record: Record::Header(header),
            ..
        }) => header,
        _ => return Err("it does not start with a header".to_owned()),
    };
    let envelopes: Vec<Option<Envelope>> = lines
        .iter()
        .map(|line| serde_json::from_slice(line).ok())
        .collect();
    let next_seq = envelopes.iter().flatten().map(|envelope| envelope.seq + 1);
    let mut folded = Folded {
        header,
        started_ms: envelopes
            .first()
            .and_then(Option::as_ref)
            .map_or(0, |at| at.t),
        conversation: Conversation::default(),
        made: HashMap::new(),
        components: HashMap::new(),
        next_seq: next_seq.max().unwrap_or(1),
        next_n: envelopes
            .iter()
            .flatten()
            .map(|envelope| envelope.n + 1)
            .max()
            .unwrap_or_default(),
        retired: envelopes
            .iter()
            .flatten()
            .any(|envelope| envelope.kind == "retired"),
    };
    for (line, envelope) in lines.iter().zip(&envelopes).skip(1) {
        // Saved components are read below; unknown record types are
        // skipped and left as they are.
        if envelope
            .as_ref()
            .is_some_and(|envelope| envelope.component.is_some())
        {
            continue;
        }
        let Ok(line) = serde_json::from_slice::<Line>(line) else {
            continue;
        };
        let conversation = &mut folded.conversation;
        let first_new = conversation.next_id();
        match line.record {
            Record::Message { message, origin } => {
                conversation.append(message.into_owned(), origin);
            }
            Record::Splice { from, to, with } => {
                conversation.splice(from, to, with);
            }
            // Only a conversation waiting for its model is halted, so a
            // halt after a halt means it was asked again, as by a retry.
            Record::Halt { reason } => {
                conversation.resume();
                if let Some(reason) = reason {
                    conversation.halt(reason);
                }
            }
            Record::Header(_) | Record::Component { .. } | Record::Retired => {}
        }
        let made = (first_new..conversation.next_id()).map(|id| (MessageId(id), line.n));
        folded.made.extend(made);
    }
    // Only the newest record of each saved component is read, wherever it
    // is in the log.
    let newest: HashMap<&str, usize> = envelopes
        .iter()
        .enumerate()
        .filter_map(|(at, envelope)| Some((envelope.as_ref()?.component.as_deref()?, at)))
        .collect();
    for at in newest.into_values() {
        let line = lines
            .get(at)
            .and_then(|line| serde_json::from_slice(line).ok());
        if let Some(Line {
            record: Record::Component { component, value },
            ..
        }) = line
        {
            folded
                .components
                .extend(value.map(|value| (component, value)));
        }
    }
    Ok(folded)
}

/// The session's `n` after the last line of every log of `folded`.
fn next_n(folded: &[Folded]) -> u64 {
    let next = folded.iter().map(|log| log.next_n);
    next.max().unwrap_or_default()
}

/// Every log of a store, folded.
struct Logs {
    /// The logs read, their images loaded.
    folded: Vec<Folded>,
    /// The logs that could not be read, by agent id, with why.
    unreadable: Vec<(String, String)>,
    /// The images whose blobs are gone, by agent id, with why.
    lost_images: Vec<(String, String)>,
}

/// Reads and folds every log of `store`, with their images. With `repair`,
/// a torn last line is cut off first, so the log can be appended to.
fn read_logs(store: &dyn JournalStore, repair: bool) -> io::Result<Logs> {
    let mut logs = Logs {
        folded: Vec::new(),
        unreadable: Vec::new(),
        lost_images: Vec::new(),
    };
    for agent in store.agents()? {
        let read = LogLines::read(store, &agent).and_then(|log| {
            if repair && log.torn() {
                store.truncate(&agent, log.len())?;
            }
            Ok(log)
        });
        let folded = read
            .map_err(|failure| failure.to_string())
            .and_then(|log| fold(&log));
        let mut folded = match folded {
            Ok(folded) => folded,
            Err(why) => {
                logs.unreadable.push((agent, why));
                continue;
            }
        };
        for message in folded.conversation.each_message() {
            if let Err(failure) = load_images(message, store) {
                logs.lost_images.push((agent.clone(), failure.to_string()));
            }
        }
        logs.folded.push(folded);
    }
    Ok(logs)
}

/// The ids of the agents a restore skips: those retired, and the agents
/// they spawned, all the way down.
fn retired(agents: &[Folded]) -> HashSet<String> {
    let parents: HashMap<&str, Option<&str>> = agents
        .iter()
        .map(|agent| (agent.header.agent.as_str(), agent.header.parent.as_deref()))
        .collect();
    let gone: HashSet<&str> = agents
        .iter()
        .filter(|agent| agent.retired)
        .map(|agent| agent.header.agent.as_str())
        .collect();
    let mut skipped = HashSet::new();
    for agent in agents {
        // A loop cannot happen, but a bound costs nothing.
        let up = std::iter::successors(Some(agent.header.agent.as_str()), |id| {
            parents.get(id).copied().flatten()
        });
        if up.take(agents.len() + 1).any(|id| gone.contains(id)) {
            skipped.insert(agent.header.agent.clone());
        }
    }
    skipped
}

/// An agent as its log left it, read by [`read_session`] and written by
/// [`write_session`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SavedAgent {
    /// Its id.
    pub id: String,
    /// The id of the agent that spawned it, if any.
    pub parent: Option<String>,
    /// When its log was started, in milliseconds since the Unix epoch.
    pub started_ms: u64,
    /// Its conversation as it now is, with its origins and message ids.
    pub conversation: Conversation,
    /// The newest value of each of its saved components, by type path, as
    /// the host serialized it.
    pub saved: BTreeMap<String, Value>,
    /// Whether it was retired: a restore skips it, and the agents it
    /// spawned.
    pub retired: bool,
}

/// A session as its logs left it, read by [`read_session`] and written by
/// [`write_session`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SavedSession {
    /// Its agents, in the order their logs were started.
    pub agents: Vec<SavedAgent>,
    /// The newest value of each of its saved values, such as a host's
    /// resources, by type path: the [`SESSION`] log's.
    pub resources: BTreeMap<String, Value>,
    /// The logs that could not be read, by agent id, with why.
    pub unreadable: Vec<(String, String)>,
    /// The images a log names whose blob is gone, by agent id, with why:
    /// each is read as an image without data.
    pub lost_images: Vec<(String, String)>,
}

/// The session of `logs`.
fn session_of(logs: Logs) -> SavedSession {
    let mut session = SavedSession {
        unreadable: logs.unreadable,
        lost_images: logs.lost_images,
        ..SavedSession::default()
    };
    for folded in logs.folded {
        if folded.header.agent == SESSION {
            session.resources = folded.components.into_iter().collect();
            continue;
        }
        session.agents.push(SavedAgent {
            id: folded.header.agent,
            parent: folded.header.parent,
            started_ms: folded.started_ms,
            conversation: folded.conversation,
            saved: folded.components.into_iter().collect(),
            retired: folded.retired,
        });
    }
    session
        .agents
        .sort_by(|a, b| (a.started_ms, &a.id).cmp(&(b.started_ms, &b.id)));
    session
}

/// Reads the session kept in `store`, another session's or a running
/// one's, without restoring it and without writing to it: a torn last line
/// is left out, not cut off. A running host writes its records when it
/// flushes them.
///
/// # Errors
///
/// When the store cannot list its logs.
pub fn read_session(store: &dyn JournalStore) -> io::Result<SavedSession> {
    read_logs(store, false).map(session_of)
}

/// Makes the session in `store` read back as `session`, such as one
/// imported from another tool's files: an agent without a log gets one,
/// and each log gets only the records that bring it to what `session`
/// says, so an agent's messages are appended when its log holds the first
/// of them. Each agent's conversation, saved values, and whether it was
/// retired are written, and the session's saved values; message ids are
/// the conversation's when it was built from the store's, such as by
/// adding to one [`read_session`] read, and new otherwise. An agent's
/// parent comes from the log when it has one. Agents and logs that
/// `session` does not name, and its `unreadable` and `lost_images`, are
/// left as they are. A torn last line is cut off first.
///
/// # Errors
///
/// When the store fails, or an agent of `session` has a log that cannot
/// be read.
pub fn write_session(store: &dyn JournalStore, session: &SavedSession) -> io::Result<()> {
    let logs = read_logs(store, true)?;
    let unreadable = logs.unreadable.iter().find(|(agent, _)| {
        agent == SESSION || session.agents.iter().any(|saved| saved.id == *agent)
    });
    if let Some((agent, why)) = unreadable {
        let why = format!("the log of {agent} cannot be read: {why}");
        return Err(io::Error::new(io::ErrorKind::InvalidData, why));
    }
    let mut book = Book {
        live: true,
        ..Book::default()
    };
    book.after(&logs.folded, next_n(&logs.folded));
    // A log is written only when it changed.
    for log in book.agents.values_mut() {
        log.started = false;
    }
    let had = |agent: &str| logs.folded.iter().find(|log| log.header.agent == agent);
    let blank = Conversation::default();
    for saved in &session.agents {
        let log = had(&saved.id);
        book.agent(&saved.id, saved.parent.as_deref());
        let old = log.map_or(&blank, |log| &log.conversation);
        let new = &saved.conversation;
        let item = |conversation: &Conversation, at| {
            let entry = conversation.entry(at)?;
            Some((entry.message, entry.origins))
        };
        let len = old.messages().len();
        let same = (0..len).take_while(|&at| item(old, at) == item(new, at));
        let same = same.count();
        let mut halted = old.halted();
        if same < len || same < new.messages().len() {
            let with: Vec<Entry> = (same..new.messages().len())
                .filter_map(|at| new.entry(at))
                .collect();
            let messages = with.iter().map(|entry| &entry.message);
            book.messages(store, &saved.id, messages, |stored| Record::Splice {
                from: same,
                to: len,
                with: stored
                    .into_iter()
                    .zip(&with)
                    .map(|(message, entry)| Entry {
                        message: message.into_owned(),
                        ..entry.clone()
                    })
                    .collect(),
            });
            halted = None;
        }
        if halted != new.halted() {
            let reason = new.halted();
            book.record(&saved.id, Record::Halt { reason });
        }
        let known = log.map(|log| &log.components);
        write_values(&mut book, &saved.id, known, &saved.saved);
        if saved.retired && !log.is_some_and(|log| log.retired) {
            book.record(&saved.id, Record::Retired);
        }
        if let Some(log) = book.agents.get_mut(&saved.id) {
            log.started = true;
        }
    }
    let known = had(SESSION).map(|log| &log.components);
    write_values(&mut book, SESSION, known, &session.resources);
    book.flush(store);
    match book.failure {
        Some(failure) => Err(io::Error::other(failure)),
        None => Ok(()),
    }
}

/// Queues the saved values of the log `agent` that differ from `known`,
/// and the removal of each of `known` that `values` does not have.
fn write_values(
    book: &mut Book,
    agent: &str,
    known: Option<&HashMap<String, Value>>,
    values: &BTreeMap<String, Value>,
) {
    let removed = known
        .into_iter()
        .flat_map(HashMap::keys)
        .filter(|component| !values.contains_key(*component));
    let removed: Vec<String> = removed.cloned().collect();
    for component in removed {
        book.component(agent, &component, None, true);
    }
    for (component, value) in values {
        book.component(agent, component, Some(value.clone()), true);
    }
}

/// Copies the session in `from` into `to`, an empty store, such as to
/// branch it: every blob, and every log as of `before`, a message of the
/// agent whose id is named there: every log, that agent's too, up to the
/// records made before the one that made that message, so the copy is the
/// session as it was then. A message keeps its id when a splice edits it,
/// so that is the message's first record. Without `before`, all of it. An
/// agent with no message, such as one spawned later, is left out. A torn
/// tail is never copied, and the effect log, which is not the store's, is
/// not copied.
///
/// # Errors
///
/// When a store fails, `from` cannot list its blobs, or the agent's log
/// cannot be read or never made message `before`.
pub fn fork_session(
    from: &dyn JournalStore,
    to: &dyn JournalStore,
    before: Option<(&str, MessageId)>,
) -> io::Result<()> {
    let Some((agent, message)) = before else {
        return fork(from, to, |_, log| Some(log.len()));
    };
    let log = LogLines::read(from, agent)?;
    let folded = fold(&log).map_err(|why| io::Error::new(io::ErrorKind::InvalidData, why))?;
    let Some(&at) = folded.made.get(&message) else {
        let why = format!("agent {agent} made no message {}", message.0);
        return Err(io::Error::new(io::ErrorKind::InvalidInput, why));
    };
    fork(from, to, |id, log| {
        let mut keep = None;
        let mut messages = false;
        for (end, line) in log.iter() {
            let Ok(stamp) = serde_json::from_slice::<Envelope>(line) else {
                continue;
            };
            if stamp.n >= at {
                break;
            }
            messages |= matches!(stamp.kind.as_str(), "message" | "splice");
            keep = Some(end);
        }
        keep.filter(|_| messages || id == SESSION)
    })
}

#[cfg(test)]
mod tests;
