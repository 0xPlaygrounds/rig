//! The session's append-only logs. Each agent, a subagent too, has one
//! JSON-lines log in the session's [`SessionStore`], created at its first
//! message: a header, then one record per change a [`Commit`] made (a
//! message, a splice, a halt, the agent retired) and per change of a
//! [`ReflectSaved`] component, such as the agent's model,
//! reasoning setting, system prompt, tool access and last usage. Records
//! are queued as they happen and written at the end of the frame, a
//! subagent's before its parent's, and at once before a tool that may
//! change something runs. Nothing is ever rewritten;
//! [`restore`](super::restore) folds the logs back at startup,
//! [`read_session`] reads a session's logs without restoring or writing
//! anything, and [`fork_session`] copies a session as it was when one of
//! its messages was made.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::io;
use std::ops::{Bound, RangeBounds};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::reflect::AppTypeRegistry;
use bevy_ecs::resource::IsResource;
use bevy_ecs::system::SystemParam;
use bevy_log::{error, warn};
use bevy_reflect::serde::TypedReflectSerializer;
use bevy_reflect::{CreateTypeData, Reflect, TypePath};
use rig_cassette::journal::{JournalStore, LogLines, fork, store_images};
use rig_core::completion::Message;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use web_time::{SystemTime, UNIX_EPOCH};

use super::agent::{
    Agent, AgentId, Conversation, Entry, Halt, MessageId, Notice, SpawnedBy, Spliced,
};
use super::inbox::Origin;
use super::{StopTurns, WriteJournal};

/// Saves an agent component or a resource with the session: derive
/// `Reflect` and add `Saved` to its `#[reflect(..)]`, as
/// `#[reflect(Component, Saved)]` or `#[reflect(Resource, Saved)]`. Each
/// change of it is logged at the end of the frame by its type path (an
/// agent's in its log, a resource in the session's [`SESSION`] log, written
/// once an agent's log is) and restored by reflection, inserted on the
/// agent it was saved with before [`SessionRestored`] exists, so an
/// `On<Insert<T>>` observer re-arms what it says; a value that serializes
/// to `null` is saved as absent, and so is an agent component removed. A
/// resource is restored in place of the one its plugin inserted while the
/// app was built.
/// Registration is a runtime fact: a generic type is saved only once
/// registered (`app.register_type::<T>()`), and a logged type no build
/// registers any more is reported on restore and kept in the log.
#[derive(Clone)]
pub struct ReflectSaved {
    watch: fn(&mut App),
}

impl<T: Component + Reflect + TypePath> CreateTypeData<T> for ReflectSaved {
    fn create_type_data(_: ()) -> Self {
        // `Changed` also sees an immutable value inserted in place of
        // another.
        let watch = |app: &mut App| {
            let log = log_changed::<T>.after(StopTurns).before(WriteJournal);
            app.add_systems(Last, log);
        };
        Self { watch }
    }
}

/// The log of the session's saved resources, beside the agents' logs.
pub const SESSION: &str = "session";

/// Logs each change of the saved component `T` on the agents or as a
/// resource, and its removal from an agent; reflection runs only for a
/// change. A change starts its agent's log, as a message does, unless it
/// is a value the agent was spawned with, so a session nobody used is not
/// kept.
fn log_changed<T: Component + Reflect + TypePath>(
    changed: Query<
        (Entity, Option<&AgentId>, &T),
        (Or<(With<Agent>, With<IsResource>)>, Changed<T>),
    >,
    mut removed: RemovedComponents<T>,
    agents: Query<&AgentId, With<Agent>>,
    spawned: Query<(), Added<Agent>>,
    log: Res<SessionLog>,
    registry: Res<AppTypeRegistry>,
) {
    if !log.is_live() {
        return;
    }
    let name = T::type_path();
    let registry = registry.read();
    for (entity, id, component) in &changed {
        let value = TypedReflectSerializer::new(component.as_partial_reflect(), &registry);
        let key = id.map_or(SESSION, |id| id.0.as_str());
        let starts = id.is_some() && !spawned.contains(entity);
        match serde_json::to_value(value) {
            Ok(value) => log.component(key, name, Some(value), starts),
            Err(failure) => error!("not logging {name}: {failure}"),
        }
    }
    for entity in removed.read() {
        if let Ok(id) = agents.get(entity)
            && !changed.contains(entity)
        {
            log.component(&id.0, name, None, !spawned.contains(entity));
        }
    }
}

/// The first record of an agent log.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) struct Header {
    /// The agent's id.
    pub(crate) agent: String,
    /// The id of the agent it was [`SpawnedBy`](super::agent::SpawnedBy),
    /// if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) parent: Option<String>,
}

/// One record of an agent log, tagged by `type`.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum Record<'a> {
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
pub(crate) struct Line<'a> {
    /// Dense from 0, the header's, in each log.
    pub(crate) seq: u64,
    /// Its place among the records of every log of the session, in the
    /// order they were made: what a fork cuts the other logs by.
    pub(crate) n: u64,
    /// When it was written, in milliseconds since the Unix epoch.
    pub(crate) t: u64,
    #[serde(flatten)]
    pub(crate) record: Record<'a>,
}

/// Milliseconds since the Unix epoch.
pub fn now_ms() -> u64 {
    millis(SystemTime::now())
}

/// Milliseconds since the Unix epoch at `at`; 0 before it.
pub(crate) fn millis(at: SystemTime) -> u64 {
    at.duration_since(UNIX_EPOCH)
        .map(|elapsed| u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or_default()
}

/// One agent's log: where it goes, what is queued for it and the newest
/// value of each saved component, so an unchanged value is not logged
/// again.
#[derive(Default)]
pub(crate) struct AgentLog {
    /// 0 for an agent the user started, 1 for its subagents, and so on.
    pub(crate) depth: usize,
    pub(crate) next_seq: u64,
    /// Whether the log holds a message or a saved change, so it is
    /// written.
    pub(crate) started: bool,
    pub(crate) pending: Vec<u8>,
    pub(crate) components: HashMap<String, Value>,
}

#[derive(Default)]
struct Book {
    store: Option<Arc<dyn JournalStore>>,
    /// The next record's [`Line::n`].
    next_n: u64,
    /// Records are queued only once the session was restored.
    live: bool,
    /// Why the log stopped: every record after a failed write is dropped.
    failure: Option<String>,
    agents: HashMap<String, AgentLog>,
}

impl Book {
    /// The log of `agent`, started when it has none with a header naming
    /// `parent`, the agent that spawned it.
    fn agent(&mut self, agent: &str, parent: Option<&str>) -> Option<&mut AgentLog> {
        if !self.live || self.failure.is_some() {
            return None;
        }
        self.store.as_ref()?;
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

/// Where the session is kept, such as a rig-cassette `MemoryStore` or
/// `JsonlDirStore`: inserted while the app is built, before it runs, by
/// any plugin. Without one nothing is kept.
#[derive(Resource, Clone)]
pub struct SessionStore(pub Arc<dyn JournalStore>);

impl SessionStore {
    /// The session kept in `store`.
    pub fn new(store: impl JournalStore) -> Self {
        Self(Arc::new(store))
    }
}

/// The session's agent logs: the open files, the records queued for them,
/// and each agent's latest-wins state. Every method takes `&self`, so any
/// system can log; nothing is logged before the session was restored, or
/// without a [`SessionStore`].
#[derive(Resource, Clone, Default)]
pub struct SessionLog(Arc<Mutex<Book>>);

impl SessionLog {
    fn book(&self) -> MutexGuard<'_, Book> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Starts logging to `store`, after the logs restored from it as
    /// `agents`, by agent id, its next record the session's `next_n`.
    pub(crate) fn resume(
        &self,
        store: Arc<dyn JournalStore>,
        agents: Vec<(String, AgentLog)>,
        next_n: u64,
    ) {
        let mut book = self.book();
        book.store = Some(store);
        book.agents.extend(agents);
        book.next_n = book.next_n.max(next_n);
        book.live = true;
    }

    /// Whether records are logged: the session was restored, and no write
    /// failed.
    pub(crate) fn is_live(&self) -> bool {
        let book = self.book();
        book.live && book.failure.is_none()
    }

    /// Logs the record `make` builds of `messages`, their images stored as
    /// blobs, for `agent`, which starts its log.
    fn log_messages<'m>(
        &self,
        agent: &AgentId,
        messages: impl IntoIterator<Item = &'m Message>,
        make: impl FnOnce(Vec<Cow<'m, Message>>) -> Record<'m>,
    ) {
        let mut book = self.book();
        if !book.live || book.failure.is_some() {
            return;
        }
        let Some(store) = book.store.clone() else {
            return;
        };
        let stored: Result<Vec<_>, _> = messages
            .into_iter()
            .map(|message| store_images(message, &*store))
            .collect();
        match stored {
            Ok(stored) => {
                if book.record(&agent.0, make(stored)).is_some()
                    && let Some(log) = book.agents.get_mut(&agent.0)
                {
                    log.started = true;
                }
            }
            Err(failure) => book.failure = Some(format!("storing an image failed: {failure}")),
        }
    }

    /// Logs the saved component `component` in the log `agent`, an agent's
    /// id or [`SESSION`], when its value changed; `None` or `null` when it
    /// was removed. With `starts`, the change starts the agent's log.
    pub(crate) fn component(
        &self,
        agent: &str,
        component: &str,
        value: Option<Value>,
        starts: bool,
    ) {
        let mut book = self.book();
        let value = value.filter(|value| !value.is_null());
        let known = book
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
        if book.record(agent, record).is_some()
            && let Some(log) = book.agents.get_mut(agent)
        {
            log.started |= starts;
            match value {
                Some(value) => log.components.insert(component.to_owned(), value),
                None => log.components.remove(component),
            };
        }
    }

    /// Writes every queued record: each started log's records with one
    /// write, a subagent's log before its parent's, so a parent never
    /// refers to a subagent message that is not on disk, and the
    /// [`SESSION`] log once any is started. No fsync. A failed write stops
    /// the logging.
    pub(crate) fn flush(&self) {
        let mut guard = self.book();
        let book = &mut *guard;
        if book.failure.is_some() {
            return;
        }
        let Some(store) = book.store.clone() else {
            return;
        };
        let any = book.agents.values().any(|log| log.started);
        let mut due: Vec<(usize, String)> = book
            .agents
            .iter()
            .filter(|(agent, log)| log.started || any && agent.as_str() == SESSION)
            .filter(|(_, log)| !log.pending.is_empty())
            .map(|(agent, log)| (log.depth, agent.clone()))
            .collect();
        due.sort_by(|a, b| b.cmp(a));
        for (_, agent) in due {
            let Some(log) = book.agents.get_mut(&agent) else {
                continue;
            };
            if let Err(failure) = store.append(&agent, &log.pending) {
                book.failure = Some(format!("writing the log of {agent} failed: {failure}"));
                return;
            }
            log.pending.clear();
        }
    }

    /// Why the log stopped, if it did.
    fn failure(&self) -> Option<String> {
        self.book().failure.clone()
    }
}

/// Inserted once the session was restored from its [`SessionStore`] and is
/// logged: a choice made afterwards is the user's, such as a model to
/// remember (`run_if(resource_exists::<SessionRestored>)`). It stays when
/// a log write fails later, which stops only the logging. An app without
/// a store never gets one.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct SessionRestored;

/// Changes agents' conversations, the one way they change: each change is
/// logged, with a message's images stored as blobs, and announced as a
/// [`Committed`], so a plugin sees every change the log records.
#[derive(SystemParam)]
pub struct Commit<'w, 's> {
    ids: Query<'w, 's, &'static AgentId>,
    log: Res<'w, SessionLog>,
    committed: MessageWriter<'w, Committed>,
    commands: Commands<'w, 's>,
}

impl Commit<'_, '_> {
    /// Adds `message` to `conversation`, that of `agent`: into its last
    /// message when both are the user's, so user and model keep taking
    /// turns.
    pub fn message(&mut self, agent: Entity, conversation: &mut Conversation, message: Message) {
        self.message_from(agent, conversation, message, None);
    }

    /// [`Self::message`], with where it came from when not from the user,
    /// the model or a tool.
    pub fn message_from(
        &mut self,
        agent: Entity,
        conversation: &mut Conversation,
        message: Message,
        origin: Option<Origin>,
    ) {
        let Ok(id) = self.ids.get(agent) else {
            return;
        };
        self.log
            .log_messages(id, [&message], |mut stored| Record::Message {
                message: stored.pop().unwrap_or(Cow::Borrowed(&message)),
                origin: origin.clone(),
            });
        self.committed.write(Committed::Message {
            agent,
            message: message.clone(),
            origin: origin.clone(),
        });
        conversation.append(message, origin);
    }

    /// Replaces messages `range` of `conversation`, that of `agent`, with
    /// `with`, merging a user message into a user message beside it: an
    /// entry taken with [`Conversation::entry`] keeps its message's id, and
    /// a merged message keeps the id that was there before. It is logged,
    /// so a restore makes the same change, and announced as a
    /// [`Committed::Splice`]. A cut is `splice(agent, c, len.., [])`; an
    /// edit in place passes `conversation.entry(i)` back changed. A splice
    /// that reaches the end clears the halt; one before it keeps it.
    pub fn splice(
        &mut self,
        agent: Entity,
        conversation: &mut Conversation,
        range: impl RangeBounds<usize>,
        with: impl IntoIterator<Item = Entry>,
    ) {
        let Ok(id) = self.ids.get(agent) else {
            return;
        };
        let len = conversation.messages().len();
        let to = match range.end_bound() {
            Bound::Included(&last) => last.saturating_add(1),
            Bound::Excluded(&end) => end,
            Bound::Unbounded => len,
        }
        .min(len);
        let from = match range.start_bound() {
            Bound::Included(&first) => first,
            Bound::Excluded(&before) => before.saturating_add(1),
            Bound::Unbounded => 0,
        }
        .min(to);
        let with: Vec<Entry> = with.into_iter().collect();
        if from == to && with.is_empty() {
            return;
        }
        let messages = with.iter().map(|entry| &entry.message);
        self.log.log_messages(id, messages, |stored| {
            let with = stored.into_iter().zip(&with);
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
        let Spliced { removed, changed } = conversation.splice(from, to, with);
        self.committed.write(Committed::Splice {
            agent,
            removed,
            changed,
        });
    }

    /// Retires `agent` for good: the log says so, so a restore brings back
    /// neither it nor the agents it spawned, and it is despawned with them
    /// and their work.
    pub fn retire(&mut self, agent: Entity) {
        let Ok(id) = self.ids.get(agent) else {
            return;
        };
        self.log.book().record(&id.0, Record::Retired);
        self.committed.write(Committed::Retired { agent });
        self.commands.entity(agent).try_despawn();
    }

    /// Leaves the user's last message in `conversation`, that of `agent`,
    /// unanswered for `reason` when the model owes an answer to it, so a
    /// restore does not send it to the model. Not for a turn the app's
    /// exit stops: the restart carries that one on.
    pub fn halt(&mut self, agent: Entity, conversation: &mut Conversation, reason: Halt) {
        if let Ok(id) = self.ids.get(agent)
            && conversation.halt(reason)
        {
            let reason = Some(reason);
            self.log.book().record(&id.0, Record::Halt { reason });
            self.committed.write(Committed::Halt { agent, reason });
        }
    }

    /// Asks the model again for the user's last message, as a retry does:
    /// a halted conversation is no longer halted, which the log records.
    /// Whether the conversation ends in the user's message.
    pub(crate) fn resume(&mut self, agent: Entity, conversation: &mut Conversation) -> bool {
        let halted = conversation.halted().is_some();
        let asks = conversation.resume();
        if halted && let Ok(id) = self.ids.get(agent) {
            self.log.book().record(&id.0, Record::Halt { reason: None });
            self.committed.write(Committed::Halt {
                agent,
                reason: None,
            });
        }
        asks
    }
}

/// A change a [`Commit`] made to an agent's conversation, in the order
/// made; a restore makes none.
#[derive(Message, Clone, Debug)]
pub enum Committed {
    /// A message was added, into the last one when both are the user's.
    Message {
        agent: Entity,
        message: Message,
        /// Where it came from, when not from the user, the model or a tool.
        origin: Option<Origin>,
    },
    /// Messages were spliced: `removed` are the ids no longer in the
    /// conversation, `changed` those of the messages the splice made or
    /// changed. Ids stay valid across several splices in one frame; who
    /// spliced is read from the changed messages' item origins.
    Splice {
        agent: Entity,
        removed: Vec<MessageId>,
        changed: Vec<MessageId>,
    },
    /// The user's last message was left unanswered for `reason`, or, with
    /// none, asked for again.
    Halt { agent: Entity, reason: Option<Halt> },
    /// The agent was retired, and despawned.
    Retired { agent: Entity },
}

/// Restores the session at startup, settles what a crash or restart left
/// half done, and logs everything after.
pub struct JournalPlugin;

impl Plugin for JournalPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreStartup, super::restore::restore_session)
            .add_systems(Startup, super::restore::reconcile)
            .add_systems(Last, write_logs.in_set(WriteJournal))
            .add_observer(open_child_log);
    }

    /// Watches every type registered as [`ReflectSaved`], once every
    /// plugin had its say in the registry.
    fn finish(&self, app: &mut App) {
        let watches: Vec<fn(&mut App)> = app
            .world()
            .get_resource::<AppTypeRegistry>()
            .map(|registry| {
                let registry = registry.read();
                let saved = registry.iter_with_data::<ReflectSaved>();
                saved.map(|(_, saved)| saved.watch).collect()
            })
            .unwrap_or_default();
        for watch in watches {
            watch(app);
        }
    }
}

/// Starts the log of an agent spawned by another, so its header names
/// that agent and a restore links them again.
fn open_child_log(
    spawned: On<Add<SpawnedBy>>,
    agents: Query<(&AgentId, &SpawnedBy)>,
    ids: Query<&AgentId>,
    log: Res<SessionLog>,
) {
    if let Ok((child, parent)) = agents.get(spawned.entity)
        && let Ok(parent) = ids.get(parent.0)
    {
        log.book().agent(&child.0, Some(&parent.0));
    }
}

/// Writes the frame's records. A failure is shown once.
fn write_logs(log: Res<SessionLog>, mut reported: Local<bool>, mut notices: MessageWriter<Notice>) {
    log.flush();
    if !*reported && let Some(failure) = log.failure() {
        *reported = true;
        error!("the session log stopped: {failure}");
        notices.write(Notice::error(
            None,
            format!("The session is no longer saved: {failure}"),
        ));
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
pub(crate) struct Folded {
    pub(crate) header: Header,
    /// When the log was started, in milliseconds since the Unix epoch.
    pub(crate) started_ms: u64,
    /// The conversation, its images not loaded yet ([`load_images`]).
    pub(crate) conversation: Conversation,
    /// The session's `n` of the record that made each message.
    pub(crate) made: HashMap<MessageId, u64>,
    /// The newest value of each saved component, by type path.
    pub(crate) components: HashMap<String, Value>,
    /// The `seq` after the last line's.
    pub(crate) next_seq: u64,
    /// The session's `n` after the log's last line's.
    pub(crate) next_n: u64,
    /// Whether the agent was retired.
    pub(crate) retired: bool,
}

/// Folds `log`, an agent's complete lines: every message, splice and halt
/// in order, as they were made live, so the ids are the live ones. Records
/// of unknown types are skipped. Images are left in the store
/// ([`load_images`]).
pub(crate) fn fold(log: &LogLines) -> Result<Folded, String> {
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

/// Reads the images of `folded`'s messages back from `store`'s blobs, only
/// for the messages the fold kept.
pub(crate) fn load_images(folded: &mut Folded, store: &dyn JournalStore) {
    for message in folded.conversation.each_message() {
        if let Err(failure) = rig_cassette::journal::load_images(message, store) {
            warn!(
                "an image agent {} logged is gone: {failure}",
                folded.header.agent
            );
        }
    }
}

/// An agent as its log left it, read by [`read_session`].
#[derive(Clone)]
pub struct SavedAgent {
    /// Its id.
    pub id: AgentId,
    /// The agent that spawned it, if any.
    pub parent: Option<AgentId>,
    /// When its log was started, in milliseconds since the Unix epoch.
    pub started_ms: u64,
    /// Its conversation as it now is, with its origins and message ids.
    pub conversation: Conversation,
    /// The newest value of each of its saved components, by type path, as
    /// reflection serialized it.
    pub saved: BTreeMap<String, Value>,
    /// Whether it was retired: a restore skips it, and the agents it
    /// spawned.
    pub retired: bool,
}

/// A session as its logs left it, read by [`read_session`].
#[derive(Clone, Default)]
pub struct SavedSession {
    /// Its agents, in the order their logs were started.
    pub agents: Vec<SavedAgent>,
    /// The newest value of each of its saved resources, by type path.
    pub resources: BTreeMap<String, Value>,
    /// The logs that could not be read, by agent id, with why.
    pub unreadable: Vec<(String, String)>,
}

/// Reads the session kept in `store`, another session's or the running
/// one's, without restoring it and without writing to it: a torn last line
/// is left out, not cut off. What the running app logged this frame is
/// written at its end ([`WriteJournal`]).
///
/// # Errors
///
/// When the store cannot list its logs.
pub fn read_session(store: &dyn JournalStore) -> io::Result<SavedSession> {
    let mut session = SavedSession::default();
    for agent in store.agents()? {
        let log = LogLines::read(store, &agent).map_err(|failure| failure.to_string());
        let folded = match log.and_then(|log| fold(&log)) {
            Ok(mut folded) => {
                load_images(&mut folded, store);
                folded
            }
            Err(why) => {
                session.unreadable.push((agent, why));
                continue;
            }
        };
        if folded.header.agent == SESSION {
            session.resources = folded.components.into_iter().collect();
            continue;
        }
        session.agents.push(SavedAgent {
            id: AgentId(folded.header.agent),
            parent: folded.header.parent.map(AgentId),
            started_ms: folded.started_ms,
            conversation: folded.conversation,
            saved: folded.components.into_iter().collect(),
            retired: folded.retired,
        });
    }
    session
        .agents
        .sort_by(|a, b| (a.started_ms, &a.id.0).cmp(&(b.started_ms, &b.id.0)));
    Ok(session)
}

/// Copies the session in `from` into `to`, an empty store, such as to
/// branch it: every blob, and every log as of `before`, a message of the
/// agent named there: every log, that agent's too, up to the records made
/// before the one that made that message, so the copy is the session as it
/// was then. A message keeps its id when a splice edits it, so that is
/// the message's first record. Without `before`, all of it. An agent with
/// no message, such as one spawned later, is left out. A torn tail is
/// never copied, and the effect log, which is not the store's, is not
/// copied.
///
/// # Errors
///
/// When a store fails, `from` cannot list its blobs, or the agent's log
/// cannot be read or never made message `before`.
pub fn fork_session(
    from: &dyn JournalStore,
    to: &dyn JournalStore,
    before: Option<(&AgentId, MessageId)>,
) -> io::Result<()> {
    let Some((agent, message)) = before else {
        return fork(from, to, |_, log| Some(log.len()));
    };
    let log = LogLines::read(from, &agent.0)?;
    let folded = fold(&log).map_err(|why| io::Error::new(io::ErrorKind::InvalidData, why))?;
    let Some(&at) = folded.made.get(&message) else {
        let why = format!("agent {} made no message {}", agent.0, message.0);
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
pub(crate) mod tests;
