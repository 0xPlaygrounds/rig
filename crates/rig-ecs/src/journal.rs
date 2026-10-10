//! The session's append-only logs. Each agent, a subagent too, has one
//! JSON-lines log in the session's [`SessionStore`], created at its first
//! message: a header, then one record per change a [`Commit`] made, per
//! [`Condensed`] and per change of a [`ReflectSaved`] component, such as
//! the agent's model, reasoning setting, system prompt, tool access and
//! last usage. Records are queued as they happen and written
//! at the end of the frame, a subagent's before its parent's, and at once
//! before a tool that may change something runs. Nothing is ever
//! rewritten; [`restore`](super::restore) folds the logs back at startup.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use bevy_app::OnAppExitSystems;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::reflect::AppTypeRegistry;
use bevy_ecs::system::SystemParam;
use bevy_log::error;
use bevy_reflect::serde::TypedReflectSerializer;
use bevy_reflect::{CreateTypeData, Reflect, TypePath};
use rig_cassette::journal::{JournalStore, store_images};
use rig_core::completion::Message;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use web_time::{SystemTime, UNIX_EPOCH};

use super::agent::{Agent, AgentId, Condensed, Conversation, Halt, Notice, SpawnedBy};
use super::inbox::Origin;
use super::{StopTurns, WriteJournal};

/// Saves an agent component with the session: derive `Reflect` and add
/// `Saved` to its `#[reflect(..)]`. Each change and removal of it on an
/// agent is logged at the end of the frame by its type path and restored
/// by reflection; a value that serializes to `null` is saved as absent.
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
            let log = log_changed::<T>.in_set(OnAppExitSystems).after(StopTurns);
            app.add_systems(Last, log.before(WriteJournal));
        };
        Self { watch }
    }
}

/// Logs each change and removal of the saved component `T` on the agents;
/// reflection runs only for a change.
fn log_changed<T: Component + Reflect + TypePath>(
    changed: Query<(&AgentId, &T), (With<Agent>, Changed<T>)>,
    mut removed: RemovedComponents<T>,
    agents: Query<&AgentId, With<Agent>>,
    log: Res<SessionLog>,
    registry: Res<AppTypeRegistry>,
) {
    if !log.is_live() {
        return;
    }
    let name = T::type_path();
    let registry = registry.read();
    for (id, component) in &changed {
        let value = TypedReflectSerializer::new(component.as_partial_reflect(), &registry);
        match serde_json::to_value(value) {
            Ok(value) => log.component(id, name, Some(value)),
            Err(failure) => error!("not logging {name}: {failure}"),
        }
    }
    for entity in removed.read() {
        if let Ok(id) = agents.get(entity)
            && !changed.contains(entity)
        {
            log.component(id, name, None);
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
        /// Where it came from, when it was not the user's own text, the
        /// model's reply or tool results.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        origin: Option<Origin>,
    },
    /// The last message was taken out, such as a user message no model
    /// could answer.
    Retract,
    /// The user's last message was left unanswered, which a restore keeps.
    Halt { reason: Halt },
    /// A saved component, by type path; `value: null` when it was
    /// removed. The latest per type wins.
    Component {
        component: String,
        value: Option<Value>,
    },
    /// A [`Condensed`] conversation: requests send `summary` in place of
    /// the messages before the one logged as `first_kept`.
    Condensed { summary: String, first_kept: u64 },
}

/// A line of an agent log.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) struct Line<'a> {
    /// Dense from 0, the header's, in each log.
    pub(crate) seq: u64,
    /// When it was written, in milliseconds since the Unix epoch.
    pub(crate) t: u64,
    #[serde(flatten)]
    pub(crate) record: Record<'a>,
}

/// Milliseconds since the Unix epoch.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
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
    /// Whether the log holds a message, so it is written.
    pub(crate) started: bool,
    pub(crate) pending: Vec<u8>,
    pub(crate) components: HashMap<String, Value>,
}

#[derive(Default)]
struct Book {
    store: Option<Arc<dyn JournalStore>>,
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
            if let Err(failure) = enqueue(&mut log, Record::Header(header)) {
                self.failure = Some(failure.to_string());
                return None;
            }
            self.agents.insert(agent.to_owned(), log);
        }
        self.agents.get_mut(agent)
    }

    /// Queues `record` for `agent`; its `seq`, or `None` when nothing is
    /// logged.
    fn record(&mut self, agent: &str, record: Record<'_>) -> Option<u64> {
        let queued = enqueue(self.agent(agent, None)?, record);
        match queued {
            Ok(seq) => Some(seq),
            Err(failure) => {
                self.failure = Some(failure.to_string());
                None
            }
        }
    }
}

/// Queues `record` as the next line of `log`; its `seq`.
fn enqueue(log: &mut AgentLog, record: Record<'_>) -> serde_json::Result<u64> {
    let seq = log.next_seq;
    let line = Line {
        seq,
        t: now_ms(),
        record,
    };
    serde_json::to_writer(&mut log.pending, &line)?;
    log.pending.push(b'\n');
    log.next_seq += 1;
    Ok(seq)
}

/// Where the session is kept, such as a rig-cassette `MemoryStore` or
/// `JsonlDirStore`: inserted before the agent plugins are built. Without
/// one nothing is kept.
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
#[derive(Resource, Clone)]
pub struct SessionLog(Arc<Mutex<Book>>);

impl SessionLog {
    /// The logs of the session kept in `store`, idle until [`Self::resume`].
    pub(crate) fn new(store: Option<Arc<dyn JournalStore>>) -> Self {
        Self(Arc::new(Mutex::new(Book {
            store,
            ..Book::default()
        })))
    }

    fn book(&self) -> MutexGuard<'_, Book> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Starts logging, after the logs restored as `agents`, by agent id.
    pub(crate) fn resume(&self, agents: Vec<(String, AgentLog)>) {
        let mut book = self.book();
        book.agents.extend(agents);
        book.live = true;
    }

    /// Whether records are logged: the session was restored, and no write
    /// failed.
    pub fn is_live(&self) -> bool {
        let book = self.book();
        book.live && book.failure.is_none()
    }

    /// Logs `message` of `agent`, which starts its log; its `seq`.
    fn log_message(
        &self,
        agent: &AgentId,
        message: &Message,
        origin: Option<Origin>,
    ) -> Option<u64> {
        let mut book = self.book();
        if !book.live || book.failure.is_some() {
            return None;
        }
        let store = book.store.clone()?;
        match store_images(message, &*store) {
            Ok(message) => {
                let seq = book.record(&agent.0, Record::Message { message, origin })?;
                book.agents.get_mut(&agent.0)?.started = true;
                Some(seq)
            }
            Err(failure) => {
                book.failure = Some(format!("storing an image failed: {failure}"));
                None
            }
        }
    }

    /// Logs the saved component `component` of `agent` when its value
    /// changed; `None` or `null` when it was removed.
    pub(crate) fn component(&self, agent: &AgentId, component: &str, value: Option<Value>) {
        let mut book = self.book();
        let value = value.filter(|value| !value.is_null());
        let known = book
            .agents
            .get(&agent.0)
            .and_then(|log| log.components.get(component));
        if known == value.as_ref() {
            return;
        }
        let record = Record::Component {
            component: component.to_owned(),
            value: value.clone(),
        };
        if book.record(&agent.0, record).is_some()
            && let Some(log) = book.agents.get_mut(&agent.0)
        {
            match value {
                Some(value) => log.components.insert(component.to_owned(), value),
                None => log.components.remove(component),
            };
        }
    }

    /// Writes every queued record: each started log's records with one
    /// write, a subagent's log before its parent's, so a parent never
    /// refers to a subagent message that is not on disk. No fsync. A failed
    /// write stops the logging.
    pub(crate) fn flush(&self) {
        let mut guard = self.book();
        let book = &mut *guard;
        if book.failure.is_some() {
            return;
        }
        let Some(store) = book.store.clone() else {
            return;
        };
        let mut due: Vec<(usize, String)> = book
            .agents
            .iter()
            .filter(|(_, log)| log.started && !log.pending.is_empty())
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

/// Changes agents' conversations, the one way they change: each change is
/// logged, with a message's images stored as blobs, and announced as a
/// [`Committed`], so a plugin sees every change the log records.
#[derive(SystemParam)]
pub struct Commit<'w, 's> {
    ids: Query<'w, 's, &'static AgentId>,
    log: Res<'w, SessionLog>,
    committed: MessageWriter<'w, Committed>,
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
        let seq = self.log.log_message(id, &message, origin.clone());
        self.committed.write(Committed::Message {
            agent,
            message: message.clone(),
            origin: origin.clone(),
        });
        conversation.append(message, origin, seq);
    }

    /// Takes the last message out of `conversation`, that of `agent`, such
    /// as a user message no model could answer.
    pub(crate) fn retract(&mut self, agent: Entity, conversation: &mut Conversation) {
        if let Ok(id) = self.ids.get(agent)
            && conversation.retract().is_some()
        {
            self.log.book().record(&id.0, Record::Retract);
            self.committed.write(Committed::Retract { agent });
        }
    }

    /// Leaves the user's last message in `conversation`, that of `agent`,
    /// unanswered for `reason` when the model owes an answer to it, so a
    /// restore does not send it to the model. Not for a turn the app's
    /// exit stops (see [`Exiting`](super::turn::Exiting)): the restart
    /// carries that one on.
    pub fn halt(&mut self, agent: Entity, conversation: &mut Conversation, reason: Halt) {
        if let Ok(id) = self.ids.get(agent)
            && conversation.halt(reason)
        {
            self.log.book().record(&id.0, Record::Halt { reason });
            self.committed.write(Committed::Halt { agent, reason });
        }
    }
}

/// Adds `message` to the conversation of `agent` as [`Commit::message`]
/// does, from an exclusive system or a test.
pub fn commit_message(world: &mut World, agent: Entity, message: Message) {
    let add = |In((agent, message)): In<(Entity, Message)>,
               mut agents: Query<&mut Conversation>,
               mut commit: Commit| {
        if let Ok(mut conversation) = agents.get_mut(agent) {
            commit.message(agent, &mut conversation, message);
        }
    };
    if let Err(error) = world.run_system_cached_with(add, (agent, message)) {
        error!("could not add a message: {error}");
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
    /// The last message was taken out.
    Retract { agent: Entity },
    /// The user's last message was left unanswered.
    Halt { agent: Entity, reason: Halt },
}

/// Restores the session at startup, settles what a crash or restart left
/// half done, and logs everything after.
pub struct JournalPlugin;

impl Plugin for JournalPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreStartup, super::restore::restore_session)
            .add_systems(Startup, super::restore::reconcile)
            .add_systems(
                Last,
                write_logs
                    .in_set(OnAppExitSystems)
                    .in_set(WriteJournal)
                    .after(StopTurns),
            )
            .add_observer(open_child_log)
            .add_observer(log_condensed);
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

/// Logs each [`Condensed`] inserted on an agent; one a restore inserts is
/// not, as nothing is logged before the session was restored.
fn log_condensed(
    insert: On<Insert<Condensed>>,
    agents: Query<(&AgentId, &Conversation, &Condensed)>,
    log: Res<SessionLog>,
) {
    let Ok((id, conversation, condensed)) = agents.get(insert.entity) else {
        return;
    };
    let mut book = log.book();
    if let Some(log) = book.agent(&id.0, None) {
        let first_kept = conversation.seq(condensed.upto).unwrap_or(log.next_seq);
        let summary = condensed.summary.clone();
        book.record(
            &id.0,
            Record::Condensed {
                summary,
                first_kept,
            },
        );
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
