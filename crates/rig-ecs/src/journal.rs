//! The session's append-only logs. Each agent, a subagent too, has one
//! JSON-lines log in the session's [`SessionStore`], created at its first
//! message: a header, then one record per committed message, compaction
//! and change of a saved component. Records are queued as they happen and
//! written at the end of the frame, a subagent's before its parent's, and
//! at once before a tool that may change something runs. Nothing is ever rewritten: a compaction
//! is one more record. [`restore`](super::restore) folds the logs back at
//! startup.
//!
//! A component is saved when it was registered with
//! [`AppSaveExt::save_component`], as the agent's model, reasoning setting,
//! system prompt, tool access and spending are; a logged component no
//! plugin registers any more is skipped on restore.

use std::collections::{BTreeMap, HashMap};
use std::io;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use base64::Engine;
use bevy_app::OnAppExitSystems;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::error;
use rig_core::completion::Message;
use rig_core::message::{
    DocumentSourceKind, Image, ImageMediaType, ToolResultContent, UserContent,
};
use rig_memory::TrackedSet;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use web_time::{SystemTime, UNIX_EPOCH};

use super::agent::{
    Agent, AgentId, Conversation, Effort, Halt, ModelChoice, Notice, SpawnedBy, SystemPrompt,
    ToolAccess,
};
use super::compaction::Compacted;
use super::effects::Effects;
use super::inbox::Origin;
use super::store::{JournalStore, SessionStore};
use super::usage::Spending;
use super::{StopTurns, WriteJournal};

/// Registers plugin components that are part of the session.
pub trait AppSaveExt {
    /// Saves the agent component `T` with the session: each change of it
    /// on an agent is logged at the end of the frame, as is its removal,
    /// and restoring the session inserts it again, in the order the
    /// components were registered. A value that serializes to `null` is
    /// saved as absent. Its type name names it in the log, so renaming the
    /// type drops what was saved.
    fn save_component<T: Component + Serialize + DeserializeOwned>(&mut self) -> &mut Self;
}

impl AppSaveExt for App {
    fn save_component<T: Component + Serialize + DeserializeOwned>(&mut self) -> &mut Self {
        self.world_mut()
            .get_resource_or_init::<SavedComponents>()
            .0
            .push((std::any::type_name::<T>(), insert_saved::<T>));
        self.add_systems(
            Last,
            log_saved::<T>
                .in_set(OnAppExitSystems)
                .after(StopTurns)
                .before(WriteJournal),
        )
    }
}

/// Inserts a saved component's logged value on an agent.
pub(crate) type InsertSaved = fn(&mut EntityWorldMut, Value) -> serde_json::Result<()>;

/// How each component registered with [`AppSaveExt::save_component`] is
/// restored, by type name, in the order registered.
#[derive(Resource, Default)]
pub(crate) struct SavedComponents(pub(crate) Vec<(&'static str, InsertSaved)>);

fn insert_saved<T: Component + DeserializeOwned>(
    agent: &mut EntityWorldMut,
    value: Value,
) -> serde_json::Result<()> {
    agent.insert(serde_json::from_value::<T>(value)?);
    Ok(())
}

/// Logs each change and removal of the saved component `T` on the agents.
fn log_saved<T: Component + Serialize>(
    changed: Query<(&AgentId, &T), (With<Agent>, Changed<T>)>,
    mut removed: RemovedComponents<T>,
    agents: Query<&AgentId, With<Agent>>,
    log: Res<SessionLog>,
) {
    if !log.is_live() {
        return;
    }
    let name = std::any::type_name::<T>();
    for (id, component) in &changed {
        match serde_json::to_value(component) {
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

/// The version of the agent logs' layout, in each header.
pub(crate) const LOG_VERSION: u32 = 1;

/// The version every saved plugin component is logged with. A record of a
/// newer version is skipped on restore.
pub(crate) const COMPONENT_VERSION: u32 = 1;

/// How an image stored as a blob is named in a logged
/// message, in place of its data: `blob:<sha256>.<ext>`.
pub(crate) const BLOB: &str = "blob:";

/// The first record of an agent log.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) struct Header {
    /// [`LOG_VERSION`].
    pub(crate) v: u32,
    /// The agent's id.
    pub(crate) agent: String,
    /// The id of the agent it was [`SpawnedBy`](super::agent::SpawnedBy),
    /// if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) parent: Option<String>,
}

/// A saved component's value and the version it was logged with.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub(crate) struct SavedValue {
    pub(crate) v: u32,
    pub(crate) value: Value,
}

/// The latest-wins state a compaction carries, so a restore need not read
/// what came before it.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub(crate) struct Snapshot {
    #[serde(default)]
    pub(crate) components: BTreeMap<String, SavedValue>,
}

/// A compaction: requests send `summary` in place of the messages before
/// the one logged as `first_kept`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) struct CompactionRecord {
    pub(crate) summary: String,
    pub(crate) first_kept: u64,
    /// The files the summarized messages read and changed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) tracked: Vec<TrackedSet>,
    pub(crate) snapshot: Snapshot,
}

/// One record of an agent log, tagged by `type`.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum Record {
    /// The first record.
    Header(Header),
    /// A message added to the conversation; a user message after a user
    /// message goes into it.
    Message {
        message: Message,
        /// Where it came from, when it was not the user's own text, the
        /// model's reply or tool results.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        origin: Option<Origin>,
    },
    /// The last message was taken out, such as a user message no model
    /// could answer.
    Retract,
    /// The user's last message was left unanswered, which a restore keeps;
    /// an older log's halt is [`Halt::Kept`].
    Halt {
        #[serde(default)]
        reason: Halt,
    },
    /// A saved component, by type path; `value: null` when it was
    /// removed. The latest per type wins.
    Component {
        component: String,
        v: u32,
        value: Option<Value>,
    },
    /// A compaction, with the latest-wins state at that point.
    Compaction(CompactionRecord),
}

/// A line of an agent log.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) struct Line {
    /// Dense from 0, the header's, in each log.
    pub(crate) seq: u64,
    /// When it was written, in milliseconds since the Unix epoch.
    pub(crate) t: u64,
    #[serde(flatten)]
    pub(crate) record: Record,
}

/// Milliseconds since the Unix epoch.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or_default()
}

/// One agent's log: where it goes, what is queued for it and the
/// latest-wins state its next records and compactions build on.
pub(crate) struct AgentLog {
    /// 0 for an agent the user started, 1 for its subagents, and so on.
    pub(crate) depth: usize,
    pub(crate) next_seq: u64,
    /// Whether the log holds a message, so it is written.
    pub(crate) started: bool,
    pub(crate) pending: Vec<u8>,
    /// The `seq` of the record that began each message of the conversation.
    pub(crate) message_seqs: Vec<u64>,
    pub(crate) components: BTreeMap<String, SavedValue>,
}

impl AgentLog {
    fn new(depth: usize) -> Self {
        Self {
            depth,
            next_seq: 0,
            started: false,
            pending: Vec::new(),
            message_seqs: Vec::new(),
            components: BTreeMap::new(),
        }
    }
}

struct Book {
    store: Option<Arc<dyn JournalStore>>,
    /// Records are queued only once the session was restored.
    live: bool,
    /// Why the log stopped: every record after a failed write is dropped.
    failure: Option<String>,
    /// Whether the failure was reported.
    reported: bool,
    agents: HashMap<String, AgentLog>,
}

impl Book {
    /// The log of `agent`, started with a header when it has none.
    fn agent(&mut self, agent: &str) -> Option<&mut AgentLog> {
        if !self.live || self.failure.is_some() {
            return None;
        }
        self.store.as_ref()?;
        if !self.agents.contains_key(agent) {
            let mut log = AgentLog::new(0);
            let header = header(agent, None);
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
    fn record(&mut self, agent: &str, record: Record) -> Option<u64> {
        let queued = enqueue(self.agent(agent)?, record);
        match queued {
            Ok(seq) => Some(seq),
            Err(failure) => {
                self.failure = Some(failure.to_string());
                None
            }
        }
    }
}

fn header(agent: &str, parent: Option<String>) -> Header {
    Header {
        v: LOG_VERSION,
        agent: agent.to_owned(),
        parent,
    }
}

/// Queues `record` as the next line of `log`; its `seq`.
fn enqueue(log: &mut AgentLog, record: Record) -> serde_json::Result<u64> {
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
            live: false,
            failure: None,
            reported: false,
            agents: HashMap::new(),
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

    /// Starts the log of `child`, which `parent` spawned, unless it has
    /// one.
    pub(crate) fn open_child(&self, child: &AgentId, parent: &AgentId) {
        let mut book = self.book();
        if !book.live || book.failure.is_some() || book.agents.contains_key(&child.0) {
            return;
        }
        if book.store.is_none() {
            return;
        }
        let depth = book.agents.get(&parent.0).map_or(0, |log| log.depth) + 1;
        let mut log = AgentLog::new(depth);
        let header = header(&child.0, Some(parent.0.clone()));
        match enqueue(&mut log, Record::Header(header)) {
            Ok(_) => {
                book.agents.insert(child.0.clone(), log);
            }
            Err(failure) => book.failure = Some(failure.to_string()),
        }
    }

    /// Adds `message` to the conversation of `agent` and logs it, with its
    /// images stored as blobs. `origin` says where it came from, when not
    /// from the user, the model or a tool. The one way messages are added,
    /// such as a note a plugin puts in an idle agent's conversation without
    /// starting a turn.
    pub fn commit(
        &self,
        agent: &AgentId,
        conversation: &mut Conversation,
        message: Message,
        origin: Option<Origin>,
    ) {
        let logged = self.log_message(agent, &message, origin.clone());
        let merged = conversation.append(message, origin);
        if let Some(seq) = logged {
            let mut book = self.book();
            if let Some(log) = book.agents.get_mut(&agent.0) {
                log.started = true;
                if !merged {
                    log.message_seqs.push(seq);
                }
            }
        }
    }

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
        match stored(message, &*store) {
            Ok(message) => book.record(&agent.0, Record::Message { message, origin }),
            Err(failure) => {
                book.failure = Some(format!("storing an image failed: {failure}"));
                None
            }
        }
    }

    /// Takes the last message out of the conversation of `agent`, and logs
    /// that.
    pub(crate) fn retract(
        &self,
        agent: &AgentId,
        conversation: &mut Conversation,
    ) -> Option<Message> {
        let message = conversation.retract()?;
        let mut book = self.book();
        if book.record(&agent.0, Record::Retract).is_some()
            && let Some(log) = book.agents.get_mut(&agent.0)
        {
            log.message_seqs.pop();
        }
        Some(message)
    }

    /// Halts the conversation of `agent` for `reason` when the model owes
    /// an answer to its last message, the user's, and logs that, so a
    /// restore does not send it to the model. Not for a turn the app's exit
    /// stops (see [`Exiting`](super::turn::Exiting)): the restart carries
    /// that one on.
    pub fn halt(&self, agent: &AgentId, conversation: &mut Conversation, reason: Halt) {
        if conversation.halt(reason) {
            self.book().record(&agent.0, Record::Halt { reason });
        }
    }

    /// Logs the compaction `compacted` of `agent`, with the state it
    /// carries.
    pub(crate) fn compaction(&self, agent: &AgentId, compacted: &Compacted) {
        let mut book = self.book();
        let Some(log) = book.agent(&agent.0) else {
            return;
        };
        let first_kept = log
            .message_seqs
            .get(compacted.upto)
            .copied()
            .unwrap_or(log.next_seq);
        let record = Record::Compaction(CompactionRecord {
            summary: compacted.summary.clone(),
            first_kept,
            tracked: compacted.tracked.clone(),
            snapshot: Snapshot {
                components: log.components.clone(),
            },
        });
        book.record(&agent.0, record);
    }

    /// Logs the saved component `component` of `agent` when its value
    /// changed; `None` or `null` when it was removed.
    pub(crate) fn component(&self, agent: &AgentId, component: &str, value: Option<Value>) {
        let mut book = self.book();
        let saved = value
            .filter(|value| !value.is_null())
            .map(|value| SavedValue {
                v: COMPONENT_VERSION,
                value,
            });
        let known = book
            .agents
            .get(&agent.0)
            .and_then(|log| log.components.get(component));
        if known == saved.as_ref() {
            return;
        }
        let record = Record::Component {
            component: component.to_owned(),
            v: COMPONENT_VERSION,
            value: saved.as_ref().map(|saved| saved.value.clone()),
        };
        if book.record(&agent.0, record).is_some()
            && let Some(log) = book.agents.get_mut(&agent.0)
        {
            match saved {
                Some(saved) => log.components.insert(component.to_owned(), saved),
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

    /// Why the log stopped, the first time it is asked after it did.
    fn take_failure(&self) -> Option<String> {
        let mut book = self.book();
        if book.reported {
            return None;
        }
        let failure = book.failure.clone()?;
        book.reported = true;
        Some(failure)
    }
}

/// `message` as it is logged: each image's data stored as a blob in
/// `blobs` and named by its hash.
fn stored(message: &Message, blobs: &dyn JournalStore) -> io::Result<Message> {
    let mut message = message.clone();
    if let Message::User { content } = &mut message {
        for item in content {
            match item {
                UserContent::Image(image) => store(image, blobs)?,
                UserContent::ToolResult(result) => {
                    for part in &mut result.content {
                        if let ToolResultContent::Image(image) = part {
                            store(image, blobs)?;
                        }
                    }
                }
                _ => {}
            }
        }
    }
    Ok(message)
}

/// Stores the data of `image` in `blobs`, once per content, and names it
/// there instead.
fn store(image: &mut Image, blobs: &dyn JournalStore) -> io::Result<()> {
    let bytes = match &image.data {
        DocumentSourceKind::Base64(data) => {
            match base64::engine::general_purpose::STANDARD.decode(data) {
                Ok(bytes) => bytes,
                // Kept inline: it is not ours to fix.
                Err(_) => return Ok(()),
            }
        }
        DocumentSourceKind::Raw(bytes) => bytes.clone(),
        _ => return Ok(()),
    };
    let name = format!(
        "{:x}.{}",
        Sha256::digest(&bytes),
        image
            .media_type
            .as_ref()
            .map_or("bin", ImageMediaType::extension)
    );
    blobs.put_blob(&name, &bytes)?;
    image.data = DocumentSourceKind::Url(format!("{BLOB}{name}"));
    Ok(())
}

/// Puts the data of each image `message` names in `blobs` back in place;
/// an image whose file is gone becomes a line saying so.
pub(crate) fn load_blobs(message: &mut Message, blobs: &dyn JournalStore) {
    let Message::User { content } = message else {
        return;
    };
    for item in content.iter_mut() {
        match item {
            UserContent::Image(image) => {
                if let Err(why) = load(image, blobs) {
                    *item = UserContent::text(why);
                }
            }
            UserContent::ToolResult(result) => {
                for part in &mut result.content {
                    if let ToolResultContent::Image(image) = part
                        && let Err(why) = load(image, blobs)
                    {
                        *part = ToolResultContent::text(why);
                    }
                }
            }
            _ => {}
        }
    }
}

fn load(image: &mut Image, blobs: &dyn JournalStore) -> Result<(), String> {
    let DocumentSourceKind::Url(url) = &image.data else {
        return Ok(());
    };
    let Some(name) = url.strip_prefix(BLOB) else {
        return Ok(());
    };
    let bytes = blobs
        .blob(name)
        .map_err(|failure| format!("[an image of this message is gone: {failure}]"))?;
    image.data =
        DocumentSourceKind::Base64(base64::engine::general_purpose::STANDARD.encode(bytes));
    Ok(())
}

/// Restores the session at startup, settles what a crash or restart left
/// half done, and logs everything after.
pub struct JournalPlugin;

impl Plugin for JournalPlugin {
    fn build(&self, app: &mut App) {
        // The reasoning setting first: inserting the model checks it.
        app.save_component::<Effort>()
            .save_component::<ModelChoice>()
            .save_component::<SystemPrompt>()
            .save_component::<ToolAccess>()
            .save_component::<Spending>()
            .add_systems(PreStartup, super::restore::restore_session)
            .add_systems(Startup, super::restore::reconcile)
            .add_systems(
                Last,
                write_logs
                    .in_set(OnAppExitSystems)
                    .in_set(WriteJournal)
                    .after(StopTurns),
            )
            .add_observer(open_child_log);
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
        log.open_child(child, parent);
    }
}

/// Writes the frame's records and the resolved effects. A failure is
/// shown once.
fn write_logs(
    log: Res<SessionLog>,
    effects: Res<Effects>,
    store: Option<Res<SessionStore>>,
    mut effects_failed: Local<bool>,
    mut notices: MessageWriter<Notice>,
) {
    log.flush();
    if let Some(failure) = log.take_failure() {
        error!("the session log stopped: {failure}");
        notices.write(Notice::error(
            None,
            format!("The session is no longer saved: {failure}"),
        ));
    }
    if let Some(store) = store
        && let Err(failure) = store.0.append_effects(&effects.take())
        && !*effects_failed
    {
        *effects_failed = true;
        error!("writing the effect log failed: {failure}");
        notices.write(Notice::error(
            None,
            format!("Writing the effect log failed: {failure}"),
        ));
    }
}
