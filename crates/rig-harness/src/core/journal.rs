//! The session's append-only logs. Each agent, a subagent too, has one
//! JSON-lines file in the session directory, [`SessionDir::agent_log`],
//! created at its first message: a header, then one record per committed
//! message, settings change, usage update, compaction and saved plugin
//! component. Records are queued as they happen and written at the end of
//! the frame, a subagent's before its parent's, and at once before a tool
//! that may change something runs. Nothing is ever rewritten: a compaction
//! is one more record. [`restore`](super::restore) folds the logs back at
//! startup.
//!
//! A plugin component is logged when its type is reflected with
//! `#[reflect(Component, Saved)]`; a logged component whose type is gone is
//! skipped on restore.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine;
use bevy_app::OnAppExitSystems;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::error;
use bevy_reflect::CreateTypeData;
use bevy_reflect::serde::TypedReflectSerializer;
use rig::harness_protocol::SessionDir;
use rig_core::completion::{Message, Usage};
use rig_core::message::{
    DocumentSourceKind, Image, ImageMediaType, ToolResultContent, UserContent,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::agent::{
    Agent, AgentId, Conversation, Effort, ModelChoice, Notice, SpawnedBy, SystemPrompt, ToolAccess,
};
use super::compaction::Compacted;
use super::effects::Effects;
use super::inbox::Origin;
use super::usage::Spending;

/// Type data marking a plugin component as part of the session: each
/// change is logged, and restoring the session inserts it again. Derive it
/// with `#[reflect(Component, Saved)]`.
#[derive(Clone)]
pub struct ReflectSaved;

impl<T> CreateTypeData<T> for ReflectSaved {
    fn create_type_data(_input: ()) -> Self {
        Self
    }
}

/// The session's directory, the only place the core writes: the agent logs
/// and the effect log [`SessionDir::effects`]. Inserted before the agent
/// plugins are built.
#[derive(Resource, Clone, Debug)]
pub struct SessionPaths(pub SessionDir);

impl Deref for SessionPaths {
    type Target = SessionDir;

    fn deref(&self) -> &SessionDir {
        &self.0
    }
}

/// The version of the agent logs' layout, in each header.
pub(crate) const LOG_VERSION: u32 = 1;

/// The version every saved plugin component is logged with. A record of a
/// newer version is skipped on restore.
pub(crate) const COMPONENT_VERSION: u32 = 1;

/// How an image stored in [`SessionDir::blobs`] is named in a logged
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

/// An agent's model, reasoning setting, system prompt when it is not the
/// default one, and tools when they are not all of them.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub(crate) struct Settings {
    #[serde(default)]
    pub(crate) model: Option<String>,
    #[serde(default)]
    pub(crate) effort: Effort,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<Vec<String>>,
}

impl Settings {
    /// The settings of an agent with these components.
    pub(crate) fn of(
        model: Option<&ModelChoice>,
        effort: Effort,
        prompt: &SystemPrompt,
        access: &ToolAccess,
    ) -> Self {
        Self {
            model: model.map(|model| model.0.clone()),
            effort,
            prompt: (prompt.0 != SystemPrompt::default().0).then(|| prompt.0.clone()),
            tools: match access {
                ToolAccess::All => None,
                ToolAccess::Only(names) => Some(names.clone()),
            },
        }
    }
}

/// An agent's model calls' usage so far, by model, and the context the
/// last one left.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub(crate) struct UsageRecord {
    #[serde(default)]
    pub(crate) models: BTreeMap<String, Spending>,
    #[serde(default)]
    pub(crate) context: Option<u64>,
}

impl UsageRecord {
    /// The usage of every model summed.
    pub(crate) fn total(&self) -> Spending {
        let mut total = Spending::default();
        for spent in self.models.values() {
            total.add(spent);
        }
        total.context = self.context;
        total
    }
}

/// A saved plugin component's value and the version it was logged with.
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
    pub(crate) settings: Option<Settings>,
    #[serde(default)]
    pub(crate) usage: UsageRecord,
    #[serde(default)]
    pub(crate) components: BTreeMap<String, SavedValue>,
}

/// A compaction: requests send `summary` in place of the messages before
/// the one logged as `first_kept`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) struct CompactionRecord {
    pub(crate) summary: String,
    pub(crate) first_kept: u64,
    #[serde(default)]
    pub(crate) read: BTreeSet<String>,
    #[serde(default)]
    pub(crate) modified: BTreeSet<String>,
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
    /// The turn ended without an answer to the last message, which a
    /// restore leaves unanswered.
    Halt,
    /// The settings; the latest wins.
    Settings(Settings),
    /// The usage so far; the latest wins.
    Usage(UsageRecord),
    /// A saved plugin component, by type path; `value: null` when it was
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
pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or_default()
}

/// One agent's log: where it goes, what is queued for it and the
/// latest-wins state its next records and compactions build on.
pub(crate) struct AgentLog {
    pub(crate) path: PathBuf,
    /// 0 for an agent the user started, 1 for its subagents, and so on.
    pub(crate) depth: usize,
    pub(crate) next_seq: u64,
    /// Whether the log holds a message, so it is written.
    pub(crate) started: bool,
    pub(crate) file: Option<File>,
    pub(crate) pending: Vec<u8>,
    /// The `seq` of the record that began each message of the conversation.
    pub(crate) message_seqs: Vec<u64>,
    /// Whether the last conversation record is a [`Record::Halt`].
    pub(crate) halted: bool,
    pub(crate) settings: Option<Settings>,
    pub(crate) usage: UsageRecord,
    pub(crate) components: BTreeMap<String, SavedValue>,
}

impl AgentLog {
    fn new(path: PathBuf, depth: usize) -> Self {
        Self {
            path,
            depth,
            next_seq: 0,
            started: false,
            file: None,
            pending: Vec::new(),
            message_seqs: Vec::new(),
            halted: false,
            settings: None,
            usage: UsageRecord::default(),
            components: BTreeMap::new(),
        }
    }
}

struct Book {
    dir: Option<SessionDir>,
    /// Records are queued only once the session was restored.
    live: bool,
    /// The app is exiting: a turn that ends now is left for the restart.
    exiting: bool,
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
        let dir = self.dir.as_ref()?;
        if !self.agents.contains_key(agent) {
            let mut log = AgentLog::new(dir.agent_log(agent), 0);
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
/// without a [`SessionPaths`].
#[derive(Resource, Clone)]
pub struct SessionLog(Arc<Mutex<Book>>);

impl SessionLog {
    /// The logs of the session in `dir`, idle until [`Self::resume`].
    pub(crate) fn new(dir: Option<SessionDir>) -> Self {
        Self(Arc::new(Mutex::new(Book {
            dir,
            live: false,
            exiting: false,
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

    /// Whether records are logged.
    pub(crate) fn is_live(&self) -> bool {
        let book = self.book();
        book.live && book.failure.is_none()
    }

    /// The app is exiting: the turns it stops are left for the restart.
    pub(crate) fn set_exiting(&self) {
        self.book().exiting = true;
    }

    /// Whether the app is exiting.
    pub(crate) fn is_exiting(&self) -> bool {
        self.book().exiting
    }

    /// Starts the log of `child`, which `parent` spawned, unless it has
    /// one.
    pub(crate) fn open_child(&self, child: &AgentId, parent: &AgentId) {
        let mut book = self.book();
        if !book.live || book.failure.is_some() || book.agents.contains_key(&child.0) {
            return;
        }
        let Some(dir) = book.dir.as_ref() else {
            return;
        };
        let depth = book.agents.get(&parent.0).map_or(0, |log| log.depth) + 1;
        let mut log = AgentLog::new(dir.agent_log(&child.0), depth);
        let header = header(&child.0, Some(parent.0.clone()));
        match enqueue(&mut log, Record::Header(header)) {
            Ok(_) => {
                book.agents.insert(child.0.clone(), log);
            }
            Err(failure) => book.failure = Some(failure.to_string()),
        }
    }

    /// Adds `message` to the conversation of `agent` and logs it, with its
    /// images stored in [`SessionDir::blobs`]. `origin` says where it came
    /// from, when not from the user, the model or a tool. The one way
    /// messages are added.
    pub(crate) fn commit(
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
                log.halted = false;
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
        let blobs = book.dir.as_ref()?.blobs();
        match stored(message, &blobs) {
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
            log.halted = false;
        }
        Some(message)
    }

    /// Logs that the turn of `agent` ended without answering the last
    /// message of `conversation`, when it is the user's, so a restore does
    /// not send it to the model. Nothing while the app exits: then the
    /// restart carries the turn on.
    pub(crate) fn halt(&self, agent: &AgentId, conversation: &Conversation) {
        if !matches!(conversation.messages().last(), Some(Message::User { .. })) {
            return;
        }
        let mut book = self.book();
        if book.exiting
            || book
                .agents
                .get(&agent.0)
                .is_none_or(|log| log.halted || !log.started)
        {
            return;
        }
        if book.record(&agent.0, Record::Halt).is_some()
            && let Some(log) = book.agents.get_mut(&agent.0)
        {
            log.halted = true;
        }
    }

    /// Logs the settings of `agent` when they changed.
    pub(crate) fn settings(&self, agent: &AgentId, settings: Settings) {
        let mut book = self.book();
        if book
            .agents
            .get(&agent.0)
            .is_some_and(|log| log.settings.as_ref() == Some(&settings))
        {
            return;
        }
        if book
            .record(&agent.0, Record::Settings(settings.clone()))
            .is_some()
            && let Some(log) = book.agents.get_mut(&agent.0)
        {
            log.settings = Some(settings);
        }
    }

    /// Adds a finished call's `usage` to what `agent` spent on `model`,
    /// and logs the totals with the `context` the agent is at.
    pub(crate) fn usage(&self, agent: &AgentId, model: &str, usage: &Usage, context: Option<u64>) {
        let mut book = self.book();
        let Some(log) = book.agent(&agent.0) else {
            return;
        };
        log.usage
            .models
            .entry(model.to_owned())
            .or_default()
            .record(usage);
        log.usage.context = context;
        let record = Record::Usage(log.usage.clone());
        book.record(&agent.0, record);
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
            read: compacted.read.clone(),
            modified: compacted.modified.clone(),
            snapshot: Snapshot {
                settings: log.settings.clone(),
                usage: log.usage.clone(),
                components: log.components.clone(),
            },
        });
        book.record(&agent.0, record);
    }

    /// The saved plugin components logged as present on `agent`, by type
    /// path.
    pub(crate) fn components(&self, agent: &AgentId) -> BTreeSet<String> {
        self.book()
            .agents
            .get(&agent.0)
            .map(|log| log.components.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Logs the saved plugin component `component` of `agent` when its
    /// value changed; `None` when it was removed.
    pub(crate) fn component(&self, agent: &AgentId, component: &str, value: Option<Value>) {
        let mut book = self.book();
        let saved = value.map(|value| SavedValue {
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
            if let Err(failure) = write_pending(log) {
                book.failure = Some(format!("writing {} failed: {failure}", log.path.display()));
                return;
            }
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

fn write_pending(log: &mut AgentLog) -> io::Result<()> {
    let file = match log.file.take() {
        Some(file) => file,
        None => OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log.path)?,
    };
    log.file.insert(file).write_all(&log.pending)?;
    log.pending.clear();
    Ok(())
}

/// `message` as it is logged: each image's data stored in `blobs` and
/// named by its hash.
fn stored(message: &Message, blobs: &Path) -> io::Result<Message> {
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
fn store(image: &mut Image, blobs: &Path) -> io::Result<()> {
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
        extension(image.media_type.as_ref())
    );
    let path = blobs.join(&name);
    if !path.exists() {
        fs::create_dir_all(blobs)?;
        let temporary = path.with_extension("tmp");
        fs::write(&temporary, &bytes)?;
        fs::rename(&temporary, &path)?;
    }
    image.data = DocumentSourceKind::Url(format!("{BLOB}{name}"));
    Ok(())
}

/// The file extension of an image of `media_type`.
fn extension(media_type: Option<&ImageMediaType>) -> &'static str {
    match media_type {
        Some(ImageMediaType::JPEG) => "jpg",
        Some(ImageMediaType::PNG) => "png",
        Some(ImageMediaType::GIF) => "gif",
        Some(ImageMediaType::WEBP) => "webp",
        Some(ImageMediaType::HEIC) => "heic",
        Some(ImageMediaType::HEIF) => "heif",
        Some(ImageMediaType::SVG) => "svg",
        None => "bin",
    }
}

/// Puts the data of each image `message` names in `blobs` back in place;
/// an image whose file is gone becomes a line saying so.
pub(crate) fn load_blobs(message: &mut Message, blobs: &Path) {
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

fn load(image: &mut Image, blobs: &Path) -> Result<(), String> {
    let DocumentSourceKind::Url(url) = &image.data else {
        return Ok(());
    };
    let Some(name) = url.strip_prefix(BLOB) else {
        return Ok(());
    };
    let bytes = fs::read(blobs.join(name))
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
        app.add_systems(PreStartup, super::restore::restore_session)
            .add_systems(Startup, super::restore::reconcile)
            .add_systems(
                Last,
                (log_saved_components, write_logs)
                    .chain()
                    .in_set(OnAppExitSystems)
                    .after(super::turn::stop_turns_on_exit),
            )
            .add_observer(log_settings)
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

/// Logs an agent's settings when its model or reasoning setting is
/// inserted, which is how both change.
fn log_settings(
    inserted: On<Insert<(ModelChoice, Effort)>>,
    agents: Query<(
        &AgentId,
        Option<&ModelChoice>,
        &Effort,
        &SystemPrompt,
        &ToolAccess,
    )>,
    log: Res<SessionLog>,
) {
    if let Ok((id, model, &effort, prompt, access)) = agents.get(inserted.entity) {
        log.settings(id, Settings::of(model, effort, prompt, access));
    }
}

/// Logs each saved plugin component that changed since the last frame,
/// and each one removed, on every agent.
fn log_saved_components(world: &mut World) {
    let (Some(log), Some(registry)) = (
        world.get_resource::<SessionLog>().cloned(),
        world.get_resource::<AppTypeRegistry>().cloned(),
    ) else {
        return;
    };
    if !log.is_live() {
        return;
    }
    let (last_run, this_run) = (world.last_change_tick(), world.change_tick());
    let registry = registry.read();
    let agents: Vec<(Entity, AgentId)> = world
        .query_filtered::<(Entity, &AgentId), With<Agent>>()
        .iter(world)
        .map(|(entity, id)| (entity, id.clone()))
        .collect();
    for (entity, id) in agents {
        let Ok(agent) = world.get_entity(entity) else {
            continue;
        };
        let known = log.components(&id);
        for (registration, _) in registry.iter_with_data::<ReflectSaved>() {
            let path = registration.type_info().type_path();
            let (Some(component), Some(component_id)) = (
                registration.data::<ReflectComponent>(),
                world.components().get_id(registration.type_id()),
            ) else {
                continue;
            };
            let Some(ticks) = agent.get_change_ticks_by_id(component_id) else {
                if known.contains(path) {
                    log.component(&id, path, None);
                }
                continue;
            };
            if known.contains(path) && !ticks.is_changed(last_run, this_run) {
                continue;
            }
            let Some(value) = component.reflect(agent) else {
                continue;
            };
            match serde_json::to_value(TypedReflectSerializer::new(
                value.as_partial_reflect(),
                &registry,
            )) {
                Ok(value) => log.component(&id, path, Some(value)),
                Err(failure) => error!("not logging {path}: {failure}"),
            }
        }
    }
}

/// Writes the frame's records and the resolved effects. A failure is
/// shown once.
fn write_logs(
    log: Res<SessionLog>,
    effects: Res<Effects>,
    paths: Option<Res<SessionPaths>>,
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
    if let Some(paths) = paths
        && let Err(failure) = effects.flush(&paths.effects())
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
