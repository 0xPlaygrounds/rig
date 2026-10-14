//! CT-13: reusable context snapshots.
//!
//! `/snapshot save <name> [--at <turn>] [--shared]` copies the session, whole or as it
//! was before turn `<turn>` began, with rig-memory's `fork` into a store in memory and
//! keeps it in the project's plugin data: every agent's messages, its own system prompt,
//! its model and every saved value and session file plugins keep. `--shared` also writes
//! it into the repository's `.rig/snapshots`, through a plugin effect whose payload any
//! redaction plugin edits in Prepare or refuses in Gate. `/snapshot list` and
//! `/snapshot delete <name>` manage them.
//!
//! `/new --from <name>` and `rig --from <name>` fork the snapshot into a new session of
//! `RIG_HOME` and switch to it: a normal session from then on, the snapshot unchanged.
//! Its first start compares the files the snapshot's tool calls read ([`FilesSeen`])
//! with the disk, and the files that changed go to the model with the user's next message
//! and to the user as a note. A snapshot whose model is unusable starts on the default.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use rig_harness::harness_protocol::SessionId;
use rig_harness::prelude::*;
use rig_harness::rig_core::effect::EffectKind;
use rig_harness::rig_core::message::UserContent;
use rig_harness::rig_core::tool::{FilesRead, content_hash};
use rig_harness::rig_ecs::journal::SessionStore;
use rig_harness::rig_memory::{
    self, DirFiles, Files, HeuristicTokenCounter, Keep, MODEL, MemoryStore, Place as At, Rewrite,
    Session, Store, TokenCounter, check_name,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Where snapshots are kept in the project's plugin data.
const FOLDER: &str = "context-snapshots";

/// Where shared snapshots are kept, from the project root.
const SHARED: &str = ".rig/snapshots";

/// The session file a session started from a snapshot finds on its first start.
const STARTED: &str = "context-snapshots/started-from.json";

/// The kind of the effect that writes a shared snapshot into the repository.
pub const SHARE_KIND: &str = "context-snapshots.share";

/// Who the plugin is, in notes and decisions.
const NAME: &str = "context-snapshots";

/// The command-line option that starts a session from a snapshot.
const FROM: &str = "from";

/// Saves, lists and deletes context snapshots, and starts sessions from them.
#[derive(Default)]
pub struct ContextSnapshotsPlugin;

/// The plugin's stage systems, for others to order against: the changed-files note
/// added to the user's next message, in `PrepareSystems::Edit`.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SnapshotSystems;

impl Plugin for ContextSnapshotsPlugin {
    fn build(&self, app: &mut App) {
        app.register_required_components::<Agent, FilesSeen>()
            .register_required_components::<Agent, ContextNote>()
            .keeps_saved::<FilesSeen>()
            .keeps_saved::<ContextNote>()
            .keeps_files(FOLDER)
            .add_flag(
                Flag::new(FROM, "Start a new session from a saved context snapshot").value("NAME"),
            )
            .add_command(
                "snapshot",
                "Context snapshots: /snapshot save <name> [--at <turn>] [--shared], list, delete <name>",
                snapshot,
            )
            .configure_sets(AgentPass, SnapshotSystems.in_set(PrepareSystems::Edit))
            .add_systems(
                AgentPass,
                (
                    attach_note.in_set(SnapshotSystems),
                    new_from.in_set(Stage::Gate).before(CommandSystems),
                ),
            )
            .add_systems(PostStartup, (arrive, start_from_flag).chain())
            .add_observer(remember_reads)
            .add_observer(start_from)
            .add_observer(write_shared);
    }
}

/// The content hash of each file the agent's tool calls read, by path, as last read:
/// what its conversation shows of the file. Saved, so a snapshot holds it as of its cut.
#[derive(Component, Reflect, Default, Debug, Clone)]
#[reflect(Component, Saved, Default)]
pub struct FilesSeen(pub HashMap<String, String>);

/// What the model is told with the user's next message, such as the files that
/// changed since the snapshot this session started from was saved.
#[derive(Component, Reflect, Default, Debug, Clone)]
#[reflect(Component, Saved, Default)]
pub struct ContextNote(pub Option<String>);

/// What `/snapshot list` shows of a snapshot, kept beside it.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct SnapshotMeta {
    /// Its name.
    pub name: String,
    /// The main agent's turns it holds.
    pub turns: usize,
    /// An estimate of the main agent's messages in tokens.
    pub tokens: usize,
    /// The main agent's model, `vendor/model`.
    pub model: Option<String>,
    /// When it was saved, in milliseconds since the Unix epoch.
    pub saved_ms: u64,
}

/// A session's store as plain data: logs as text, blobs and files.
#[derive(Serialize, Deserialize, Default, Clone, Debug)]
pub struct StoreDump {
    /// Each log, its JSON lines.
    pub logs: BTreeMap<String, String>,
    /// Each blob.
    pub blobs: BTreeMap<String, Content>,
    /// Each file.
    pub files: BTreeMap<String, Content>,
}

/// A file's content: text when it is UTF-8, so a redaction reads it.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum Content {
    /// UTF-8 text.
    Text(String),
    /// Other bytes.
    Bytes(Vec<u8>),
}

impl Content {
    fn of(bytes: Vec<u8>) -> Self {
        String::from_utf8(bytes).map_or_else(|text| Self::Bytes(text.into_bytes()), Self::Text)
    }

    fn bytes(&self) -> &[u8] {
        match self {
            Self::Text(text) => text.as_bytes(),
            Self::Bytes(bytes) => bytes,
        }
    }
}

impl StoreDump {
    /// Everything `store` holds.
    ///
    /// # Errors
    ///
    /// When the store fails or a log is not UTF-8.
    pub fn of(store: &dyn Store) -> std::io::Result<Self> {
        let mut dump = Self::default();
        for log in store.logs()? {
            let text = String::from_utf8(store.read_log(&log)?).map_err(std::io::Error::other)?;
            dump.logs.insert(log, text);
        }
        for blob in store.blobs()? {
            dump.blobs
                .insert(blob.clone(), Content::of(store.blob(&blob)?));
        }
        for file in store.files()? {
            dump.files
                .insert(file.clone(), Content::of(store.read_file(&file)?));
        }
        Ok(dump)
    }

    /// A store in memory holding it.
    ///
    /// # Errors
    ///
    /// When a name is no name in a store.
    pub fn store(&self) -> std::io::Result<MemoryStore> {
        let store = MemoryStore::default();
        for (log, text) in &self.logs {
            store.append_log(log, text.as_bytes())?;
        }
        for (blob, content) in &self.blobs {
            store.put_blob(blob, content.bytes())?;
        }
        for (file, content) in &self.files {
            store.write_file(file, content.bytes())?;
        }
        Ok(store)
    }
}

/// The payload of a [`SHARE_KIND`] effect: the snapshot it writes into the
/// repository, which a redaction plugin edits in Prepare.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SharePayload {
    /// What it is.
    pub meta: SnapshotMeta,
    /// Its session.
    pub store: StoreDump,
}

/// On a [`SHARE_KIND`] effect of this plugin.
#[derive(Component)]
struct Sharing;

/// What a session started from a snapshot does on its first start.
#[derive(Serialize, Deserialize)]
struct StartedFrom {
    snapshot: String,
    notice: Option<String>,
}

/// Starts a new session from the snapshot `name` and switches to it.
#[derive(Event, Clone, Debug)]
pub struct StartFromSnapshot {
    /// The agent that asked, for notices.
    pub agent: Option<Entity>,
    /// The snapshot.
    pub name: String,
}

/// Where snapshots are kept: the project's plugin data, and the repository's
/// shared ones.
struct Places {
    own: Arc<dyn Files>,
    shared: DirFiles,
}

impl Places {
    fn new(data: &PluginData, root: &ProjectRoot) -> Self {
        Self {
            own: data.project.clone(),
            shared: DirFiles::new(root.0.join(SHARED)),
        }
    }

    /// The snapshot `name`'s meta and store, from the project's data, else shared.
    fn load(&self, name: &str) -> Result<(SnapshotMeta, StoreDump), String> {
        let own = read_snapshot(&*self.own, &format!("{FOLDER}/{name}"));
        own.or_else(|_| read_snapshot(&self.shared, name))
            .map_err(|_| format!("There is no snapshot named “{name}”."))
    }

    /// Every snapshot's meta, and whether it is shared, by name.
    fn list(&self) -> BTreeMap<String, (SnapshotMeta, bool)> {
        let mut all = BTreeMap::new();
        for (files, prefix, shared) in [
            (&self.shared as &dyn Files, String::new(), true),
            (&*self.own, format!("{FOLDER}/"), false),
        ] {
            for file in files.files().unwrap_or_default() {
                let Some(dir) = file
                    .strip_prefix(&prefix)
                    .and_then(|it| it.strip_suffix("/meta.json"))
                else {
                    continue;
                };
                let meta = files.read_file(&file).ok();
                if let Some(meta) =
                    meta.and_then(|it| serde_json::from_slice::<SnapshotMeta>(&it).ok())
                {
                    let shared = shared || all.contains_key(dir);
                    all.insert(dir.to_owned(), (meta, shared));
                }
            }
        }
        all
    }
}

fn read_snapshot(files: &dyn Files, dir: &str) -> Result<(SnapshotMeta, StoreDump), String> {
    let read = |name: &str| {
        files
            .read_file(&format!("{dir}/{name}"))
            .map_err(|e| e.to_string())
    };
    let meta = serde_json::from_slice(&read("meta.json")?).map_err(|e| e.to_string())?;
    let store = serde_json::from_slice(&read("store.json")?).map_err(|e| e.to_string())?;
    Ok((meta, store))
}

fn write_snapshot(
    files: &dyn Files,
    dir: &str,
    meta: &SnapshotMeta,
    store: &StoreDump,
) -> Result<(), String> {
    let store = serde_json::to_vec(store).map_err(|e| e.to_string())?;
    let meta = serde_json::to_vec(meta).map_err(|e| e.to_string())?;
    // The meta last: a snapshot is listed once its store is whole.
    files
        .write_file(&format!("{dir}/store.json"), &store)
        .and_then(|()| files.write_file(&format!("{dir}/meta.json"), &meta))
        .map_err(|e| e.to_string())
}

/// Remembers the hash of each file a successful tool call read, as it settles.
fn remember_reads(
    settled: On<Add<Settled>>,
    calls: Query<(&Outcome, &Published, &ForAgent), ToolCalls>,
    mut agents: Query<&mut FilesSeen>,
) {
    let Ok((outcome, published, &ForAgent(agent))) = calls.get(settled.entity) else {
        return;
    };
    let Ok(Some(FilesRead(read))) = published.0.get::<FilesRead>() else {
        return;
    };
    if let Ok(mut seen) = agents.get_mut(agent)
        && outcome.tool_result().is_success()
    {
        for file in read {
            seen.0.insert(file.path, file.hash.0);
        }
    }
}

/// `/snapshot save|list|delete`.
fn snapshot(
    In(args): In<CommandArgs>,
    (data, root, store): (Res<PluginData>, Res<ProjectRoot>, Option<Res<SessionStore>>),
    agents: Query<(&AgentId, &Conversation, Has<ActiveTurn>)>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let places = Places::new(&data, &root);
    let words: Vec<&str> = args.args.split_whitespace().collect();
    let agent = args.agent;
    let result = match words.as_slice() {
        ["save", name, rest @ ..] => {
            let agent_state = agents.get(agent).ok();
            save(&places, store.as_deref(), agent_state, name, rest).map(|(meta, shared)| {
                if let Some(store) = shared {
                    let payload = SharePayload {
                        meta: meta.clone(),
                        store,
                    };
                    let payload = serde_json::to_value(payload).unwrap_or_default();
                    let kind = EffectKind::Custom {
                        kind: SHARE_KIND.into(),
                        payload,
                    };
                    let request = EffectRequest {
                        kind,
                        handler: None,
                    };
                    commands.spawn((request, Sharing, ForAgent(agent)));
                }
                format!("Saved snapshot “{}”: {} turns.", meta.name, meta.turns)
            })
        }
        ["list"] => {
            let all = places.list();
            if all.is_empty() {
                Ok("No snapshots; /snapshot save <name> makes one.".to_owned())
            } else {
                Ok(all
                    .values()
                    .map(|(meta, shared)| listed(meta, *shared))
                    .collect::<Vec<_>>()
                    .join("\n"))
            }
        }
        ["delete", name] => delete(&places, name),
        _ => Err(
            "Use /snapshot save <name> [--at <turn>] [--shared], list or delete <name>.".to_owned(),
        ),
    };
    notices.write(match result {
        Ok(text) => Notice::info(agent, text),
        Err(why) => Notice::error(agent, why),
    });
}

/// One line of `/snapshot list`.
fn listed(meta: &SnapshotMeta, shared: bool) -> String {
    let saved = UNIX_EPOCH + Duration::from_millis(meta.saved_ms);
    let age = rig_harness::rig_tools::ago(saved.elapsed().unwrap_or_default());
    let model = meta.model.as_deref().unwrap_or("no model");
    let shared = if shared { " · shared" } else { "" };
    format!(
        "{}: {} turns · ~{} tokens · {model} · {age}{shared}",
        meta.name, meta.turns, meta.tokens
    )
}

/// Saves the snapshot `name` of the session in `store`; with `--shared`, also the
/// store to share.
fn save(
    places: &Places,
    store: Option<&SessionStore>,
    agent: Option<(&AgentId, &Conversation, bool)>,
    name: &str,
    rest: &[&str],
) -> Result<(SnapshotMeta, Option<StoreDump>), String> {
    let (Some(store), Some((id, conversation, busy))) = (store, agent) else {
        return Err("There is no saved session to snapshot.".to_owned());
    };
    if busy {
        return Err("A turn is running. Press Esc to stop it first.".to_owned());
    }
    if check_name(name).is_err() || name.contains('/') {
        return Err(format!("“{name}” is no snapshot name."));
    }
    let mut at = None;
    let mut shared = false;
    let mut words = rest.iter();
    while let Some(word) = words.next() {
        match *word {
            "--shared" => shared = true,
            "--at" => {
                let turn = words.next().and_then(|turn| turn.parse::<u64>().ok());
                at = Some(turn.ok_or("--at needs a number of turns")?);
            }
            other => return Err(format!("Unknown option {other}.")),
        }
    }
    // Turn `at` (counted from 0) begins at the user's message it answers.
    let before = at.and_then(|at| {
        let (_, turn) = conversation.turns().find(|(_, turn)| turn.index == at)?;
        Some((id.0.as_str(), turn.start))
    });
    let copy = MemoryStore::default();
    rig_memory::fork(&*store.0, &copy, before, &Keep).map_err(|e| e.to_string())?;
    let session = Session::read(&copy).map_err(|e| e.to_string())?;
    let main = session.main_agent();
    let counter = HeuristicTokenCounter::default();
    let meta = SnapshotMeta {
        name: name.to_owned(),
        turns: main.map_or(0, |main| main.conversation.turns().count()),
        tokens: main.map_or(0, |main| {
            main.conversation
                .messages()
                .iter()
                .map(|m| counter.count(m))
                .sum()
        }),
        model: main.and_then(|main| main.model()),
        saved_ms: rig_memory::now_ms(),
    };
    let dump = StoreDump::of(&copy).map_err(|e| e.to_string())?;
    write_snapshot(&*places.own, &format!("{FOLDER}/{name}"), &meta, &dump)?;
    Ok((meta, shared.then_some(dump)))
}

/// Deletes the snapshot `name`, the project's and the shared copy.
fn delete(places: &Places, name: &str) -> Result<String, String> {
    if places.load(name).is_err() {
        return Err(format!("There is no snapshot named “{name}”."));
    }
    for (files, dir) in [
        (&*places.own, format!("{FOLDER}/{name}")),
        (&places.shared as &dyn Files, name.to_owned()),
    ] {
        for file in ["meta.json", "store.json"] {
            files
                .remove_file(&format!("{dir}/{file}"))
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(format!("Deleted snapshot “{name}”."))
}

/// Writes each shared snapshot once its effect starts, as its payload stands after
/// every plugin's Prepare edits, such as a redaction's.
fn write_shared(
    started: On<Add<Started>>,
    mut effects: Query<(&EffectRequest, &mut Decision, &ForAgent), With<Sharing>>,
    root: Res<ProjectRoot>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((request, mut decision, &ForAgent(agent))) = effects.get_mut(started.entity) else {
        return;
    };
    let EffectKind::Custom { payload, .. } = &request.kind else {
        return;
    };
    let written = serde_json::from_value::<SharePayload>(payload.clone())
        .map_err(|e| e.to_string())
        .and_then(|share| {
            let files = DirFiles::new(root.0.join(SHARED));
            write_snapshot(&files, &share.meta.name, &share.meta, &share.store)
                .map(|()| share.meta.name)
        });
    let outcome = match written {
        Ok(name) => {
            notices.write(Notice::info(
                agent,
                format!("Shared snapshot “{name}” in {SHARED}."),
            ));
            Outcome::custom("shared")
        }
        Err(why) => {
            notices.write(Notice::error(
                agent,
                format!("Could not share the snapshot: {why}"),
            ));
            Outcome::refused(why)
        }
    };
    decision.decide(Decider::plugin(NAME), outcome);
}

/// Takes over the user's `/new --from <name>` lines before the slash commands run;
/// a plain `/new` stays rig-sessions'.
fn new_from(
    mut typed: Query<(&Incoming, &ForAgent, &mut Decision), (Undecided, IncomingMessages)>,
    mut commands: Commands,
) {
    for (incoming, &ForAgent(agent), mut decision) in &mut typed {
        if incoming.author.kind != AuthorKind::User || incoming.literal {
            continue;
        }
        let mut words = incoming.text().split_whitespace();
        if let (Some("/new"), Some("--from"), Some(name), None) =
            (words.next(), words.next(), words.next(), words.next())
        {
            decision.decide(
                Decider::plugin(NAME),
                Outcome::custom("new from a snapshot"),
            );
            commands.trigger(StartFromSnapshot {
                agent: Some(agent),
                name: name.to_owned(),
            });
        }
    }
}

/// `rig --from <name>`: a new session whose conversation is still empty starts the
/// snapshot's session in its place.
fn start_from_flag(
    invoked: Option<Res<Invoked>>,
    data: Res<PluginData>,
    agents: Query<(Entity, &Conversation), (With<Agent>, Without<SpawnedBy>)>,
    mut commands: Commands,
) {
    let Some(name) = invoked
        .as_ref()
        .and_then(|invoked| invoked.flag_value(FROM))
    else {
        return;
    };
    let empty = agents
        .iter()
        .all(|(_, conversation)| conversation.messages().is_empty());
    if empty && data.session.read_file(STARTED).is_err() {
        let agent = agents.iter().next().map(|(agent, _)| agent);
        commands.trigger(StartFromSnapshot {
            agent,
            name: name.to_owned(),
        });
    }
}

/// Drops the saved model `model` from a copy.
struct WithoutModel(String);

impl Rewrite for WithoutModel {
    fn value(&self, _: &At<'_>, key: &str, value: &mut Value) -> bool {
        let drop = key == MODEL && value.as_str() == Some(&self.0);
        if drop {
            *value = Value::Null;
        }
        drop
    }

    fn file(&self, _: &str, _: &mut Vec<u8>) -> bool {
        false
    }
}

/// Forks the snapshot into a new session and switches to it.
fn start_from(
    start: On<StartFromSnapshot>,
    (data, root, home, stores): (
        Res<PluginData>,
        Res<ProjectRoot>,
        Res<RigHome>,
        Res<SessionStores>,
    ),
    models: Res<Models>,
    turns: Query<(), With<TurnOf>>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    if !turns.is_empty() {
        notices.write(Notice::info(
            start.agent,
            "A turn is running. Press Esc to stop it first.",
        ));
        return;
    }
    let places = Places::new(&data, &root);
    let started = places.load(&start.name).and_then(|(meta, dump)| {
        let copy = dump.store().map_err(|e| e.to_string())?;
        let unusable = meta.model.clone().filter(|model| models.connect(model).is_err());
        let notice = unusable.as_ref().map(|model| {
            format!("The snapshot's model {model} is unavailable; this session starts on the default model.")
        });
        let (id, dir) = home.0.new_session().map_err(|e| e.to_string())?;
        let store = stores.open(&dir).map_err(|e| e.to_string())?;
        let with: Box<dyn Rewrite> = match unusable {
            Some(model) => Box::new(WithoutModel(model)),
            None => Box::new(Keep),
        };
        rig_memory::fork(&copy, &*store, None, &*with).map_err(|e| e.to_string())?;
        let marker = StartedFrom { snapshot: meta.name, notice };
        let marker = serde_json::to_vec(&marker).map_err(|e| e.to_string())?;
        store.write_file(&format!("plugins/{STARTED}"), &marker).map_err(|e| e.to_string())?;
        std::fs::write(dir.directory(), root.0.to_string_lossy().as_bytes())
            .map_err(|e| e.to_string())?;
        Ok::<SessionId, String>(id)
    });
    match started {
        Ok(id) => commands.trigger(Exit(ExitReason::Switch(Some(id)))),
        Err(why) => {
            notices.write(Notice::error(start.agent, why));
        }
    }
}

/// On a session's first start from a snapshot: tells the user, and the model with
/// the user's next message, which files the snapshot read have changed since.
fn arrive(
    data: Res<PluginData>,
    workspace: Option<Res<ProjectWorkspace>>,
    mut agents: Query<
        (Entity, &mut Conversation, &FilesSeen, &mut ContextNote),
        (With<Agent>, Without<SpawnedBy>),
    >,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(bytes) = data.session.read_file(STARTED) else {
        return;
    };
    data.session.remove_file(STARTED).ok();
    let Ok(started) = serde_json::from_slice::<StartedFrom>(&bytes) else {
        return;
    };
    let Some((agent, mut conversation, seen, mut note)) = agents.iter_mut().next() else {
        return;
    };
    if let Some(notice) = started.notice {
        notices.write(Notice::info(agent, notice));
    }
    let mut changed: Vec<&String> = seen
        .0
        .iter()
        .filter(|(path, hash)| now_hash(workspace.as_deref(), path).as_ref() != Some(*hash))
        .map(|(path, _)| path)
        .collect();
    if changed.is_empty() {
        return;
    }
    changed.sort();
    let list = changed
        .iter()
        .map(|path| path.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let text = format!("These files changed since this context was captured: {list}");
    notices.write(Notice::info(agent, text.clone()));
    let title = format!("Started from snapshot “{}”", started.snapshot);
    let after = conversation.entries().next_back().map(|entry| entry.id);
    conversation.add_aside(Note::new(title, text.clone()).aside(Author::plugin(NAME), after));
    note.0 = Some(text);
}

/// The content hash of the file at `path` now, as `read` hashes it; `None` when it
/// cannot be read.
fn now_hash(workspace: Option<&ProjectWorkspace>, path: &str) -> Option<String> {
    let path = PathBuf::from(path);
    let text = match workspace {
        Some(workspace) => workspace.0.read_text(&path).ok()?,
        None => std::fs::read_to_string(&path).ok()?,
    };
    Some(content_hash(text.as_bytes()).0)
}

/// Adds the agent's [`ContextNote`] to the user's next message to it, once.
fn attach_note(
    mut messages: Query<
        (&mut Incoming, &ForAgent),
        (Added<Admitted>, IncomingMessages, Without<Started>),
    >,
    mut agents: Query<&mut ContextNote>,
) {
    for (mut incoming, &ForAgent(agent)) in &mut messages {
        let command = !incoming.literal && incoming.text().starts_with('/');
        if incoming.author.kind != AuthorKind::User || command {
            continue;
        }
        let Ok(mut note) = agents.get_mut(agent) else {
            continue;
        };
        if let Some(text) = note.0.take() {
            let content = UserContent::text(text);
            let label = "context snapshot".to_owned();
            incoming
                .parts
                .push(MessagePart::Attachment(Attachment { label, content }));
        }
    }
}

#[cfg(test)]
mod tests;
