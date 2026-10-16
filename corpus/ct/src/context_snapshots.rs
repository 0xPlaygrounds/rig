//! CT-13: reusable context snapshots.
//!
//! A snapshot is a saved session of `RIG_HOME` that holds a [`Snapshot`] value (listed,
//! so `/snapshot list` and a session picker read it without a log).
//! `/snapshot save <name> [--at <turn>] [--shared]` forks this session into one, whole
//! or as it was before turn `<turn>` began (`ForkSession`, `ForkCut::Before`): every
//! agent's messages, its own system prompt, its model and every saved value and session
//! file plugins keep. `--shared` also copies it into the repository's
//! `.rig/snapshots/<name>`, through a plugin effect a redaction plugin acts on first
//! (`rewrite_stopped` on the snapshot session it names, or a refusal in Gate).
//! `/snapshot list` opens a picker over them; `/snapshot delete <name>` removes one.
//!
//! `/new --from <name>` and `rig --from <name>` fork the snapshot into a new session and
//! switch to it: a normal session from then on, the snapshot unchanged. Its first start
//! compares the files the snapshot's tool calls read ([`FilesSeen`]) with the disk, and
//! the files that changed go to the model with the user's next message and to the user
//! as a note. A snapshot whose model is unusable starts on the default.

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::time::UNIX_EPOCH;

use rig_harness::harness_protocol::{SessionDir, SessionId};
use rig_harness::prelude::*;
use rig_harness::rig_core::effect::EffectKind;
use rig_harness::rig_core::message::UserContent;
use rig_harness::rig_core::tool::{FilesRead, content_hash};
use rig_harness::rig_memory::{
    self, DirStore, HeuristicTokenCounter, MODEL, NoRewrite, SessionMeta, TokenCounter, check_name,
};
use serde::{Deserialize, Serialize};

/// The key a snapshot session's [`Snapshot`] is saved and listed under.
pub const KEY: &str = "context-snapshots.snapshot";

/// The key of [`StartedFrom`].
const STARTED: &str = "context-snapshots.started";

/// Where shared snapshots are kept, from the project root.
const SHARED: &str = ".rig/snapshots";

/// The kind of the effect that copies a snapshot into the repository.
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
        if !app.is_plugin_added::<SessionPickerPlugin>() {
            app.add_plugins(SessionPickerPlugin);
        }
        app.register_required_components::<Agent, FilesSeen>()
            .register_required_components::<Agent, ContextNote>()
            .keeps_saved::<FilesSeen>()
            .keeps_saved::<ContextNote>()
            .register_type::<Snapshot>()
            .register_type::<StartedFrom>()
            .saved_as::<Snapshot>(KEY)
            .listed();
        // Saved, not listed: `listed` is the only thing to say of a saved key.
        let _ = app.saved_as::<StartedFrom>(STARTED);
        app.add_flag(
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
            .add_systems(Last, forked.after(ForkSessions))
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

/// What makes a saved session a snapshot, and what `/snapshot list` shows of it.
#[derive(Resource, Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(opaque, Resource, Saved, Default, Clone, Serialize, Deserialize)]
pub struct Snapshot {
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

/// In a session started from a snapshot, until its first start: which one, and what
/// the user is told.
#[derive(Resource, Reflect, Serialize, Deserialize, Clone, Debug, Default)]
#[reflect(opaque, Resource, Saved, Default, Clone, Serialize, Deserialize)]
pub struct StartedFrom(pub Option<(String, Option<String>)>);

/// The payload of a [`SHARE_KIND`] effect: the snapshot session it copies into the
/// repository, which a redaction plugin rewrites (`rewrite_stopped`) before it starts.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SharePayload {
    /// The snapshot's name.
    pub name: String,
    /// Its session's id.
    pub session: String,
}

/// On a [`SHARE_KIND`] effect of this plugin.
#[derive(Component)]
struct Sharing;

/// The asker of a fork that makes the snapshot `name`.
#[derive(Component)]
struct Saving {
    name: String,
    shared: bool,
    agent: Entity,
}

/// The asker of a fork that starts a session from the snapshot `name`.
#[derive(Component)]
struct Starting {
    name: String,
    agent: Option<Entity>,
}

/// Starts a new session from the snapshot `name` (or of that session id) and
/// switches to it.
#[derive(Event, Clone, Debug)]
pub struct StartFromSnapshot {
    /// The agent that asked, for notices.
    pub agent: Option<Entity>,
    /// The snapshot.
    pub name: String,
}

/// The snapshots of `home` for the project in `root`, by session, from their listings.
pub fn snapshots(
    home: &Home,
    stores: &StoreOpener,
    root: &ProjectRoot,
) -> Vec<(SessionId, Snapshot)> {
    home.saved_sessions()
        .iter()
        .filter_map(|(_, dir)| SessionEntry::read(dir, stores))
        .filter(|entry| entry.directory.as_ref().is_none_or(|it| *it == root.0))
        .filter_map(|entry| Some((entry.id.clone(), snapshot_of(&entry)?)))
        .collect()
}

/// The snapshot `entry` is, from its listing.
fn snapshot_of(entry: &SessionEntry) -> Option<Snapshot> {
    let saved = entry.listing.meta.listed.get(KEY)?;
    serde_json::from_value(saved.clone()).ok()
}

/// The snapshot `name` (or of that session id) of `home`.
fn find(home: &Home, stores: &StoreOpener, root: &ProjectRoot, name: &str) -> Option<SessionId> {
    let all = snapshots(home, stores, root);
    let found = all
        .into_iter()
        .find(|(id, it)| it.name == name || id.as_str() == name);
    found.map(|(id, _)| id)
}

/// The shared snapshots in the repository, by name.
fn shared(root: &ProjectRoot) -> BTreeSet<String> {
    let entries = std::fs::read_dir(root.0.join(SHARED)).into_iter().flatten();
    entries
        .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
        .collect()
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
    (home, root, stores): (Res<RigHome>, Res<ProjectRoot>, Res<StoreOpener>),
    agents: Query<(&AgentId, &Conversation, Has<ActiveTurn>)>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let words: Vec<&str> = args.args.split_whitespace().collect();
    let agent = args.agent;
    let all = || snapshots(&home.0, &stores, &root);
    let result = match words.as_slice() {
        ["save", name, rest @ ..] => {
            let taken = all().iter().any(|(_, it)| it.name == *name);
            let cut = agents
                .get(agent)
                .map_err(|_| "There is no session to snapshot.".to_owned());
            cut.and_then(|agent| save_cut(agent, name, rest, taken))
                .map(|(cut, shared)| {
                    let name = (*name).to_owned();
                    let asker = commands
                        .spawn(Saving {
                            name,
                            shared,
                            agent,
                        })
                        .id();
                    commands.write_message(ForkSession {
                        from: None,
                        cut,
                        asker: Some(asker),
                    });
                    String::new()
                })
        }
        ["list"] => list(all(), &shared(&root), agent, &mut commands),
        ["delete", name] => {
            let found = find(&home.0, &stores, &root, name);
            let share = root.0.join(SHARED).join(name);
            if let Some(id) = &found {
                commands.trigger(DeleteSession(id.clone()));
            }
            let unshared = check_name(name).is_ok() && std::fs::remove_dir_all(share).is_ok();
            match found.is_some() || unshared {
                true => Ok(format!("Deleted snapshot “{name}”.")),
                false => Err(format!("There is no snapshot named “{name}”.")),
            }
        }
        _ => Err(
            "Use /snapshot save <name> [--at <turn>] [--shared], list or delete <name>.".to_owned(),
        ),
    };
    match result {
        Ok(text) if text.is_empty() => {}
        Ok(text) => {
            notices.write(Notice::info(agent, text));
        }
        Err(why) => {
            notices.write(Notice::error(agent, why));
        }
    }
}

/// Where `/snapshot save <name> <rest>` cuts the session, and whether it is shared.
fn save_cut(
    (id, conversation, busy): (&AgentId, &Conversation, bool),
    name: &str,
    rest: &[&str],
    taken: bool,
) -> Result<(ForkCut, bool), String> {
    if busy {
        return Err("A turn is running. Press Esc to stop it first.".to_owned());
    }
    if check_name(name).is_err() || name.contains('/') {
        return Err(format!("“{name}” is no snapshot name."));
    }
    if taken {
        return Err(format!(
            "There is a snapshot “{name}”; /snapshot delete {name} first."
        ));
    }
    let mut cut = ForkCut::Whole;
    let mut shared = false;
    let mut words = rest.iter();
    while let Some(word) = words.next() {
        match *word {
            "--shared" => shared = true,
            "--at" => {
                let at = words.next().and_then(|turn| turn.parse::<u64>().ok());
                let at = at.ok_or("--at needs a number of turns")?;
                // Turn `at` (counted from 0) begins at the user's message it answers.
                if let Some((_, turn)) = conversation.turns().find(|(_, turn)| turn.index == at) {
                    cut = ForkCut::Before(id.clone(), turn.start);
                }
            }
            other => return Err(format!("Unknown option {other}.")),
        }
    }
    Ok((cut, shared))
}

/// `/snapshot list`: a picker over the project's snapshots, previewed; the one picked
/// starts a new session. Shared ones not brought in yet are named.
fn list(
    all: Vec<(SessionId, Snapshot)>,
    shared: &BTreeSet<String>,
    agent: Entity,
    commands: &mut Commands,
) -> Result<String, String> {
    let names: BTreeSet<&str> = all.iter().map(|(_, it)| it.name.as_str()).collect();
    let not_in: Vec<&str> = shared
        .iter()
        .map(String::as_str)
        .filter(|it| !names.contains(it))
        .collect();
    let mut text = String::new();
    if !not_in.is_empty() {
        text = format!(
            "Shared in {SHARED}: {}; /new --from <name> starts one.",
            not_in.join(", ")
        );
    }
    if all.is_empty() {
        return Ok(match text.is_empty() {
            true => "No snapshots; /snapshot save <name> makes one.".to_owned(),
            false => text,
        });
    }
    let source = SessionSource::Sessions(all.into_iter().map(|(id, _)| id).collect());
    let picker = Picker::new("Start a session from which snapshot?").for_agent(agent);
    let view = SessionPicker {
        source,
        show: Some(show),
        ..Default::default()
    };
    commands.spawn((picker, view)).observe(pick);
    Ok(text)
}

/// A snapshot's line in the picker: name, turns, tokens, model and age.
fn show(item: &SessionItem<'_>) -> (String, Option<String>) {
    let SessionItem::Session(entry) = item else {
        return (String::new(), None);
    };
    let Some(it) = snapshot_of(entry) else {
        return (entry.label(), None);
    };
    let saved = UNIX_EPOCH + Duration::from_millis(it.saved_ms);
    let age = rig_harness::rig_tools::ago(saved.elapsed().unwrap_or_default());
    let model = it.model.as_deref().unwrap_or("no model");
    let label = format!(
        "{}: {} turns · ~{} tokens · {model}",
        it.name, it.turns, it.tokens
    );
    (label, Some(age))
}

/// Starts a session from the snapshot picked.
fn pick(picked: On<Picked>, chosen: Query<(&SessionPicked, &Picker)>, mut commands: Commands) {
    let Ok((
        SessionPicked {
            session: Some(id), ..
        },
        picker,
    )) = chosen.get(picked.entity)
    else {
        return;
    };
    let name = id.to_string();
    commands.trigger(StartFromSnapshot {
        agent: picker.agent,
        name,
    });
}

/// Copies each shared snapshot into the repository once its effect starts, as its
/// session stands after every plugin's Prepare, such as a redaction's.
fn write_shared(
    started: On<Add<Started>>,
    mut effects: Query<(&EffectRequest, &mut Decision, &ForAgent), With<Sharing>>,
    (root, home, stores): (Res<ProjectRoot>, Res<RigHome>, Res<StoreOpener>),
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
            let id: SessionId = share
                .session
                .parse()
                .map_err(|_| "no session id".to_owned())?;
            let from = stores
                .open(&home.0.session(&id))
                .map_err(|e| e.to_string())?;
            let path = root.0.join(SHARED).join(&share.name);
            std::fs::remove_dir_all(&path).ok();
            let to = DirStore::open(path).map_err(|e| e.to_string())?;
            rig_memory::fork(&*from, &to, None, &NoRewrite).map_err(|e| e.to_string())?;
            Ok(share.name)
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
    mut typed: Query<(&Incoming, &ForAgent, &mut Decision), (ReadyToStart, IncomingMessages)>,
    mut commands: Commands,
) {
    for (incoming, &ForAgent(agent), mut decision) in &mut typed {
        let mut words = incoming.command().unwrap_or_default().split_whitespace();
        if let (Some("new"), Some("--from"), Some(name), None) =
            (words.next(), words.next(), words.next(), words.next())
        {
            decision.decide(
                Decider::plugin(NAME),
                Outcome::custom("new from a snapshot"),
            );
            let (agent, name) = (Some(agent), name.to_owned());
            commands.trigger(StartFromSnapshot { agent, name });
        }
    }
}

/// `rig --from <name>`: a new session whose conversation is still empty starts the
/// snapshot's session in its place (the harness removes the empty one).
fn start_from_flag(
    invoked: Option<Res<Invoked>>,
    started: Option<Res<StartedFrom>>,
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
    if empty && started.is_none() {
        let agent = agents.iter().next().map(|(agent, _)| agent);
        commands.trigger(StartFromSnapshot {
            agent,
            name: name.to_owned(),
        });
    }
}

/// Forks the snapshot into a new session; [`forked`] switches to it.
fn start_from(
    start: On<StartFromSnapshot>,
    (home, root, stores): (Res<RigHome>, Res<ProjectRoot>, Res<StoreOpener>),
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
    let found = find(&home.0, &stores, &root, &start.name)
        .map(Ok)
        .or_else(|| {
            shared(&root)
                .contains(&start.name)
                .then(|| bring_in(&home.0, &stores, &root, &start.name))
        });
    match found {
        Some(Ok(id)) => {
            let asker = Starting {
                name: start.name.clone(),
                agent: start.agent,
            };
            let asker = commands.spawn(asker).id();
            commands.write_message(ForkSession {
                from: Some(id),
                cut: ForkCut::Whole,
                asker: Some(asker),
            });
        }
        Some(Err(why)) => {
            notices.write(Notice::error(start.agent, why));
        }
        None => {
            let why = format!("There is no snapshot named “{}”.", start.name);
            notices.write(Notice::error(start.agent, why));
        }
    }
}

/// Copies the shared snapshot `name` into a session of `home`, so it is a snapshot
/// `ForkSession` forks: a fork from a store that is no session of `home`.
fn bring_in(
    home: &Home,
    stores: &StoreOpener,
    root: &ProjectRoot,
    name: &str,
) -> Result<SessionId, String> {
    let from = DirStore::open(root.0.join(SHARED).join(name)).map_err(|e| e.to_string())?;
    let (id, dir) = home.new_session().map_err(|e| e.to_string())?;
    let copied = stores.open(&dir).map_err(|e| e.to_string()).and_then(|to| {
        rig_memory::fork(&from, &*to, None, &NoRewrite).map_err(|e| e.to_string())?;
        std::fs::write(dir.directory(), root.0.to_string_lossy().as_bytes())
            .and_then(|()| dir.mark_saved())
            .map_err(|e| e.to_string())
    });
    match copied {
        Ok(()) => Ok(id),
        Err(why) => {
            std::fs::remove_dir_all(dir.path()).ok();
            Err(why)
        }
    }
}

/// Finishes this plugin's forks: marks a new snapshot as one (and shares it), or
/// readies a session started from one and switches to it.
fn forked(
    mut forked: MessageReader<SessionForked>,
    mut failed: MessageReader<SessionForkFailed>,
    askers: Query<(Option<&Saving>, Option<&Starting>)>,
    (home, paths, stores): (Res<RigHome>, Res<SessionPaths>, Res<StoreOpener>),
    models: Res<Models>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    for fork in forked.read() {
        let Some(asker) = fork.asker else { continue };
        let Ok((saving, starting)) = askers.get(asker) else {
            continue;
        };
        let dir = home.0.session(&fork.to);
        if let Some(Saving {
            name,
            shared,
            agent,
        }) = saving
        {
            let marked = mark(&stores, &paths, &dir, name);
            let text = marked.map(|it| {
                if *shared {
                    let session = fork.to.to_string();
                    let share = SharePayload {
                        name: name.clone(),
                        session,
                    };
                    let payload = serde_json::to_value(share).unwrap_or_default();
                    let kind = EffectKind::Custom {
                        kind: SHARE_KIND.into(),
                        payload,
                    };
                    let request = EffectRequest {
                        kind,
                        handler: None,
                    };
                    commands.spawn((request, Sharing, ForAgent(*agent)));
                }
                format!("Saved snapshot “{name}”: {} turns.", it.turns)
            });
            notices.write(match text {
                Ok(text) => Notice::info(*agent, text),
                Err(why) => Notice::error(*agent, why),
            });
        }
        if let Some(Starting { name, agent }) = starting {
            match ready(&stores, &paths, &dir, name, &models) {
                Ok(()) => commands.trigger(Exit(ExitReason::Switch(Some(fork.to.clone())))),
                Err(why) => {
                    notices.write(Notice::error(*agent, why));
                }
            }
        }
        commands.entity(asker).despawn();
    }
    for fail in failed.read() {
        if let Some(asker) = fail.asker.filter(|it| askers.contains(*it)) {
            commands.entity(asker).despawn();
        }
    }
}

/// Makes the session in `dir` the snapshot `name`, saved and listed.
fn mark(
    stores: &StoreOpener,
    paths: &SessionPaths,
    dir: &SessionDir,
    name: &str,
) -> Result<Snapshot, String> {
    let mut made = None;
    edit_stopped(stores, paths, dir, |session| {
        let main = session.main_agent();
        let counter = HeuristicTokenCounter::default();
        let it = Snapshot {
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
            saved_ms: rig_harness::rig_core::now_ms(),
        };
        let value = serde_json::to_value(&it).map_err(|e| e.to_string())?;
        session.saved.insert(KEY.to_owned(), value);
        made = Some(it);
        Ok(())
    })?;
    let made = made.ok_or("the snapshot was not written")?;
    let value = serde_json::to_value(&made).map_err(|e| e.to_string())?;
    relist(stores, dir, Some(value))?;
    Ok(made)
}

/// Readies the session in `dir`, a copy of the snapshot `name`: no snapshot itself,
/// told so on its first start, and without the snapshot's model when it is unusable.
fn ready(
    stores: &StoreOpener,
    paths: &SessionPaths,
    dir: &SessionDir,
    name: &str,
    models: &Models,
) -> Result<(), String> {
    edit_stopped(stores, paths, dir, |session| {
        session.saved.remove(KEY);
        let main = session
            .main_agent()
            .map(|main| (main.id.clone(), main.model()));
        let mut notice = None;
        if let Some((id, Some(model))) = main
            && models.connect(&model).is_err()
            && let Some(agent) = session.agents.iter_mut().find(|agent| agent.id == id)
        {
            agent.saved.remove(MODEL);
            notice = Some(format!(
                "The snapshot's model {model} is unavailable; this session starts on the default model."
            ));
        }
        let started = StartedFrom(Some((name.to_owned(), notice)));
        let value = serde_json::to_value(started).map_err(|e| e.to_string())?;
        session.saved.insert(STARTED.to_owned(), value);
        Ok(())
    })?;
    relist(stores, dir, None)
}

/// Sets the listed [`Snapshot`] of the stopped session in `dir`: `edit_stopped`
/// leaves its listing as it was.
fn relist(
    stores: &StoreOpener,
    dir: &SessionDir,
    value: Option<serde_json::Value>,
) -> Result<(), String> {
    let store = stores.open(dir).map_err(|e| e.to_string())?;
    let mut meta = SessionMeta::read(&*store).unwrap_or_default();
    match value {
        Some(value) => meta.listed.insert(KEY.to_owned(), value),
        None => meta.listed.remove(KEY),
    };
    meta.write(&*store).map_err(|e| e.to_string())
}

/// On a session's first start from a snapshot: tells the user, and the model with
/// the user's next message, which files the snapshot read have changed since.
fn arrive(
    started: Option<ResMut<StartedFrom>>,
    workspace: Option<Res<ProjectWorkspace>>,
    mut agents: Query<
        (Entity, &mut Conversation, &FilesSeen, &mut ContextNote),
        (With<Agent>, Without<SpawnedBy>),
    >,
    mut notices: MessageWriter<Notice>,
) {
    let Some((snapshot, notice)) = started.and_then(|mut started| started.0.take()) else {
        return;
    };
    let Some((agent, mut conversation, seen, mut note)) = agents.iter_mut().next() else {
        return;
    };
    if let Some(notice) = notice {
        notices.write(Notice::info(agent, notice));
    }
    let mut changed: Vec<&str> = seen
        .0
        .iter()
        .filter(|(path, hash)| now_hash(workspace.as_deref(), path).as_ref() != Some(*hash))
        .map(|(path, _)| path.as_str())
        .collect();
    if changed.is_empty() {
        return;
    }
    changed.sort_unstable();
    let text = format!(
        "These files changed since this context was captured: {}",
        changed.join(", ")
    );
    notices.write(Notice::info(agent, text.clone()));
    let title = format!("Started from snapshot “{snapshot}”");
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
        if incoming.author.kind != AuthorKind::User || incoming.command().is_some() {
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
