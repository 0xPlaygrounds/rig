//! An inbox folder: `dir` in the `[inbox]` settings
//! watches a directory, and each Markdown file dropped there becomes one
//! message to an agent, answered next to it as `<name>.answer.md`.
//!
//! A file is read once its size and change time held for [`SETTLE`];
//! hidden files, and the answer and error files this plugin writes, are
//! never read. Optional front matter between `---` lines names the `agent`
//! (an agent's title; a name no agent has spawns a helper of that title
//! under the primary agent) and a `title`. Items never interrupt a turn:
//! each agent takes them one at a time, in file-name order, once it is
//! idle. The message is the user's, titled `inbox: <file name>`, so the
//! transcript tags it.
//!
//! When its turn ends with an answer, `<name>.answer.md` holds the reply,
//! the files the turn's calls changed and the calls refused for want of
//! the user's approval (asked in the terminal view when one is attached,
//! refused at once without one), and the request moves to `done/`. A
//! failed or stopped turn writes `<name>.error.md` and moves it to
//! `failed/`. The items in flight are kept in the hidden `.inbox.json` of
//! the folder: after a restart, an item that was mid-turn is retried once,
//! its answer marked `(retried)`, or follows the turn the restored session
//! carries on. The watch keeps the session's worker loaded ([`KeepLoaded`]),
//! so it answers the folder with nobody at the terminal.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use bevy_log::warn;
use rig_basics::approval::{Approved, NeedsApproval};
use rig_harness::worker::KeepLoaded;
use rig_basics::subagents::{SpawnSubagent, SubagentsPlugin};
use rig_harness::prelude::*;
use rig_harness::rig_core::tool::FilesChanged;
use rig_harness::rig_tools::workspace::{Listing, Stat, Workspace};
use serde::{Deserialize, Serialize};

/// The option and the settings table that name the folder.
pub const INBOX: &str = "inbox";

/// How long a file's size and change time must hold before it is read.
pub const SETTLE: Duration = Duration::from_secs(1);

/// The hidden file in the folder that lists the items in flight.
pub const STATE: &str = ".inbox.json";

/// Where answered requests go, under the folder.
pub const DONE: &str = "done";

/// Where failed requests go, under the folder.
pub const FAILED: &str = "failed";

/// The plugin's name, as it writes notices.
const NAME: &str = "inbox-folder";

/// The largest request read.
const MAX_BYTES: u64 = 1 << 20;

/// The `[inbox]` settings: `dir = "~/notes/inbox"`, relative to the
/// project.
#[derive(Resource, Deserialize, Default, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct InboxSettings {
    /// The folder watched, if any.
    pub dir: Option<PathBuf>,
}

/// Watches the inbox folder; see the [crate] docs.
#[derive(Default)]
pub struct InboxFolderPlugin;

impl Plugin for InboxFolderPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<rig_harness::watch::WatchPlugin>() {
            app.add_plugins(rig_harness::watch::WatchPlugin);
        }
        if !app.is_plugin_added::<rig_harness::clock::ClockPlugin>() {
            app.add_plugins(rig_harness::clock::ClockPlugin);
        }
        if !app.is_plugin_added::<SubagentsPlugin>() {
            app.add_plugins(SubagentsPlugin);
        }
        app.add_setting::<InboxSettings>(INBOX)
            .init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .init_resource::<Inbox>()
            .add_systems(
                Update,
                (arm, take_changes, start_items).chain().before(AgentLoop),
            )
            .add_observer(settled)
            .add_observer(mark_turn)
            .add_observer(note_call)
            .add_observer(undelivered)
            .add_observer(not_created)
            .add_observer(finish);
    }
}

/// Where the inbox stands.
#[derive(Resource, Default, Debug)]
pub struct Inbox {
    /// The folder, once armed.
    dir: Option<PathBuf>,
    /// The entity with the watch.
    watch: Option<Entity>,
    /// The items in flight when the last run stopped, by file name.
    interrupted: BTreeMap<String, InFlight>,
    /// Each file settling, by name, and the entity whose wake reads it.
    settling: HashMap<String, Entity>,
    /// The items read and waiting for their agent, by file name.
    queued: BTreeMap<String, Item>,
    /// The items sent and not answered yet, by file name.
    running: BTreeMap<String, InFlight>,
}

impl Inbox {
    /// The names of the items read and waiting, in the order they start.
    pub fn queued(&self) -> impl Iterator<Item = &str> {
        self.queued.keys().map(String::as_str)
    }
}

/// An item in flight, as the folder's [`STATE`] keeps it.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
struct InFlight {
    /// The agent named, or none for the primary agent.
    agent: Option<String>,
    /// Whether it was sent again after a restart.
    retried: bool,
}

/// A request read from the folder.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Item {
    /// Its file name in the folder.
    file: String,
    /// What it asks, without its front matter.
    text: String,
    /// The agent its front matter names, by title.
    agent: Option<String>,
    /// Its front matter's title.
    title: Option<String>,
    /// Whether it is sent again after a restart cut its turn short.
    retried: bool,
}

/// On a file's entity while its size settles.
#[derive(Component, Debug)]
struct Settling {
    /// The file name.
    file: String,
    /// Its size and change time when last looked at.
    seen: Option<(u64, Option<SystemTime>)>,
}

/// On the message that sends an item.
#[derive(Component, Clone, Debug)]
struct InboxMessage(Item);

/// On a turn that answers an item: what it changed and what was denied.
#[derive(Component, Clone, Debug)]
struct InboxTurn {
    /// The item.
    item: Item,
    /// The files its calls changed.
    changed: BTreeSet<String>,
    /// The calls refused for want of the user's approval.
    denied: Vec<String>,
}

/// On a request for a helper an item names, with the helper's title.
#[derive(Component, Clone, Debug)]
struct Creating(String);

/// Arms the watch of the configured folder once, on this machine.
fn arm(
    mut inbox: ResMut<Inbox>,
    settings: Res<InboxSettings>,
    workspace: Res<ProjectWorkspace>,
    root: Res<ProjectRoot>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    if inbox.watch.is_some() {
        return;
    }
    let Some(dir) = settings.dir.as_ref() else {
        return;
    };
    let dir = root.0.join(dir);
    if let Some(host) = workspace.0.host() {
        let why = format!("The inbox folder is read on this machine; the project is on {host}.");
        notices.write(Notice::error(None, why));
        inbox.watch = Some(Entity::PLACEHOLDER);
        return;
    }
    let state = workspace.0.read(&dir.join(STATE), MAX_BYTES).ok();
    inbox.interrupted = state
        .and_then(|state| serde_json::from_slice(&state.bytes).ok())
        .unwrap_or_default();
    let watch = FileWatch::new(&dir).listing(Listing::All);
    let name = Name::new(format!("watching the inbox {}", dir.display()));
    inbox.watch = Some(commands.spawn((name, watch, KeepLoaded)).id());
    inbox.dir = Some(dir);
}

/// Starts or puts off the settling of each request the watch reports.
fn take_changes(
    mut changes: MessageReader<FileChanges>,
    mut inbox: ResMut<Inbox>,
    mut files: Query<&mut Settling>,
    workspace: Res<ProjectWorkspace>,
    mut commands: Commands,
) {
    for change in changes.read() {
        if Some(change.watch) != inbox.watch {
            continue;
        }
        for path in &change.paths {
            let Some(file) = request_name(path) else {
                continue;
            };
            if inbox.queued.contains_key(&file) || inbox.running.contains_key(&file) {
                continue;
            }
            let seen = look(&*workspace.0, path);
            let wait = WakeAt::after(SETTLE);
            match inbox.settling.get(&file) {
                Some(&entity) => {
                    if let Ok(mut settling) = files.get_mut(entity) {
                        settling.seen = seen;
                    }
                    commands.entity(entity).insert(wait);
                }
                None => {
                    let settling = Settling {
                        file: file.clone(),
                        seen,
                    };
                    let entity = commands.spawn((settling, wait)).id();
                    inbox.settling.insert(file, entity);
                }
            }
        }
    }
}

/// The size and change time of the file `path`, unless it is no file.
fn look(workspace: &dyn Workspace, path: &Path) -> Option<(u64, Option<SystemTime>)> {
    match workspace.stat(path) {
        Ok(Stat::File { size, modified, .. }) => Some((size, modified)),
        _ => None,
    }
}

/// The file name of `path` if it is a request: a visible `*.md` file that
/// is not an answer or an error.
fn request_name(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    let request = name.ends_with(".md")
        && !name.starts_with('.')
        && !name.ends_with(".answer.md")
        && !name.ends_with(".error.md");
    request.then(|| name.to_owned())
}

/// Reads a settling file once its size and change time held for
/// [`SETTLE`], and queues it; looks again later while it still changes.
fn settled(
    due: On<Add<Due>>,
    mut files: Query<&mut Settling>,
    mut inbox: ResMut<Inbox>,
    workspace: Res<ProjectWorkspace>,
    mut commands: Commands,
) {
    let Ok(mut settling) = files.get_mut(due.entity) else {
        return;
    };
    let Some(dir) = inbox.dir.clone() else {
        return;
    };
    let path = dir.join(&settling.file);
    let now = look(&*workspace.0, &path);
    if now.is_some() && now != settling.seen {
        settling.seen = now;
        commands.entity(due.entity).insert(WakeAt::after(SETTLE));
        return;
    }
    let file = settling.file.clone();
    commands.entity(due.entity).despawn();
    inbox.settling.remove(&file);
    if now.is_none() {
        return;
    }
    let Ok(contents) = workspace.0.read(&path, MAX_BYTES) else {
        return;
    };
    let text = String::from_utf8_lossy(&contents.bytes);
    let (agent, title, text) = front_matter(&text);
    let interrupted = inbox.interrupted.remove(&file);
    let item = Item {
        file: file.clone(),
        text,
        agent,
        title,
        retried: interrupted.is_some(),
    };
    if interrupted.is_some_and(|it| it.retried) {
        let why = "Its turn was cut short by a restart twice; it is not retried again.";
        fail(&*workspace.0, &dir, &item, why);
        return;
    }
    inbox.queued.insert(file, item);
}

/// The `agent` and `title` of `text`'s front matter, and the text after it.
fn front_matter(text: &str) -> (Option<String>, Option<String>, String) {
    let mut lines = text.lines();
    if lines.next().map(str::trim_end) != Some("---") {
        return (None, None, text.trim().to_owned());
    }
    let (mut agent, mut title) = (None, None);
    let mut body = Vec::new();
    let mut inside = true;
    for line in lines {
        if inside && line.trim_end() == "---" {
            inside = false;
        } else if inside {
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            let value = value.trim().trim_matches(['"', '\'']).to_owned();
            match key.trim() {
                "agent" if !value.is_empty() => agent = Some(value),
                "title" if !value.is_empty() => title = Some(value),
                _ => {}
            }
        } else {
            body.push(line);
        }
    }
    if inside {
        return (None, None, text.trim().to_owned());
    }
    (agent, title, body.join("\n").trim().to_owned())
}

/// Sends each agent its first queued item, in file-name order, once it is
/// idle; spawns the helper an item names when no agent has its title; and
/// lets a retried item follow the turn its restored agent carries on.
#[allow(clippy::too_many_arguments)]
fn start_items(
    mut inbox: ResMut<Inbox>,
    workspace: Res<ProjectWorkspace>,
    agents: MainAgentQuery,
    titles: Query<(Entity, &AgentTitle), With<Agent>>,
    busy: Query<&ActiveTurn>,
    restored: Query<(), (With<Restored>, Without<InboxTurn>)>,
    sending: Query<&ForAgent, With<InboxMessage>>,
    creating: Query<&Creating>,
    mut commands: Commands,
) {
    let Some(dir) = inbox.dir.clone() else {
        return;
    };
    let main = main_agent(&agents);
    let mut taken: HashSet<Entity> = sending.iter().map(|&ForAgent(agent)| agent).collect();
    let mut asked: HashSet<String> = creating.iter().map(|it| it.0.clone()).collect();
    let mut started = Vec::new();
    for item in inbox.queued.values() {
        let agent = match &item.agent {
            None => main,
            Some(name) => titles
                .iter()
                .find(|(_, title)| title.0 == *name)
                .map(|(agent, _)| agent),
        };
        let Some(agent) = agent else {
            if let (Some(name), Some(main)) = (&item.agent, main)
                && asked.insert(name.clone())
            {
                let helper = SpawnSubagent::helper(name.clone());
                commands.spawn((helper, ForAgent(main), Creating(name.clone())));
            }
            continue;
        };
        if !taken.insert(agent) {
            continue;
        }
        if let Ok(active) = busy.get(agent) {
            if item.retried && restored.contains(active.turn()) {
                commands
                    .entity(active.turn())
                    .insert(InboxTurn::of(item.clone()));
                started.push(item.clone());
            }
            continue;
        }
        let tag = match &item.title {
            Some(title) => format!("inbox: {} ({title})", item.file),
            None => format!("inbox: {}", item.file),
        };
        let message = Incoming::new(item.text.clone(), DeliveryMode::Queue)
            .with_author(Author::user().titled(tag))
            .literal();
        commands.spawn((message, ForAgent(agent), InboxMessage(item.clone())));
        started.push(item.clone());
    }
    if started.is_empty() {
        return;
    }
    for item in started {
        inbox.queued.remove(&item.file);
        let flight = InFlight {
            agent: item.agent.clone(),
            retried: item.retried,
        };
        inbox.running.insert(item.file, flight);
    }
    save_state(&*workspace.0, &dir, &inbox.running);
}

impl InboxTurn {
    /// A turn answering `item`, with nothing changed or denied yet.
    fn of(item: Item) -> Self {
        Self {
            item,
            changed: BTreeSet::new(),
            denied: Vec::new(),
        }
    }
}

/// Writes the items in flight to the folder's [`STATE`].
fn save_state(workspace: &dyn Workspace, dir: &Path, running: &BTreeMap<String, InFlight>) {
    let path = dir.join(STATE);
    let written = match serde_json::to_vec_pretty(running) {
        Ok(bytes) => workspace.write(&path, &bytes),
        Err(error) => Err(std::io::Error::other(error)),
    };
    if let Err(error) = written {
        warn!("{NAME}: cannot write {}: {error}", path.display());
    }
}

/// Moves an item from its message to the turn it went into.
fn mark_turn(
    delivered: On<Add<DeliveredIn>>,
    messages: Query<(&InboxMessage, &DeliveredIn)>,
    mut commands: Commands,
) {
    if let Ok((InboxMessage(item), &DeliveredIn(turn))) = messages.get(delivered.entity) {
        commands
            .entity(turn)
            .try_insert(InboxTurn::of(item.clone()));
    }
}

/// Keeps what each settled call of an item's turn changed, and the call
/// if it was refused for want of the user's approval.
fn note_call(
    settled: On<Add<Settled>>,
    calls: Query<
        (
            &CallOf,
            &ToolRequest,
            &Outcome,
            Option<&Published>,
            Has<NeedsApproval>,
        ),
        Without<Approved>,
    >,
    mut turns: Query<&mut InboxTurn>,
) {
    let Ok((&CallOf(turn), request, outcome, published, asked)) = calls.get(settled.entity) else {
        return;
    };
    let Ok(mut answering) = turns.get_mut(turn) else {
        return;
    };
    if let Some(Published(facts)) = published
        && let Ok(Some(FilesChanged(changes))) = facts.get::<FilesChanged>()
    {
        answering
            .changed
            .extend(changes.into_iter().map(|change| change.path));
    }
    if asked && let Err(report) = &outcome.0 {
        let call = &request.call.function;
        let arguments = serde_json::to_string(&call.arguments).unwrap_or_default();
        let denied = format!("`{}` {arguments}: {}", call.name, report.message);
        answering.denied.push(denied);
    }
}

/// Fails an item whose message settled without going into a turn, such as
/// one another plugin took over.
fn undelivered(
    settled: On<Add<Settled>>,
    messages: Query<&InboxMessage, Without<DeliveredIn>>,
    mut inbox: ResMut<Inbox>,
    workspace: Res<ProjectWorkspace>,
) {
    let Ok(InboxMessage(item)) = messages.get(settled.entity) else {
        return;
    };
    let why = "The message never reached the agent's model.";
    done_with(&mut inbox, &*workspace.0, item, Err(why));
}

/// Fails the items naming a helper that could not be spawned.
fn not_created(
    settled: On<Add<Settled>>,
    requests: Query<(&Creating, &Outcome)>,
    mut inbox: ResMut<Inbox>,
    workspace: Res<ProjectWorkspace>,
) {
    let Ok((Creating(name), Outcome(Err(report)))) = requests.get(settled.entity) else {
        return;
    };
    let Some(dir) = inbox.dir.clone() else {
        return;
    };
    let why = format!("No agent `{name}` could be started: {}", report.message);
    let named = |item: &Item| item.agent.as_ref() == Some(name);
    let failed: Vec<Item> = inbox
        .queued
        .values()
        .filter(|item| named(item))
        .cloned()
        .collect();
    for item in failed {
        inbox.queued.remove(&item.file);
        fail(&*workspace.0, &dir, &item, &why);
    }
}

/// Answers an item once its turn ended: its answer, or why it failed.
fn finish(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &InboxTurn)>,
    mut inbox: ResMut<Inbox>,
    workspace: Res<ProjectWorkspace>,
) {
    let Ok((outcome, turn)) = turns.get(settled.entity) else {
        return;
    };
    let answer = match outcome {
        TurnOutcome::Answered(message) => Ok(answer(turn, message.final_answer())),
        TurnOutcome::Failed(why) => Err(format!("The turn failed: {why}")),
        TurnOutcome::Stopped(None) => Err("The turn was stopped.".to_owned()),
        TurnOutcome::Stopped(Some(why)) => Err(format!("The turn was stopped: {why}")),
    };
    let answer = answer.as_deref().map_err(String::as_str);
    done_with(&mut inbox, &*workspace.0, &turn.item, answer);
}

/// The answer file of `turn`, whose reply is `reply`.
fn answer(turn: &InboxTurn, reply: Option<String>) -> String {
    let item = &turn.item;
    let retried = if item.retried { " (retried)" } else { "" };
    let heading = item.title.as_deref().unwrap_or(&item.file);
    let mut text = format!("# {heading}{retried}\n\n");
    text.push_str(reply.as_deref().unwrap_or("(no text in the reply)"));
    text.push_str("\n\n## Files changed\n\n");
    if turn.changed.is_empty() {
        text.push_str("None.\n");
    }
    for path in &turn.changed {
        text.push_str(&format!("- {path}\n"));
    }
    if !turn.denied.is_empty() {
        text.push_str("\n## Denied\n\nNobody approved these calls:\n\n");
        for call in &turn.denied {
            text.push_str(&format!("- {call}\n"));
        }
    }
    text
}

/// Writes an item's answer, or its error, and moves its request.
fn done_with(
    inbox: &mut Inbox,
    workspace: &dyn Workspace,
    item: &Item,
    answer: Result<&str, &str>,
) {
    let Some(dir) = inbox.dir.clone() else {
        return;
    };
    match answer {
        Ok(text) => {
            let stem = stem(&item.file);
            write(workspace, &dir.join(format!("{stem}.answer.md")), text);
            let from = dir.join(&item.file);
            if let Err(error) = workspace.rename(&from, &dir.join(DONE).join(&item.file)) {
                warn!("{NAME}: cannot move {}: {error}", from.display());
            }
        }
        Err(why) => fail(workspace, &dir, item, why),
    }
    inbox.running.remove(&item.file);
    save_state(workspace, &dir, &inbox.running);
}

/// Writes an item's error file and moves its request to [`FAILED`].
fn fail(workspace: &dyn Workspace, dir: &Path, item: &Item, why: &str) {
    let stem = stem(&item.file);
    let text = format!("# {} failed\n\n{why}\n", item.file);
    write(workspace, &dir.join(format!("{stem}.error.md")), &text);
    let from = dir.join(&item.file);
    if let Err(error) = workspace.rename(&from, &dir.join(FAILED).join(&item.file)) {
        warn!("{NAME}: cannot move {}: {error}", from.display());
    }
}

/// `file` without its `.md`.
fn stem(file: &str) -> &str {
    file.strip_suffix(".md").unwrap_or(file)
}

/// Writes `text` to `path`, logging a failure.
fn write(workspace: &dyn Workspace, path: &Path, text: &str) {
    if let Err(error) = workspace.write(path, text.as_bytes()) {
        warn!("{NAME}: cannot write {}: {error}", path.display());
    }
}

#[cfg(test)]
mod tests;
