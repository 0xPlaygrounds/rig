//! CT-09: prompt revisions with version switching.
//!
//! Versions are the conversation's own branches (rig-core's `branch_off`,
//! `switch_branch`, `branches`): revising user message `m` keeps it and
//! everything after it as a branch following the message before it, and
//! the agent answers the new text. Every version of a message is a
//! branch this plugin kept at the same point of the active line, plus the
//! active entry there; they are ordered by when their first message was
//! made. A branch carries the saved asides of its entries, branches kept
//! inside it too, so versions nest and an inner choice is remembered
//! while an outer version is switched away and back. Branches are saved,
//! so a resume restores the active path; `messages()` is only the active
//! path, and `branches()`/`line()` give every version.
//!
//! - `/revise [<m>] [<text>]` revises user message `m` (the last by
//!   default): with a text at once, else the message is opened for
//!   editing as a question with its text proposed.
//! - `/version <n> [<m>]` makes version `n` of message `m` active (the
//!   last message with versions by default). It never calls a model.
//! - In the terminal view, on a selected user message (Alt+Up), `e`
//!   revises it and Left/Right switch its version.
//! - Each message with versions shows `‹ 2/3 ›` as an unsaved note after
//!   it.
//! - Each tool call's `FilesChanged` is kept as an aside after the entry
//!   that made it, so it goes with its version. After a switch, the files
//!   either continuation changed are compared with the working tree, and a
//!   one-line notice names those that differ from the selected version.

use std::collections::BTreeMap;
use std::path::PathBuf;

use rig_harness::prelude::*;
use rig_harness::rig_core::conversation::{Aside, AsideId, MessageKey};
use rig_harness::rig_core::message::Message;
use rig_harness::rig_core::tool::{FileChange, FileState, FilesChanged};
use serde::{Deserialize, Serialize};

/// The author of this plugin's conversation edits, branches and notes.
pub const NAME: &str = "prompt-revisions";

/// The kind of the aside after an entry whose tool call changed files: a
/// [`FilesRecord`].
pub const FILES: &str = "prompt-revisions.files";

/// The most bytes of one file read to compare it with a version's.
const READ_LIMIT: u64 = 64 << 20;

/// The author of this plugin's edits.
fn author() -> Author {
    Author::plugin(NAME)
}

/// The files a tool call changed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FilesRecord {
    /// The changes, as the tool published them.
    pub files: Vec<FileChange>,
}

/// One version of a user message.
#[derive(Clone, Debug, PartialEq)]
pub struct Version {
    /// The branch holding it; `None` for the active one.
    pub branch: Option<AsideId>,
    /// Its first message's key.
    pub key: MessageKey,
}

/// A user message of the active line and its versions.
#[derive(Clone, Debug, PartialEq)]
pub struct Slot {
    /// Where it is in the conversation.
    pub at: usize,
    /// Its number among the user's messages of the active line, from 1.
    pub message: usize,
    /// Its versions, oldest first.
    pub versions: Vec<Version>,
    /// Which of them is active, from 0.
    pub active: usize,
}

impl Slot {
    /// `‹ 2/3 ›`.
    pub fn badge(&self) -> String {
        format!(
            "‹ {}/{} ›",
            self.active.saturating_add(1),
            self.versions.len()
        )
    }
}

/// Every user message of the active line with its versions, in order.
pub fn slots(conversation: &Conversation) -> Vec<Slot> {
    let kept = conversation.branches();
    let ours = |id: AsideId| {
        conversation
            .aside(id)
            .is_some_and(|aside| aside.author == author())
    };
    let users = conversation.entries().filter(|entry| entry.is_users_own());
    users
        .enumerate()
        .map(|(number, entry)| {
            let others = kept.iter().filter(|kept| {
                let first = kept.branch.entries.first();
                kept.inside.is_none()
                    && kept.at == entry.index
                    && ours(kept.id)
                    && first.is_some_and(|first| matches!(*first.message, Message::User { .. }))
            });
            let others = others.filter_map(|kept| {
                let first = kept.branch.entries.first()?;
                let key = MessageKey {
                    made: first.made?,
                    id: first.id?,
                };
                Some(Version {
                    branch: Some(kept.id),
                    key,
                })
            });
            let mut versions: Vec<Version> = others.collect();
            let key = entry.key();
            versions.push(Version { branch: None, key });
            versions.sort_by_key(|version| version.key);
            let active = versions.iter().position(|version| version.branch.is_none());
            Slot {
                at: entry.index,
                message: number.saturating_add(1),
                versions,
                active: active.unwrap_or_default(),
            }
        })
        .collect()
}

/// A question opening a message for editing.
#[derive(Component, Clone, Debug)]
struct Revising {
    agent: Entity,
    message: MessageId,
}

/// Reading the working tree for a switch's notice.
#[derive(Component, Clone, Copy, Debug)]
struct Comparing(Entity);

/// Prompt revisions: `/revise`, `/version`, the badge and the keys.
#[derive(Default)]
pub struct PromptRevisionsPlugin;

impl Plugin for PromptRevisionsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectWorkspace>()
            .add_observer(record_changes)
            .add_observer(answered)
            .add_observer(compared)
            .add_systems(Update, badges.after(AgentLoop))
            .add_command(
                "revise",
                "Revise a message of yours: /revise [<message>] [<text>]",
                revise,
            )
            .add_command(
                "version",
                "Switch a revised message's version: /version <n> [<message>]",
                version,
            );
        #[cfg(feature = "tui")]
        tui::build(app);
    }
}

/// `/revise [<m>] [<text>]`.
fn revise(
    In(args): In<CommandArgs>,
    agents: Query<&Conversation, Without<ActiveTurn>>,
    answerers: Answerers,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let agent = args.agent;
    let Ok(conversation) = agents.get(agent) else {
        notices.write(Notice::error(agent, "/revise waits for the turn to end"));
        return;
    };
    let words = args.args.trim();
    let (number, text) = match words.split_once(char::is_whitespace) {
        Some((first, rest)) => match first.parse::<usize>() {
            Ok(number) => (Some(number), rest.trim()),
            Err(_) => (None, words),
        },
        None => match words.parse::<usize>() {
            Ok(number) => (Some(number), ""),
            Err(_) => (None, words),
        },
    };
    let found = slots(conversation);
    let slot = match number {
        Some(number) => found.iter().find(|slot| slot.message == number),
        None => found.last(),
    };
    let Some((at, id)) = slot.and_then(|slot| Some((slot.at, conversation.id(slot.at)?))) else {
        notices.write(Notice::error(agent, "/revise: no such message of yours"));
        return;
    };
    if !text.is_empty() {
        let text = text.to_owned();
        commands.queue(move |world: &mut World| revise_now(world, agent, id, text));
        return;
    }
    if !answerers.can_answer(Some(agent)) {
        notices.write(Notice::error(agent, "/revise needs a text here"));
        return;
    }
    let old = typed(conversation, at).unwrap_or_default();
    let number = slot.map(|slot| slot.message).unwrap_or_default();
    let ask = Ask::new(format!("Revise message {number}"))
        .initial(old)
        .multiline();
    let message = id;
    commands.spawn((ask, ForAgent(agent), Revising { agent, message }));
}

/// Revises the message an answered question opened.
fn answered(
    settled: On<Add<Settled>>,
    asks: Query<(&Revising, &Outcome), With<Ask>>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok((revising, outcome)) = asks.get(settled.entity) else {
        return;
    };
    let text = Answer::of(outcome).and_then(|answer| answer.text);
    let Some(text) = text.filter(|text| !text.trim().is_empty()) else {
        notices.write(Notice::info(revising.agent, "Revision cancelled."));
        return;
    };
    let Revising { agent, message } = revising.clone();
    commands.queue(move |world: &mut World| revise_now(world, agent, message, text));
}

/// `/version <n> [<m>]`.
fn version(
    In(args): In<CommandArgs>,
    agents: Query<&Conversation, Without<ActiveTurn>>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let agent = args.agent;
    let Ok(conversation) = agents.get(agent) else {
        notices.write(Notice::error(agent, "/version waits for the turn to end"));
        return;
    };
    let numbers: Result<Vec<usize>, _> = args.args.split_whitespace().map(str::parse).collect();
    let found = slots(conversation);
    let mut revised = found.iter().filter(|slot| slot.versions.len() > 1);
    let (n, slot) = match numbers.as_deref() {
        Ok([n]) => (*n, revised.next_back()),
        Ok([n, m]) => (*n, found.iter().find(|slot| slot.message == *m)),
        _ => {
            notices.write(Notice::error(agent, "/version <n> [<message>]"));
            return;
        }
    };
    let Some(slot) = slot else {
        notices.write(Notice::error(agent, "/version: no such revised message"));
        return;
    };
    let chosen = n.checked_sub(1).and_then(|at| slot.versions.get(at));
    let Some(chosen) = chosen else {
        let why = format!(
            "Message {} has {} versions.",
            slot.message,
            slot.versions.len()
        );
        notices.write(Notice::error(agent, why));
        return;
    };
    if let Some(branch) = chosen.branch {
        let at = slot.at;
        commands.queue(move |world: &mut World| switch_now(world, agent, at, branch));
    }
}

/// Keeps the message `message` and what follows it as a version and sends
/// `text` in its place, once no turn runs.
fn revise_now(world: &mut World, agent: Entity, message: MessageId, text: String) {
    if world.get::<ActiveTurn>(agent).is_some() {
        notice(
            world,
            Notice::error(agent, "The revision waits for the turn to end."),
        );
        return;
    }
    let Some(mut conversation) = world.get_mut::<Conversation>(agent) else {
        return;
    };
    let Some(at) = conversation.index_of(message) else {
        return;
    };
    let before = files_after(&conversation, at);
    let title: String = text
        .lines()
        .next()
        .unwrap_or_default()
        .chars()
        .take(40)
        .collect();
    if let Err(failure) = conversation.branch_off(at, title, author()) {
        notice(world, Notice::error(agent, failure.to_string()));
        return;
    }
    compare(world, agent, before, BTreeMap::new());
    let deliver = Deliver::new(agent, text, DeliveryMode::Queue).with_author(Author::user());
    world.trigger(deliver);
}

/// Makes the version kept as `branch` at entry `at` the active one, once
/// no turn runs.
fn switch_now(world: &mut World, agent: Entity, at: usize, branch: AsideId) {
    if world.get::<ActiveTurn>(agent).is_some() {
        notice(
            world,
            Notice::error(agent, "The switch waits for the turn to end."),
        );
        return;
    }
    let Some(mut conversation) = world.get_mut::<Conversation>(agent) else {
        return;
    };
    let before = files_after(&conversation, at);
    if let Err(failure) = conversation.switch_branch(branch, author()) {
        notice(world, Notice::error(agent, failure.to_string()));
        return;
    }
    let after = files_after(&conversation, at);
    compare(world, agent, before, after);
}

fn notice(world: &mut World, notice: Notice) {
    world.write_message(notice);
}

/// Keeps the files each settled tool call changed after the entry that
/// made the call.
fn record_changes(
    settled: On<Add<Settled>>,
    calls: Query<(&Published, &ToolRequest, &ForAgent)>,
    mut agents: Query<&mut Conversation>,
) {
    let Ok((published, request, &ForAgent(agent))) = calls.get(settled.entity) else {
        return;
    };
    let Ok(Some(FilesChanged(files))) = published.0.get::<FilesChanged>() else {
        return;
    };
    let Ok(mut conversation) = agents.get_mut(agent) else {
        return;
    };
    let made = conversation
        .entries()
        .rev()
        .find(|entry| match entry.message {
            Message::Assistant(reply) => reply.tool_calls().any(|call| call.id == request.call.id),
            _ => false,
        });
    let Some(after) = made.map(|entry| entry.id) else {
        return;
    };
    if let Ok(value) = serde_json::to_value(FilesRecord { files }) {
        conversation.add_aside(Aside::new(FILES, author(), value).after(Some(after)));
    }
}

/// The changes the active line made from entry `at` on, by file, in order.
fn files_after(conversation: &Conversation, at: usize) -> BTreeMap<String, Vec<FileChange>> {
    let mut found: Vec<(usize, FileChange)> = conversation
        .asides()
        .filter(|(_, aside)| aside.kind == FILES)
        .map(|(_, aside)| (conversation.aside_position(aside), aside))
        .filter(|(position, _)| *position > at)
        .filter_map(|(position, aside)| {
            Some((position, FilesRecord::deserialize(&aside.value).ok()?))
        })
        .flat_map(|(position, record)| record.files.into_iter().map(move |file| (position, file)))
        .collect();
    found.sort_by_key(|(position, _)| *position);
    let mut files: BTreeMap<String, Vec<FileChange>> = BTreeMap::new();
    for (_, change) in found {
        files.entry(change.path.clone()).or_default().push(change);
    }
    files
}

/// Compares the working tree, off the main thread, with what the selected
/// continuation left: each file either continuation changed is as the
/// selected one's last change left it, or, where only the one left
/// changed it, as it was before that one's first change.
fn compare(
    world: &mut World,
    agent: Entity,
    left: BTreeMap<String, Vec<FileChange>>,
    selected: BTreeMap<String, Vec<FileChange>>,
) {
    let mut expected: BTreeMap<String, Option<FileState>> = BTreeMap::new();
    for (path, changes) in &left {
        if let Some(first) = changes.first() {
            expected.insert(path.clone(), first.before.clone());
        }
    }
    for (path, changes) in &selected {
        if let Some(last) = changes.last() {
            expected.insert(path.clone(), last.after.clone());
        }
    }
    if expected.is_empty() {
        return;
    }
    let workspace = world.resource::<ProjectWorkspace>().0.clone();
    let wake = world.resource::<Wake>().clone();
    let read = blocking(move || {
        let differ: Vec<String> = expected
            .into_iter()
            .filter(|(path, want)| {
                let now = workspace.read(&PathBuf::from(path), READ_LIMIT).ok();
                let now = now.map(|contents| FileState::of(&contents.bytes));
                match (now, want) {
                    (None, None) => false,
                    (Some(now), Some(want)) => match (&now.hash, &want.hash) {
                        (Some(now), Some(want)) => now != want,
                        _ => now.bytes != want.bytes,
                    },
                    _ => true,
                }
            })
            .map(|(path, _)| path)
            .collect();
        Ok(differ.join(", "))
    });
    let pool = bevy_tasks::IoTaskPool::get_or_init(bevy_tasks::TaskPool::default);
    let running = Running::spawn(pool, &wake, async move { read.await.unwrap_or_default() });
    world.spawn((Comparing(agent), running));
}

/// Says which files differ from the selected version.
fn compared(
    done: On<Insert<Done<String>>>,
    comparing: Query<(&Comparing, &Done<String>)>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok((&Comparing(agent), Done(differ))) = comparing.get(done.entity) else {
        return;
    };
    if !differ.is_empty() {
        let said =
            format!("Files not as this version left them (switching changes no file): {differ}");
        notices.write(Notice::warn(agent, said));
    }
    commands.entity(done.entity).despawn();
}

/// Keeps one unsaved `‹ a/b ›` note after each message with versions.
fn badges(mut agents: Query<&mut Conversation, Changed<Conversation>>) {
    for mut conversation in &mut agents {
        let wanted: Vec<(Option<MessageId>, String)> = slots(&conversation)
            .into_iter()
            .filter(|slot| slot.versions.len() > 1)
            .map(|slot| (conversation.id(slot.at), slot.badge()))
            .collect();
        let ours =
            |aside: &Aside| aside.author == author() && !aside.saved && Note::of(aside).is_some();
        let shown: Vec<(AsideId, Option<MessageId>, String)> = conversation
            .asides()
            .filter(|(_, aside)| ours(aside))
            .filter_map(|(id, aside)| Some((id, aside.after, Note::of(aside)?.title)))
            .collect();
        let same = shown.len() == wanted.len()
            && shown
                .iter()
                .zip(&wanted)
                .all(|((_, after, title), (want_after, want_title))| {
                    after == want_after && title == want_title
                });
        if same {
            continue;
        }
        for (id, _, _) in shown {
            conversation.remove_aside(id);
        }
        for (after, badge) in wanted {
            let text = "/version <n> <message>, or Left and Right on the selected message";
            let note = Note::new(badge, text).aside(author(), after).unsaved();
            conversation.add_aside(note);
        }
    }
}

#[cfg(feature = "tui")]
mod tui {
    //! The badge's look, and the keys on a selected user message.

    use rig_harness::prelude::*;
    use rig_tui::crossterm::event::KeyCode;
    use rig_tui::ratatui::style::Stylize;
    use rig_tui::ratatui::text::Line;
    use rig_tui::{
        AppKeyBindingsExt, AppToolRenderersExt, KeyBinding, KeyPress, KeyPressed, KeyScope,
        MainPane, TranscriptItem, TranscriptView,
    };

    use super::{NAME, slots};

    pub(super) fn build(app: &mut App) {
        let on = |press: KeyPress, label: &str| {
            KeyBinding::new(press, label).in_scope(KeyScope::Transcript)
        };
        app.add_entry_renderer(NAME, |entry| {
            vec![Line::from(format!("  {}", entry.title)).dim()]
        });
        app.add_key_binding(
            on(KeyPress::key(KeyCode::Left), "Previous version"),
            previous,
        );
        app.add_key_binding(on(KeyPress::key(KeyCode::Right), "Next version"), next);
        app.add_key_binding(
            on(KeyPress::key(KeyCode::Char('e')), "Revise this message"),
            edit,
        );
    }

    /// The selected user message's number, the version shown and how
    /// many there are.
    fn selected(
        view: &TranscriptView,
        conversation: &Conversation,
    ) -> Option<(usize, usize, usize)> {
        let id = match view.selected.as_ref()? {
            TranscriptItem::Message(id) => *id,
            TranscriptItem::Aside(aside) => conversation.aside(*aside)?.after?,
            _ => return None,
        };
        let at = conversation.index_of(id)?;
        let found = slots(conversation);
        let slot = found.into_iter().find(|slot| slot.at == at)?;
        Some((
            slot.message,
            slot.active.saturating_add(1),
            slot.versions.len(),
        ))
    }

    fn step(
        pressed: &KeyPressed,
        view: &TranscriptView,
        agents: &Query<&Conversation>,
        by: isize,
    ) -> Option<RunCommand> {
        let agent = pressed.agent.filter(|_| !pressed.released)?;
        let (message, shown, count) = selected(view, agents.get(agent).ok()?)?;
        let to = shown
            .checked_add_signed(by)
            .filter(|to| (1..=count).contains(to))?;
        Some(RunCommand::new(agent, format!("version {to} {message}")))
    }

    fn previous(
        pressed: On<KeyPressed>,
        view: Single<&TranscriptView, With<MainPane>>,
        agents: Query<&Conversation>,
        mut commands: Commands,
    ) {
        if let Some(run) = step(&pressed, &view, &agents, -1) {
            commands.trigger(run);
        }
    }

    fn next(
        pressed: On<KeyPressed>,
        view: Single<&TranscriptView, With<MainPane>>,
        agents: Query<&Conversation>,
        mut commands: Commands,
    ) {
        if let Some(run) = step(&pressed, &view, &agents, 1) {
            commands.trigger(run);
        }
    }

    fn edit(
        pressed: On<KeyPressed>,
        view: Single<&TranscriptView, With<MainPane>>,
        agents: Query<&Conversation>,
        mut commands: Commands,
    ) {
        let Some(agent) = pressed.agent.filter(|_| !pressed.released) else {
            return;
        };
        let found = agents
            .get(agent)
            .ok()
            .and_then(|conversation| selected(&view, conversation));
        if let Some((message, _, _)) = found {
            commands.trigger(RunCommand::new(agent, format!("revise {message}")));
        }
    }
}

#[cfg(test)]
mod tests;
