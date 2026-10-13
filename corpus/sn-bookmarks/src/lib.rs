//! Plugin corpus: 17. Conversation bookmarks (held-out spec `17-bookmarks.md`).
//!
//! `/bookmark [label]` marks the latest message of the agent the line was
//! typed for, labelled with its first words when no label is given;
//! `/bookmark rm <label>` takes one away, and `/bookmarks` lists them in a
//! picker, where choosing one scrolls the terminal view to the message and
//! selects it. The bookmarks are a saved resource ([`Bookmarks`]), so they
//! come back on resume; each is drawn as a transcript note after its
//! message, which no model sees. A fork gets every bookmark of its parent
//! that points at a message made before the fork point. A bookmark whose
//! message a condensing took out opens the saved history around it, read
//! from a copy of the session as it was before that condensing.

use rig_harness::prelude::*;
use rig_harness::rig_core::conversation::Conversation as Saved;
use rig_harness::rig_core::message::{AssistantContent, Message, UserContent};
use rig_harness::rig_ecs::journal::SessionStore;
use rig_harness::rig_memory::MemoryStore;
use rig_tui::ratatui::style::Style;
use rig_tui::ratatui::text::Line;
use rig_tui::{AppToolRenderersExt, EntryView, ItemRow, ScrollTo, TranscriptItem, TranscriptView};
use serde::{Deserialize, Serialize};

/// The author (a plugin name) of a bookmark's marker in the transcript.
pub const ORIGIN: &str = "bookmark";

/// The author (a plugin name) of the saved history shown for a
/// condensed-away bookmark.
pub const HISTORY_ORIGIN: &str = "bookmark-history";

/// The file of a forked session's store that holds its parent's
/// bookmarks, written by the parent as it forks.
const HANDED: &str = "bookmarks/forked.json";

/// How many words of a message label a bookmark given no label.
const LABEL_WORDS: usize = 5;

/// How many messages before and after a condensed-away one its saved
/// history shows.
const AROUND: usize = 3;

/// How many condensings deep the saved history is looked for.
const DEPTH: usize = 16;

/// One bookmark: a message of an agent, by id.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bookmark {
    /// Its label, unique in the session.
    pub label: String,
    /// The id of the agent whose message it marks.
    pub agent: String,
    /// The message it marks.
    pub message: MessageId,
    /// When it was made, in milliseconds since the Unix epoch.
    pub at_ms: u64,
    /// The first line of the message's text.
    pub first_line: String,
}

/// The session's bookmarks, oldest first, saved with the session (through
/// serde: a [`MessageId`] has no reflection).
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[reflect(opaque, Resource, Saved, Default, Clone, Serialize, Deserialize)]
pub struct Bookmarks(pub Vec<Bookmark>);

/// `/bookmark`, `/bookmarks` and the markers in the transcript.
#[derive(Default)]
pub struct BookmarksPlugin;

impl Plugin for BookmarksPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Bookmarks>()
            .register_type::<Bookmarks>()
            .add_message::<SessionForked>()
            .add_command(
                "bookmark",
                "Bookmark the latest message: /bookmark [label], /bookmark rm <label>",
                bookmark,
            )
            .add_command("bookmarks", "List the bookmarks and jump to one", list)
            .add_entry_renderer(ORIGIN, marker)
            .add_systems(Startup, inherit.run_if(resource_exists::<FirstStart>))
            .add_systems(
                Update,
                (
                    hand_to_forks,
                    show_markers.run_if(
                        resource_changed::<Bookmarks>.or_else(any_match_filter::<Added<AgentId>>),
                    ),
                ),
            );
    }
}

/// On a `/bookmarks` picker: the label of each item.
#[derive(Component)]
struct Choices(Vec<String>);

/// The notes by plugin `plugin` in `conversation`.
fn notes_of(conversation: &Saved, plugin: &str) -> Vec<AsideId> {
    let author = Author::plugin(plugin);
    let notes = conversation.asides();
    notes
        .filter(|(_, aside)| aside.kind == Note::KIND && aside.author == author)
        .map(|(id, _)| id)
        .collect()
}

/// Takes the notes by plugin `plugin` out of `conversation`, marking it
/// changed only when it held any.
fn clear_notes(conversation: &mut Mut<Conversation>, plugin: &str) {
    for id in notes_of(conversation, plugin) {
        conversation.remove_aside(id);
    }
}

/// The marker's look: a line naming the bookmark.
fn marker(entry: &EntryView<'_>) -> Vec<Line<'static>> {
    vec![Line::styled(
        format!("◆ bookmark: {}", entry.title),
        Style::new().yellow(),
    )]
}

/// The text of `message`: what the user or the model wrote, with the
/// tools the model called.
fn text_of(message: &Message) -> String {
    let parts: Vec<String> = match message {
        Message::System { content } => vec![content.clone()],
        Message::User { content } => content
            .iter()
            .filter_map(|item| match item {
                UserContent::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .collect(),
        Message::Assistant(reply) => reply
            .content
            .iter()
            .filter_map(|item| match item {
                AssistantContent::Text(text) => Some(text.text.clone()),
                AssistantContent::ToolCall(call) => {
                    Some(format!("[calls {}]", call.function.name.as_str()))
                }
                _ => None,
            })
            .collect(),
    };
    parts.join("\n").trim().to_owned()
}

/// `/bookmark [label]` and `/bookmark rm <label>`.
fn bookmark(
    In(args): In<CommandArgs>,
    agents: Query<(&AgentId, &Conversation)>,
    mut bookmarks: ResMut<Bookmarks>,
    mut notices: MessageWriter<Notice>,
) {
    let words = args.args.trim();
    if let Some(label) = words
        .strip_prefix("rm")
        .filter(|rest| rest.is_empty() || rest.starts_with(' '))
    {
        let label = label.trim();
        let before = bookmarks.0.len();
        bookmarks.0.retain(|kept| kept.label != label);
        if bookmarks.0.len() == before {
            notices.write(Notice::error(
                args.agent,
                format!("No bookmark is labelled `{label}`."),
            ));
        } else {
            notices.write(Notice::info(
                args.agent,
                format!("Removed bookmark `{label}`."),
            ));
        }
        return;
    }
    let Ok((id, conversation)) = agents.get(args.agent) else {
        return;
    };
    let latest = conversation.messages().len().checked_sub(1);
    let marked =
        latest.and_then(|at| Some((conversation.id(at)?, conversation.messages().get(at)?)));
    let Some((message, latest)) = marked else {
        notices.write(Notice::error(
            args.agent,
            "There is no message to bookmark yet.",
        ));
        return;
    };
    let text = text_of(latest);
    let label = match words {
        "" => text
            .split_whitespace()
            .take(LABEL_WORDS)
            .collect::<Vec<_>>()
            .join(" "),
        label => label.to_owned(),
    };
    if label.is_empty() {
        notices.write(Notice::error(
            args.agent,
            "That message has no text: give a label.",
        ));
        return;
    }
    // A label is unique in the session: reusing one moves it.
    bookmarks.0.retain(|kept| kept.label != label);
    bookmarks.0.push(Bookmark {
        label: label.clone(),
        agent: id.0.clone(),
        message,
        at_ms: now_ms(),
        first_line: text.lines().next().unwrap_or_default().to_owned(),
    });
    notices.write(Notice::info(args.agent, format!("Bookmarked `{label}`.")));
}

/// How long ago, roughly.
fn age(elapsed_ms: u64) -> String {
    match elapsed_ms / 1000 {
        0..60 => "just now".to_owned(),
        seconds @ 60..3600 => format!("{}m ago", seconds / 60),
        seconds @ 3600..86_400 => format!("{}h ago", seconds / 3600),
        seconds => format!("{}d ago", seconds / 86_400),
    }
}

/// `/bookmarks`: a picker of every bookmark, with when it was made, the
/// agent and the first line of its message.
fn list(
    In(args): In<CommandArgs>,
    bookmarks: Res<Bookmarks>,
    agents: Query<(&AgentId, &Name)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if bookmarks.0.is_empty() {
        notices.write(Notice::info(
            args.agent,
            "No bookmarks yet: /bookmark [label].",
        ));
        return;
    }
    let now = now_ms();
    let mut picker = Picker::new("Bookmarks").for_agent(args.agent);
    for bookmark in &bookmarks.0 {
        let agent = agents.iter().find(|(id, _)| id.0 == bookmark.agent);
        let agent = agent.map_or(bookmark.agent.as_str(), |(_, name)| name.as_str());
        let when = age(now.saturating_sub(bookmark.at_ms));
        picker.items.push(PickItem {
            label: bookmark.label.clone(),
            command: String::new(),
            detail: Some(format!("{when} · {agent} · {}", bookmark.first_line)),
        });
    }
    let labels = bookmarks
        .0
        .iter()
        .map(|bookmark| bookmark.label.clone())
        .collect();
    commands.spawn((picker, Choices(labels))).observe(open);
}

/// Opens the bookmark chosen in a `/bookmarks` picker: scrolls to its
/// message and selects it, or, when a condensing took the message out,
/// shows the saved history around it after what stands for it now.
fn open(
    picked: On<Picked>,
    choices: Query<&Choices>,
    bookmarks: Res<Bookmarks>,
    mut agents: Query<(Entity, &AgentId, &mut Conversation)>,
    (store, view): (Option<Res<SessionStore>>, Option<ResMut<TranscriptView>>),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Some(Chosen::Item(chosen)) = &picked.chosen else {
        return;
    };
    let label = choices
        .get(picked.entity)
        .ok()
        .and_then(|choices| choices.0.get(*chosen));
    let Some(bookmark) =
        label.and_then(|label| bookmarks.0.iter().find(|kept| kept.label == *label))
    else {
        return;
    };
    for (_, _, mut conversation) in &mut agents {
        clear_notes(&mut conversation, HISTORY_ORIGIN);
    }
    let found = agents.iter_mut().find(|(_, id, _)| id.0 == bookmark.agent);
    let Some((agent, id, mut conversation)) = found else {
        notices.write(Notice::error(
            None,
            format!("The agent of `{}` is gone.", bookmark.label),
        ));
        return;
    };
    let item = if conversation.index_of(bookmark.message).is_some() {
        TranscriptItem::Message(bookmark.message)
    } else {
        let history = store
            .and_then(|store| saved_history(&*store.0, &id.0, &conversation, bookmark.message));
        let Some(history) = history else {
            let why = format!("The saved history of `{}` cannot be read.", bookmark.label);
            notices.write(Notice::error(agent, why));
            return;
        };
        let title = format!("Bookmark `{}`, before it was condensed", bookmark.label);
        let note =
            Note::new(title, history).aside(Author::plugin(HISTORY_ORIGIN), Some(bookmark.message));
        TranscriptItem::Note(conversation.add_aside(note.unsaved()))
    };
    if let Some(mut view) = view {
        if matches!(item, TranscriptItem::Note(_)) {
            view.expanded.insert(item.clone());
        }
        view.selected = Some(item.clone());
    }
    commands.trigger(ScrollTo {
        agent,
        item: Some(item),
        row: ItemRow::Reveal,
    });
}

/// The messages around `message` of agent `agent`, as the session kept
/// them before the condensing that took it out of `now`. A copy of the
/// session as it was before the oldest message of `now` newer than
/// `message` was made holds it, unless an older condensing took it out
/// too: then the copy's oldest newer message is tried, and so on.
fn saved_history(
    store: &dyn Store,
    agent: &str,
    now: &Saved,
    message: MessageId,
) -> Option<String> {
    // Ids rise in the order messages were made: the oldest newer message
    // was made after `message` and no later than what stands for it.
    let oldest_newer = |conversation: &Saved| {
        let ids = (0..conversation.messages().len()).filter_map(|at| conversation.id(at));
        ids.filter(|id| *id > message).min()
    };
    let mut before = oldest_newer(now)?;
    for _ in 0..DEPTH {
        let copy = MemoryStore::default();
        fork(store, &copy, Some((agent, before))).ok()?;
        let session = Session::read(&copy).ok()?;
        let then = session
            .agents
            .into_iter()
            .find(|saved| saved.id == agent)?
            .conversation;
        if let Some(at) = then.index_of(message) {
            return Some(around(&then, at));
        }
        before = oldest_newer(&then).filter(|older| *older < before)?;
    }
    None
}

/// Messages `at - AROUND ..= at + AROUND` of `conversation`, the one at
/// `at` marked.
fn around(conversation: &Saved, at: usize) -> String {
    let messages = conversation.messages();
    let first = at.saturating_sub(AROUND);
    let shown = messages.iter().enumerate().skip(first).take(2 * AROUND + 1);
    let lines: Vec<String> = shown
        .map(|(index, message)| {
            let who = match message {
                Message::Assistant(_) => "model",
                Message::System { .. } => "system",
                Message::User { .. } => "user",
            };
            let mark = if index == at { "◆ " } else { "" };
            format!("{mark}{who}: {}", text_of(message))
        })
        .collect();
    lines.join("\n\n")
}

/// Draws each bookmark as a note after its message, in place of the
/// markers drawn before.
fn show_markers(bookmarks: Res<Bookmarks>, mut agents: Query<(&AgentId, &mut Conversation)>) {
    for (id, mut conversation) in &mut agents {
        clear_notes(&mut conversation, ORIGIN);
        for bookmark in bookmarks.0.iter().filter(|bookmark| bookmark.agent == id.0) {
            let note = Note::new(&bookmark.label, &bookmark.first_line);
            let marker = note.aside(Author::plugin(ORIGIN), Some(bookmark.message));
            // The saved resource is what is kept; the markers are drawn from it.
            conversation.add_aside(marker.unsaved());
        }
    }
}

/// Hands every bookmark to each fork of this session, in its store; the
/// fork keeps those before its fork point ([`inherit`]).
fn hand_to_forks(
    mut forked: MessageReader<SessionForked>,
    bookmarks: Res<Bookmarks>,
    (stores, home): (Res<SessionStores>, Option<Res<RigHome>>),
    mut notices: MessageWriter<Notice>,
) {
    for fork in forked.read() {
        let home = home
            .as_ref()
            .map_or_else(Home::from_env, |home| home.0.clone());
        let written = serde_json::to_vec(&bookmarks.0)
            .map_err(std::io::Error::other)
            .and_then(|json| {
                stores
                    .open(&home.session(&fork.to))?
                    .write_file(HANDED, &json)
            });
        if let Err(failure) = written {
            let why = format!("The fork {} did not get the bookmarks: {failure}", fork.to);
            notices.write(Notice::error(None, why));
        }
    }
}

/// In a fork's first start, keeps the bookmarks its parent handed it, else
/// those the copy restored, that point at a message made before the fork
/// point: one its agent's copy has a message at least as new as.
fn inherit(
    store: Option<Res<SessionStore>>,
    agents: Query<(&AgentId, &Conversation)>,
    mut bookmarks: ResMut<Bookmarks>,
) {
    let Some(store) = store.filter(|store| ForkedFrom::of(&*store.0).is_some()) else {
        return;
    };
    let handed = store.0.read_file(HANDED).ok();
    let handed: Option<Vec<Bookmark>> = handed.and_then(|json| serde_json::from_slice(&json).ok());
    let mut kept = handed.unwrap_or_else(|| bookmarks.0.clone());
    kept.retain(|bookmark| {
        let copy = agents.iter().find(|(id, _)| id.0 == bookmark.agent);
        let newest = copy.and_then(|(_, conversation)| {
            (0..conversation.messages().len())
                .filter_map(|at| conversation.id(at))
                .max()
        });
        newest.is_some_and(|newest| bookmark.message <= newest)
    });
    bookmarks.set_if_neq(Bookmarks(kept));
}

#[cfg(test)]
mod tests;
