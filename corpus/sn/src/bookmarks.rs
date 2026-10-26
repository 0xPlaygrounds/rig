//! Plugin corpus: 17. Conversation bookmarks (held-out spec `17-bookmarks.md`).
//!
//! `/bookmark [label]` marks the latest message of the agent the line was
//! typed for, labelled with its first words when no label is given;
//! `/bookmark rm <label>` takes one away, and `/bookmarks` lists them in a
//! picker, where choosing one scrolls the terminal view to the message and
//! selects it.
//!
//! A bookmark is a saved aside of kind [`KIND`] after its message: it is
//! saved and restored with the conversation, never reaches a model, and is
//! drawn as the marker. A condensing leaves the aside in place; opening
//! it reads the agent's log as it was just before the splice that took
//! the message out (`rig_memory::as_of`) and shows that in a transcript
//! pane over the screen, scrolled to the message.

use rig_harness::prelude::*;
use rig_harness::rig_core::conversation::{Change, Conversation as Saved};
use rig_harness::rig_core::message::{Message, UserContent};
use rig_harness::rig_ecs::session_log::SessionStore;
use rig_harness::rig_memory::{LogRecord, SessionError, as_of, history};
use rig_tui::crossterm::event::KeyCode;
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::style::Style;
use rig_tui::ratatui::text::Line;
use rig_tui::{
    AsideView, ItemRow, KeyBinding, KeyFocus, KeyPress, KeyPressed, KeyScope, MainPane, PaneKeys,
    Placement, RenderBatch, RenderSystems, ScrollTo, ShowAgent, TranscriptItem, TranscriptPane,
    TranscriptRender, TranscriptSource, TranscriptView, TuiPanel,
};
use serde::{Deserialize, Serialize};

/// The kind of a bookmark's aside.
pub const KIND: &str = "bookmark";

/// The plugin, as the author of bookmarks.
const PLUGIN: &str = "bookmarks";

/// How many words of a message label a bookmark given no label.
const LABEL_WORDS: usize = 5;

/// A bookmark's aside value; the aside's `after` is the message it marks.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bookmark {
    /// Its label, unique in the session.
    pub label: String,
    /// When it was made, in milliseconds since the Unix epoch.
    pub at_ms: u64,
    /// The first line of the message's text.
    pub first_line: String,
}

impl Bookmark {
    /// The bookmark `aside` holds, if it holds one.
    pub fn of(aside: &Aside) -> Option<Self> {
        (aside.kind == KIND)
            .then(|| serde_json::from_value(aside.value.clone()).ok())
            .flatten()
    }
}

/// `/bookmark`, `/bookmarks` and the markers in the transcript.
#[derive(Default)]
pub struct BookmarksPlugin;

impl Plugin for BookmarksPlugin {
    fn build(&self, app: &mut App) {
        app.add_command(
            "bookmark",
            "Bookmark the latest message: /bookmark [label], /bookmark rm <label>",
            bookmark,
        )
        .add_command("bookmarks", "List the bookmarks and jump to one", list)
        .add_systems(TranscriptRender, draw.in_set(RenderSystems::Render));
    }
}

/// On a `/bookmarks` picker: the agent and aside of each item.
#[derive(Component)]
struct Choices(Vec<(Entity, AsideId)>);

/// On the pane showing a condensed-away bookmark's saved history.
#[derive(Component)]
struct HistoryPane;

/// Draws the bookmarks' asides.
fn draw(mut batch: ResMut<RenderBatch>) {
    for (aside, output) in batch.asides_mut(KIND) {
        output.lines = marker(&aside.view());
    }
}

/// The marker's look: a line naming the bookmark.
fn marker(view: &AsideView<'_>) -> Vec<Line<'static>> {
    let label = Bookmark::of(view.aside).map(|bookmark| bookmark.label);
    let label = label.unwrap_or_default();
    vec![Line::styled(
        format!("◆ bookmark: {label}"),
        Style::new().yellow(),
    )]
}

/// Every bookmark of every agent: the agent, the aside and the bookmark,
/// oldest first.
fn all<'a>(
    agents: impl Iterator<Item = (Entity, &'a Saved)>,
) -> Vec<(Entity, AsideId, MessageId, Bookmark)> {
    let mut found: Vec<_> = agents
        .flat_map(|(agent, conversation)| {
            conversation.asides().filter_map(move |(id, aside)| {
                Some((agent, id, aside.after?, Bookmark::of(aside)?))
            })
        })
        .collect();
    found.sort_by_key(|(_, id, _, bookmark)| (bookmark.at_ms, *id));
    found
}

/// The text of `message`, what the user or the model wrote.
fn text_of(message: &Message) -> String {
    let text = match message {
        Message::System { content } => content.clone(),
        Message::User { content } => content
            .iter()
            .filter_map(|item| match item {
                UserContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Message::Assistant(reply) => reply.text(),
    };
    text.trim().to_owned()
}

/// `/bookmark [label]` and `/bookmark rm <label>`.
fn bookmark(
    In(args): In<CommandArgs>,
    mut agents: Query<(Entity, &mut Conversation)>,
    mut notices: MessageWriter<Notice>,
) {
    let words = args.args.trim();
    let removing = words
        .strip_prefix("rm")
        .filter(|rest| rest.is_empty() || rest.starts_with(' '))
        .map(str::trim);
    let latest = agents.get(args.agent).ok().and_then(|(_, conversation)| {
        let at = conversation.messages().len().checked_sub(1)?;
        Some((
            conversation.id(at)?,
            text_of(conversation.messages().get(at)?),
        ))
    });
    let (label, latest) = match (removing, latest) {
        (Some(label), _) => (label.to_owned(), None),
        (None, None) => {
            let why = "There is no message to bookmark yet.";
            notices.write(Notice::error(args.agent, why));
            return;
        }
        (None, Some((message, text))) => {
            let first = text.split_whitespace().take(LABEL_WORDS);
            let label = match words {
                "" => first.collect::<Vec<_>>().join(" "),
                label => label.to_owned(),
            };
            (label, Some((message, text)))
        }
    };
    if label.is_empty() {
        let why = "Give a label: that message has no text.";
        notices.write(Notice::error(args.agent, why));
        return;
    }
    // A label is unique in the session: reusing one moves it.
    let taken = all(agents
        .iter()
        .map(|(agent, conversation)| (agent, &conversation.0)));
    let taken = taken.into_iter().find(|(.., kept)| kept.label == label);
    if let Some((agent, aside, ..)) = taken
        && let Ok((_, mut conversation)) = agents.get_mut(agent)
    {
        conversation.remove_aside(aside);
    }
    let Some((message, text)) = latest else {
        let said = match taken {
            Some(_) => Notice::info(args.agent, format!("Removed bookmark `{label}`.")),
            None => Notice::error(args.agent, format!("No bookmark is labelled `{label}`.")),
        };
        notices.write(said);
        return;
    };
    let value = Bookmark {
        label: label.clone(),
        at_ms: now_ms(),
        first_line: text.lines().next().unwrap_or_default().to_owned(),
    };
    let Ok(value) = serde_json::to_value(value) else {
        return;
    };
    if let Ok((_, mut conversation)) = agents.get_mut(args.agent) {
        let aside = Aside::new(KIND, Author::plugin(PLUGIN), value).after(Some(message));
        conversation.add_aside(aside);
    }
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
    agents: Query<(Entity, &Conversation, &AgentId, Option<&Name>)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let bookmarks = all(agents
        .iter()
        .map(|(agent, conversation, ..)| (agent, &conversation.0)));
    if bookmarks.is_empty() {
        let none = "No bookmarks yet: /bookmark [label].";
        notices.write(Notice::info(args.agent, none));
        return;
    }
    let now = now_ms();
    let mut picker = Picker::new("Bookmarks").for_agent(args.agent);
    for (agent, _, _, bookmark) in &bookmarks {
        let name = agents.get(*agent).map_or("", |(_, _, id, name)| {
            name.map_or(id.0.as_str(), Name::as_str)
        });
        let when = age(now.saturating_sub(bookmark.at_ms));
        picker.items.push(PickItem {
            label: bookmark.label.clone(),
            command: String::new(),
            detail: Some(format!("{when} · {name} · {}", bookmark.first_line)),
            checked: false,
        });
    }
    let choices = bookmarks.iter().map(|(agent, id, ..)| (*agent, *id));
    commands
        .spawn((picker, Choices(choices.collect())))
        .observe(open);
}

/// Opens the bookmark chosen in a `/bookmarks` picker: shows its agent in
/// the main pane, scrolled to its message and selected, or, when a
/// condensing took the message out, the saved history around it.
fn open(
    picked: On<Picked>,
    choices: Query<&Choices>,
    agents: Query<(&AgentId, &Conversation)>,
    (store, main, shown): (
        Option<Res<SessionStore>>,
        Option<Single<(Entity, &mut TranscriptView), With<MainPane>>>,
        Query<Entity, With<HistoryPane>>,
    ),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let chosen = picked
        .chosen
        .as_ref()
        .and_then(|chosen| chosen.items.first());
    let choice = chosen.and_then(|at| choices.get(picked.entity).ok()?.0.get(*at));
    let Some(&(agent, aside)) = choice else {
        return;
    };
    let Ok((id, conversation)) = agents.get(agent) else {
        notices.write(Notice::error(None, "That bookmark's agent is gone."));
        return;
    };
    let Some(message) = conversation.aside(aside).and_then(|aside| aside.after) else {
        return;
    };
    for pane in &shown {
        commands.entity(pane).despawn();
    }
    let item = TranscriptItem::Message(message);
    if conversation.index_of(message).is_some() {
        let Some(main) = main else {
            return;
        };
        let (pane, mut view) = main.into_inner();
        view.selected = Some(item.clone());
        commands.trigger(ShowAgent { entity: agent });
        commands.trigger(ScrollTo {
            pane,
            item: Some(item),
            row: ItemRow::Reveal,
        });
        return;
    }
    let then = store.map(|store| before_taken_out(&*store.0, &id.0, message));
    let Some(Ok(Some(then))) = then else {
        let why = "The saved history of that bookmark cannot be read.";
        notices.write(Notice::error(agent, why));
        return;
    };
    show_history(&mut commands, then, item);
}

/// The conversation of `log` in `store` just before the last change that
/// took `message` out of it; `None` when none did.
fn before_taken_out(
    store: &dyn Store,
    log: &str,
    message: MessageId,
) -> Result<Option<Saved>, SessionError> {
    let took_out = |change: &Change| match change {
        Change::Splice { removed, with, .. } => {
            removed.contains(&message) && with.iter().all(|entry| entry.id != Some(message))
        }
        Change::Reset { state } => state.index_of(message).is_none(),
        _ => false,
    };
    let records = history(store, log)?;
    let last = records
        .iter()
        .rev()
        .find_map(|(stamp, record)| match record {
            LogRecord::Change(recorded) if took_out(&recorded.change) => Some(stamp.n),
            _ => None,
        });
    let then = last.map(|n| as_of(store, log, n)).transpose()?;
    Ok(then.filter(|then| then.index_of(message).is_some()))
}

/// Shows `then` read-only in a pane over the screen, `item` selected at
/// its top; Esc in it closes it.
fn show_history(commands: &mut Commands, then: Saved, item: TranscriptItem) {
    let over = Placement::Over {
        width: Constraint::Percentage(90),
        height: Constraint::Percentage(90),
    };
    let mut view = TranscriptView::default();
    view.selected = Some(item.clone());
    let title = "Bookmark, as saved before it was condensed (Esc closes)";
    let source = TranscriptPane::new(TranscriptSource::Value(then)).titled(title);
    let pane = (source, view, TuiPanel::new(over), PaneKeys, HistoryPane);
    let pane = commands.spawn(pane).id();
    let close =
        KeyBinding::new(KeyPress::key(KeyCode::Esc), "Close").in_scope(KeyScope::Panel(pane));
    commands.spawn((close, ChildOf(pane))).observe(
        move |pressed: On<KeyPressed>, mut commands: Commands| {
            if !pressed.released {
                commands.entity(pane).try_despawn();
            }
        },
    );
    commands.insert_resource(KeyFocus(Some(pane)));
    commands.trigger(ScrollTo {
        pane,
        item: Some(item),
        row: ItemRow::Reveal,
    });
}

#[cfg(test)]
mod tests;
