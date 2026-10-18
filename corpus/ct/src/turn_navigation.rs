//! CT-17: turn navigation and a sticky prompt header in the terminal view.
//!
//! `Ctrl+Up` and `Ctrl+Down` scroll the transcript of the agent shown to
//! the prompt before or after the top row, `Ctrl+Home` to the first prompt
//! and `Ctrl+End` back to the live bottom; the `[turn-navigation]` table of
//! the settings rebinds them ([`NavigationKeys`]). While the view is
//! scrolled up, a one-row panel over the transcript shows the prompt the
//! top row belongs to ([`PromptHeader`]): its turn number, its first line
//! cut to the width, how long the turn took and how many tools it called.
//! A prompt is a request by the user or by an agent (a subagent's task),
//! never a plugin's message. The view itself keeps a scrolled view where it
//! is as output arrives and marks the new output below it. The plugin is a
//! view half only: `plugins.toml` names it as an entry's `view`, and it
//! reads the mirrored conversation.

use std::collections::HashSet;
use std::str::FromStr;

use rig_harness::prelude::*;
use rig_harness::rig_core::conversation::{self, AuthorKind, EntryRef};
use rig_harness::rig_core::message::Message as Said;
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::style::Style;
use rig_tui::ratatui::text::Line;
use rig_tui::ratatui::widgets::Paragraph;
use rig_tui::{
    AppKeyBindingsExt, DrawnTranscript, ItemRow, KeyPressed, Keys, MainPane, PanelCanvas,
    Placement, ScrollTo, TranscriptItem, TranscriptView, TuiPanel, TuiPlugin, TuiSystems,
};
use serde::{Deserialize, Deserializer};
use unicode_width::UnicodeWidthChar;

/// The `[turn-navigation]` table: the keys, each as the terminal view
/// reads keys (`ctrl+up`, `alt+k`, `ctrl+x p`).
#[derive(Resource, Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NavigationKeys {
    /// To the prompt before the top row.
    #[serde(deserialize_with = "keys")]
    pub previous: Keys,
    /// To the prompt after the top row, else the live bottom.
    #[serde(deserialize_with = "keys")]
    pub next: Keys,
    /// To the first prompt.
    #[serde(deserialize_with = "keys")]
    pub first: Keys,
    /// Back to the live bottom.
    #[serde(deserialize_with = "keys")]
    pub last: Keys,
}

impl Default for NavigationKeys {
    fn default() -> Self {
        let key = |text: &str| Keys::from_str(text).unwrap_or_else(|_| Keys::from([]));
        Self {
            previous: key("ctrl+up"),
            next: key("ctrl+down"),
            first: key("ctrl+home"),
            last: key("ctrl+end"),
        }
    }
}

/// Reads keys from their text.
fn keys<'de, D: Deserializer<'de>>(input: D) -> Result<Keys, D::Error> {
    let text = String::deserialize(input)?;
    Keys::from_str(&text).map_err(serde::de::Error::custom)
}

/// The turn of the prompt the top row of a scrolled view belongs to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TurnHeader {
    /// The turn's number among the prompts, from 1.
    pub turn: usize,
    /// The prompt's first line with text.
    pub first_line: String,
    /// From the prompt to the turn's last entry so far.
    pub elapsed: Duration,
    /// The tool calls the turn made.
    pub tool_calls: usize,
}

impl TurnHeader {
    /// The header as one row `width` columns wide: the prompt is cut,
    /// with `…`, to leave room for the number, the time and the calls.
    pub fn line(&self, width: u16) -> Line<'static> {
        let calls = match self.tool_calls {
            1 => "1 tool".to_owned(),
            n => format!("{n} tools"),
        };
        let head = format!("▸ {} · ", self.turn);
        let tail = format!(" · {} · {calls}", short_time(self.elapsed));
        let room = usize::from(width).saturating_sub(columns(&head) + columns(&tail));
        let text = format!("{head}{}{tail}", cut(&self.first_line, room));
        Line::styled(cut(&text, usize::from(width)), Style::new().reversed())
    }
}

/// The header shown over the transcript: `Some` while the view of the
/// agent shown is scrolled up and its top row is in a prompt's turn.
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct PromptHeader(pub Option<TurnHeader>);

/// On the header's panel.
#[derive(Component)]
struct HeaderPanel;

/// Where a navigation key goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Jump {
    Previous,
    Next,
    First,
    Last,
}

/// The keys and the header.
#[derive(Default)]
pub struct TurnNavigationPlugin;

impl Plugin for TurnNavigationPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<TuiPlugin>() {
            app.add_plugins(TuiPlugin);
        }
        app.add_setting::<NavigationKeys>("turn-navigation")
            .init_resource::<PromptHeader>()
            .add_systems(
                PostUpdate,
                (
                    keep_drawn.in_set(TuiSystems::Prepare),
                    note_header
                        .after(TuiSystems::Layout)
                        .before(TuiSystems::Draw),
                    draw_header.in_set(TuiSystems::Draw),
                ),
            );
        let panel = TuiPanel::new(Placement::Top(Constraint::Length(0)));
        app.world_mut()
            .spawn((Name::new("panel:prompt-header"), HeaderPanel, panel));
        let keys = app.world().resource::<NavigationKeys>().clone();
        for (keys, label, to) in [
            (keys.previous, "Previous prompt", Jump::Previous),
            (keys.next, "Next prompt", Jump::Next),
            (keys.first, "First prompt", Jump::First),
            (keys.last, "Back to the bottom", Jump::Last),
        ] {
            app.add_key_binding(rig_tui::KeyBinding::new(keys, label), jump(to));
        }
    }
}

/// Whether `entry` is a prompt: a request by the user or an agent.
fn is_prompt(entry: &EntryRef<'_>) -> bool {
    entry.is_request()
        && entry
            .author
            .is_none_or(|author| !matches!(author.kind, AuthorKind::Plugin(_)))
}

/// Each prompt of `conversation`, with its turn: the request of each turn
/// the user or an agent asked for, numbered among them. A turn a plugin's
/// message began, and what it adds, belong to the prompt before it.
fn turns(conversation: &conversation::Conversation) -> Vec<(MessageId, TurnHeader)> {
    let asked: HashSet<MessageId> = conversation
        .turns()
        .filter_map(|(_, record)| conversation.request(record.start))
        .filter(is_prompt)
        .map(|request| request.id)
        .collect();
    let mut turns: Vec<(MessageId, u64, TurnHeader)> = Vec::new();
    for entry in conversation.entries() {
        if asked.contains(&entry.id) {
            let header = TurnHeader {
                turn: turns.len() + 1,
                first_line: first_line(entry.message),
                elapsed: Duration::ZERO,
                tool_calls: 0,
            };
            turns.push((entry.id, entry.made, header));
        } else if let Some((_, started, header)) = turns.last_mut() {
            header.elapsed = Duration::from_millis(entry.made.saturating_sub(*started));
            if let Said::Assistant(reply) = entry.message {
                let calls = reply.content.iter();
                header.tool_calls += calls
                    .filter(|item| matches!(item, AssistantContent::ToolCall(_)))
                    .count();
            }
        }
    }
    turns
        .into_iter()
        .map(|(id, _, header)| (id, header))
        .collect()
}

/// The first line with text of a user message.
fn first_line(message: &Said) -> String {
    let Said::User { content } = message else {
        return String::new();
    };
    let texts = content.iter().filter_map(|item| match item {
        UserContent::Text(text) => Some(text.text.as_str()),
        _ => None,
    });
    let mut lines = texts.flat_map(str::lines).map(str::trim);
    lines
        .find(|line| !line.is_empty())
        .unwrap_or_default()
        .to_owned()
}

/// Each prompt drawn in `drawn`, with its turn and its first row, top to
/// bottom.
fn drawn_prompts(
    conversation: &conversation::Conversation,
    drawn: &DrawnTranscript,
) -> Vec<(MessageId, TurnHeader, usize)> {
    let turns = turns(conversation).into_iter();
    let rows = turns.filter_map(|(id, header)| {
        let rows = drawn.rows_of(&TranscriptItem::Message(id))?;
        Some((id, header, rows.start))
    });
    rows.collect()
}

/// Has the main transcript pane keep the rows it draws, which the keys
/// and the header read.
fn keep_drawn(
    panes: Query<Entity, (With<MainPane>, Without<DrawnTranscript>)>,
    mut commands: Commands,
) {
    for pane in &panes {
        commands.entity(pane).insert(DrawnTranscript::default());
    }
}

/// The main pane and what it drew.
type MainDrawn<'w, 's> = Query<'w, 's, (Entity, &'static DrawnTranscript), With<MainPane>>;

/// The observer of a navigation key: scrolls the main pane, showing the
/// agent shown, as `to` says, from the rows the last frame drew.
fn jump(
    to: Jump,
) -> impl FnMut(On<KeyPressed>, MainDrawn, Query<&Conversation>, Commands) + Send + Sync + 'static {
    move |pressed, panes, conversations, mut commands| {
        let (Some(agent), false, Ok((pane, drawn))) =
            (pressed.agent, pressed.released, panes.single())
        else {
            return;
        };
        let bottom = ScrollTo {
            pane,
            item: None,
            row: ItemRow::Top(0),
        };
        if to == Jump::Last {
            commands.trigger(bottom);
            return;
        }
        let Some(conversation) = conversations
            .get(agent)
            .ok()
            .filter(|_| drawn.agent() == Some(agent))
        else {
            return;
        };
        let prompts = drawn_prompts(conversation, drawn);
        let top = drawn.on_screen().start;
        let target = match to {
            Jump::First => prompts.first(),
            Jump::Previous => prompts.iter().rev().find(|(.., start)| *start < top),
            Jump::Next | Jump::Last => prompts.iter().find(|(.., start)| *start > top),
        };
        match target {
            Some((id, ..)) => commands.trigger(ScrollTo {
                pane,
                item: Some(TranscriptItem::Message(*id)),
                row: ItemRow::Top(0),
            }),
            None if to == Jump::Next => commands.trigger(bottom),
            None => {}
        }
    }
}

/// Once the frame is laid out, the header for where the main pane is
/// scrolled, drawn in this frame. When it comes or goes, its row (the
/// panel's height) is laid out only at the next frame, which the changed
/// panel asks for and the wake brings now.
fn note_header(
    pane: Single<(Ref<DrawnTranscript>, Ref<TranscriptView>), With<MainPane>>,
    conversations: Query<&Conversation>,
    mut header: ResMut<PromptHeader>,
    mut panels: Query<&mut TuiPanel, With<HeaderPanel>>,
    wake: Option<Res<Wake>>,
) {
    let (drawn, view) = &*pane;
    if !drawn.is_changed() && !view.is_changed() {
        return;
    }
    let scrolled = view.top().and(drawn.agent());
    let conversation = scrolled.and_then(|agent| conversations.get(agent).ok());
    let top = drawn.on_screen().start;
    let shown = conversation.and_then(|conversation| {
        let prompts = drawn_prompts(conversation, drawn).into_iter();
        let mut above = prompts.filter(|(.., start)| *start <= top);
        above.next_back().map(|(_, header, _)| header)
    });
    if !header.set_if_neq(PromptHeader(shown)) {
        return;
    }
    let rows = u16::from(header.0.is_some());
    let placed = TuiPanel::new(Placement::Top(Constraint::Length(rows)));
    // Every panel is placed, so no short-circuiting `any`.
    let mut moved = false;
    for mut panel in &mut panels {
        moved |= panel.set_if_neq(placed.clone());
    }
    if let Some(wake) = wake.filter(|_| moved) {
        wake.wake();
    }
}

/// Draws the header into its panel.
fn draw_header(header: Res<PromptHeader>, mut panels: Query<&mut PanelCanvas, With<HeaderPanel>>) {
    let Some(shown) = &header.0 else {
        return;
    };
    for mut canvas in &mut panels {
        let width = canvas.area().width;
        if width > 0 {
            canvas.render(Paragraph::new(shown.line(width)));
        }
    }
}

/// How wide `text` is drawn, in columns.
fn columns(text: &str) -> usize {
    text.chars().map(|c| c.width().unwrap_or(0)).sum()
}

/// `text` cut to `width` columns, ending in `…` when cut.
fn cut(text: &str, width: usize) -> String {
    if columns(text) <= width {
        return text.to_owned();
    }
    let mut kept = String::new();
    let mut used = 0;
    for c in text.chars() {
        let wide = c.width().unwrap_or(0);
        if used + wide + 1 > width {
            break;
        }
        used += wide;
        kept.push(c);
    }
    if width > 0 {
        kept.push('…');
    }
    kept
}

/// A duration in a few characters: `42s`, `3m 05s`, `2h 10m`.
fn short_time(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs();
    match seconds {
        0..60 => format!("{seconds}s"),
        60..3600 => format!("{}m {:02}s", seconds / 60, seconds % 60),
        _ => format!("{}h {:02}m", seconds / 3600, seconds / 60 % 60),
    }
}

#[cfg(test)]
mod tests;
