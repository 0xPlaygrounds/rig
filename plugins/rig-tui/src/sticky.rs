//! [`StickyPromptPlugin`]: pins the most recent user message scrolled above
//! a transcript pane's top row over that row, so the prompt stays in sight
//! through long replies and while scrolling through earlier turns. It hides
//! whenever that message is visible on screen.
//!
//! ```no_run
//! use rig_harness::prelude::*;
//!
//! App::new().add_plugins((rig_tui::TuiPlugin, rig_tui::sticky::StickyPromptPlugin));
//! ```

use std::collections::HashMap;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Widget};
use rig_core::completion::Message;
use rig_core::conversation::{Conversation, EntryRef, attached_file, unattached};
use rig_core::message::UserContent;
use rig_ecs::agent::Conversation as AgentConversation;
use rig_ecs::prelude::MessageId;
use rig_harness::main_agent::ShownAgent;
use rig_tools::shorten;

use crate::pane::{DrawPanes, PaneState, TranscriptPane, TranscriptSource};
use crate::panel::{PanelCanvas, TuiSystems};
use crate::theme::Theme;
use crate::view::{TranscriptItem, TuiView};

/// Columns taken by the `"› "` prefix before the prompt text.
const PREFIX_WIDTH: u16 = 2;

/// Draws the user's most recent off-screen prompt over the top row of each
/// transcript pane. The terminal view works without it;
/// [`TuiPlugins`](crate::TuiPlugins) adds it with the view. Without the
/// [`TuiViewPlugin`](crate::TuiViewPlugin), added before it or after, it
/// adds nothing.
#[derive(Default)]
pub struct StickyPromptPlugin;

impl Plugin for StickyPromptPlugin {
    fn build(&self, _app: &mut App) {}

    // In `finish`, after every plugin's `build`, so the order they were
    // added in does not matter.
    fn finish(&self, app: &mut App) {
        if !app.is_plugin_added::<crate::TuiViewPlugin>() {
            return;
        }
        app.add_systems(
            PostUpdate,
            draw_sticky_prompts
                .in_set(TuiSystems::Draw)
                .after(DrawPanes)
                .run_if(resource_exists::<TuiView>),
        );
    }
}

/// Overlays each transcript pane's top row with the most recent user
/// message scrolled completely above it.
fn draw_sticky_prompts(
    mut panes: Query<(&TranscriptPane, &PaneState, &mut PanelCanvas)>,
    conversations: Query<&AgentConversation>,
    shown: Res<ShownAgent>,
    theme: Res<Theme>,
) {
    for (pane, state, mut canvas) in &mut panes {
        let area = state.area;
        // A one-row pane keeps its only row for transcript text.
        if area.height < 2 || area.width <= PREFIX_WIDTH {
            continue;
        }
        let Some(conversation) = conversation_of(pane, shown.0, &conversations) else {
            continue;
        };
        let Some(prompt) = sticky_prompt(conversation, state) else {
            continue;
        };
        let Some(row) = area.rows().next() else {
            continue;
        };
        draw_prompt(canvas.buffer_mut(), row, &prompt, &theme);
    }
}

/// The conversation `pane` draws, if any.
fn conversation_of<'a>(
    pane: &'a TranscriptPane,
    shown: Option<Entity>,
    conversations: &'a Query<&AgentConversation>,
) -> Option<&'a Conversation> {
    match &pane.source {
        TranscriptSource::Value(conversation) => Some(conversation),
        TranscriptSource::Agent(agent) => conversations.get(*agent).ok().map(|conv| &**conv),
        TranscriptSource::Shown => shown
            .and_then(|agent| conversations.get(agent).ok())
            .map(|conv| &**conv),
    }
}

/// The user prompt to pin at the top of `state`: the last user message at
/// or above the pane's top row, only when none of its rows remain on
/// screen.
fn sticky_prompt(conversation: &Conversation, state: &PaneState) -> Option<String> {
    let top = state.on_screen.start;
    if state.on_screen.is_empty() || top == 0 {
        return None;
    }
    let prompts: HashMap<MessageId, EntryRef<'_>> = conversation
        .entries()
        .filter(|entry| is_user_prompt(*entry))
        .map(|entry| (entry.id, entry))
        .collect();
    if prompts.is_empty() {
        return None;
    }
    let mut found = None;
    for (item, rows) in state.transcript.marks_in(&state.parts) {
        if rows.start > top {
            break;
        }
        let TranscriptItem::Message(id) = item else {
            continue;
        };
        if rows.is_empty() {
            continue;
        }
        if let Some(&entry) = prompts.get(id) {
            found = Some((entry, rows));
        }
    }
    let (entry, rows) = found?;
    if rows.end > top {
        return None;
    }
    prompt_text(entry)
}

/// Whether `entry` is a message the user sent: user-authored, with no tool
/// results, and with content the transcript draws.
fn is_user_prompt(entry: EntryRef<'_>) -> bool {
    if !entry.is_users_own() {
        return false;
    }
    let Message::User { content } = entry.message else {
        return false;
    };
    if content
        .iter()
        .any(|part| matches!(part, UserContent::ToolResult(_)))
    {
        return false;
    }
    content.iter().any(|part| match part {
        UserContent::Text(text) => unattached(&text.text).is_none(),
        UserContent::ToolResult(_) => false,
        _ => true,
    })
}

/// `entry` collapsed to one line: its user-typed text with line breaks as
/// spaces, or the first attachment label when it has no typed text.
fn prompt_text(entry: EntryRef<'_>) -> Option<String> {
    let Message::User { content } = entry.message else {
        return None;
    };
    let mut lines = Vec::new();
    let mut attachment = None;
    for item in content {
        match item {
            UserContent::Text(text) => {
                if let Some((label, _)) = attached_file(&text.text) {
                    attachment.get_or_insert_with(|| format!("[{label}]"));
                    continue;
                }
                if unattached(&text.text).is_some() {
                    continue;
                }
                for line in text.text.lines() {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        lines.push(trimmed.replace('\t', " "));
                    }
                }
            }
            UserContent::Image(_) => {
                attachment.get_or_insert_with(|| "[image]".to_owned());
            }
            UserContent::Document(_) => {
                attachment.get_or_insert_with(|| "[document]".to_owned());
            }
            UserContent::Audio(_) => {
                attachment.get_or_insert_with(|| "[audio]".to_owned());
            }
            UserContent::Video(_) => {
                attachment.get_or_insert_with(|| "[video]".to_owned());
            }
            UserContent::ToolResult(_) => {}
        }
    }
    if !lines.is_empty() {
        Some(lines.join(" "))
    } else {
        attachment
    }
}

/// Draws `prompt` onto `row` on the user band, or reversed when
/// [`Theme::plain`] is set.
fn draw_prompt(buffer: &mut Buffer, row: Rect, prompt: &str, theme: &Theme) {
    let room = usize::from(row.width.saturating_sub(PREFIX_WIDTH));
    let text = shorten(prompt, room);
    let background = match theme.plain {
        true => Style::new().reversed(),
        false => Style::new().bg(theme.user),
    };
    Clear.render(row, buffer);
    buffer.set_style(row, background);
    Line::from(vec![Span::from("› ").dim(), Span::from(text).bold()]).render(row, buffer);
}

#[cfg(test)]
mod tests;
