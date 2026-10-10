//! The transcript's wrapped rows, kept per message. A frame re-renders and
//! re-wraps only the messages that changed (a message's look also depends
//! on its neighbours: a tool call is drawn with the result the next message
//! carries), re-wraps without re-rendering when the width changed, and
//! copies out only the rows on screen. Before, every frame and every
//! streamed delta re-wrapped the whole history.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hasher};
use std::sync::Arc;

use bevy_ecs::entity::Entity;
use ratatui::style::{Style, Stylize};
use ratatui::text::Line;
use rig_core::completion::{AssistantContent, Message};
use rig_core::message::{ToolResult, UserContent};

use super::markdown;
use super::renderers::{
    RESULT_LINES, RenderToolCall, ToolCallView, excerpt, result_style, result_text,
};
use super::wrap::wrap_all;
use crate::front::attached_file;
use crate::host::launcher::BUILD_ORIGIN;
use rig_ecs::agent::{Conversation, STOPPED};
use rig_ecs::inbox::{Origin, OriginKind};

/// The renderers by tool name.
pub(crate) type Renderers<'a> = HashMap<&'a str, &'a Arc<RenderToolCall>>;

/// One message's look: its lines, and those lines wrapped to the width.
struct Entry {
    fingerprint: u64,
    rows: Vec<Line<'static>>,
    lines: Vec<Line<'static>>,
}

/// The wrapped rows of the shown agent's messages.
#[derive(Default)]
pub(crate) struct Transcript {
    agent: Option<Entity>,
    width: u16,
    entries: Vec<Entry>,
}

/// A part of the transcript, in order: a message's cached rows, or rows
/// made for this frame (notices, a summary, the reply streaming in).
pub(crate) enum Part {
    Message(usize),
    Rows(Vec<Line<'static>>),
}

impl Transcript {
    /// Brings the rows up to date with the conversation of `agent` at
    /// `width`.
    /// `changed` says whether the messages or the renderers may have
    /// changed since the last call; when they did not, only a new agent or
    /// width costs anything.
    pub(crate) fn update(
        &mut self,
        agent: Entity,
        conversation: &Conversation,
        changed: bool,
        renderers: &Renderers<'_>,
        width: u16,
    ) {
        if self.agent != Some(agent) {
            self.agent = Some(agent);
            self.entries.clear();
        }
        let usable = usize::from(width.max(1));
        if self.width != width {
            self.width = width;
            for entry in &mut self.entries {
                entry.rows = wrap_all(&entry.lines, usable);
            }
        }
        let messages = conversation.messages();
        if !changed && self.entries.len() == messages.len() {
            return;
        }
        self.entries.truncate(messages.len());
        for (index, message) in messages.iter().enumerate() {
            let previous = index.checked_sub(1).and_then(|before| messages.get(before));
            let next = messages.get(index + 1);
            let fingerprint = fingerprint([previous, Some(message), next]);
            if self
                .entries
                .get(index)
                .is_some_and(|entry| entry.fingerprint == fingerprint)
            {
                continue;
            }
            let lines = message_lines(message, (conversation, index), previous, next, renderers);
            let entry = Entry {
                fingerprint,
                rows: wrap_all(&lines, usable),
                lines,
            };
            match self.entries.get_mut(index) {
                Some(slot) => *slot = entry,
                None => self.entries.push(entry),
            }
        }
    }

    /// Drops every cached row, after the renderers changed.
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }

    /// A part's rows.
    fn rows<'a>(&'a self, part: &'a Part) -> &'a [Line<'static>] {
        match part {
            Part::Message(index) => self
                .entries
                .get(*index)
                .map(|entry| entry.rows.as_slice())
                .unwrap_or_default(),
            Part::Rows(rows) => rows.as_slice(),
        }
    }

    /// The rows of `parts` that fit `height` rows at `width`, where
    /// `scroll` places them, and what is below them.
    pub(crate) fn visible(
        &self,
        parts: &[Part],
        (width, height): (u16, usize),
        scroll: &mut Scroll,
    ) -> (Vec<Line<'static>>, Below) {
        let counts: Vec<usize> = parts.iter().map(|part| self.rows(part).len()).collect();
        let (top, below) = scroll.place(&counts, height, width);
        let mut shown = Vec::with_capacity(height);
        let mut start = 0;
        for (part, count) in parts.iter().zip(&counts) {
            let end = start + count;
            if end > top && start < top + height {
                let from = top.saturating_sub(start);
                let to = (top + height - start).min(*count);
                shown.extend(
                    self.rows(part)
                        .get(from..to)
                        .unwrap_or_default()
                        .iter()
                        .cloned(),
                );
            }
            start = end;
            if start >= top + height {
                break;
            }
        }
        (shown, below)
    }
}

/// Where the transcript is scrolled: following its bottom, or held at the
/// row the user scrolled to, which then stays put as rows arrive below.
#[derive(Default)]
pub(crate) struct Scroll {
    /// The row held at the top of the screen; `None` follows the bottom.
    held: Option<Held>,
    /// Rows the keys moved the view since the last frame, up positive.
    moved: isize,
}

/// The top row of a view scrolled up.
struct Held {
    /// Its index in the whole transcript.
    top: usize,
    /// The part it is in, its row there and that part's rows, to find it
    /// again after a resize re-wraps everything above it.
    part: usize,
    row: usize,
    rows: usize,
    /// The transcript's width and rows at the last frame.
    width: u16,
    total: usize,
    /// Whether rows arrived below since the view was held.
    grew: bool,
}

/// What is below a view scrolled up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Below {
    /// Nothing: the view follows the bottom.
    Nothing,
    /// Rows that were there when the view was scrolled up.
    More,
    /// Rows that arrived since.
    New,
}

impl Scroll {
    /// Moves the view `rows` up.
    pub(crate) fn up(&mut self, rows: usize) {
        self.moved = self
            .moved
            .saturating_add(isize::try_from(rows).unwrap_or(isize::MAX));
    }

    /// Moves the view `rows` down; reaching the bottom follows it again.
    pub(crate) fn down(&mut self, rows: usize) {
        self.moved = self
            .moved
            .saturating_sub(isize::try_from(rows).unwrap_or(isize::MAX));
    }

    /// Follows the bottom again.
    pub(crate) fn follow(&mut self) {
        *self = Self::default();
    }

    /// The top row of a view `height` rows high over parts of `counts`
    /// rows at `width`, after the keys' moves; holds it while it is above
    /// the bottom.
    fn place(&mut self, counts: &[usize], height: usize, width: u16) -> (usize, Below) {
        let total: usize = counts.iter().sum();
        let bottom = total.saturating_sub(height);
        let moved = std::mem::take(&mut self.moved);
        let (top, grew) = match &self.held {
            None => (bottom, false),
            Some(held) if held.width == width => (held.top, held.grew || total > held.total),
            Some(held) => {
                let start: usize = counts.iter().take(held.part).sum();
                let rows = counts.get(held.part).copied().unwrap_or(0);
                let row = (held.row * rows / held.rows.max(1)).min(rows.saturating_sub(1));
                (start + row, held.grew)
            }
        };
        let top = top.saturating_add_signed(moved.saturating_neg());
        if top >= bottom {
            self.held = None;
            return (bottom, Below::Nothing);
        }
        let (mut part, mut start) = (0, 0);
        for (index, count) in counts.iter().enumerate() {
            part = index;
            if start + count > top {
                break;
            }
            start += count;
        }
        self.held = Some(Held {
            top,
            part,
            row: top - start,
            rows: counts.get(part).copied().unwrap_or(0),
            width,
            total,
            grew,
        });
        (top, if grew { Below::New } else { Below::More })
    }
}

/// A hash of the messages a message's look depends on.
fn fingerprint(messages: [Option<&Message>; 3]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for message in messages {
        match message {
            Some(message) => {
                hasher.write_u8(1);
                hasher.write(&serde_json::to_vec(message).unwrap_or_default());
            }
            None => hasher.write_u8(0),
        }
    }
    hasher.finish()
}

/// The tool results a message carries.
fn tool_results(message: Option<&Message>) -> impl Iterator<Item = &ToolResult> {
    let content = match message {
        Some(Message::User { content }) => Some(content.iter()),
        _ => None,
    };
    content.into_iter().flatten().filter_map(|item| match item {
        UserContent::ToolResult(result) => Some(result),
        _ => None,
    })
}

/// Whether `message` is a reply holding the call `result` answers.
fn answers(message: Option<&Message>, result: &ToolResult) -> bool {
    match message {
        Some(Message::Assistant(assistant)) => assistant
            .content
            .iter()
            .any(|item| matches!(item, AssistantContent::ToolCall(call) if call.id == result.call)),
        _ => false,
    }
}

/// Draws `message`, the `at`-th of `conversation`. A tool
/// call's result, which comes in the `next` message, is drawn with the
/// call, so the results of a reply's calls are not drawn after all of its
/// calls.
fn message_lines(
    message: &Message,
    (conversation, at): (&Conversation, usize),
    previous: Option<&Message>,
    next: Option<&Message>,
    renderers: &Renderers<'_>,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    match message {
        Message::System { .. } => {}
        Message::User { content } => {
            for (item_at, item) in content.iter().enumerate() {
                match item {
                    UserContent::Text(text) => {
                        if let Some((label, count)) = attached_file(&text.text) {
                            lines.push(Line::from(format!("  [{label}, {count} lines]")).dim());
                            continue;
                        }
                        if text.text == STOPPED {
                            lines.push(Line::from("  [stopped before it was answered]").dim());
                            continue;
                        }
                        lines.push(Line::default());
                        match conversation.origin(at, item_at) {
                            Some(origin) => delivered_lines(origin, &text.text, &mut lines),
                            None => user_lines(&text.text, &mut lines),
                        }
                    }
                    // Drawn under its call already.
                    UserContent::ToolResult(result) if answers(previous, result) => {}
                    UserContent::ToolResult(result) => lines.extend(excerpt(
                        &result_text(result),
                        RESULT_LINES,
                        result_style(result.is_error),
                    )),
                    UserContent::Image(_) => lines.push(Line::from("  [image]").dim()),
                    _ => lines.push(Line::from("  [attachment]").dim()),
                }
            }
        }
        Message::Assistant(assistant) => {
            for item in &assistant.content {
                match item {
                    AssistantContent::Text(text) => {
                        lines.push(Line::default());
                        lines.extend(markdown::render(&text.text));
                    }
                    AssistantContent::Reasoning(reasoning) => {
                        lines.extend(plain_lines(&reasoning.text, Style::new().dim().italic()));
                    }
                    AssistantContent::ToolCall(call) => {
                        let view = ToolCallView {
                            call,
                            result: tool_results(next).find(|result| result.call == call.id),
                        };
                        match renderers.get(call.function.name.as_str()) {
                            Some(render) => lines.extend(render(&view)),
                            None => lines.extend(view.default_lines()),
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    lines
}

/// What the user typed, under a `›`.
fn user_lines(text: &str, lines: &mut Vec<Line<'static>>) {
    let style = Style::new().cyan().bold();
    for (index, line) in text.lines().enumerate() {
        let prefix = if index == 0 { "› " } else { "  " };
        lines.push(Line::styled(
            format!("{prefix}{}", line.replace('\t', "    ")),
            style,
        ));
    }
}

/// Text an agent or a plugin sent: where it came from, then its start
/// without the header the model reads.
/// `/agents` shows an agent's whole transcript.
fn delivered_lines(origin: &Origin, text: &str, lines: &mut Vec<Line<'static>>) {
    let body = origin
        .header()
        .and_then(|header| text.strip_prefix(&header))
        .map_or(text, |body| body.trim_start_matches('\n'));
    // A failed build's note, which the model reads: told apart by its
    // origin.
    if matches!(&origin.kind, OriginKind::Plugin(name) if name == BUILD_ORIGIN) {
        lines.push(Line::styled(
            "✗ build failed (noted for the agent)",
            Style::new().red().bold(),
        ));
        lines.extend(excerpt(body, RESULT_LINES, Style::new().red().dim()));
        return;
    }
    lines.push(Line::styled(
        format!("⤶ {}", origin.label()),
        Style::new().magenta().bold(),
    ));
    lines.extend(excerpt(body, RESULT_LINES, Style::new().dim()));
}

/// `text`'s lines in one style.
pub(crate) fn plain_lines(text: &str, style: Style) -> Vec<Line<'static>> {
    text.lines()
        .map(|line| Line::styled(line.replace('\t', "    "), style))
        .collect()
}

#[cfg(test)]
mod tests;
