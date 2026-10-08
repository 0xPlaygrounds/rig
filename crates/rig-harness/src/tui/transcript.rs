//! The transcript's wrapped rows, kept per message. A frame re-renders and
//! re-wraps only the messages that changed (a message's look also depends
//! on its neighbours: a tool call is drawn with the result the next message
//! carries), re-wraps without re-rendering when the width changed, and
//! copies out only the rows on screen. Before, every frame and every
//! streamed delta re-wrapped the whole history.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hasher};
use std::io;
use std::sync::Arc;

use bevy_ecs::entity::Entity;
use ratatui::style::{Style, Stylize};
use ratatui::text::Line;
use rig_core::completion::{AssistantContent, Message};
use rig_core::message::{ToolResult, UserContent};

use super::markdown;
use super::renderers::{RESULT_LINES, RenderToolCall, ToolCallView, excerpt};
use super::wrap::wrap_all;
use crate::core::subagents::REPORT_PREFIX;

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
    /// Brings the rows up to date with `messages` of `agent` at `width`.
    /// `changed` says whether the messages or the renderers may have
    /// changed since the last call; when they did not, only a new agent or
    /// width costs anything.
    pub(crate) fn update(
        &mut self,
        agent: Entity,
        messages: &[Message],
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
            let lines = message_lines(message, previous, next, renderers);
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

    /// The rows of `parts` that fit `height` rows, `scroll` rows up from
    /// the bottom. `scroll` is clamped to the top.
    pub(crate) fn visible(
        &self,
        parts: &[Part],
        height: usize,
        scroll: &mut usize,
    ) -> Vec<Line<'static>> {
        let rows_of = |part: &Part| -> usize {
            match part {
                Part::Message(index) => {
                    self.entries.get(*index).map_or(0, |entry| entry.rows.len())
                }
                Part::Rows(rows) => rows.len(),
            }
        };
        let total: usize = parts.iter().map(rows_of).sum();
        let bottom = total.saturating_sub(height);
        *scroll = (*scroll).min(bottom);
        let top = bottom - *scroll;
        let mut shown = Vec::with_capacity(height);
        let mut start = 0;
        for part in parts {
            let count = rows_of(part);
            let end = start + count;
            if end > top && start < top + height {
                let rows = match part {
                    Part::Message(index) => self
                        .entries
                        .get(*index)
                        .map(|entry| entry.rows.as_slice())
                        .unwrap_or_default(),
                    Part::Rows(rows) => rows.as_slice(),
                };
                let from = top.saturating_sub(start);
                let to = (top + height - start).min(count);
                shown.extend(rows.get(from..to).unwrap_or_default().iter().cloned());
            }
            start = end;
            if start >= top + height {
                break;
            }
        }
        shown
    }
}

/// Feeds what is written to a hasher.
struct HashWriter<'a>(&'a mut DefaultHasher);

impl io::Write for HashWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.write(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A hash of the messages a message's look depends on.
fn fingerprint(messages: [Option<&Message>; 3]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for message in messages {
        match message {
            Some(message) => {
                hasher.write_u8(1);
                serde_json::to_writer(HashWriter(&mut hasher), message).ok();
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

/// Draws `message`. A tool call's result, which comes in the `next`
/// message, is drawn with the call, so the results of a reply's calls are
/// not drawn after all of its calls.
fn message_lines(
    message: &Message,
    previous: Option<&Message>,
    next: Option<&Message>,
    renderers: &Renderers<'_>,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    match message {
        Message::System { .. } => {}
        Message::User { content } => {
            for item in content {
                match item {
                    UserContent::Text(text) => {
                        lines.push(Line::default());
                        if text.text.starts_with(REPORT_PREFIX) {
                            report_lines(&text.text, &mut lines);
                        } else {
                            user_lines(&text.text, &mut lines);
                        }
                    }
                    // Drawn under its call already.
                    UserContent::ToolResult(result) if answers(previous, result) => {}
                    UserContent::ToolResult(result) => {
                        let text: Vec<&str> = result
                            .content
                            .iter()
                            .filter_map(|content| content.as_text())
                            .collect();
                        let style = if result.is_error {
                            Style::new().red()
                        } else {
                            Style::new().dim()
                        };
                        lines.extend(excerpt(&text.join("\n"), RESULT_LINES, style));
                    }
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

/// A subagent's answer: its header, then the start of the answer.
/// `/agents` shows the subagent's whole transcript.
fn report_lines(text: &str, lines: &mut Vec<Line<'static>>) {
    let (head, answer) = text.split_once('\n').unwrap_or((text, ""));
    lines.push(Line::styled(
        format!("⤶ {head}"),
        Style::new().magenta().bold(),
    ));
    lines.extend(excerpt(answer, RESULT_LINES, Style::new().dim()));
}

/// `text`'s lines in one style.
pub(crate) fn plain_lines(text: &str, style: Style) -> Vec<Line<'static>> {
    text.lines()
        .map(|line| Line::styled(line.replace('\t', "    "), style))
        .collect()
}
