//! Drawing: the conversation, notices, a status line, the input line and
//! the picker.

use bevy::ecs::query::QueryData;
use bevy::prelude::*;
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, Paragraph};
use rig_core::completion::{AssistantContent, Message};
use rig_core::message::{ToolResultContent, UserContent};
use unicode_width::UnicodeWidthChar;

use crate::commands::BuildProgress;
use crate::core::{
    AgentStatus, Conversation, EffortChoice, ModelChoice, ModelEndpoint, NoticeLevel,
    StreamingText, ToolCallDone, ToolCallRun, Work, effort_label,
};

use super::view::{Picker, TuiView};

/// Result lines shown under a tool call.
const RESULT_LINES: usize = 4;
/// Reasoning lines shown for one reply.
const REASONING_LINES: usize = 3;

/// The components of an agent the view shows.
#[derive(QueryData)]
pub(super) struct AgentView {
    conversation: &'static Conversation,
    status: &'static AgentStatus,
    model: &'static ModelChoice,
    effort: &'static EffortChoice,
    endpoint: Option<&'static ModelEndpoint>,
    work: Option<&'static Work>,
}

/// What one frame draws, in lines.
pub(super) struct Scene {
    conversation: Vec<Line<'static>>,
    status: Line<'static>,
}

impl Scene {
    pub(super) fn new(
        agent: &AgentViewItem,
        streams: &Query<&StreamingText>,
        calls: &Query<(&ToolCallRun, Option<&ToolCallDone>)>,
        build: Option<&BuildProgress>,
    ) -> Self {
        let mut conversation = Vec::new();
        for message in &agent.conversation.0 {
            message_lines(message, &mut conversation);
        }
        let work: Vec<Entity> = agent.work.iter().flat_map(|work| work.iter()).collect();
        for entity in &work {
            if let Ok(streamed) = streams.get(*entity) {
                reasoning_lines(&streamed.reasoning, &mut conversation);
                text_lines(&streamed.text, Style::new(), "", &mut conversation);
            }
            if let Ok((run, done)) = calls.get(*entity) {
                let state = if done.is_some() { "done" } else { "running" };
                conversation.push(Line::styled(
                    format!("  … {} {state}", run.call.function.name.as_str()),
                    Style::new().fg(Color::DarkGray),
                ));
            }
        }
        Self {
            conversation,
            status: status_line(agent, build),
        }
    }
}

fn status_line(agent: &AgentViewItem, build: Option<&BuildProgress>) -> Line<'static> {
    let model = match (agent.endpoint, &agent.model.0) {
        (Some(endpoint), _) => endpoint.spec.display_name.clone(),
        (None, Some(reference)) => format!("{reference} (not connected)"),
        (None, None) => "no model: /model".to_owned(),
    };
    let effort = match (agent.endpoint, &agent.effort.0) {
        (Some(endpoint), Some(reasoning)) => {
            format!(" · effort {}", effort_label(endpoint.spec, reasoning))
        }
        _ => String::new(),
    };
    let (state, color) = match agent.status {
        AgentStatus::Idle => ("ready".to_owned(), Color::Green),
        AgentStatus::Streaming => ("thinking… (Esc stops)".to_owned(), Color::Yellow),
        AgentStatus::RunningTools => ("running tools… (Esc stops)".to_owned(), Color::Yellow),
        AgentStatus::Failed(reason) => (
            format!("failed: {}", reason.lines().next().unwrap_or_default()),
            Color::Red,
        ),
    };
    let mut spans = vec![
        Span::styled(
            format!("{model}{effort} · "),
            Style::new().fg(Color::DarkGray),
        ),
        Span::styled(state, Style::new().fg(color)),
    ];
    if let Some(build) = build {
        let progress = match (build.total, build.current.as_str()) {
            (0, "") => " · build starting".to_owned(),
            (0, line) => format!(" · build: {line}"),
            (total, names) => format!(" · compiling {}/{total}: {names}", build.done),
        };
        spans.push(Span::styled(progress, Style::new().fg(Color::Magenta)));
    }
    Line::from(spans)
}

fn message_lines(message: &Message, lines: &mut Vec<Line<'static>>) {
    match message {
        Message::User { content } => {
            for part in content {
                match part {
                    UserContent::Text(text) => {
                        lines.push(Line::default());
                        text_lines(
                            &text.text,
                            Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                            "› ",
                            lines,
                        );
                    }
                    UserContent::ToolResult(result) => {
                        let style = if result.is_error {
                            Style::new().fg(Color::Red)
                        } else {
                            Style::new().fg(Color::DarkGray)
                        };
                        let text: Vec<String> = result
                            .content
                            .iter()
                            .map(|content| match content {
                                ToolResultContent::Text(text) => text.text.clone(),
                                ToolResultContent::Json { value } => value.to_string(),
                                ToolResultContent::Image(_) => "[image]".to_owned(),
                            })
                            .collect();
                        let text = text.join("\n");
                        let total = text.lines().count();
                        let mut shown: Vec<&str> = text.lines().take(RESULT_LINES).collect();
                        let more = format!("… {} more lines", total.saturating_sub(RESULT_LINES));
                        if total > RESULT_LINES {
                            shown.push(&more);
                        }
                        text_lines(&shown.join("\n"), style, "  ⎿ ", lines);
                    }
                    _ => lines.push(Line::raw("  [attachment]")),
                }
            }
        }
        Message::Assistant(message) => {
            lines.push(Line::default());
            for part in &message.content {
                match part {
                    AssistantContent::Text(text) => text_lines(&text.text, Style::new(), "", lines),
                    AssistantContent::Reasoning(reasoning) => {
                        reasoning_lines(&reasoning.text, lines)
                    }
                    AssistantContent::ToolCall(call) => {
                        let args: String = call
                            .function
                            .arguments_value()
                            .to_string()
                            .chars()
                            .take(200)
                            .collect();
                        lines.push(Line::from(vec![
                            Span::styled("● ", Style::new().fg(Color::Yellow)),
                            Span::styled(
                                call.function.name.as_str().to_owned(),
                                Style::new().add_modifier(Modifier::BOLD),
                            ),
                            Span::styled(format!(" {args}"), Style::new().fg(Color::DarkGray)),
                        ]));
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

fn reasoning_lines(text: &str, lines: &mut Vec<Line<'static>>) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    let shown: Vec<&str> = text.lines().rev().take(REASONING_LINES).collect();
    let shown: Vec<&str> = shown.into_iter().rev().collect();
    text_lines(
        &shown.join("\n"),
        Style::new()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::ITALIC),
        "┊ ",
        lines,
    );
}

/// One line per line of `text`, the first behind `prefix` and the rest
/// indented to match.
fn text_lines(text: &str, style: Style, prefix: &str, lines: &mut Vec<Line<'static>>) {
    let indent = " ".repeat(prefix.chars().count());
    for (number, line) in text.lines().enumerate() {
        let lead = if number == 0 {
            prefix.to_owned()
        } else {
            indent.clone()
        };
        lines.push(Line::from(vec![
            Span::styled(lead, style),
            Span::styled(line.replace('\t', "    "), style),
        ]));
    }
}

/// Splits `line` into rows at most `width` columns wide.
fn wrap(line: &Line<'static>, width: usize, rows: &mut Vec<Line<'static>>) {
    let mut row: Vec<Span<'static>> = Vec::new();
    let mut used = 0;
    for span in &line.spans {
        let mut piece = String::new();
        for c in span.content.chars() {
            let columns = c.width().unwrap_or(0);
            if used + columns > width && used > 0 {
                if !piece.is_empty() {
                    row.push(Span::styled(std::mem::take(&mut piece), span.style));
                }
                rows.push(Line::from(std::mem::take(&mut row)));
                used = 0;
            }
            piece.push(c);
            used += columns;
        }
        if !piece.is_empty() {
            row.push(Span::styled(piece, span.style));
        }
    }
    rows.push(Line::from(row));
}

/// Wraps `lines` and keeps the `height` rows that end `scroll` rows above
/// the bottom.
fn window(lines: &[Line<'static>], area: Rect, scroll: usize) -> Vec<Line<'static>> {
    let mut rows = Vec::new();
    for line in lines {
        wrap(line, usize::from(area.width).max(1), &mut rows);
    }
    let height = usize::from(area.height);
    let scroll = scroll.min(rows.len().saturating_sub(height));
    let start = rows.len().saturating_sub(height + scroll);
    rows.into_iter().skip(start).take(height).collect()
}

pub(super) fn frame(frame: &mut Frame, view: &TuiView, scene: &Scene) {
    let area = frame.area();
    let mut notices = Vec::new();
    for (level, text) in &view.notices {
        let style = match level {
            NoticeLevel::Info => Style::new().fg(Color::Blue),
            NoticeLevel::Error => Style::new().fg(Color::Red),
        };
        text_lines(text, style, "", &mut notices);
    }
    let mut notice_rows = Vec::new();
    for line in &notices {
        wrap(line, usize::from(area.width).max(1), &mut notice_rows);
    }
    let notice_height = u16::try_from(notice_rows.len())
        .unwrap_or(u16::MAX)
        .min(area.height / 2);
    let [conversation, notice_area, status, input] = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(notice_height),
        Constraint::Length(1),
        Constraint::Length(3),
    ])
    .areas(area);

    frame.render_widget(
        Paragraph::new(window(&scene.conversation, conversation, view.scroll)),
        conversation,
    );
    // From the top, so the first error of a long build failure shows.
    frame.render_widget(Paragraph::new(notice_rows), notice_area);
    frame.render_widget(Paragraph::new(scene.status.clone()), status);

    let inner_width = usize::from(input.width.saturating_sub(2)).max(1);
    let (shown, columns) = tail(&view.input, inner_width.saturating_sub(1));
    frame.render_widget(
        Paragraph::new(shown)
            .block(Block::bordered().border_style(Style::new().fg(Color::DarkGray))),
        input,
    );
    if let Some(picker) = &view.picker {
        draw_picker(frame, area, picker);
    } else {
        let x = input
            .x
            .saturating_add(1)
            .saturating_add(u16::try_from(columns).unwrap_or(0));
        frame.set_cursor_position((x, input.y.saturating_add(1)));
    }
}

/// The end of `text` that fits in `width` columns, and its width.
fn tail(text: &str, width: usize) -> (String, usize) {
    let mut columns = 0;
    let mut kept = Vec::new();
    for c in text.chars().rev() {
        let next = columns + c.width().unwrap_or(0);
        if next > width {
            break;
        }
        columns = next;
        kept.push(c);
    }
    (kept.into_iter().rev().collect(), columns)
}

fn draw_picker(frame: &mut Frame, area: Rect, picker: &Picker) {
    let visible = picker.visible();
    let width = area.width.saturating_sub(4).min(90);
    let wanted = u16::try_from(visible.len())
        .unwrap_or(u16::MAX)
        .saturating_add(3);
    let height = area.height.saturating_sub(4).min(wanted.max(4));
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(width)) / 2,
        y: area.y + (area.height.saturating_sub(height)) / 2,
        width,
        height,
    };
    let rows = usize::from(height.saturating_sub(3));
    let first = picker.selected.saturating_sub(rows.saturating_sub(1));
    let mut lines = vec![Line::from(vec![
        Span::styled("filter: ", Style::new().fg(Color::DarkGray)),
        Span::raw(picker.filter.clone()),
    ])];
    for (index, choice) in visible.iter().enumerate().skip(first).take(rows) {
        let style = if index == picker.selected {
            Style::new().add_modifier(Modifier::REVERSED)
        } else {
            Style::new()
        };
        lines.push(Line::styled(choice.label.clone(), style));
    }
    if visible.is_empty() {
        lines.push(Line::styled("no match", Style::new().fg(Color::DarkGray)));
    }
    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(lines)
            .block(Block::bordered().title(format!(" {} · ↑↓ Enter Esc ", picker.title))),
        popup,
    );
}
