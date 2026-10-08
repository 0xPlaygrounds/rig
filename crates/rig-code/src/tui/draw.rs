//! Rendering with ratatui: the focused agent's conversation and notices, a
//! status line, the composer, and an open picker on top.

use bevy::prelude::*;
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Position, Rect},
    style::{Color, Modifier, Style},
    text::Line,
    widgets::{Block, Clear, Paragraph},
};
use rig_core::message::{AssistantContent, Message, ToolResultContent, UserContent};

use super::{
    Tui,
    view::{Composer, Focus, NoticeEntry, NoticeLog, Picker, PickerState, Scroll},
};
use crate::ecs::{
    NoticeLevel,
    agent::{AgentStatus, Conversation, Draft, EffortChoice, ModelChoice},
};

/// Most lines of reasoning and tool output shown per block.
const PREVIEW_LINES: usize = 4;
/// Most characters of tool call arguments shown.
const ARGUMENT_CHARS: usize = 200;
/// Most composer lines shown.
const COMPOSER_LINES: usize = 5;

/// One logical transcript line and its style, wrapped when drawn.
type Styled = (String, Style);

fn user_style() -> Style {
    Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD)
}

fn dim() -> Style {
    Style::new().fg(Color::DarkGray)
}

fn tool_style() -> Style {
    Style::new().fg(Color::Yellow)
}

fn error_style() -> Style {
    Style::new().fg(Color::Red)
}

/// Draw the focused agent.
pub(super) fn draw(
    mut tui: ResMut<Tui>,
    focus: Res<Focus>,
    agents: Query<(
        &Conversation,
        &ModelChoice,
        &EffortChoice,
        &AgentStatus,
        Option<&Draft>,
    )>,
    composer: Res<Composer>,
    mut scroll: ResMut<Scroll>,
    picker: Res<Picker>,
    notices: Res<NoticeLog>,
) {
    let Some(agent) = focus.0.and_then(|agent| agents.get(agent).ok()) else {
        return;
    };
    let result = tui.terminal.draw(|frame| {
        let (conversation, model, effort, status, draft) = agent;
        let width = frame.area().width.saturating_sub(2).max(1) as usize;
        let composer_lines = wrap(&composer_text(&composer), width)
            .len()
            .clamp(1, COMPOSER_LINES);
        let [transcript_area, status_area, composer_area] = Layout::vertical([
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(composer_lines as u16 + 2),
        ])
        .areas(frame.area());
        let lines = transcript(conversation, draft, &notices.0);
        draw_transcript(frame, transcript_area, &lines, &mut scroll.0);
        draw_status(frame, status_area, model, *effort, *status);
        draw_composer(frame, composer_area, &composer, picker.0.is_none());
        if let Some(state) = &picker.0 {
            draw_picker(frame, state);
        }
    });
    if let Err(error) = result {
        error!("cannot draw the terminal: {error}");
    }
}

/// The conversation as styled lines, notices in place, then the reply
/// streaming in.
fn transcript(
    conversation: &Conversation,
    draft: Option<&Draft>,
    notices: &[NoticeEntry],
) -> Vec<Styled> {
    let mut lines = Vec::new();
    let notices_after = |lines: &mut Vec<Styled>, index: usize| {
        for notice in notices.iter().filter(|notice| notice.after == index) {
            let style = match notice.level {
                NoticeLevel::Info => dim(),
                NoticeLevel::Error => error_style(),
            };
            push_text(lines, &notice.text, style, usize::MAX);
            lines.push((String::new(), Style::new()));
        }
    };
    for (index, message) in conversation.0.iter().enumerate() {
        notices_after(&mut lines, index);
        match message {
            Message::System { .. } => continue,
            Message::User { content } => {
                for content in content {
                    match content {
                        UserContent::Text(text) => {
                            push_text(
                                &mut lines,
                                &format!("> {}", text.text),
                                user_style(),
                                usize::MAX,
                            );
                        }
                        UserContent::ToolResult(result) => {
                            let style = if result.is_error {
                                error_style()
                            } else {
                                dim()
                            };
                            let output = result
                                .content
                                .iter()
                                .map(result_text)
                                .collect::<Vec<_>>()
                                .join("\n");
                            push_text(
                                &mut lines,
                                &format!("  {} -> {output}", result.name.as_str()),
                                style,
                                PREVIEW_LINES,
                            );
                        }
                        _ => lines.push(("> [attachment]".to_owned(), user_style())),
                    }
                }
            }
            Message::Assistant(message) => {
                for content in &message.content {
                    match content {
                        AssistantContent::Text(text) => {
                            push_text(&mut lines, &text.text, Style::new(), usize::MAX);
                        }
                        AssistantContent::Reasoning(reasoning) => {
                            push_text(
                                &mut lines,
                                &reasoning.text,
                                dim().add_modifier(Modifier::ITALIC),
                                PREVIEW_LINES,
                            );
                        }
                        AssistantContent::ToolCall(call) => {
                            let arguments =
                                serde_json::Value::Object(call.function.arguments.clone())
                                    .to_string()
                                    .chars()
                                    .take(ARGUMENT_CHARS)
                                    .collect::<String>();
                            lines.push((
                                format!("* {} {arguments}", call.function.name.as_str()),
                                tool_style(),
                            ));
                        }
                        _ => {}
                    }
                }
            }
        }
        lines.push((String::new(), Style::new()));
    }
    notices_after(&mut lines, conversation.0.len());
    if let Some(draft) = draft {
        push_text(
            &mut lines,
            &draft.reasoning,
            dim().add_modifier(Modifier::ITALIC),
            usize::MAX,
        );
        push_text(&mut lines, &draft.text, Style::new(), usize::MAX);
    }
    lines
}

/// Push the lines of `text` in `style`, at most `limit` of them. Tabs
/// become spaces, since the wrapping counts one cell per character.
fn push_text(lines: &mut Vec<Styled>, text: &str, style: Style, limit: usize) {
    let total = text.lines().count();
    lines.extend(
        text.lines()
            .take(limit)
            .map(|line| (line.replace('\t', "    "), style)),
    );
    if total > limit {
        lines.push((format!("  ... {} more lines", total - limit), dim()));
    }
}

fn result_text(content: &ToolResultContent) -> String {
    match content {
        ToolResultContent::Text(text) => text.text.clone(),
        ToolResultContent::Json { value } => value.to_string(),
        ToolResultContent::Image(_) => "[image]".to_owned(),
    }
}

/// Hard-wrap `text` at `width` characters.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let chars = text.chars().collect::<Vec<_>>();
    if chars.is_empty() {
        return vec![String::new()];
    }
    chars
        .chunks(width.max(1))
        .map(|chunk| chunk.iter().collect())
        .collect()
}

/// Draw the bottom of the transcript, `scroll` lines up, clamped to the
/// transcript's length.
fn draw_transcript(frame: &mut Frame, area: Rect, lines: &[Styled], scroll: &mut usize) {
    let width = area.width.max(1) as usize;
    let wrapped = lines
        .iter()
        .flat_map(|(text, style)| {
            wrap(text, width)
                .into_iter()
                .map(|line| Line::styled(line, *style))
        })
        .collect::<Vec<_>>();
    let height = area.height as usize;
    *scroll = (*scroll).min(wrapped.len().saturating_sub(height));
    let end = wrapped.len() - *scroll;
    let start = end.saturating_sub(height);
    let visible = wrapped.get(start..end).unwrap_or_default().to_vec();
    frame.render_widget(Paragraph::new(visible), area);
}

fn draw_status(
    frame: &mut Frame,
    area: Rect,
    model: &ModelChoice,
    effort: EffortChoice,
    status: AgentStatus,
) {
    let model = model.0.as_deref().unwrap_or("no model: /model");
    let status = match status {
        AgentStatus::Idle => "idle",
        AgentStatus::Thinking => "thinking (Esc stops)",
        AgentStatus::Tools => "running tools (Esc stops)",
    };
    let line = format!(
        " {model} | effort {} | {status} | /help, Ctrl-C quits",
        effort.name()
    );
    frame.render_widget(
        Paragraph::new(line).style(Style::new().fg(Color::Black).bg(Color::Gray)),
        area,
    );
}

/// The composer text as shown: line breaks as a visible mark, so each
/// character takes one cell and the cursor maps directly.
fn composer_text(composer: &Composer) -> String {
    composer.text.replace('\n', "\u{21b5}")
}

fn draw_composer(frame: &mut Frame, area: Rect, composer: &Composer, show_cursor: bool) {
    let block = Block::bordered().title(" message ");
    let inner = block.inner(area);
    let width = inner.width.max(1) as usize;
    let lines = wrap(&composer_text(composer), width);
    let row = composer.cursor / width;
    let first = (row + 1).saturating_sub(inner.height as usize);
    let visible = lines
        .into_iter()
        .skip(first)
        .map(Line::from)
        .collect::<Vec<_>>();
    frame.render_widget(Paragraph::new(visible).block(block), area);
    if show_cursor {
        let x = inner.x + (composer.cursor % width) as u16;
        let y = inner.y + (row - first) as u16;
        frame.set_cursor_position(Position { x, y });
    }
}

fn draw_picker(frame: &mut Frame, state: &PickerState) {
    let area = frame.area();
    let width = (area.width * 4 / 5).max(20).min(area.width);
    let height = area.height.saturating_sub(4).clamp(3, 24);
    let rect = Rect {
        x: area.x + (area.width - width) / 2,
        y: area.y + (area.height.saturating_sub(height)) / 2,
        width,
        height: height.min(area.height),
    };
    let block = Block::bordered().title(format!(
        " {}: type to filter, Enter picks, Esc closes ",
        state.title
    ));
    let inner = block.inner(rect);
    let filtered = state.filtered();
    let rows = (inner.height as usize).saturating_sub(1).max(1);
    let first = (state.selected + 1).saturating_sub(rows);
    let mut lines = vec![Line::styled(format!("> {}", state.filter), user_style())];
    if filtered.is_empty() {
        lines.push(Line::styled("no match", dim()));
    }
    for (index, (label, _)) in filtered.iter().enumerate().skip(first).take(rows) {
        let style = if index == state.selected {
            Style::new().add_modifier(Modifier::REVERSED)
        } else {
            Style::new()
        };
        lines.push(Line::styled(label.clone(), style));
    }
    frame.render_widget(Clear, rect);
    frame.render_widget(Paragraph::new(lines).block(block), rect);
}
