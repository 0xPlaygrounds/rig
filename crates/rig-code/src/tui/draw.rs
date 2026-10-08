use bevy_ecs::prelude::*;
use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
};
use rig_core::{
    completion::AssistantContent,
    message::{Message, StopReason, ToolResultContent, UserContent},
};

use super::{Screen, view::TuiView};
use crate::{
    agent::{
        Agent, AgentCalls, AgentStatus, Conversation, Effort, ModelCall, ModelChoice, ToolCallSlot,
    },
    model,
};

/// Lines of a tool result shown in the transcript.
const RESULT_PREVIEW: usize = 3;
/// Rows of the picker list.
const PICKER_ROWS: usize = 16;

/// One transcript row before wrapping.
struct Row {
    text: String,
    style: Style,
}

type Shown<'a> = (
    &'a Conversation,
    &'a ModelChoice,
    &'a Effort,
    &'a AgentStatus,
    Option<&'a AgentCalls>,
);

/// Draw the shown agent when anything it shows changed.
pub(super) fn draw(
    mut screen: ResMut<Screen>,
    mut view: ResMut<TuiView>,
    agents: Query<Shown, With<Agent>>,
    changed: Query<
        (),
        Or<(
            Changed<Conversation>,
            Changed<AgentStatus>,
            Changed<ModelChoice>,
            Changed<Effort>,
        )>,
    >,
    model_calls: Query<Ref<ModelCall>>,
    slots: Query<Ref<ToolCallSlot>>,
) {
    let calls_changed = model_calls.iter().any(|call| call.is_changed())
        || slots.iter().any(|slot| slot.is_changed());
    if !view.dirty && changed.is_empty() && !calls_changed {
        return;
    }
    view.dirty = false;
    let Some(shown) = view.agent.and_then(|agent| agents.get(agent).ok()) else {
        return;
    };
    let view = &mut *view;
    let drawn = screen.0.draw(|frame| {
        render(frame, view, shown, &model_calls, &slots);
    });
    if let Err(error) = drawn {
        bevy_log::warn!("cannot draw: {error}");
    }
}

fn render(
    frame: &mut Frame,
    view: &mut TuiView,
    (conversation, choice, effort, status, calls): Shown,
    model_calls: &Query<Ref<ModelCall>>,
    slots: &Query<Ref<ToolCallSlot>>,
) {
    let area = frame.area();
    if area.height < 6 || area.width < 12 {
        return;
    }
    let width = usize::from(area.width);
    let notice_rows: Vec<Row> = view
        .notices
        .iter()
        .flat_map(|notice| notice.lines())
        .map(|line| row(line, Style::default().fg(Color::Yellow)))
        .collect();
    let notice_rows = wrap(notice_rows, width);
    let notice_height = u16::try_from(notice_rows.len())
        .unwrap_or(u16::MAX)
        .min(area.height.saturating_sub(5) / 2);
    let input_height = 3.min(area.height);
    let transcript_height = area.height.saturating_sub(1 + notice_height + input_height);

    let header = Rect::new(area.x, area.y, area.width, 1.min(area.height));
    let transcript = Rect::new(area.x, area.y + 1, area.width, transcript_height);
    let notices = Rect::new(area.x, transcript.bottom(), area.width, notice_height);
    let input = Rect::new(area.x, notices.bottom(), area.width, input_height);

    let model = choice.0.as_deref().unwrap_or("no model (/model)");
    let state = match status {
        AgentStatus::Idle => "idle".to_owned(),
        AgentStatus::Thinking => "thinking (Esc stops)".to_owned(),
        AgentStatus::Tools(count) => format!("running {count} tool(s) (Esc stops)"),
    };
    let title = format!(
        " rig-code · {model} · effort {} · {state}",
        model::describe(effort.0)
    );
    frame.render_widget(
        Paragraph::new(title).style(Style::default().add_modifier(Modifier::REVERSED)),
        header,
    );

    let mut rows = transcript_rows(conversation);
    for call in calls.into_iter().flat_map(|calls| calls.iter()) {
        if let Ok(call) = model_calls.get(call) {
            rows.extend(call.reasoning.lines().map(|line| row(line, dim())));
            rows.extend(call.text.lines().map(|line| row(line, Style::default())));
        }
        if let Ok(slot) = slots.get(call)
            && slot.result.is_none()
        {
            let name = slot.call.function.name.as_str();
            rows.push(row(&format!("… running {name}"), dim()));
        }
    }
    let rows = wrap(rows, width);
    let height = usize::from(transcript.height);
    view.scroll = view.scroll.min(rows.len().saturating_sub(height));
    let start = rows.len().saturating_sub(height + view.scroll);
    let visible: Vec<Line> = rows
        .into_iter()
        .skip(start)
        .take(height)
        .map(|row| Line::styled(row.text, row.style))
        .collect();
    frame.render_widget(Paragraph::new(visible), transcript);

    let notice_lines: Vec<Line> = notice_rows
        .into_iter()
        .rev()
        .take(usize::from(notice_height))
        .rev()
        .map(|row| Line::styled(row.text, row.style))
        .collect();
    frame.render_widget(Paragraph::new(notice_lines), notices);

    let inner = usize::from(input.width.saturating_sub(4)).max(1);
    let skip = view.cursor.saturating_sub(inner - 1);
    let shown: String = view.input.chars().skip(skip).take(inner).collect();
    frame.render_widget(
        Paragraph::new(shown).block(Block::default().borders(Borders::ALL).title(" > ")),
        input,
    );
    let cursor = u16::try_from(view.cursor - skip).unwrap_or(0);
    frame.set_cursor_position((input.x + 1 + cursor, input.y + 1));

    if let Some(picker) = &view.picker {
        let options = picker.filtered();
        let rows = PICKER_ROWS.min(options.len().max(1));
        let height = (u16::try_from(rows).unwrap_or(1) + 2).min(area.height);
        let width = (area.width * 4 / 5).max(10).min(area.width);
        let popup = Rect::new(
            area.x + (area.width - width) / 2,
            area.y + (area.height - height) / 2,
            width,
            height,
        );
        let first = picker.selected.saturating_sub(rows - 1);
        let lines: Vec<Line> = options
            .iter()
            .enumerate()
            .skip(first)
            .take(rows)
            .map(|(index, option)| {
                let style = if index == picker.selected {
                    Style::default().add_modifier(Modifier::REVERSED)
                } else {
                    Style::default()
                };
                Line::from(Span::styled(option.label.clone(), style))
            })
            .collect();
        let title = format!(
            " {} · filter: {}_ · Enter picks, Esc closes ",
            picker.title, picker.filter
        );
        frame.render_widget(Clear, popup);
        frame.render_widget(
            Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(title)),
            popup,
        );
    }
}

/// The conversation as display rows.
fn transcript_rows(conversation: &Conversation) -> Vec<Row> {
    let mut rows = Vec::new();
    for message in &conversation.0 {
        match message {
            Message::System { .. } => {}
            Message::User { content } => {
                for item in content {
                    match item {
                        UserContent::Text(text) => {
                            rows.push(row("", Style::default()));
                            for line in text.text.lines() {
                                rows.push(row(&format!("> {line}"), user()));
                            }
                        }
                        UserContent::ToolResult(result) => {
                            let style = if result.is_error {
                                Style::default().fg(Color::Red)
                            } else {
                                dim()
                            };
                            let text: Vec<&str> = result
                                .content
                                .iter()
                                .filter_map(ToolResultContent::as_text)
                                .flat_map(str::lines)
                                .collect();
                            for line in text.iter().take(RESULT_PREVIEW) {
                                rows.push(row(&format!("  │ {line}"), style));
                            }
                            if text.len() > RESULT_PREVIEW {
                                let more = text.len() - RESULT_PREVIEW;
                                rows.push(row(&format!("  │ … {more} more lines"), style));
                            }
                        }
                        _ => {}
                    }
                }
            }
            Message::Assistant(turn) => {
                for block in &turn.content {
                    match block {
                        AssistantContent::Text(text) => {
                            rows.extend(text.text.lines().map(|line| row(line, Style::default())));
                        }
                        AssistantContent::Reasoning(reasoning) => {
                            rows.extend(reasoning.text.lines().map(|line| row(line, dim())));
                        }
                        AssistantContent::ToolCall(call) => {
                            let line = format!(
                                "→ {} {}",
                                call.function.name.as_str(),
                                call.function.arguments_value()
                            );
                            rows.push(row(&line, Style::default().fg(Color::Cyan)));
                        }
                        _ => {}
                    }
                }
                if let Some(StopReason::Aborted(reason)) = &turn.stop {
                    rows.push(row(&format!("[{reason}]"), dim()));
                }
            }
        }
    }
    rows
}

fn row(text: &str, style: Style) -> Row {
    Row {
        text: text.replace('\t', "    "),
        style,
    }
}

fn dim() -> Style {
    Style::default().fg(Color::DarkGray)
}

fn user() -> Style {
    Style::default()
        .fg(Color::Green)
        .add_modifier(Modifier::BOLD)
}

/// Split rows longer than `width` characters into several.
fn wrap(rows: Vec<Row>, width: usize) -> Vec<Row> {
    let mut wrapped = Vec::with_capacity(rows.len());
    for row in rows {
        let chars: Vec<char> = row.text.chars().collect();
        if chars.len() <= width {
            wrapped.push(row);
            continue;
        }
        for chunk in chars.chunks(width) {
            wrapped.push(Row {
                text: chunk.iter().collect(),
                style: row.style,
            });
        }
    }
    wrapped
}
