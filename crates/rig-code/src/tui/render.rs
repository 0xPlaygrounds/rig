//! Draws the focused agent: transcript, status line, input line and picker.

use bevy_ecs::prelude::*;
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, List, ListState, Paragraph, Wrap};
use rig_core::completion::{AssistantContent, Message};
use rig_core::message::{ToolResult, UserContent};

use super::terminal::Tui;
use super::view::TuiView;
use crate::core::agent::{AgentStatus, CallOf, Conversation, Effort, ModelChoice, Partial};
use crate::core::models;
use crate::reload::ReloadBuild;

/// Lines of a tool result shown in the transcript.
const RESULT_LINES: usize = 4;
/// Characters of tool call arguments shown in the transcript.
const ARGUMENT_CHARS: usize = 160;
/// Most lines the input box shows.
const INPUT_LINES: usize = 8;
/// Width of the rebuild progress bar.
const GAUGE_WIDTH: u32 = 20;

/// Whether anything drawn changed since the last frame: the view state (a
/// key, a notice, a resize), an agent's drawn components, or a streaming
/// reply. The rebuild's progress is checked separately.
pub fn needs_redraw(
    view: Res<TuiView>,
    agents: Query<
        (),
        Or<(
            Changed<Conversation>,
            Changed<ModelChoice>,
            Changed<Effort>,
            Changed<AgentStatus>,
        )>,
    >,
    partials: Query<(), Changed<Partial>>,
) -> bool {
    view.is_changed() || !agents.is_empty() || !partials.is_empty()
}

/// Draws one frame.
pub fn render(
    mut tui: ResMut<Tui>,
    mut view: ResMut<TuiView>,
    agents: Query<(&Conversation, &ModelChoice, &Effort, &AgentStatus)>,
    partials: Query<(&CallOf, &Partial)>,
    build: Option<Res<ReloadBuild>>,
) -> Result {
    // Clamping the scroll is drawing's own bookkeeping, not a change to
    // redraw for.
    let view = view.bypass_change_detection();
    let shown = view.agent.and_then(|agent| agents.get(agent).ok());
    let partial = view.agent.and_then(|agent| {
        partials
            .iter()
            .find(|(call_of, _)| call_of.0 == agent)
            .map(|(_, partial)| partial)
    });
    tui.terminal.draw(|frame| {
        // The input box grows with a pasted or long input, up to a limit;
        // the rest of it stays scrolled to its end.
        let input_text = Paragraph::new(format!("> {}▏", view.input)).wrap(Wrap { trim: false });
        let input_lines = input_text.line_count(frame.area().width.saturating_sub(2));
        let input_height = input_lines.clamp(1, INPUT_LINES);
        let input_scroll =
            u16::try_from(input_lines.saturating_sub(input_height)).unwrap_or(u16::MAX);
        let [transcript, status, input] = Layout::vertical([
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(u16::try_from(input_height + 2).unwrap_or(3)),
        ])
        .areas(frame.area());
        let mut lines = Vec::new();
        if let Some((conversation, _, _, _)) = shown {
            for message in &conversation.0 {
                message_lines(message, &mut lines);
            }
        }
        if let Some(partial) = partial {
            text_lines(&partial.reasoning, Style::new().dim().italic(), &mut lines);
            text_lines(&partial.text, Style::new(), &mut lines);
        }
        for notice in &view.notices {
            text_lines(notice, Style::new().fg(Color::Magenta), &mut lines);
        }
        draw_transcript(frame, transcript, lines, &mut view.scroll);
        let mut line = status_line(shown.map(|(_, model, effort, status)| (model, effort, status)));
        if let Some(build) = &build {
            line.push_span(reload_span(build));
        }
        frame.render_widget(line, status);
        frame.render_widget(
            input_text
                .scroll((input_scroll, 0))
                .block(Block::bordered()),
            input,
        );
        if let Some(picker) = &view.picker {
            draw_picker(frame, picker);
        }
    })?;
    Ok(())
}

fn draw_transcript(frame: &mut Frame, area: Rect, lines: Vec<Line<'static>>, scroll: &mut usize) {
    let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
    let total = paragraph.line_count(area.width);
    let bottom = total.saturating_sub(usize::from(area.height));
    *scroll = (*scroll).min(bottom);
    let top = u16::try_from(bottom - *scroll).unwrap_or(u16::MAX);
    frame.render_widget(paragraph.scroll((top, 0)), area);
}

fn status_line(shown: Option<(&ModelChoice, &Effort, &AgentStatus)>) -> Line<'static> {
    let Some((model, effort, status)) = shown else {
        return Line::from("no agent").dim();
    };
    let model = model
        .0
        .clone()
        .unwrap_or_else(|| "no model: /model picks one".to_owned());
    let status = match status {
        AgentStatus::Idle => Span::from("idle").green(),
        AgentStatus::Thinking => Span::from("thinking… (Esc stops)").yellow(),
        AgentStatus::RunningTools => Span::from("running tools… (Esc stops)").yellow(),
    };
    Line::from(vec![
        Span::from(model).bold(),
        Span::from(format!("  reasoning {}  ", models::effort_label(effort.0))).dim(),
        status,
    ])
}

fn reload_span(build: &ReloadBuild) -> Span<'static> {
    if build.is_ready() {
        return Span::from("  Reloading: restarting…").cyan();
    }
    match build.progress() {
        Some((done, total)) => {
            let filled = (done.saturating_mul(GAUGE_WIDTH) / total.max(1)).min(GAUGE_WIDTH);
            let bar: String = (0..GAUGE_WIDTH)
                .map(|cell| if cell < filled { '█' } else { '░' })
                .collect();
            Span::from(format!(
                "  Reloading: Compiling {done}/{total} {bar} (Esc cancels)"
            ))
            .cyan()
        }
        None => Span::from("  Reloading: resolving… (Esc cancels)").cyan(),
    }
}

fn draw_picker(frame: &mut Frame, picker: &super::view::Picker) {
    let area = frame.area();
    let width = area.width.saturating_mul(4) / 5;
    let height = area.height.saturating_mul(3) / 4;
    let popup = Rect::new(
        area.x + (area.width - width) / 2,
        area.y + (area.height - height) / 2,
        width,
        height,
    );
    frame.render_widget(Clear, popup);
    let block = Block::bordered().title(format!(
        " {} · type to filter, Enter picks, Esc closes ",
        picker.title
    ));
    let inner = block.inner(popup);
    frame.render_widget(block, popup);
    let [filter, list] = Layout::vertical([Constraint::Length(1), Constraint::Min(1)]).areas(inner);
    frame.render_widget(Line::from(format!("filter: {}▏", picker.filter)), filter);
    let items: Vec<String> = picker
        .visible()
        .into_iter()
        .map(|(label, _)| label.clone())
        .collect();
    let mut state = ListState::default().with_selected(Some(picker.selected));
    frame.render_stateful_widget(
        List::new(items).highlight_style(Style::new().add_modifier(Modifier::REVERSED)),
        list,
        &mut state,
    );
}

fn message_lines(message: &Message, lines: &mut Vec<Line<'static>>) {
    match message {
        Message::System { .. } => {}
        Message::User { content } => {
            for item in content {
                match item {
                    UserContent::Text(text) => {
                        lines.push(Line::default());
                        text_lines(
                            &format!("› {}", text.text),
                            Style::new().cyan().bold(),
                            lines,
                        );
                    }
                    UserContent::ToolResult(result) => result_lines(result, lines),
                    _ => {}
                }
            }
        }
        Message::Assistant(assistant) => {
            for item in &assistant.content {
                match item {
                    AssistantContent::Text(text) => {
                        lines.push(Line::default());
                        text_lines(&text.text, Style::new(), lines);
                    }
                    AssistantContent::Reasoning(reasoning) => {
                        text_lines(&reasoning.text, Style::new().dim().italic(), lines);
                    }
                    AssistantContent::ToolCall(call) => {
                        let arguments =
                            serde_json::Value::Object(call.function.arguments.clone()).to_string();
                        lines.push(Line::from(vec![
                            Span::from("● ").yellow(),
                            Span::from(call.function.name.as_str().to_owned())
                                .yellow()
                                .bold(),
                            Span::from(format!(" {}", clip(&arguments, ARGUMENT_CHARS))).dim(),
                        ]));
                    }
                    _ => {}
                }
            }
        }
    }
}

fn result_lines(result: &ToolResult, lines: &mut Vec<Line<'static>>) {
    let text: String = result
        .content
        .iter()
        .filter_map(|content| content.as_text())
        .collect::<Vec<_>>()
        .join("\n");
    let style = if result.is_error {
        Style::new().red()
    } else {
        Style::new().dim()
    };
    let total = text.lines().count();
    for (index, line) in text.lines().take(RESULT_LINES).enumerate() {
        let prefix = if index == 0 { "  ⎿ " } else { "    " };
        lines.push(Line::styled(
            format!("{prefix}{}", line.replace('\t', "    ")),
            style,
        ));
    }
    if total > RESULT_LINES {
        lines.push(Line::styled(
            format!("    … {} more lines", total - RESULT_LINES),
            style,
        ));
    }
}

fn text_lines(text: &str, style: Style, lines: &mut Vec<Line<'static>>) {
    lines.extend(
        text.lines()
            .map(|line| Line::styled(line.replace('\t', "    "), style)),
    );
}

fn clip(text: &str, limit: usize) -> String {
    match text.char_indices().nth(limit) {
        Some((end, _)) => format!("{}…", text.get(..end).unwrap_or(text)),
        None => text.to_owned(),
    }
}
