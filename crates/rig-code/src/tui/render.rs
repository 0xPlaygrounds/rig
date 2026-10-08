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
use super::view::{ShownNotice, TuiView};
use crate::core::agent::{
    ActiveTurn, Calls, Conversation, Effort, ModelChoice, NoticeLevel, Partial, ToolCallRun,
};
use crate::core::commands::SlashCommand;
use crate::core::models;
use crate::host::reload::ReloadBuild;

/// Lines of a tool result shown in the transcript.
const RESULT_LINES: usize = 4;
/// Characters of tool call arguments shown in the transcript.
const ARGUMENT_CHARS: usize = 160;
/// Most lines the input box shows.
const INPUT_LINES: usize = 8;
/// Width of the rebuild progress bar.
const GAUGE_WIDTH: u32 = 20;

/// What the shown agent is doing.
#[derive(Clone, Copy)]
enum Activity {
    Idle,
    Thinking,
    RunningTools,
}

/// Whether anything drawn changed since the last frame: the view state (a
/// key, a notice, a resize), an agent's drawn components, a turn's calls,
/// or a streaming reply. A turn's end changes its conversation or comes
/// with a notice. The rebuild's progress is checked separately.
pub fn needs_redraw(
    view: Res<TuiView>,
    agents: Query<
        (),
        Or<(
            Changed<Conversation>,
            Changed<ModelChoice>,
            Changed<Effort>,
            Changed<ActiveTurn>,
        )>,
    >,
    turns: Query<(), Changed<Calls>>,
    partials: Query<(), Changed<Partial>>,
) -> bool {
    view.is_changed() || !agents.is_empty() || !turns.is_empty() || !partials.is_empty()
}

/// Draws one frame.
pub fn render(
    mut tui: ResMut<Tui>,
    mut view: ResMut<TuiView>,
    agents: Query<(
        &Conversation,
        Option<&ModelChoice>,
        &Effort,
        Option<&ActiveTurn>,
    )>,
    turns: Query<&Calls>,
    partials: Query<&Partial>,
    tool_calls: Query<(), With<ToolCallRun>>,
    slash: Query<&SlashCommand>,
    build: Option<Res<ReloadBuild>>,
) -> Result {
    // Clamping the scroll is drawing's own bookkeeping, not a change to
    // redraw for.
    let view = view.bypass_change_detection();
    let shown = view.agent.and_then(|agent| agents.get(agent).ok());
    let calls = shown
        .and_then(|(.., turn)| turn)
        .and_then(|turn| turns.get(turn.turn()).ok());
    let partial = calls.and_then(|calls| calls.iter().find_map(|call| partials.get(call).ok()));
    let activity = if shown.and_then(|(.., turn)| turn).is_none() {
        Activity::Idle
    } else if calls.is_some_and(|calls| calls.iter().any(|call| tool_calls.contains(call))) {
        Activity::RunningTools
    } else {
        Activity::Thinking
    };
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
        // Notices go between the messages, where they arrived; the ones
        // after the last message go before the reply streaming in.
        let messages = shown
            .map(|(conversation, ..)| conversation.0.as_slice())
            .unwrap_or_default();
        let mut notices = view
            .notices
            .iter()
            .filter(|notice| notice.is_for(view.agent))
            .peekable();
        let mut lines = Vec::new();
        for (index, message) in messages.iter().enumerate() {
            while let Some(notice) = notices.next_if(|notice| notice.after <= index) {
                notice_lines(notice, &mut lines);
            }
            let previous = index.checked_sub(1).and_then(|before| messages.get(before));
            message_lines(message, previous, messages.get(index + 1), &mut lines);
        }
        for notice in notices {
            notice_lines(notice, &mut lines);
        }
        if let Some(partial) = partial {
            text_lines(&partial.reasoning, Style::new().dim().italic(), &mut lines);
            text_lines(&partial.text, Style::new(), &mut lines);
        }
        draw_transcript(frame, transcript, lines, &mut view.scroll);
        // /model comes from a plugin, so point at it only when loaded.
        let model_hint = slash.iter().any(|command| command.name == "model");
        let mut line = status_line(
            shown.map(|(_, model, effort, _)| (model, effort, activity)),
            model_hint,
        );
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
        if let Some(output) = &view.reload_failure {
            draw_reload_failure(frame, output);
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

fn status_line(
    shown: Option<(Option<&ModelChoice>, &Effort, Activity)>,
    model_hint: bool,
) -> Line<'static> {
    let Some((model, effort, status)) = shown else {
        return Line::from("no agent").dim();
    };
    let model = match model {
        Some(model) => model.0.clone(),
        None if model_hint => "no model: /model picks one".to_owned(),
        None => "no model".to_owned(),
    };
    let status = match status {
        Activity::Idle => Span::from("idle").green(),
        Activity::Thinking => Span::from("thinking… (Esc stops)").yellow(),
        Activity::RunningTools => Span::from("running tools… (Esc stops)").yellow(),
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
        // The launcher's phase, then cargo's own lines while it resolves
        // and downloads dependencies.
        None => Span::from(format!(
            "  Reloading: {} (Esc cancels)",
            build.latest().unwrap_or("Resolving dependencies…")
        ))
        .cyan(),
    }
}

/// A centred box over the transcript, `width` and `height` in fifths and
/// quarters of the screen, cleared for drawing on.
fn popup(frame: &mut Frame, fifths: u16, quarters: u16) -> Rect {
    let area = frame.area();
    let width = area.width.saturating_mul(fifths) / 5;
    let height = area.height.saturating_mul(quarters) / 4;
    let popup = Rect::new(
        area.x + (area.width - width) / 2,
        area.y + (area.height - height) / 2,
        width,
        height,
    );
    frame.render_widget(Clear, popup);
    popup
}

/// A failed rebuild's output, from its first error on, over the transcript.
fn draw_reload_failure(frame: &mut Frame, output: &str) {
    let popup = popup(frame, 5, 4);
    let block = Block::bordered()
        .border_style(Style::new().red())
        .title(" The rebuild failed; this build keeps running · Esc closes ");
    let mut lines = Vec::new();
    text_lines(output, Style::new(), &mut lines);
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(block),
        popup,
    );
}

fn draw_picker(frame: &mut Frame, picker: &super::view::Picker) {
    let popup = popup(frame, 4, 3);
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

/// Draws `message`. A tool call's result, which comes in the `next`
/// message, is drawn right under the call, so the results of a reply's
/// calls are not drawn after all of its calls.
fn message_lines(
    message: &Message,
    previous: Option<&Message>,
    next: Option<&Message>,
    lines: &mut Vec<Line<'static>>,
) {
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
                    // Drawn under its call already.
                    UserContent::ToolResult(result) if answers(previous, result) => {}
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
                        if let Some(result) =
                            tool_results(next).find(|result| result.call == call.id)
                        {
                            result_lines(result, lines);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
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

fn notice_lines(notice: &ShownNotice, lines: &mut Vec<Line<'static>>) {
    let style = match notice.level {
        NoticeLevel::Info => Style::new().fg(Color::Magenta),
        NoticeLevel::Error => Style::new().fg(Color::Red),
    };
    text_lines(&notice.text, style, lines);
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
