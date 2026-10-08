//! Drawing the view: the transcript, a status line, the composer and an
//! open picker. Nothing is drawn in a frame where nothing it shows changed.

use bevy_ecs::prelude::*;
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};
use rig_core::message::{AssistantContent, Message, StopReason, ToolResultContent, UserContent};

use super::Tui;
use super::view::{Picker, View};
use crate::core::agent::{Calls, Conversation, Effort, Model, Status};
use crate::core::models::effort_label;
use crate::core::registry::NoticeLevel;
use crate::core::turn::{ModelCall, ToolCall, ToolState};
use crate::host::reload::BuildJob;

/// How many lines of a tool result the transcript shows.
const RESULT_LINES: usize = 4;

/// What the transcript shows of one agent.
#[derive(bevy_ecs::query::QueryData)]
pub(crate) struct Shown {
    conversation: &'static Conversation,
    status: &'static Status,
    model: Option<&'static Model>,
    effort: &'static Effort,
    calls: Option<&'static Calls>,
}

/// Draws the view when it, an agent or a call changed.
pub(crate) fn render(
    mut tui: ResMut<Tui>,
    view: Res<View>,
    agents: Query<Shown>,
    model_calls: Query<&ModelCall>,
    tool_calls: Query<&ToolCall>,
    changed_agents: Query<
        (),
        Or<(
            Changed<Conversation>,
            Changed<Status>,
            Changed<Effort>,
            Changed<Model>,
        )>,
    >,
    changed_calls: Query<(), Or<(Changed<ModelCall>, Changed<ToolCall>)>>,
    builds: Query<Ref<BuildJob>>,
    mut removed_builds: RemovedComponents<BuildJob>,
) -> Result {
    let changed = tui.is_added()
        || view.is_changed()
        || !changed_agents.is_empty()
        || !changed_calls.is_empty()
        || builds.iter().any(|build| build.is_changed())
        || removed_builds.read().next().is_some();
    if !changed {
        return Ok(());
    }
    let shown = view.agent.and_then(|agent| agents.get(agent).ok());
    let build = builds.iter().next();
    tui.terminal.draw(|frame| {
        draw(
            frame,
            &view,
            shown.as_ref(),
            build.as_deref(),
            &model_calls,
            &tool_calls,
        );
    })?;
    Ok(())
}

fn draw(
    frame: &mut Frame,
    view: &View,
    shown: Option<&ShownItem>,
    build: Option<&BuildJob>,
    model_calls: &Query<&ModelCall>,
    tool_calls: &Query<&ToolCall>,
) {
    let composer_lines = view.composer.lines().count().clamp(1, 8);
    let [transcript, status, composer] = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(1),
        Constraint::Length(u16::try_from(composer_lines).unwrap_or(8) + 1),
    ])
    .areas(frame.area());
    if let Some(shown) = shown {
        let lines = transcript_lines(view, shown, model_calls, tool_calls);
        let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
        let total = paragraph.line_count(transcript.width);
        let bottom = total.saturating_sub(usize::from(transcript.height));
        let top = bottom.saturating_sub(view.scroll);
        frame.render_widget(
            paragraph.scroll((u16::try_from(top).unwrap_or(u16::MAX), 0)),
            transcript,
        );
        frame.render_widget(status_line(shown, build, view.scroll > 0), status);
    }
    let input = Paragraph::new(format!("> {}", view.composer))
        .wrap(Wrap { trim: false })
        .block(Block::default().borders(Borders::TOP).dim());
    frame.render_widget(input, composer);
    if let Some(picker) = &view.picker {
        draw_picker(frame, picker);
    } else {
        let last = view.composer.lines().last().unwrap_or_default();
        let column = u16::try_from(last.chars().count() + 2).unwrap_or(u16::MAX);
        let row = u16::try_from(view.composer.lines().count().max(1)).unwrap_or(1);
        frame.set_cursor_position((
            composer
                .x
                .saturating_add(column)
                .min(composer.right().saturating_sub(1)),
            composer
                .y
                .saturating_add(row)
                .min(composer.bottom().saturating_sub(1)),
        ));
    }
}

/// The agent's conversation, the reply streaming in, the tool calls of the
/// turn, and the notices in between.
fn transcript_lines(
    view: &View,
    shown: &ShownItem,
    model_calls: &Query<&ModelCall>,
    tool_calls: &Query<&ToolCall>,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let messages = &shown.conversation.0;
    for (index, message) in messages.iter().enumerate() {
        notices(view, index, &mut lines);
        match message {
            Message::System { .. } => continue,
            Message::User { content } => user_lines(content, &mut lines),
            Message::Assistant(turn) => {
                assistant_lines(&turn.content, &mut lines);
                match &turn.stop {
                    Some(StopReason::Aborted(reason)) => {
                        lines.push(Line::from(format!("(stopped: {reason})")).red());
                    }
                    Some(StopReason::Error(reason)) => {
                        lines.push(Line::from(format!("(failed: {reason})")).red());
                    }
                    _ => {}
                }
            }
        }
        lines.push(Line::default());
    }
    let mut tools = Vec::new();
    for entity in shown.calls.into_iter().flat_map(|calls| calls.iter()) {
        if let Ok(call) = model_calls.get(entity) {
            assistant_lines(&call.delivered(), &mut lines);
            lines.push(Line::from("…").dim());
        }
        if let Ok(tool) = tool_calls.get(entity) {
            tools.push(tool);
        }
    }
    tools.sort_by_key(|tool| tool.index);
    for tool in tools {
        let state = match tool.state {
            ToolState::Queued => "queued",
            ToolState::Running(_) => "running",
            ToolState::Done(_) => "done",
        };
        lines.push(Line::from(format!("  {} {state}", tool.call.function.name)).dim());
    }
    notices(view, messages.len(), &mut lines);
    lines
}

/// The notices placed after the first `index` messages; the last call
/// passes the conversation's length and gets every later notice too.
fn notices(view: &View, index: usize, lines: &mut Vec<Line<'static>>) {
    let agent = view.agent;
    for notice in view
        .notices
        .iter()
        .filter(|notice| Some(notice.agent) == agent && notice.after == index)
    {
        let style = match notice.level {
            NoticeLevel::Info => Style::new().fg(Color::Blue),
            NoticeLevel::Error => Style::new().fg(Color::Red),
        };
        for text in notice.text.lines() {
            lines.push(Line::styled(text.to_owned(), style));
        }
        lines.push(Line::default());
    }
}

fn user_lines(content: &[UserContent], lines: &mut Vec<Line<'static>>) {
    for part in content {
        match part {
            UserContent::Text(text) => {
                for (index, line) in text.text.lines().enumerate() {
                    let prefix = if index == 0 { "› " } else { "  " };
                    lines.push(Line::from(format!("{prefix}{line}")).cyan().bold());
                }
            }
            UserContent::ToolResult(result) => {
                let style = if result.is_error {
                    Style::new().fg(Color::Red)
                } else {
                    Style::new().add_modifier(Modifier::DIM)
                };
                let text: String = result
                    .content
                    .iter()
                    .map(|content| match content {
                        ToolResultContent::Text(text) => text.text.clone(),
                        ToolResultContent::Json { value } => value.to_string(),
                        ToolResultContent::Image(_) => "[image]".to_owned(),
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let total = text.lines().count();
                for (index, line) in text.lines().take(RESULT_LINES).enumerate() {
                    let prefix = if index == 0 { "  ⎿ " } else { "    " };
                    lines.push(Line::styled(format!("{prefix}{line}"), style));
                }
                if total > RESULT_LINES {
                    lines.push(Line::styled(
                        format!("    … {} more lines", total - RESULT_LINES),
                        style,
                    ));
                }
            }
            _ => lines.push(Line::from("› [attachment]").cyan()),
        }
    }
}

fn assistant_lines(content: &[AssistantContent], lines: &mut Vec<Line<'static>>) {
    for part in content {
        match part {
            AssistantContent::Text(text) => {
                lines.extend(text.text.lines().map(|line| Line::from(line.to_owned())));
            }
            AssistantContent::Reasoning(reasoning) => {
                lines.extend(
                    reasoning
                        .text
                        .lines()
                        .map(|line| Line::from(line.to_owned()).dim().italic()),
                );
            }
            AssistantContent::ToolCall(call) => {
                let mut arguments = call.function.arguments_value().to_string();
                if arguments.chars().count() > 200 {
                    arguments = arguments.chars().take(200).collect::<String>() + "…";
                }
                lines.push(Line::from(vec![
                    Span::from("● ").yellow(),
                    Span::from(call.function.name.to_string()).yellow().bold(),
                    Span::from(format!(" {arguments}")).dim(),
                ]));
            }
            _ => {}
        }
    }
}

fn status_line(shown: &ShownItem, build: Option<&BuildJob>, scrolled: bool) -> Line<'static> {
    let model = shown
        .model
        .map_or_else(|| "no model (/model)".to_owned(), |model| model.0.clone());
    let status = match shown.status {
        Status::Idle => "ready",
        Status::Queued => "waiting",
        Status::Streaming => "streaming (Esc stops)",
        Status::Tools => "running tools (Esc stops)",
    };
    let mut spans = vec![
        Span::from(format!(" {model}")).bold(),
        Span::from(format!("  effort {}", effort_label(shown.effort.0))),
        Span::from(format!("  {status}")).green(),
    ];
    if let Some(build) = build {
        let text = match &build.progress {
            _ if build.built => "  built; reloading when idle".to_owned(),
            Some(progress) => format!(
                "  Compiling {}/{} crates ({})",
                progress.done, progress.total, progress.building
            ),
            None => "  building…".to_owned(),
        };
        spans.push(Span::from(text).yellow());
    }
    if scrolled {
        spans.push(Span::from("  scrolled (Down to return)").dim());
    }
    Line::from(spans)
}

fn draw_picker(frame: &mut Frame, picker: &Picker) {
    let area = centered(frame.area(), 80, 70);
    let matches = picker.matches();
    let items: Vec<ListItem> = matches
        .iter()
        .map(|option| {
            ListItem::new(Line::from(vec![
                Span::from(option.label.clone()),
                Span::from(format!("  {}", option.detail)).dim(),
            ]))
        })
        .collect();
    let block = Block::bordered()
        .title(format!(" {} ", picker.title))
        .title_bottom(format!(
            " filter: {}_  ({} of {})  Enter picks, Esc closes ",
            picker.filter,
            matches.len(),
            picker.options.len()
        ));
    let list = List::new(items)
        .block(block)
        .highlight_style(Style::new().reversed());
    frame.render_widget(Clear, area);
    frame.render_stateful_widget(
        list,
        area,
        &mut ListState::default().with_selected(Some(picker.selected)),
    );
}

/// The rectangle of `width` and `height` percent in the middle of `area`.
fn centered(area: Rect, width: u16, height: u16) -> Rect {
    let [row] = Layout::vertical([Constraint::Percentage(height)])
        .flex(ratatui::layout::Flex::Center)
        .areas(area);
    let [middle] = Layout::horizontal([Constraint::Percentage(width)])
        .flex(ratatui::layout::Flex::Center)
        .areas(row);
    middle
}
