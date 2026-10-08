//! Draws the focused agent: transcript, status line, input editor with its
//! completion list, and the overlays.

use bevy_ecs::prelude::*;
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, List, ListState, Paragraph, Wrap};

use super::complete::{Completion, Kind as CompletionKind};
use super::markdown;
use super::renderers::ToolRenderer;
use super::terminal::Tui;
use super::transcript::{Part, Renderers, Transcript, plain_lines};
use super::view::{Overlay, Picker, ShownNotice, TuiView};
use super::wrap::wrap_all;
use crate::core::agent::{
    ActiveTurn, Agent, Calls, Connection, Conversation, Effort, ModelChoice, NoticeLevel, Partial,
    ToolCallRun,
};
use crate::core::commands::SlashCommand;
use crate::core::compaction::{Compacted, Summarizing};
use crate::core::inbox::Inbox;
use crate::core::models;
use crate::core::recovery::{Backoff, MAX_RETRIES};
use crate::core::subagents::{Assignee, Delegated};
use crate::core::usage::{self, Spending, TurnSpending};
use crate::host::reload::ReloadBuild;
use crate::host::sessions::SessionName;

/// Most lines the input box shows.
const INPUT_LINES: usize = 10;
/// Most items the completion list shows at once.
const COMPLETION_ROWS: usize = 8;
/// Lines of a compaction's summary shown in the transcript.
const SUMMARY_LINES: usize = 12;
/// Width of the rebuild progress bar.
const GAUGE_WIDTH: u32 = 20;

/// What the shown agent is doing.
#[derive(Clone, Copy)]
enum Activity {
    Idle,
    Thinking,
    RunningTools,
    /// Waiting for this many subagents' answers.
    Delegating(usize),
    /// Summarizing the older conversation.
    Compacting,
    /// Waiting `seconds` before retry `attempt` of a failed model call.
    Retrying {
        attempt: u32,
        seconds: u64,
    },
}

/// Whether anything drawn changed since the last frame: the view state (a
/// key, a notice, a resize), an agent's drawn components, a turn's calls,
/// or a streaming reply. A turn's end changes its conversation or comes
/// with a notice. A retry's countdown redraws every frame while it waits.
/// The rebuild's progress is checked separately.
pub(crate) fn needs_redraw(
    view: Res<TuiView>,
    agents: Query<
        (),
        Or<(
            Changed<Conversation>,
            Changed<ModelChoice>,
            Changed<Effort>,
            Changed<ActiveTurn>,
            Changed<Spending>,
            Changed<Compacted>,
            Changed<Inbox>,
        )>,
    >,
    turns: Query<(), Or<(Changed<Calls>, Changed<TurnSpending>)>>,
    partials: Query<(), Changed<Partial>>,
    waits: Query<(), With<Backoff>>,
    name: Res<SessionName>,
) -> bool {
    view.is_changed()
        || name.is_changed()
        || !agents.is_empty()
        || !turns.is_empty()
        || !partials.is_empty()
        || !waits.is_empty()
}

/// Draws one frame. The transcript's rows are kept between frames in a
/// [`Transcript`], so only changed messages are laid out again.
pub(crate) fn render(
    mut tui: ResMut<Tui>,
    mut view: ResMut<TuiView>,
    mut transcript: Local<Transcript>,
    agents: Query<(
        &Conversation,
        &Compacted,
        Option<&ModelChoice>,
        &Effort,
        Option<&ActiveTurn>,
        &Spending,
        Option<&Connection>,
        &Inbox,
    )>,
    changed: Query<(), Changed<Conversation>>,
    turns: Query<(Option<&Calls>, &TurnSpending)>,
    partials: Query<&Partial>,
    (tool_calls, summaries, assigned): (
        Query<(), With<ToolCallRun>>,
        Query<(), With<Summarizing>>,
        Query<(), With<Assignee>>,
    ),
    waits: Query<&Backoff>,
    everyone: Query<(Entity, Has<ActiveTurn>, Option<&Delegated>), With<Agent>>,
    slash: Query<&SlashCommand>,
    renderers: Query<Ref<ToolRenderer>>,
    mut removed_renderers: RemovedComponents<ToolRenderer>,
    build: Option<Res<ReloadBuild>>,
    name: Res<SessionName>,
) -> Result {
    // Clamping the scroll is drawing's own bookkeeping, not a change to
    // redraw for.
    let view = view.bypass_change_detection();
    let shown = view
        .agent
        .and_then(|agent| Some((agent, agents.get(agent).ok()?)));
    let turn = shown
        .and_then(|(_, (_, _, _, _, turn, ..))| turn)
        .and_then(|turn| turns.get(turn.turn()).ok());
    let calls = turn.and_then(|(calls, _)| calls);
    let partial = calls.and_then(|calls| calls.iter().find_map(|call| partials.get(call).ok()));
    let wait = calls.and_then(|calls| calls.iter().find_map(|call| waits.get(call).ok()));
    let activity = if turn.is_none() {
        Activity::Idle
    } else if let Some(wait) = wait {
        Activity::Retrying {
            attempt: wait.attempt,
            seconds: wait.seconds_left(),
        }
    } else if calls.is_some_and(|calls| calls.iter().any(|call| summaries.contains(call))) {
        Activity::Compacting
    } else if let Some(waiting) = calls
        .map(|calls| calls.iter().filter(|call| assigned.contains(*call)).count())
        .filter(|waiting| *waiting > 0)
    {
        Activity::Delegating(waiting)
    } else if calls.is_some_and(|calls| calls.iter().any(|call| tool_calls.contains(call))) {
        Activity::RunningTools
    } else {
        Activity::Thinking
    };
    let renderers_changed = removed_renderers.read().count() > 0
        || renderers.iter().any(|renderer| renderer.is_changed());
    if renderers_changed {
        transcript.clear();
    }
    let by_tool: Renderers<'_> = renderers
        .iter()
        .map(|renderer| {
            let renderer = renderer.into_inner();
            (renderer.tool.as_str(), &renderer.render)
        })
        .collect();
    tui.terminal.draw(|frame| {
        let width = frame.area().width;
        // The input box grows with the input, up to a limit, and scrolls
        // to keep the cursor in sight.
        let layout = view.editor.layout(width.saturating_sub(2), Style::new());
        let input_height = layout.rows.len().clamp(1, INPUT_LINES);
        let input_top = (layout.cursor_row + 1).saturating_sub(input_height);
        let [transcript_area, status_area, input] = Layout::vertical([
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(u16::try_from(input_height + 2).unwrap_or(3)),
        ])
        .areas(frame.area());
        // Notices go between the messages, where they arrived; the ones
        // after the last message go before the reply streaming in.
        let rows_width = usize::from(transcript_area.width.max(1));
        let mut parts = Vec::new();
        if let Some((agent, (conversation, compacted, ..))) = shown {
            transcript.update(
                agent,
                &conversation.0,
                changed.contains(agent),
                &by_tool,
                transcript_area.width,
            );
            let mut notices = view
                .notices
                .iter()
                .filter(|notice| notice.is_for(view.agent))
                .peekable();
            for index in 0..conversation.0.len() {
                let mut extra = Vec::new();
                while let Some(notice) = notices.next_if(|notice| notice.after <= index) {
                    notice_lines(notice, &mut extra);
                }
                if compacted.upto == index && !compacted.summary.is_empty() {
                    summary_lines(compacted, &mut extra);
                }
                if !extra.is_empty() {
                    parts.push(Part::Rows(wrap_all(&extra, rows_width)));
                }
                parts.push(Part::Message(index));
            }
            let mut extra = Vec::new();
            for notice in notices {
                notice_lines(notice, &mut extra);
            }
            if let Some(partial) = partial {
                extra.extend(plain_lines(&partial.reasoning, Style::new().dim().italic()));
                if !partial.text.is_empty() {
                    extra.push(Line::default());
                    extra.extend(markdown::render(&partial.text));
                }
            }
            if let Some((.., inbox)) = shown.map(|(_, shown)| shown) {
                inbox_lines(inbox, &mut extra);
            }
            parts.push(Part::Rows(wrap_all(&extra, rows_width)));
        } else {
            let mut extra = Vec::new();
            for notice in view.notices.iter().filter(|notice| notice.is_for(None)) {
                notice_lines(notice, &mut extra);
            }
            parts.push(Part::Rows(wrap_all(&extra, rows_width)));
        }
        let rows = transcript.visible(
            &parts,
            usize::from(transcript_area.height),
            &mut view.scroll,
        );
        frame.render_widget(Paragraph::new(rows), transcript_area);
        // /model comes from a plugin, so point at it only when loaded.
        let model_hint = slash.iter().any(|command| command.name == "model");
        let shown = shown.map(|(_, shown)| shown);
        let mut line = status_line(
            shown.map(|(_, _, model, effort, ..)| (model, effort, activity)),
            model_hint,
        );
        // Which agent this is, when it is a subagent, and how many others
        // are at work.
        let focused = view.agent;
        if let Some(task) = focused
            .and_then(|agent| everyone.get(agent).ok())
            .and_then(|(_, _, delegated)| delegated)
        {
            line.spans
                .insert(0, Span::from(format!("⤷ {}  ", task.task)).magenta());
        }
        if let Some(name) = &name.0 {
            line.spans.insert(0, Span::from(format!("{name}  ")).cyan());
        }
        let working = everyone
            .iter()
            .filter(|(agent, busy, _)| *busy && Some(*agent) != focused)
            .count();
        if working > 0 {
            line.push_span(Span::from(format!("  +{working} more working (/agents)")).magenta());
        }
        if let Some((_, spent)) = turn
            && let Some(cost) = spent.0.cost_label()
        {
            line.push_span(Span::from(format!("  this turn {cost}")).dim());
        }
        if let Some(build) = &build {
            line.push_span(reload_span(build));
        }
        let usage = shown
            .map(|(.., spent, connection, _)| usage_line(spent, connection))
            .unwrap_or_default();
        let usage_width = u16::try_from(usage.width()).unwrap_or(u16::MAX);
        let [status, meter] =
            Layout::horizontal([Constraint::Min(0), Constraint::Length(usage_width)])
                .areas(status_area);
        frame.render_widget(line, status);
        frame.render_widget(usage, meter);
        let hint = match (turn.is_some(), view.editor.is_empty()) {
            (true, _) => " Enter steers this turn · Tab sends after it · Esc stops ",
            (false, true) => {
                " Enter sends · Shift+Enter or Ctrl+J new line · Ctrl+G editor · Ctrl+V image \
                 · / commands · @ files "
            }
            (false, false) => "",
        };
        frame.render_widget(
            Paragraph::new(layout.rows)
                .scroll((u16::try_from(input_top).unwrap_or(u16::MAX), 0))
                .block(Block::bordered().title_bottom(Line::from(hint).dim().right_aligned())),
            input,
        );
        match &view.overlay {
            Some(Overlay::Picker(picker)) => draw_picker(frame, picker),
            Some(Overlay::ReloadFailure(output)) => draw_reload_failure(frame, output),
            None => {
                if let Some(completion) = &view.completion {
                    draw_completion(frame, completion, input);
                }
                let column = u16::try_from(layout.cursor_column).unwrap_or(u16::MAX);
                let row = u16::try_from(layout.cursor_row - input_top).unwrap_or(0);
                frame.set_cursor_position((
                    input.x.saturating_add(1).saturating_add(column),
                    input.y.saturating_add(1).saturating_add(row),
                ));
            }
        }
    })?;
    Ok(())
}

/// The completion list, just above the input box.
fn draw_completion(frame: &mut Frame, completion: &Completion, input: Rect) {
    let rows = completion.items.len().min(COMPLETION_ROWS);
    let height = u16::try_from(rows + 2).unwrap_or(3).min(input.y);
    if height < 3 {
        return;
    }
    let area = Rect::new(input.x, input.y - height, input.width, height);
    frame.render_widget(Clear, area);
    let marker = match completion.kind {
        CompletionKind::Command => "/",
        CompletionKind::Path => "@",
    };
    let items: Vec<Line<'static>> = completion
        .items
        .iter()
        .map(|item| {
            let mut spans = vec![Span::from(format!("{marker}{}", item.text))];
            if !item.detail.is_empty() {
                spans.push(Span::from(format!("  {}", item.detail)).dim());
            }
            Line::from(spans)
        })
        .collect();
    let mut state = ListState::default().with_selected(Some(completion.selected));
    frame.render_stateful_widget(
        List::new(items)
            .block(
                Block::bordered()
                    .title(" Tab or Enter completes · Esc closes ")
                    .dim(),
            )
            .highlight_style(Style::new().add_modifier(Modifier::REVERSED)),
        area,
        &mut state,
    );
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
        Activity::Delegating(1) => {
            Span::from("waiting on a subagent… (Esc stops, /agents shows it)").yellow()
        }
        Activity::Delegating(count) => Span::from(format!(
            "waiting on {count} subagents… (Esc stops, /agents shows them)"
        ))
        .yellow(),
        Activity::Compacting => Span::from("compacting… (Esc stops)").yellow(),
        Activity::Retrying { attempt, seconds } => Span::from(format!(
            "retry {attempt}/{MAX_RETRIES} in {seconds}s… (Esc stops)"
        ))
        .red(),
    };
    Line::from(vec![
        Span::from(model).bold(),
        Span::from(format!("  reasoning {}  ", models::effort_label(effort.0))).dim(),
        status,
    ])
}

/// The agent's tokens, cost and context use: uncached input, output, cache
/// reads and writes, then the context against the model's window, yellow
/// past 70% and red past 90%.
fn usage_line(spent: &Spending, connection: Option<&Connection>) -> Line<'static> {
    if spent.calls == 0 {
        return Line::default();
    }
    let mut parts = vec![
        format!("↑{}", usage::tokens(spent.uncached_input())),
        format!(
            "↓{}",
            usage::tokens(spent.tokens.output_tokens.unwrap_or(0))
        ),
    ];
    if let Some(read) = spent.tokens.cached_input_tokens.filter(|read| *read > 0) {
        parts.push(format!("R{}", usage::tokens(read)));
    }
    if let Some(written) = spent
        .tokens
        .cache_creation_input_tokens
        .filter(|written| *written > 0)
    {
        parts.push(format!("W{}", usage::tokens(written)));
    }
    parts.extend(spent.cost_label());
    let mut spans = vec![Span::from(parts.join(" ")).dim()];
    if let Some(context) = spent.context_use(connection.map(|connection| connection.spec)) {
        let style = match context.percent() {
            Some(90..) => Style::new().red(),
            Some(70..) => Style::new().yellow(),
            _ => Style::new().dim(),
        };
        spans.push(Span::styled(format!("  ctx {} ", context.label()), style));
    }
    Line::from(spans)
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
fn popup(frame: &mut Frame, fifths: u32, quarters: u32) -> Rect {
    let popup = frame
        .area()
        .centered(Constraint::Ratio(fifths, 5), Constraint::Ratio(quarters, 4));
    frame.render_widget(Clear, popup);
    popup
}

/// A failed rebuild's output, from its first error on, over the transcript.
fn draw_reload_failure(frame: &mut Frame, output: &str) {
    let popup = popup(frame, 5, 4);
    let block = Block::bordered()
        .border_style(Style::new().red())
        .title(" The rebuild failed; this build keeps running · Esc closes ");
    frame.render_widget(
        Paragraph::new(plain_lines(output, Style::new()))
            .wrap(Wrap { trim: false })
            .block(block),
        popup,
    );
}

fn draw_picker(frame: &mut Frame, picker: &Picker) {
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

/// Draws where a compaction cut the conversation: the messages above are
/// sent to the model as the summary under the line.
fn summary_lines(compacted: &Compacted, lines: &mut Vec<Line<'static>>) {
    let style = Style::new().fg(Color::Magenta);
    lines.push(Line::default());
    lines.push(Line::styled(
        format!(
            "── {} earlier messages are sent as this summary ──",
            compacted.upto
        ),
        style.bold(),
    ));
    let total = compacted.summary.lines().count();
    for line in compacted.summary.lines().take(SUMMARY_LINES) {
        lines.push(Line::styled(line.replace('\t', "    "), style.dim()));
    }
    if total > SUMMARY_LINES {
        lines.push(Line::styled(
            format!("  … {} more lines", total - SUMMARY_LINES),
            style.dim(),
        ));
    }
}

/// What waits in the agent's inbox, under the transcript.
fn inbox_lines(inbox: &Inbox, lines: &mut Vec<Line<'static>>) {
    if inbox.is_empty() {
        return;
    }
    lines.push(Line::default());
    let waiting = inbox.steering.iter().map(|text| ("steering", text)).chain(
        inbox
            .follow_ups
            .iter()
            .map(|text| ("after this turn", text)),
    );
    for (when, text) in waiting {
        let first = text.lines().next().unwrap_or_default();
        let more = if text.lines().nth(1).is_some() {
            " …"
        } else {
            ""
        };
        lines.push(Line::from(vec![
            Span::from(format!("  ⏵ {when}: ")).cyan(),
            Span::from(format!("{first}{more}")).dim(),
        ]));
    }
}

fn notice_lines(notice: &ShownNotice, lines: &mut Vec<Line<'static>>) {
    let style = match notice.level {
        NoticeLevel::Info => Style::new().fg(Color::Magenta),
        NoticeLevel::Error => Style::new().fg(Color::Red),
    };
    lines.extend(plain_lines(&notice.text, style));
}
