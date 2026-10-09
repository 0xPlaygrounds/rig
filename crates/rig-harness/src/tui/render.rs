//! Draws the focused agent: transcript, status line, input editor with its
//! completion list, and the overlays.

use bevy_ecs::prelude::*;
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, List, ListState, Paragraph, Wrap};

use super::complete::{Completion, Kind as CompletionKind};
use super::editor::Layout as InputLayout;
use super::markdown;
use super::panel::{self, PanelCanvas, Placement, RequestRedraw, TuiPanel};
use super::renderers::ToolRenderer;
use super::terminal::Tui;
use super::transcript::{Part, Renderers, Transcript, plain_lines};
use super::view::{Overlay, Picker, ShownNotice, TuiView};
use super::wrap::wrap_all;
use crate::host::reload::{ReloadBuild, ReloadQueued};
use crate::host::sessions::SessionName;
use rig_ecs::activity::{Activity, Status};
use rig_ecs::agent::{
    ActiveTurn, Agent, Calls, Connection, Conversation, Effort, ModelChoice, NoticeLevel, Partial,
    Spawned, SpawnedBy,
};
use rig_ecs::commands::SlashCommand;
use rig_ecs::compaction::Compacted;
use rig_ecs::inbox::Inbox;
use rig_ecs::models;
use rig_ecs::recovery::RETRY;
use rig_ecs::usage::{self, Spending, TurnSpending};

/// Most lines the input box shows.
const INPUT_LINES: usize = 10;
/// Most items the completion list shows at once.
const COMPLETION_ROWS: usize = 8;
/// Lines of a compaction's summary shown in the transcript.
const SUMMARY_LINES: usize = 12;
/// Width of the rebuild progress bar.
const GAUGE_WIDTH: u32 = 20;

/// Whether anything drawn changed since the last frame: the view state (a
/// key, a notice, a resize), an agent's drawn components (its
/// [`Activity`] too, which counts a retry's wait down), a turn's calls, a
/// streaming reply, a panel, or a plugin's [`RequestRedraw`]. A turn's end
/// changes its conversation or comes with a notice. The rebuild's progress
/// is checked separately.
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
            Changed<Activity>,
        )>,
    >,
    turns: Query<(), Or<(Changed<Calls>, Changed<TurnSpending>)>>,
    partials: Query<(), Changed<Partial>>,
    name: Res<SessionName>,
    mut requests: MessageReader<RequestRedraw>,
    panels: Query<(), Changed<TuiPanel>>,
    mut removed_panels: RemovedComponents<TuiPanel>,
) -> bool {
    // Every reader is drained, so none redraws again for the same change.
    let requested = requests.read().count() > 0;
    let removed = removed_panels.read().count() > 0;
    requested
        || removed
        || view.is_changed()
        || name.is_changed()
        || !agents.is_empty()
        || !turns.is_empty()
        || !partials.is_empty()
        || !panels.is_empty()
}

/// The frame [`layout`] laid out for [`render`]; `due` while one is to be
/// drawn.
#[derive(Resource, Default)]
pub(crate) struct FrameLayout {
    due: bool,
    input_rows: Option<InputLayout>,
    input_height: usize,
    transcript: Rect,
    status: Rect,
    input: Rect,
}

/// Whether [`layout`] laid out a frame that is not drawn yet: the
/// condition of [`TuiSystems::Draw`](super::TuiSystems::Draw) and
/// [`render`].
pub(crate) fn frame_due(frame: Res<FrameLayout>) -> bool {
    frame.due
}

/// Lays out a frame: the input box grows with the input, up to a limit;
/// the status line sits above it; the panels take their sides of the rest,
/// in entity order, and the transcript what they leave.
pub(crate) fn layout(
    tui: Res<Tui>,
    view: Res<TuiView>,
    mut frame: ResMut<FrameLayout>,
    mut panels: Query<(Entity, &TuiPanel, &mut PanelCanvas)>,
) {
    let Ok(size) = tui.terminal.size() else {
        return;
    };
    let screen = Rect::new(0, 0, size.width, size.height);
    let rows = view
        .editor
        .layout(size.width.saturating_sub(2), Style::new());
    let input_height = rows.rows.len().clamp(1, INPUT_LINES);
    let [above, status, input] = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(1),
        Constraint::Length(u16::try_from(input_height + 2).unwrap_or(3)),
    ])
    .areas(screen);
    let mut panels: Vec<_> = panels.iter_mut().collect();
    panels.sort_by_key(|(entity, ..)| *entity);
    let transcript = panel::lay_out(
        panels.into_iter().map(|(_, panel, canvas)| (panel, canvas)),
        above,
        screen,
    );
    *frame = FrameLayout {
        due: true,
        input_rows: Some(rows),
        input_height,
        transcript,
        status,
        input,
    };
}

/// The agents the status line counts: each with whether it works, its
/// name, whether another spawned it and the agents it spawned.
type Everyone<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        Has<ActiveTurn>,
        Option<&'static Name>,
        Has<SpawnedBy>,
        Option<&'static Spawned>,
    ),
    With<Agent>,
>;

/// The transcript's parts after the shown agent's messages are laid out:
/// each message, after the notices that came before it and the
/// compaction's summary; then the notices after the last message, the
/// reply streaming in and what waits in the inbox. Without an agent, the
/// app's notices.
fn transcript_parts(
    view: &TuiView,
    shown: Option<(&Conversation, &Compacted, &Inbox)>,
    partial: Option<&Partial>,
    width: usize,
) -> Vec<Part> {
    let mut parts = Vec::new();
    let Some((conversation, compacted, inbox)) = shown else {
        let mut extra = Vec::new();
        for notice in view.notices.iter().filter(|notice| notice.is_for(None)) {
            notice_lines(notice, &mut extra);
        }
        parts.push(Part::Rows(wrap_all(&extra, width)));
        return parts;
    };
    let mut notices = view
        .notices
        .iter()
        .filter(|notice| notice.is_for(view.agent))
        .peekable();
    for index in 0..conversation.messages().len() {
        let mut extra = Vec::new();
        while let Some(notice) = notices.next_if(|notice| notice.after <= index) {
            notice_lines(notice, &mut extra);
        }
        if compacted.upto == index && !compacted.summary.is_empty() {
            summary_lines(compacted, &mut extra);
        }
        if !extra.is_empty() {
            parts.push(Part::Rows(wrap_all(&extra, width)));
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
    inbox_lines(inbox, &mut extra);
    parts.push(Part::Rows(wrap_all(&extra, width)));
    parts
}

/// The agent counts after the status line: which agent this is when
/// another spawned it, how many of the agents it spawned are at work, and
/// how many others.
fn agent_spans(line: &mut Line<'static>, focused: Option<Entity>, everyone: &Everyone, hint: &str) {
    let focused_agent = focused.and_then(|agent| everyone.get(agent).ok());
    if let Some((_, _, Some(title), true, _)) = focused_agent {
        line.spans
            .insert(0, Span::from(format!("⤷ {title}  ")).magenta());
    }
    let mine: Vec<Entity> = focused_agent
        .and_then(|(.., spawned)| spawned)
        .map(|spawned| {
            spawned
                .iter()
                .filter(|child| everyone.get(*child).is_ok_and(|(_, busy, ..)| busy))
                .collect()
        })
        .unwrap_or_default();
    match mine.len() {
        0 => {}
        1 => line.push_span(Span::from(format!("  an agent it spawned works{hint}")).magenta()),
        count => {
            line.push_span(Span::from(format!("  {count} agents it spawned work{hint}")).magenta());
        }
    }
    let working = everyone
        .iter()
        .filter(|(agent, busy, ..)| *busy && Some(*agent) != focused && !mine.contains(agent))
        .count();
    if working > 0 {
        line.push_span(Span::from(format!("  +{working} more working{hint}")).magenta());
    }
}

/// The hint under the input box.
fn input_hint(turn_running: bool, empty: bool) -> &'static str {
    match (turn_running, empty) {
        (true, _) => " Enter steers this turn · Tab sends after it · Esc stops ",
        (false, true) => {
            " Enter sends · Shift+Enter or Ctrl+J new line · Ctrl+V image · / commands · @ files "
        }
        (false, false) => "",
    }
}

/// Draws the frame [`layout`] laid out, with what the plugins drew in
/// their panels. The transcript's rows are kept between frames in a
/// [`Transcript`], so only changed messages are laid out again.
pub(crate) fn render(
    mut tui: ResMut<Tui>,
    mut view: ResMut<TuiView>,
    mut frame_layout: ResMut<FrameLayout>,
    mut transcript: Local<Transcript>,
    agents: Query<(
        &Conversation,
        &Compacted,
        Option<&ModelChoice>,
        &Effort,
        &Activity,
        &Spending,
        Option<&Connection>,
        &Inbox,
    )>,
    changed: Query<(), Changed<Conversation>>,
    turns: Query<(Option<&Calls>, &TurnSpending)>,
    (partials, active): (Query<&Partial>, Query<&ActiveTurn>),
    panels: Query<(Entity, &TuiPanel, &PanelCanvas)>,
    everyone: Everyone,
    slash: Query<&SlashCommand>,
    renderers: Query<Ref<ToolRenderer>>,
    mut removed_renderers: RemovedComponents<ToolRenderer>,
    (build, queued): (Option<Res<ReloadBuild>>, Option<Res<ReloadQueued>>),
    name: Res<SessionName>,
) -> Result {
    let frame_layout = std::mem::take(&mut *frame_layout);
    let Some(input_rows) = frame_layout.input_rows else {
        return Ok(());
    };
    // Clamping the scroll is drawing's own bookkeeping, not a change to
    // redraw for.
    let view = view.bypass_change_detection();
    let shown = view
        .agent
        .and_then(|agent| Some((agent, agents.get(agent).ok()?)));
    let turn = view
        .agent
        .and_then(|agent| active.get(agent).ok())
        .and_then(|turn| turns.get(turn.turn()).ok());
    let calls = turn.and_then(|(calls, _)| calls);
    let partial = calls.and_then(|calls| calls.iter().find_map(|call| partials.get(call).ok()));
    let mut panels: Vec<_> = panels.iter().collect();
    panels.sort_by_key(|(entity, ..)| *entity);
    if removed_renderers.read().count() > 0
        || renderers.iter().any(|renderer| renderer.is_changed())
    {
        transcript.clear();
    }
    let by_tool: Renderers<'_> = renderers
        .iter()
        .map(|renderer| {
            let renderer = renderer.into_inner();
            (renderer.tool.as_str(), &renderer.render)
        })
        .collect();
    // /model and /agents come from plugins, so point at them only when
    // loaded.
    let loaded = |name: &str| slash.iter().any(|command| command.name == name);
    let agents_hint = if loaded("agents") { " (/agents)" } else { "" };
    tui.terminal.draw(|frame| {
        // A resize since the layout leaves nothing outside the screen.
        let screen = frame.area();
        let transcript_area = frame_layout.transcript.intersection(screen);
        let status_area = frame_layout.status.intersection(screen);
        let input = frame_layout.input.intersection(screen);
        let layout = input_rows;
        // The input box scrolls to keep the cursor in sight.
        let input_top = (layout.cursor_row + 1).saturating_sub(frame_layout.input_height);
        if let Some((agent, (conversation, ..))) = shown {
            transcript.update(
                agent,
                conversation,
                changed.contains(agent),
                &by_tool,
                transcript_area.width,
            );
        }
        let parts = transcript_parts(
            view,
            shown.map(|(_, (conversation, compacted, .., inbox))| (conversation, compacted, inbox)),
            partial,
            usize::from(transcript_area.width.max(1)),
        );
        let rows = transcript.visible(
            &parts,
            usize::from(transcript_area.height),
            &mut view.scroll,
        );
        frame.render_widget(Paragraph::new(rows), transcript_area);
        let shown = shown.map(|(_, shown)| shown);
        let mut line = status_line(
            shown.map(|(_, _, model, effort, activity, ..)| (model, effort, activity.status)),
            loaded("model"),
        );
        agent_spans(&mut line, view.agent, &everyone, agents_hint);
        if let Some(name) = &name.0 {
            line.spans.insert(0, Span::from(format!("{name}  ")).cyan());
        }
        if let Some((_, spent)) = turn
            && spent.0.calls > 0
        {
            let used = spent.0.cost_or_tokens();
            line.push_span(Span::from(format!("  this turn {used}")).dim());
        }
        if let Some(build) = &build {
            line.push_span(reload_span(build));
        } else if queued.is_some() {
            line.push_span(
                Span::from("  Reload queued: once no turn runs (/reload cancel)").cyan(),
            );
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
        let hint = input_hint(turn.is_some(), view.editor.is_empty());
        frame.render_widget(
            Paragraph::new(layout.rows)
                .scroll((u16::try_from(input_top).unwrap_or(u16::MAX), 0))
                .block(Block::bordered().title_bottom(Line::from(hint).dim().right_aligned())),
            input,
        );
        // The panels at the sides, then the boxes over the screen.
        let over = |panel: &TuiPanel| matches!(panel.placement, Placement::Over { .. });
        for top in [false, true] {
            for (_, panel, canvas) in &panels {
                if over(panel) == top {
                    canvas.copy_to(frame.buffer_mut());
                }
            }
        }
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
    shown: Option<(Option<&ModelChoice>, &Effort, Status)>,
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
        Status::Idle => Span::from("idle").green(),
        Status::Thinking => Span::from("thinking… (Esc stops)").yellow(),
        Status::RunningTools => Span::from("running tools… (Esc stops)").yellow(),
        Status::Compacting => Span::from("compacting… (Esc stops)").yellow(),
        Status::Retrying { attempt, seconds } => Span::from(format!(
            "retry {attempt}/{} in {seconds}s… (Esc stops)",
            RETRY.max_retries
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
    if let Some(context) = spent.context_use(connection.map(|connection| &*connection.spec)) {
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
        .map(|item| item.label.clone())
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
    let waiting = inbox
        .steering
        .iter()
        .map(|pending| ("steering", &pending.text))
        .chain(
            inbox
                .queued
                .iter()
                .map(|pending| ("after this turn", &pending.text)),
        )
        .chain(inbox.notes.iter().map(|pending| ("noted", &pending.text)));
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
