//! Draws the focused agent: transcript, status line, input editor with its
//! completion list, and the picker.

use std::time::{Duration, Instant};

use bevy_ecs::prelude::*;
use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::terminal::{BeginSynchronizedUpdate, EndSynchronizedUpdate};
use crossterm::{execute, queue};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, List, ListState, Paragraph};

use super::complete::{Completion, Kind as CompletionKind};
use super::editor::Layout as InputLayout;
use super::markdown;
use super::panel::{self, PanelCanvas, Placement, RequestRedraw, TuiPanel};
use super::renderers::{ToolRenderer, excerpt};
use super::terminal::Tui;
use super::transcript::{Below, Part, Renderers, Transcript, plain_lines};
use super::view::{Picker, ShownNotice, TuiView};
use super::wrap::wrap_all;
use crate::prelude::{Activity, ReloadStatus, SessionTitle, Spending, Status, TurnSpending};
use rig_core::completion::{ContextUse, tokens_label};
use rig_ecs::agent::{
    ActiveTurn, Agent, Calls, Condensed, Conversation, LastUsage, NoticeLevel, Partial, Spawned,
    SpawnedBy,
};
use rig_ecs::commands::SlashCommand;
use rig_ecs::inbox::Inbox;
use rig_ecs::model::{Connection, Effort, ModelChoice};
use rig_ecs::turn::RETRY;

/// Most lines the input box shows.
const INPUT_LINES: usize = 10;
/// Most items the completion list shows at once.
const COMPLETION_ROWS: usize = 8;
/// Lines of a condensed conversation's summary shown in the transcript.
const SUMMARY_LINES: usize = 12;

/// The shortest time between two frames drawn only for a streaming
/// reply's new text: the loop's own limit of 60 frames a second would lay
/// the whole reply out again for every few tokens.
const STREAM_FRAME: Duration = Duration::from_millis(33);

/// When [`needs_redraw`] last drew, and whether streamed text waits to be
/// drawn.
#[derive(Default)]
pub(crate) struct Paced {
    drawn: Option<Instant>,
    waiting: bool,
}

/// Whether anything drawn changed since the last frame: the view state (a
/// key, a notice, a resize), an agent's drawn components (its
/// [`Activity`] too, which counts a retry's wait down), a turn's calls, a
/// streaming reply, a panel, or a plugin's [`RequestRedraw`]. A turn's end
/// changes its conversation or comes with a notice. The rebuild's progress
/// is checked separately. A streaming reply alone draws at most every
/// [`STREAM_FRAME`]; the text that waits is drawn by one of the frames
/// that [`settle`](rig_ecs::calls::settle) runs after the wake that brought
/// it, 16 ms apart, under any loop.
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
            Changed<LastUsage>,
            Changed<Condensed>,
            Changed<Inbox>,
            Changed<Activity>,
        )>,
    >,
    turns: Query<(), Or<(Changed<Calls>, Changed<TurnSpending>)>>,
    partials: Query<(), Changed<Partial>>,
    title: Option<Res<SessionTitle>>,
    mut requests: MessageReader<RequestRedraw>,
    panels: Query<(), Changed<TuiPanel>>,
    mut removed_panels: RemovedComponents<TuiPanel>,
    mut paced: Local<Paced>,
) -> bool {
    // Every reader is drained, so none redraws again for the same change.
    let requested = requests.read().count() > 0;
    let removed = removed_panels.read().count() > 0;
    let changed = requested
        || removed
        || view.is_changed()
        || title.is_some_and(|title| title.is_changed())
        || !agents.is_empty()
        || !turns.is_empty()
        || !panels.is_empty();
    paced.waiting |= !partials.is_empty();
    let now = Instant::now();
    let next = paced.drawn.map(|drawn| drawn + STREAM_FRAME);
    if changed || (paced.waiting && next.is_none_or(|next| now >= next)) {
        paced.drawn = Some(now);
        paced.waiting = false;
        return true;
    }
    false
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
    let rows = view.editor.layout(size.width.saturating_sub(2));
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

/// The agents the status line counts: each with its [`Activity`], its
/// name, whether another spawned it and the agents it spawned.
type Everyone<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static Activity,
        Option<&'static Name>,
        Has<SpawnedBy>,
        Option<&'static Spawned>,
    ),
    With<Agent>,
>;

/// The transcript's parts after the shown agent's messages are laid out:
/// each message, after the notices that came before it and the summary
/// sent in place of the messages above; then the notices after the last message, the
/// reply streaming in and what waits in the inbox. Without an agent, the
/// app's notices.
fn transcript_parts(
    view: &TuiView,
    shown: Option<(&Conversation, Option<&Condensed>, &Inbox)>,
    partial: Option<&Partial>,
    width: usize,
) -> Vec<Part> {
    let mut parts = Vec::new();
    let Some((conversation, condensed, inbox)) = shown else {
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
        if let Some(condensed) = condensed.filter(|condensed| condensed.upto == index) {
            summary_lines(condensed, &mut extra);
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

/// The focused agent's title, before its model, when another agent
/// spawned it.
fn spawned_title(focused: Option<Entity>, everyone: &Everyone) -> Option<Piece> {
    let (_, _, title, spawned_by, _) = everyone.get(focused?).ok()?;
    let title = title.filter(|_| spawned_by)?;
    Some(Piece::new(
        keep::TITLE,
        Span::from(format!("⤷ {title}")).magenta(),
    ))
}

/// The agent counts after the status: how many of the agents the focused
/// one spawned are at work, idle or not, and how many others.
fn agent_pieces(pieces: &mut Vec<Piece>, focused: Option<Entity>, everyone: &Everyone, hint: &str) {
    let busy = |agent: Entity| {
        everyone
            .get(agent)
            .is_ok_and(|(_, activity, ..)| activity.is_busy())
    };
    let mine: Vec<Entity> = focused
        .and_then(|agent| everyone.get(agent).ok())
        .and_then(|(.., spawned)| spawned)
        .map(|spawned| spawned.iter().filter(|child| busy(*child)).collect())
        .unwrap_or_default();
    let text = match mine.len() {
        0 => None,
        1 => Some(format!("1 subagent working{hint}")),
        count => Some(format!("{count} subagents working{hint}")),
    };
    if let Some(text) = text {
        pieces.push(Piece::new(keep::SUBAGENTS, Span::from(text).magenta()));
    }
    let others = everyone
        .iter()
        .filter(|(agent, activity, ..)| {
            activity.is_busy() && Some(*agent) != focused && !mine.contains(agent)
        })
        .count();
    if others > 0 {
        pieces.push(Piece::new(
            keep::OTHERS,
            Span::from(format!("+{others} other agents working{hint}")).magenta(),
        ));
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
        Option<&Condensed>,
        Option<&ModelChoice>,
        &Effort,
        &Activity,
        (Option<&Spending>, &LastUsage),
        Option<&Connection>,
        &Inbox,
    )>,
    changed: Query<(), Changed<Conversation>>,
    turns: Query<(Option<&Calls>, Option<&TurnSpending>)>,
    (partials, active): (Query<&Partial>, Query<&ActiveTurn>),
    panels: Query<(Entity, &TuiPanel, &PanelCanvas)>,
    everyone: Everyone,
    slash: Query<&Name, With<SlashCommand>>,
    renderers: Query<Ref<ToolRenderer>>,
    mut removed_renderers: RemovedComponents<ToolRenderer>,
    (reload, title): (Option<Res<ReloadStatus>>, Option<Res<SessionTitle>>),
) -> Result {
    let frame_layout = std::mem::take(&mut *frame_layout);
    let Some(input_rows) = frame_layout.input_rows else {
        return Ok(());
    };
    // Placing the scroll is drawing's own bookkeeping, not a change to
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
    // The panels at the sides, then the boxes over the screen.
    let mut panels: Vec<_> = panels.iter().collect();
    panels.sort_by_key(|(entity, panel, _)| {
        (matches!(panel.placement, Placement::Over { .. }), *entity)
    });
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
    let loaded = |name: &str| slash.iter().any(|command| command.as_str() == name);
    let agents_hint = if loaded("/agents") { " (/agents)" } else { "" };
    // The frame is written in one synchronized update with the cursor
    // hidden, so the cursor never shows travelling across the screen; it
    // is shown at the input once the frame is out, when the input has the
    // keys.
    let mut cursor = None;
    queue!(tui.terminal.backend_mut(), BeginSynchronizedUpdate, Hide)?;
    let drawn = tui
        .terminal
        .draw(|frame| {
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
                shown.map(|(_, (conversation, condensed, .., inbox))| {
                    (conversation, condensed, inbox)
                }),
                partial,
                usize::from(transcript_area.width.max(1)),
            );
            let (rows, below) = transcript.visible(
                &parts,
                (transcript_area.width, usize::from(transcript_area.height)),
                &mut view.scroll,
            );
            frame.render_widget(Paragraph::new(rows), transcript_area);
            draw_below(frame, below, transcript_area);
            let shown = shown.map(|(_, shown)| shown);
            let mut left = Vec::new();
            if let Some(name) = title.as_ref().and_then(|title| title.name.as_ref()) {
                left.push(Piece::new(keep::SESSION, Span::from(name.clone()).cyan()));
            }
            left.extend(spawned_title(view.agent, &everyone));
            left.extend(status_pieces(
                shown.map(|(_, _, model, effort, activity, ..)| (model, effort, &activity.status)),
                loaded("/model"),
            ));
            agent_pieces(&mut left, view.agent, &everyone, agents_hint);
            if let Some((_, Some(spent))) = turn
                && spent.0.calls > 0
            {
                let used = spent.0.cost_or_tokens();
                left.push(Piece::new(
                    keep::TURN,
                    Span::from(format!("this turn {used}")).dim(),
                ));
            }
            if let Some(reload) = reload.as_deref().and_then(reload_span) {
                left.push(Piece::new(keep::RELOAD, reload));
            }
            let mut right = shown
                .map(|(.., (spent, last), connection, _)| usage_pieces(spent, last, connection))
                .unwrap_or_default();
            fit(&mut left, &mut right, usize::from(status_area.width));
            let (line, usage) = (join(left, LEFT_GAP), join(right, RIGHT_GAP));
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
            for (.., canvas) in &panels {
                canvas.copy_to(frame.buffer_mut());
            }
            match &view.picker {
                Some(picker) => draw_picker(frame, picker),
                None => {
                    if let Some(completion) = &view.completion {
                        draw_completion(frame, completion, input);
                    }
                    let column = u16::try_from(layout.cursor_column).unwrap_or(u16::MAX);
                    let row = u16::try_from(layout.cursor_row - input_top).unwrap_or(0);
                    cursor = Some(MoveTo(
                        input.x.saturating_add(1).saturating_add(column),
                        input.y.saturating_add(1).saturating_add(row),
                    ));
                }
            }
        })
        .map(|_| ());
    let backend = tui.terminal.backend_mut();
    if drawn.is_ok()
        && let Some(cursor) = cursor
    {
        queue!(backend, cursor, Show)?;
    }
    // Ended even after a failed draw, so the terminal shows the screen
    // again.
    execute!(backend, EndSynchronizedUpdate)?;
    drawn?;
    Ok(())
}

/// A marker at the bottom right of a transcript scrolled up: what is below
/// and how to get there.
fn draw_below(frame: &mut Frame, below: Below, area: Rect) {
    let text = match below {
        Below::Nothing => return,
        Below::More => " ↓ more below (PgDn) ",
        Below::New => " ↓ new output (PgDn) ",
    };
    let width = u16::try_from(Span::from(text).width()).unwrap_or(u16::MAX);
    if area.height == 0 || area.width < width {
        return;
    }
    let marker = Rect::new(area.right() - width, area.bottom() - 1, width, 1);
    frame.render_widget(Clear, marker);
    frame.render_widget(Line::from(text).cyan().reversed(), marker);
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

/// A piece of the status line. When the line does not fit, [`fit`] drops
/// the pieces with the lowest `keep` first.
struct Piece {
    keep: u8,
    span: Span<'static>,
}

impl Piece {
    fn new(keep: u8, span: Span<'static>) -> Self {
        Self { keep, span }
    }

    fn width(&self) -> usize {
        self.span.width()
    }
}

/// How long each [`Piece`] of the status line stays as it narrows: the
/// cache reads go first, then the session's name (which the user chose and
/// `/resume` lists, so a long one must not hide what the session spends),
/// then cost, tokens and context; then the left's extras. The model and
/// the status always stay.
mod keep {
    pub(super) const CACHE: u8 = 1;
    pub(super) const SESSION: u8 = 2;
    pub(super) const COST: u8 = 3;
    pub(super) const TOKENS: u8 = 4;
    pub(super) const CONTEXT: u8 = 5;
    pub(super) const TURN: u8 = 11;
    pub(super) const REASONING: u8 = 12;
    pub(super) const OTHERS: u8 = 13;
    pub(super) const TITLE: u8 = 14;
    pub(super) const SUBAGENTS: u8 = 15;
    pub(super) const RELOAD: u8 = 16;
    pub(super) const ALWAYS: u8 = u8::MAX;
}

/// The gap between the pieces on the left, and between the left and the
/// meter.
const LEFT_GAP: &str = "  ";
/// The gap between the meter's pieces.
const RIGHT_GAP: &str = " ";

/// The width of `pieces` joined by `gap`.
fn joined_width(pieces: &[Piece], gap: &str) -> usize {
    let gaps = pieces.len().saturating_sub(1) * gap.len();
    pieces.iter().map(Piece::width).sum::<usize>() + gaps
}

/// Drops the pieces of `left` and of the meter on the `right` with the
/// lowest `keep` until both fit `width` with a gap between them. What
/// [`keep::ALWAYS`] stays; past that the line is cut at the edge.
fn fit(left: &mut Vec<Piece>, right: &mut Vec<Piece>, width: usize) {
    loop {
        let separator = if right.is_empty() { 0 } else { LEFT_GAP.len() };
        let used = joined_width(left, LEFT_GAP) + separator + joined_width(right, RIGHT_GAP);
        if used <= width {
            return;
        }
        let lowest = |pieces: &[Piece]| {
            pieces
                .iter()
                .enumerate()
                .filter(|(_, piece)| piece.keep < keep::ALWAYS)
                .min_by_key(|(_, piece)| piece.keep)
                .map(|(index, piece)| (piece.keep, index))
        };
        match (lowest(right.as_slice()), lowest(left.as_slice())) {
            (Some((on_right, index)), on_left)
                if on_left.is_none_or(|(on_left, _)| on_right <= on_left) =>
            {
                right.remove(index);
            }
            (_, Some((_, index))) => {
                left.remove(index);
            }
            (_, None) => return,
        }
    }
}

/// `pieces` as one line, `gap` between them.
fn join(pieces: Vec<Piece>, gap: &'static str) -> Line<'static> {
    let mut spans = Vec::with_capacity(pieces.len() * 2);
    for (index, piece) in pieces.into_iter().enumerate() {
        if index > 0 {
            spans.push(Span::from(gap));
        }
        spans.push(piece.span);
    }
    Line::from(spans)
}

/// The focused agent's model, reasoning setting and status.
fn status_pieces(
    shown: Option<(Option<&ModelChoice>, &Effort, &Status)>,
    model_hint: bool,
) -> Vec<Piece> {
    let Some((model, effort, status)) = shown else {
        return vec![Piece::new(keep::ALWAYS, Span::from("no agent").dim())];
    };
    let model = match model {
        Some(model) => model.0.clone(),
        None if model_hint => "no model: /model picks one".to_owned(),
        None => "no model".to_owned(),
    };
    let status = match status {
        Status::Idle => Span::from("idle").green(),
        Status::Thinking | Status::RunningTools | Status::Busy(_) => {
            Span::from(format!("{status}… (Esc stops)")).yellow()
        }
        Status::Retrying { attempt, seconds } => Span::from(format!(
            "retry {attempt}/{} in {seconds}s… (Esc stops)",
            RETRY.max_retries
        ))
        .red(),
    };
    vec![
        Piece::new(keep::ALWAYS, Span::from(model).bold()),
        Piece::new(
            keep::REASONING,
            Span::from(format!("reasoning {effort}")).dim(),
        ),
        Piece::new(keep::ALWAYS, status),
    ]
}

/// The meter: the agent's uncached input and output tokens, cache reads,
/// cost, then the context against the model's window, yellow past 70% and
/// red past 90%. `/usage` details the cache writes and reasoning.
fn usage_pieces(
    spent: Option<&Spending>,
    last: &LastUsage,
    connection: Option<&Connection>,
) -> Vec<Piece> {
    let Some(Spending(spent)) = spent.filter(|spent| spent.0.calls > 0) else {
        return Vec::new();
    };
    let mut pieces = vec![Piece::new(
        keep::TOKENS,
        Span::from(format!(
            "↑{} ↓{}",
            tokens_label(spent.uncached_input()),
            tokens_label(spent.tokens.output_tokens.unwrap_or(0))
        ))
        .dim(),
    )];
    if let Some(read) = spent.tokens.cached_input_tokens.filter(|read| *read > 0) {
        pieces.push(Piece::new(
            keep::CACHE,
            Span::from(format!("cache {}", tokens_label(read))).dim(),
        ));
    }
    if let Some(cost) = spent.cost_label() {
        pieces.push(Piece::new(keep::COST, Span::from(cost).dim()));
    }
    let window = connection.and_then(|connection| connection.spec.context_window);
    if let Some(context) = last.context().map(|tokens| ContextUse { tokens, window }) {
        let style = match context.percent() {
            Some(90..) => Style::new().red(),
            Some(70..) => Style::new().yellow(),
            _ => Style::new().dim(),
        };
        pieces.push(Piece::new(
            keep::CONTEXT,
            Span::styled(format!("ctx {context}"), style),
        ));
    }
    pieces
}

fn reload_span(reload: &ReloadStatus) -> Option<Span<'static>> {
    let text = match reload {
        ReloadStatus::Idle | ReloadStatus::Failed => return None,
        ReloadStatus::Queued { .. } => {
            "Reload queued: once no turn runs (/reload cancel)".to_owned()
        }
        ReloadStatus::Ready => "Reloading: restarting…".to_owned(),
        // The launcher's phase, then cargo's latest line.
        ReloadStatus::Building { latest } => format!(
            "Reloading: {} (Esc cancels)",
            latest.as_deref().unwrap_or("Resolving dependencies…")
        ),
    };
    Some(Span::from(text).cyan())
}

/// The picker in a centred box over the transcript.
fn draw_picker(frame: &mut Frame, picker: &Picker) {
    let popup = frame
        .area()
        .centered(Constraint::Ratio(4, 5), Constraint::Ratio(3, 4));
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
        .map(|item| item.label.clone())
        .collect();
    let mut state = ListState::default().with_selected(Some(picker.selected));
    frame.render_stateful_widget(
        List::new(items).highlight_style(Style::new().add_modifier(Modifier::REVERSED)),
        list,
        &mut state,
    );
}

/// Draws where the conversation was condensed: the messages above are sent
/// to the model as the summary under the line.
fn summary_lines(condensed: &Condensed, lines: &mut Vec<Line<'static>>) {
    let style = Style::new().fg(Color::Magenta);
    lines.push(Line::default());
    lines.push(Line::styled(
        format!(
            "── {} earlier messages are sent as this summary ──",
            condensed.upto
        ),
        style.bold(),
    ));
    lines.extend(excerpt(&condensed.summary, SUMMARY_LINES, style.dim()));
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

#[cfg(test)]
mod tests;
