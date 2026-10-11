//! Draws the focused agent: transcript, status line, input editor with its
//! completion list, and the picker or question shown.

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
use unicode_width::UnicodeWidthChar;

use super::complete::Completion;
use super::editor::{Layout as InputLayout, TuiInput};
use super::keys::KeyFocus;
use super::markdown;
use super::panel::{self, PanelCanvas, Placement, RequestRedraw, TuiPanel};
use super::pick::{Choices, Choosing, Listing, Row};
use super::renderers::{ToolRenderer, excerpt};
use super::status::{AppStatus, Side, StatusItem, StatusItems, Tone};
use super::terminal::Tui;
use super::transcript::{Below, Part, Renderers, Transcript, plain_lines};
use super::view::{ShownNotice, TuiView};
use super::wrap::wrap_all;
use rig_ecs::agent::{ActiveTurn, Calls, Condensed, Conversation, NoticeLevel, Partial};
use rig_ecs::inbox::{DeliveryMode, Incoming};
use rig_ecs::stage::{Effect, ForAgent, Outcome, Started};
use rig_harness::primary::Shown;

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
/// key, a notice, a resize, a picker or question), the input and who has
/// the keys, an agent's drawn components, a turn's calls, a
/// streaming reply, a status line item (a retry's wait counts down there),
/// a panel, or a plugin's [`RequestRedraw`]. A turn's end changes its
/// conversation or comes with a notice. A streaming reply alone draws at
/// most every [`STREAM_FRAME`]; the text that waits is drawn by one of the
/// frames the kernel runs after the wake that brought it (see
/// [`Wake`](rig_ecs::calls::Wake)), 16 ms apart, under any loop.
pub(crate) fn needs_redraw(
    view: Res<TuiView>,
    (input, focus, shown): (Res<TuiInput>, Res<KeyFocus>, Res<Shown>),
    changed: Query<
        (),
        Or<(
            Changed<Conversation>,
            Changed<ActiveTurn>,
            Changed<Condensed>,
            Changed<Calls>,
            Changed<StatusItems>,
            Changed<TuiPanel>,
        )>,
    >,
    partials: Query<(), Changed<Partial>>,
    messages: Query<(), (With<Incoming>, Or<(Added<Started>, Added<Outcome>)>)>,
    app_status: Option<Res<AppStatus>>,
    mut requests: MessageReader<RequestRedraw>,
    mut removed: (RemovedComponents<TuiPanel>, RemovedComponents<StatusItems>),
    mut paced: Local<Paced>,
) -> bool {
    // Every reader is drained, so none redraws again for the same change.
    let requested = requests.read().count() > 0;
    let removed = removed.0.read().count() + removed.1.read().count() > 0;
    let changed = requested
        || removed
        || view.is_changed()
        || shown.is_changed()
        || input.is_changed()
        || focus.is_changed()
        || app_status.is_some_and(|status| status.is_changed())
        || !changed.is_empty()
        || !messages.is_empty();
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
    input: Res<TuiInput>,
    mut frame: ResMut<FrameLayout>,
    mut panels: Query<(Entity, &TuiPanel, &mut PanelCanvas)>,
) {
    let Ok(size) = tui.terminal.size() else {
        return;
    };
    let screen = Rect::new(0, 0, size.width, size.height);
    let rows = input.layout(size.width.saturating_sub(2));
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

/// The transcript's parts after the shown agent's messages are laid out:
/// each message, after the notices that came before it and the summary
/// sent in place of the messages above; then the notices after the last message, the
/// reply streaming in and the messages the agent accepted and has not
/// sent yet. Without an agent, the app's notices.
fn transcript_parts(
    view: &TuiView,
    agent: Option<Entity>,
    shown: Option<(&Conversation, Option<&Condensed>, &[&Incoming])>,
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
        .filter(|notice| notice.is_for(agent))
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

/// The hint under the input box.
fn input_hint(turn_running: bool, empty: bool, panel_keys: bool) -> &'static str {
    match (panel_keys, turn_running, empty) {
        (true, ..) => " A panel has the keys · Esc gives them back ",
        (false, true, _) => " Enter steers this turn · Tab sends after it · Esc stops ",
        (false, false, true) => {
            " Enter sends · Shift+Enter or Ctrl+J new line · Ctrl+V image · / commands · @ files "
        }
        (false, false, false) => "",
    }
}

/// Draws the frame [`layout`] laid out, with what the plugins drew in
/// their panels. The transcript's rows are kept between frames in a
/// [`Transcript`], so only changed messages are laid out again.
pub(crate) fn render(
    mut tui: ResMut<Tui>,
    mut view: ResMut<TuiView>,
    (typed, focus, choices, shown_agent): (Res<TuiInput>, Res<KeyFocus>, Choices, Res<Shown>),
    mut frame_layout: ResMut<FrameLayout>,
    mut transcript: Local<Transcript>,
    agents: Query<(&Conversation, Option<&Condensed>)>,
    changed: Query<(), Changed<Conversation>>,
    (turns, partials, active, accepted): (
        Query<&Calls>,
        Query<&Partial>,
        Query<&ActiveTurn>,
        Query<(&Effect, &Incoming, &ForAgent), (With<Started>, Without<Outcome>)>,
    ),
    panels: Query<(Entity, &TuiPanel, &PanelCanvas)>,
    status: (Query<&StatusItems>, Option<Res<AppStatus>>),
    renderers: Query<Ref<ToolRenderer>>,
    mut removed_renderers: RemovedComponents<ToolRenderer>,
) -> Result {
    let frame_layout = std::mem::take(&mut *frame_layout);
    let Some(input_rows) = frame_layout.input_rows else {
        return Ok(());
    };
    // Placing the scroll is drawing's own bookkeeping, not a change to
    // redraw for.
    let view = view.bypass_change_detection();
    let shown_agent = shown_agent.0;
    let shown = shown_agent.and_then(|agent| Some((agent, agents.get(agent).ok()?)));
    let turn = shown_agent.and_then(|agent| active.get(agent).ok());
    let calls = turn.and_then(|turn| turns.get(turn.turn()).ok());
    let partial = calls.and_then(|calls| calls.iter().find_map(|call| partials.get(call).ok()));
    let mut inbox: Vec<(&Effect, &Incoming)> = accepted
        .iter()
        .filter(|(.., to)| Some(to.0) == shown_agent)
        .map(|(effect, message, _)| (effect, message))
        .collect();
    inbox.sort_by_key(|(effect, _)| effect.id());
    let inbox: Vec<&Incoming> = inbox.into_iter().map(|(_, message)| message).collect();
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
            if let Some((agent, (conversation, _))) = shown {
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
                shown_agent,
                shown.map(|(_, (conversation, condensed))| (conversation, condensed, &inbox[..])),
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
            let agent = shown_agent.and_then(|agent| status.0.get(agent).ok());
            let turn = turn.and_then(|turn| status.0.get(turn.turn()).ok());
            let app = status.1.as_deref().map(|app| &app.0);
            let (mut left, mut right) = status_pieces(shown_agent.is_some(), [app, agent, turn]);
            fit(&mut left, &mut right, usize::from(status_area.width));
            let (line, usage) = (join(left, LEFT_GAP), join(right, RIGHT_GAP));
            let usage_width = u16::try_from(usage.width()).unwrap_or(u16::MAX);
            let [status, meter] =
                Layout::horizontal([Constraint::Min(0), Constraint::Length(usage_width)])
                    .areas(status_area);
            frame.render_widget(line, status);
            frame.render_widget(usage, meter);
            let hint = input_hint(turn.is_some(), typed.is_empty(), focus.0.is_some());
            frame.render_widget(
                Paragraph::new(layout.rows)
                    .scroll((u16::try_from(input_top).unwrap_or(u16::MAX), 0))
                    .block(Block::bordered().title_bottom(Line::from(hint).dim().right_aligned())),
                input,
            );
            for (.., canvas) in &panels {
                canvas.copy_to(frame.buffer_mut());
            }
            let shown = view
                .choosing
                .as_ref()
                .and_then(|choosing| Some((choosing, choices.shown(choosing.entity)?)));
            match shown {
                Some((choosing, shown)) => draw_choice(frame, choosing, &shown),
                None if focus.0.is_some() => {}
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
    let marker = completion.kind.marker();
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

/// A piece of the status line, from a [`StatusItem`]. When the line does
/// not fit, [`fit`] cuts or drops the pieces with the lowest `keep` first.
struct Piece {
    keep: u8,
    /// How narrow it may be cut; 0 for never.
    min: usize,
    span: Span<'static>,
}

impl Piece {
    fn new(keep: u8, span: Span<'static>) -> Self {
        Self { keep, min: 0, span }
    }

    /// The same piece, cut down to no fewer than `min` columns before it
    /// goes.
    fn shrinking_to(mut self, min: usize) -> Self {
        self.min = min;
        self
    }

    fn width(&self) -> usize {
        self.span.width()
    }

    /// Whether it can still be cut.
    fn can_shrink(&self) -> bool {
        self.min > 0 && self.width() > self.min
    }

    /// Cuts it by `over` columns, to no fewer than its `min`, ending in `…`.
    fn shrink(&mut self, over: usize) {
        let target = self.width().saturating_sub(over).max(self.min);
        let mut text = String::new();
        let mut width = 0;
        for character in self.span.content.chars() {
            let next = width + character.width().unwrap_or(0);
            if next + 1 > target {
                break;
            }
            text.push(character);
            width = next;
        }
        text.push('…');
        self.span.content = text.into();
    }
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

/// Cuts, then drops, the pieces of `left` and of the meter on the `right`
/// with the lowest `keep` until both fit `width` with a gap between them:
/// one that may shrink is cut to its `min` before it goes. What keeps
/// `u8::MAX` stays, cut where it may; past that the line is cut at the
/// edge.
fn fit(left: &mut Vec<Piece>, right: &mut Vec<Piece>, width: usize) {
    loop {
        let separator = if right.is_empty() { 0 } else { LEFT_GAP.len() };
        let used = joined_width(left, LEFT_GAP) + separator + joined_width(right, RIGHT_GAP);
        let Some(over) = used.checked_sub(width).filter(|over| *over > 0) else {
            return;
        };
        let lowest = |pieces: &[Piece]| {
            pieces
                .iter()
                .enumerate()
                .filter(|(_, piece)| piece.keep < u8::MAX || piece.can_shrink())
                .min_by_key(|(_, piece)| piece.keep)
                .map(|(index, piece)| (piece.keep, index))
        };
        let (side, index) = match (lowest(right.as_slice()), lowest(left.as_slice())) {
            (Some((on_right, index)), on_left)
                if on_left.is_none_or(|(on_left, _)| on_right <= on_left) =>
            {
                (&mut *right, index)
            }
            (_, Some((_, index))) => (&mut *left, index),
            (_, None) => return,
        };
        match side.get_mut(index) {
            Some(piece) if piece.can_shrink() => piece.shrink(over),
            _ => {
                side.remove(index);
            }
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

/// The status line's pieces, left and right: the app's items, the shown
/// agent's and its turn's, by `order`, with `no agent` where an agent's
/// model would be when none is shown.
fn status_pieces(shown: bool, items: [Option<&StatusItems>; 3]) -> (Vec<Piece>, Vec<Piece>) {
    let no_agent = StatusItem::at(Side::Left, 30, u8::MAX).says("no agent", Tone::Dim);
    let mut items: Vec<&StatusItem> = items
        .into_iter()
        .flatten()
        .flat_map(StatusItems::iter)
        .collect();
    if !shown {
        items.push(&no_agent);
    }
    items.sort_by_key(|item| item.order);
    let piece = |item: &StatusItem| {
        let style = match item.tone {
            Tone::Plain => Style::new(),
            Tone::Bold => Style::new().bold(),
            Tone::Dim => Style::new().dim(),
            Tone::Green => Style::new().green(),
            Tone::Yellow => Style::new().yellow(),
            Tone::Red => Style::new().red(),
            Tone::Cyan => Style::new().cyan(),
            Tone::Magenta => Style::new().magenta(),
        };
        Piece::new(item.keep, Span::styled(item.text.clone(), style))
            .shrinking_to(usize::from(item.min_width))
    };
    let (left, right): (Vec<_>, Vec<_>) =
        items.into_iter().partition(|item| item.side == Side::Left);
    (
        left.into_iter().map(piece).collect(),
        right.into_iter().map(piece).collect(),
    )
}

/// The picker or question in a centred box over the transcript: its body,
/// what is typed, then the rows it leaves, and for free text, the typed
/// answer.
fn draw_choice(frame: &mut Frame, choosing: &Choosing, shown: &Listing<'_>) {
    let area = frame.area();
    let width = area.width.saturating_mul(4) / 5;
    let body = wrap_all(
        &plain_lines(shown.body, Style::new()),
        usize::from(width.saturating_sub(2).max(1)),
    );
    let rows = shown.rows(&choosing.filter);
    // Room for the body, the typed line and the rows, up to 3/4 of the
    // screen.
    let wanted = body.len() + usize::from(!body.is_empty()) + 1 + rows.len() + 2;
    let height = u16::try_from(wanted)
        .unwrap_or(u16::MAX)
        .min(area.height.saturating_mul(3) / 4)
        .max(5);
    let popup = area.centered(Constraint::Length(width), Constraint::Length(height));
    frame.render_widget(Clear, popup);
    let action = if shown.free_text {
        "type to filter or answer"
    } else {
        "type to filter"
    };
    let block = Block::bordered().title(format!(
        " {} · {action}, Enter picks, Esc closes ",
        shown.title
    ));
    let inner = block.inner(popup);
    frame.render_widget(block, popup);
    let body_height = u16::try_from(body.len())
        .unwrap_or(u16::MAX)
        .min(inner.height.saturating_sub(2) / 2);
    let gap = u16::from(body_height > 0);
    let [body_area, _, typed, list] = Layout::vertical([
        Constraint::Length(body_height),
        Constraint::Length(gap),
        Constraint::Length(1),
        Constraint::Min(1),
    ])
    .areas(inner);
    frame.render_widget(Paragraph::new(body), body_area);
    frame.render_widget(Line::from(format!("› {}▏", choosing.filter)), typed);
    let items: Vec<Line<'static>> = rows
        .iter()
        .map(|row| match row {
            Row::Item(index) => Line::from(
                shown
                    .labels
                    .get(*index)
                    .copied()
                    .unwrap_or_default()
                    .to_owned(),
            ),
            Row::Typed if choosing.filter.trim().is_empty() => {
                Line::from("✎ or type an answer").dim()
            }
            Row::Typed => Line::from(format!("✎ {}", choosing.filter.trim())),
        })
        .collect();
    let mut state = ListState::default().with_selected(Some(choosing.selected));
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

/// The messages the agent accepted and has not sent yet, under the
/// transcript, in the order they came.
fn inbox_lines(inbox: &[&Incoming], lines: &mut Vec<Line<'static>>) {
    if inbox.is_empty() {
        return;
    }
    lines.push(Line::default());
    for message in inbox {
        let when = match message.mode {
            DeliveryMode::Steer => "steering",
            DeliveryMode::Queue => "after this turn",
            DeliveryMode::Note => "noted",
        };
        let text = &message.text;
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
