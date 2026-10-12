//! Draws the focused agent: transcript, status line, input editor with its
//! completion list, and the picker or question shown.

use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};

use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::terminal::{BeginSynchronizedUpdate, EndSynchronizedUpdate};
use crossterm::{execute, queue};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, List, ListState, Paragraph};
use unicode_width::UnicodeWidthChar;

use super::complete::Completion;
use super::editor::{Layout as InputLayout, TuiInput};
use super::keys::KeyFocus;
use super::markdown;
use super::panel::{self, MIN_MAIN, PanelCanvas, Placement, RequestRedraw, TuiPanel};
use super::pick::{Choices, Choosing, Listing, Row};
use super::renderers::{
    CallState, EntryRenderer, EntryView, ToolCallView, ToolRenderer, UNNAMED_HOLD,
};
use super::status::{AppStatus, Side, StatusItem, StatusItems, Tone};
use super::terminal::Tui;
use super::transcript::{
    Below, CallFacts, EntryRenderers, Inside, Look, Marks, Part, Renderers, StreamingCalls,
    Transcript, call_lines, entry_lines, notice_lines, plain_lines,
};
use super::view::{
    DrawnTranscript, Highlight, ShownNotice, TranscriptItem, TranscriptSlot, TranscriptView,
    TuiView,
};
use super::wrap::wrap_all;
use rig_core::message::{CallId, ToolCall};
use rig_ecs::agent::{ActiveTurn, AgentId, Calls, Conversation, Partial};
use rig_ecs::inbox::{DeliveryMode, Incoming};
use rig_ecs::stage::{Effect, ForAgent, Holds, Outcome, Started};
use rig_ecs::tools::ToolRequest;
use rig_harness::prelude::{CallOutput, ToolProgress};
use rig_harness::primary::Shown;
use rig_harness::transcript::{TranscriptEntries, TranscriptEntry};

/// Most lines the input box shows.
const INPUT_LINES: usize = 10;
/// Most items the completion list shows at once.
const COMPLETION_ROWS: usize = 8;

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
/// the keys, an agent's drawn components, a turn's calls and where a tool
/// call stands (held, waiting, started, its progress, answered), a
/// streaming reply, a status line item (a retry's wait counts down there),
/// a panel, or a plugin's [`RequestRedraw`]. A turn's end changes its
/// conversation or comes with a notice. A streaming reply alone draws at
/// most every [`STREAM_FRAME`]; the text that waits is drawn by one of the
/// frames the kernel runs after the wake that brought it (see
/// [`Wake`](rig_ecs::calls::Wake)), 16 ms apart, under any loop.
pub(crate) fn needs_redraw(
    (view, transcript, facts, streaming): (
        Res<TuiView>,
        Res<TranscriptView>,
        Res<CallFacts>,
        Res<StreamingCalls>,
    ),
    (input, focus, shown): (Res<TuiInput>, Res<KeyFocus>, Res<Shown>),
    changed: Query<
        (),
        Or<(
            Changed<Conversation>,
            Changed<ActiveTurn>,
            Changed<Calls>,
            Changed<StatusItems>,
            Changed<TuiPanel>,
            Changed<TranscriptEntry>,
            Changed<TranscriptEntries>,
            Changed<EntryRenderer>,
        )>,
    >,
    partials: Query<(), Changed<Partial>>,
    messages: Query<(), (With<Incoming>, Or<(Added<Started>, Added<Outcome>)>)>,
    calls: Query<
        (),
        (
            With<ToolRequest>,
            Or<(
                Added<Started>,
                Added<Outcome>,
                Changed<Holds>,
                Changed<ToolProgress>,
            )>,
        ),
    >,
    app_status: Option<Res<AppStatus>>,
    mut requests: MessageReader<RequestRedraw>,
    mut removed: (
        RemovedComponents<TuiPanel>,
        RemovedComponents<StatusItems>,
        RemovedComponents<Holds>,
        RemovedComponents<ToolProgress>,
        RemovedComponents<TranscriptEntry>,
        RemovedComponents<EntryRenderer>,
    ),
    mut paced: Local<Paced>,
) -> bool {
    // Every reader is drained, so none redraws again for the same change.
    let requested = requests.read().count() > 0;
    let removed = removed.0.read().count()
        + removed.1.read().count()
        + removed.2.read().count()
        + removed.3.read().count()
        + removed.4.read().count()
        + removed.5.read().count()
        > 0;
    let changed = requested
        || removed
        || view.is_changed()
        || transcript.is_changed()
        || facts.is_changed()
        || shown.is_changed()
        || input.is_changed()
        || focus.is_changed()
        || app_status.is_some_and(|status| status.is_changed())
        || !changed.is_empty()
        || !messages.is_empty()
        || !calls.is_empty();
    paced.waiting |= !partials.is_empty() || streaming.is_changed();
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
    strip: Rect,
    input: Rect,
}

/// Whether [`layout`] laid out a frame that is not drawn yet: the
/// condition of [`TuiSystems::Draw`](super::TuiSystems::Draw) and
/// [`render`].
pub(crate) fn frame_due(frame: Res<FrameLayout>) -> bool {
    frame.due
}

/// Lays out a frame: the input box grows with the input, up to a limit;
/// the status line sits above it, and above the input the strip of a
/// question shown beside the work; the panels below the input stack under
/// it; the other panels take their sides of the rest, in entity order, and
/// the transcript what they leave.
pub(crate) fn layout(
    tui: Res<Tui>,
    (input, view): (Res<TuiInput>, Res<TuiView>),
    mut frame: ResMut<FrameLayout>,
    mut panels: Query<(Entity, &TuiPanel, &mut PanelCanvas)>,
) {
    let Ok(size) = tui.terminal.size() else {
        return;
    };
    let screen = Rect::new(0, 0, size.width, size.height);
    let rows = input.layout(size.width.saturating_sub(2));
    let input_height = rows.rows.len().clamp(1, INPUT_LINES);
    let mut panels: Vec<_> = panels.iter_mut().collect();
    panels.sort_by_key(|(entity, ..)| *entity);
    let asked: Vec<u16> = panels
        .iter()
        .filter_map(|(_, panel, _)| match panel.placement {
            Placement::BelowInput(size) => Some(panel::rows_of(size, screen.height)),
            _ => None,
        })
        .collect();
    let strip = u16::from(!view.inline.is_empty());
    let [above, status, strip, input, below] = split(screen, input_height, strip, &asked);
    let transcript = panel::lay_out(
        panels.into_iter().map(|(_, panel, canvas)| (panel, canvas)),
        above,
        below,
        screen,
    );
    *frame = FrameLayout {
        due: true,
        input_rows: Some(rows),
        input_height,
        transcript,
        status,
        strip,
        input,
    };
}

/// The screen's rows, top to bottom: what is above the status line, the
/// status line, `strip` rows for a question beside the work, the input box
/// for `input_height` lines, and the panels below the input that asked
/// for `asked` rows and fit while the transcript keeps its least.
pub(crate) fn split(screen: Rect, input_height: usize, strip: u16, asked: &[u16]) -> [Rect; 5] {
    let input = u16::try_from(input_height + 2).unwrap_or(3);
    let fixed = 1 + strip + input;
    let room = screen
        .height
        .saturating_sub(fixed)
        .saturating_sub(MIN_MAIN.1);
    let below: u16 = panel::below_rows(asked.iter().copied(), room).iter().sum();
    Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(1),
        Constraint::Length(strip),
        Constraint::Length(input),
        Constraint::Length(below),
    ])
    .areas(screen)
}

/// What the transcript reads of where a turn's tool calls stand.
#[derive(SystemParam)]
pub(crate) struct CallStates<'w, 's> {
    calls: Query<
        'w,
        's,
        (
            &'static ToolRequest,
            Option<&'static Holds>,
            Has<Started>,
            Has<Outcome>,
            Option<&'static ToolProgress>,
        ),
    >,
    names: Query<'w, 's, &'static Name>,
    outputs: Query<
        'w,
        's,
        (&'static ToolRequest, &'static CallOutput),
        (With<Started>, Without<Outcome>),
    >,
}

impl CallStates<'_, '_> {
    /// Where each tool call of `calls` stands, by its id: held while it
    /// has holds, else done once answered, running (with its progress
    /// line) once started, else waiting.
    pub(crate) fn of(
        &self,
        calls: impl IntoIterator<Item = Entity>,
    ) -> BTreeMap<CallId, CallState> {
        let found = calls
            .into_iter()
            .filter_map(|call| self.calls.get(call).ok());
        found
            .map(|(request, holds, started, answered, progress)| {
                let state = match holds {
                    Some(holds) => {
                        CallState::Held(holds.iter().map(|hold| self.name(hold)).collect())
                    }
                    None if answered => CallState::Done,
                    None if started => {
                        CallState::Running(progress.map(|ToolProgress(line)| line.clone()))
                    }
                    None => CallState::Waiting,
                };
                (request.call.id.clone(), state)
            })
            .collect()
    }

    /// The output so far of each running call of `calls` whose tool
    /// reported some, by its id.
    pub(crate) fn outputs(
        &self,
        calls: impl IntoIterator<Item = Entity>,
    ) -> BTreeMap<CallId, String> {
        let found = calls
            .into_iter()
            .filter_map(|call| self.outputs.get(call).ok());
        let outputs = found.map(|(request, output)| (request.call.id.clone(), output.text()));
        outputs.filter(|(_, text)| !text.is_empty()).collect()
    }

    /// A hold's name, which says why it holds.
    fn name(&self, hold: Entity) -> String {
        let name = self.names.get(hold).map_or(UNNAMED_HOLD, Name::as_str);
        name.to_owned()
    }
}

/// The agent shown, as its transcript draws it.
pub(crate) struct ShownAgent<'a> {
    pub(crate) entity: Entity,
    pub(crate) id: &'a AgentId,
    pub(crate) conversation: &'a Conversation,
    /// Whether its conversation changed since the last frame.
    pub(crate) changed: bool,
    /// The messages it accepted and has not sent yet, in the order they
    /// came.
    pub(crate) inbox: &'a [&'a Incoming],
}

/// What the transcript is drawn from besides the agent's conversation.
pub(crate) struct Sources<'a> {
    /// The transcript as plugins steer it: what is expanded, the notices.
    pub(crate) view: &'a TranscriptView,
    /// The tool renderers by tool name.
    pub(crate) renderers: &'a Renderers<'a>,
    /// The entry renderers by origin.
    pub(crate) entry_renderers: &'a EntryRenderers<'a>,
    /// Where the running turn's calls stand.
    pub(crate) calls: &'a BTreeMap<CallId, CallState>,
    /// The output so far of its running calls that report it.
    pub(crate) outputs: &'a BTreeMap<CallId, String>,
    /// What settled calls published and how long they ran.
    pub(crate) facts: &'a CallFacts,
    /// The reply streaming in, if any.
    pub(crate) partial: Option<&'a Partial>,
    /// The tool calls it streams.
    pub(crate) streaming: &'a [ToolCall],
    /// The shown agent's entries, in order.
    pub(crate) entries: &'a [(Entity, &'a TranscriptEntry)],
}

/// Brings `transcript` up to date with the agent shown and returns its
/// parts at `width`: each message, with the notices written inside it,
/// after the entries and notices written before it; then those after the
/// last message, the reply streaming in with the tool calls it streams,
/// and the messages the agent accepted and has not sent yet. Without an
/// agent, the notices about every agent.
pub(crate) fn lay_out_transcript(
    transcript: &mut Transcript,
    sources: &Sources<'_>,
    shown: Option<&ShownAgent<'_>>,
    width: u16,
) -> Vec<Part> {
    let wrap_width = usize::from(width.max(1));
    let mut parts = Vec::new();
    let view = sources.view;
    let Some(shown) = shown else {
        let (mut extra, mut marks) = (Vec::new(), Marks::new());
        let notices = view.notices().iter().enumerate();
        for notice in notices.filter(|(_, notice)| notice.is_for_every_agent()) {
            notice_lines(notice, &mut extra, &mut marks);
        }
        parts.push(Part::rows(&extra, &marks, wrap_width));
        return parts;
    };
    let (conversation, count) = (shown.conversation, shown.conversation.messages().len());
    // Entries and notices before the first message, after each message,
    // and notices inside one.
    let mut after: Vec<Vec<(usize, &ShownNotice)>> = vec![Vec::new(); count + 1];
    let mut entries_after: Vec<Vec<(Entity, &TranscriptEntry)>> = vec![Vec::new(); count + 1];
    let mut look = Look {
        renderers: sources.renderers,
        entries: sources.entry_renderers,
        calls: sources.calls,
        outputs: sources.outputs,
        inside: HashMap::new(),
        expanded: &view.expanded,
        facts: sources.facts,
    };
    for (at, notice) in view.notices().iter().enumerate() {
        let before = match notice.slot(shown.id, conversation) {
            None => continue,
            Some(TranscriptSlot::Top) => 0,
            Some(TranscriptSlot::After(index)) => index + 1,
            Some(TranscriptSlot::Inside(index, item)) => {
                let inside = look.inside.entry(index).or_default();
                inside.push((item, Inside::Notice(at, notice)));
                continue;
            }
        };
        if let Some(after) = after.get_mut(before) {
            after.push((at, notice));
        }
    }
    for &(entity, entry) in sources.entries {
        let before = match TranscriptSlot::of_entry(entry, conversation) {
            TranscriptSlot::Top => 0,
            TranscriptSlot::After(index) => index + 1,
            TranscriptSlot::Inside(index, item) => {
                let inside = look.inside.entry(index).or_default();
                inside.push((item, Inside::Entry(entity, entry)));
                continue;
            }
        };
        if let Some(at) = entries_after.get_mut(before) {
            at.push((entity, entry));
        }
    }
    transcript.update(shown.entity, conversation, shown.changed, &look, width);
    let mut after = after.into_iter().zip(entries_after);
    for index in 0..=count {
        let (notices, entries) = after.next().unwrap_or_default();
        for (entity, entry) in entries {
            let item = TranscriptItem::Entry(entity);
            let entry = EntryView {
                origin: &entry.origin,
                title: &entry.title,
                text: &entry.text,
                expanded: view.expanded.contains(&item),
            };
            let mut lines = vec![Line::default()];
            lines.extend(entry_lines(&entry, sources.entry_renderers));
            let marks = [(item, 0..lines.len())];
            parts.push(Part::rows(&lines, &marks, wrap_width));
        }
        let (mut extra, mut marks) = (Vec::new(), Marks::new());
        for notice in notices {
            notice_lines(notice, &mut extra, &mut marks);
        }
        if index == count {
            if let Some(partial) = sources.partial {
                extra.extend(plain_lines(&partial.reasoning, Style::new().dim().italic()));
                if !partial.text.is_empty() {
                    extra.push(Line::default());
                    extra.extend(markdown::render(&partial.text));
                }
            }
            for call in sources.streaming {
                let view = ToolCallView::new(call, None, CallState::Streaming);
                extra.extend(call_lines(&view, sources.renderers));
            }
            inbox_lines(shown.inbox, &mut extra);
            parts.push(Part::rows(&extra, &marks, wrap_width));
            break;
        }
        if !extra.is_empty() {
            parts.push(Part::rows(&extra, &marks, wrap_width));
        }
        parts.push(Part::Message(index));
    }
    parts
}

/// `rows`, the first being row `first` of the transcript, with the rows of
/// the `selected` item marked and the `highlights` lit.
pub(crate) fn decorate(
    rows: Vec<Line<'static>>,
    first: usize,
    marks: &Marks,
    (selected, highlights): (Option<&TranscriptItem>, &[Highlight]),
) -> Vec<Line<'static>> {
    let selected = selected.and_then(|item| marks.iter().find(|(marked, _)| marked == item));
    rows.into_iter()
        .enumerate()
        .map(|(index, row)| {
            let row = light(row, highlights);
            match selected {
                Some((_, rows)) if rows.contains(&(first + index)) => {
                    row.patch_style(Style::new().on_dark_gray())
                }
                _ => row,
            }
        })
        .collect()
}

/// `line` with each match of `highlights` in its style.
fn light(line: Line<'static>, highlights: &[Highlight]) -> Line<'static> {
    if highlights.iter().all(|highlight| highlight.text.is_empty()) {
        return line;
    }
    let (cells, matches) = lit(&line, highlights);
    if matches.iter().all(|matches| *matches == 0) {
        return line;
    }
    let mut spans: Vec<Span<'static>> = Vec::new();
    for (character, style, _) in cells {
        match spans.last_mut() {
            Some(span) if span.style == style => span.content.to_mut().push(character),
            _ => spans.push(Span::styled(character.to_string(), style)),
        }
    }
    let mut lit = Line::from(spans).style(line.style);
    lit.alignment = line.alignment;
    lit
}

/// The characters of `line`, each with its style once `highlights` are
/// lit and whether one lit it, and how many matches of each highlight lit
/// them. A match over a character an earlier highlight lit is no match.
pub(crate) fn lit(
    line: &Line<'_>,
    highlights: &[Highlight],
) -> (Vec<(char, Style, bool)>, Vec<usize>) {
    let mut cells: Vec<(char, Style, bool)> = line
        .spans
        .iter()
        .flat_map(|span| span.content.chars().map(move |c| (c, span.style, false)))
        .collect();
    let mut matches = vec![0; highlights.len()];
    for (highlight, matches) in highlights.iter().zip(&mut matches) {
        let needle: Vec<char> = highlight.text.chars().collect();
        let same = |a: char, b: char| {
            if highlight.match_case {
                a == b
            } else {
                a.to_lowercase().eq(b.to_lowercase())
            }
        };
        let mut at = 0;
        while !needle.is_empty() && at + needle.len() <= cells.len() {
            let window = cells.get(at..at + needle.len()).unwrap_or_default();
            let free = window.iter().all(|(.., lit)| !lit);
            if free && window.iter().zip(&needle).all(|((c, ..), n)| same(*c, *n)) {
                for (_, style, lit) in cells.iter_mut().skip(at).take(needle.len()) {
                    *style = style.patch(highlight.style);
                    *lit = true;
                }
                *matches += 1;
                at += needle.len();
            } else {
                at += 1;
            }
        }
    }
    (cells, matches)
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
    (mut view, mut transcript_view): (ResMut<TuiView>, ResMut<TranscriptView>),
    (typed, focus, choices, shown_agent): (Res<TuiInput>, Res<KeyFocus>, Choices, Res<Shown>),
    mut frame_layout: ResMut<FrameLayout>,
    (mut transcript, mut drawn): (Local<Transcript>, Option<ResMut<DrawnTranscript>>),
    (agents, changed, agent_entries, entries): (
        Query<(&AgentId, &Conversation)>,
        Query<(), Changed<Conversation>>,
        Query<&TranscriptEntries>,
        Query<&TranscriptEntry>,
    ),
    (turns, partials, active, accepted): (
        Query<&Calls>,
        Query<&Partial>,
        Query<&ActiveTurn>,
        Query<(&Effect, &Incoming, &ForAgent), (With<Started>, Without<Outcome>)>,
    ),
    panels: Query<(Entity, &TuiPanel, &PanelCanvas)>,
    status: (Query<&StatusItems>, Option<Res<AppStatus>>),
    (renderers, entry_renderers, call_states, facts, streaming): (
        Query<Ref<ToolRenderer>>,
        Query<Ref<EntryRenderer>>,
        CallStates,
        Res<CallFacts>,
        Res<StreamingCalls>,
    ),
    mut removed_renderers: (
        RemovedComponents<ToolRenderer>,
        RemovedComponents<EntryRenderer>,
    ),
) -> Result {
    let frame_layout = std::mem::take(&mut *frame_layout);
    let Some(input_rows) = frame_layout.input_rows else {
        return Ok(());
    };
    // Placing the scroll is drawing's own bookkeeping, not a change to
    // redraw for; so is noting where it is.
    let view = view.bypass_change_detection();
    let transcript_view = transcript_view.bypass_change_detection();
    let shown_agent = shown_agent.0;
    let turn = shown_agent.and_then(|agent| active.get(agent).ok());
    let calls = turn.and_then(|turn| turns.get(turn.turn()).ok());
    let partial = calls.and_then(|calls| {
        let mut streams = calls.iter();
        streams.find_map(|call| Some((call, partials.get(call).ok()?)))
    });
    let streamed: Vec<ToolCall> = partial
        .map(|(call, _)| {
            streaming
                .of(call)
                .iter()
                .map(|(_, call)| call.clone())
                .collect()
        })
        .unwrap_or_default();
    let partial = partial.map(|(_, partial)| partial);
    let call_outputs = call_states.outputs(calls.into_iter().flat_map(Calls::iter));
    let call_states = call_states.of(calls.into_iter().flat_map(Calls::iter));
    let shown_entries: Vec<(Entity, &TranscriptEntry)> = shown_agent
        .and_then(|agent| agent_entries.get(agent).ok())
        .into_iter()
        .flat_map(TranscriptEntries::iter)
        .filter_map(|entry| Some((entry, entries.get(entry).ok()?)))
        .collect();
    let mut inbox: Vec<(&Effect, &Incoming)> = accepted
        .iter()
        .filter(|(.., to)| Some(to.0) == shown_agent)
        .map(|(effect, message, _)| (effect, message))
        .collect();
    inbox.sort_by_key(|(effect, _)| effect.id());
    let inbox: Vec<&Incoming> = inbox.into_iter().map(|(_, message)| message).collect();
    let shown = shown_agent.and_then(|entity| {
        let (id, conversation) = agents.get(entity).ok()?;
        Some(ShownAgent {
            entity,
            id,
            conversation,
            changed: changed.contains(entity),
            inbox: &inbox,
        })
    });
    // The panels at the sides, then the boxes over the screen.
    let mut panels: Vec<_> = panels.iter().collect();
    panels.sort_by_key(|(entity, panel, _)| {
        (matches!(panel.placement, Placement::Over { .. }), *entity)
    });
    // A renderer changed: what it drew is drawn again.
    if removed_renderers.0.read().count() + removed_renderers.1.read().count() > 0
        || renderers.iter().any(|renderer| renderer.is_changed())
        || entry_renderers.iter().any(|renderer| renderer.is_changed())
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
    let by_origin: EntryRenderers<'_> = entry_renderers
        .iter()
        .map(|renderer| {
            let renderer = renderer.into_inner();
            (renderer.origin.as_str(), &renderer.render)
        })
        .collect();
    let edit = view
        .choosing
        .as_ref()
        .filter(|choosing| choosing.draft.is_some())
        .and_then(|choosing| choices.shown(choosing.entity));
    let inline = view.inline.first().and_then(|&ask| choices.shown(ask));
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
            let sources = Sources {
                view: transcript_view,
                renderers: &by_tool,
                entry_renderers: &by_origin,
                calls: &call_states,
                outputs: &call_outputs,
                facts: &facts,
                partial,
                streaming: &streamed,
                entries: &shown_entries,
            };
            let parts = lay_out_transcript(
                &mut transcript,
                &sources,
                shown.as_ref(),
                transcript_area.width,
            );
            let visible = transcript.visible(
                &parts,
                (transcript_area.width, usize::from(transcript_area.height)),
                &mut view.scroll,
                &transcript_view.highlights,
            );
            if let Some(drawn) = &mut drawn {
                let (agent, highlights) = (shown.as_ref(), &transcript_view.highlights);
                let agent = agent.map(|shown| shown.entity);
                drawn.set_if_neq(transcript.drawn(agent, &parts, &visible, highlights));
            }
            let rows = decorate(
                visible.rows,
                visible.first,
                &visible.marks,
                (
                    transcript_view.selected.as_ref(),
                    &transcript_view.highlights,
                ),
            );
            frame.render_widget(Paragraph::new(rows), transcript_area);
            draw_below(frame, visible.below, transcript_area);
            view.items = visible.marks.into_iter().map(|(item, _)| item).collect();
            transcript_view.set_top(visible.top);
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
            let hint = match &edit {
                Some(_) => " Enter answers · Shift+Enter or Ctrl+J new line · Esc refuses ",
                None => input_hint(turn.is_some(), typed.is_empty(), focus.0.is_some()),
            };
            let mut block = Block::bordered().title_bottom(Line::from(hint).dim().right_aligned());
            if let Some(edit) = &edit {
                block = block.title(Line::from(format!(" {} ", edit.title)).bold());
                draw_body(frame, edit.body, input);
            }
            frame.render_widget(
                Paragraph::new(layout.rows)
                    .scroll((u16::try_from(input_top).unwrap_or(u16::MAX), 0))
                    .block(block),
                input,
            );
            let strip = frame_layout.strip.intersection(screen);
            if let Some(inline) = &inline {
                frame.render_widget(inline_line(inline), strip);
            }
            for (.., canvas) in &panels {
                canvas.copy_to(frame.buffer_mut());
            }
            let shown = view
                .choosing
                .as_ref()
                .filter(|choosing| choosing.draft.is_none())
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

/// What a question answered in the input is about, in a box just above
/// the input box.
fn draw_body(frame: &mut Frame, body: &str, input: Rect) {
    if body.trim().is_empty() {
        return;
    }
    let lines = wrap_all(
        &plain_lines(body, Style::new()),
        usize::from(input.width.saturating_sub(2).max(1)),
    );
    let rows = u16::try_from(lines.len()).unwrap_or(u16::MAX);
    let height = rows.saturating_add(2).min(input.y);
    if height < 3 {
        return;
    }
    let area = Rect::new(input.x, input.y - height, input.width, height);
    frame.render_widget(Clear, area);
    frame.render_widget(Paragraph::new(lines).block(Block::bordered().dim()), area);
}

/// The strip of a question shown beside the work: its title and the keys
/// that answer it.
pub(crate) fn inline_line(ask: &Listing<'_>) -> Line<'static> {
    let mut spans = vec![
        Span::from("? ").yellow().bold(),
        Span::from(ask.title.to_owned()).bold(),
        Span::from(" "),
    ];
    let typed = ask.free_text.then_some("Answer…");
    let labels = ask.items.iter().map(|item| item.label).chain(typed);
    for (index, label) in labels.take(9).enumerate() {
        spans.push(Span::from(format!(" [Alt+{} {label}]", index + 1)).cyan());
    }
    spans.push(Span::from(" [Alt+0 Dismiss]").dim());
    Line::from(spans)
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
/// what is typed, then the rows it leaves, each with its detail at the
/// right, and for free text, the typed answer.
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
    let width = usize::from(list.width);
    let items: Vec<Line<'static>> = rows
        .iter()
        .map(|row| match row {
            Row::Item(index) => shown
                .items
                .get(*index)
                .map(|item| item.line(width))
                .unwrap_or_default(),
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

#[cfg(test)]
mod tests;
