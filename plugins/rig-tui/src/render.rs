//! Draws the focused agent: transcript, status line, input editor with its
//! completion list, and the picker or question shown.

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::time::{Duration, Instant};

use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use crossterm::cursor::MoveTo;
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
    CallDecorator, CallState, ContentRenderer, EntryRenderer, Looks, Renderers, ToolCallView,
    ToolRenderer, UNNAMED_HOLD,
};
use super::status::{AppStatus, Side, StatusItem, StatusItems, Tone};
use super::terminal::{Tui, TuiTestScreen};
use super::transcript::{
    Below, CallFacts, CallTree, ChildCalls, Inside, Look, Marks, Part, ShownNote, StreamingCalls,
    Transcript, live_call, notice_lines, plain_lines,
};
use super::view::{
    Continuation, DrawnTranscript, Highlight, ShownNotice, TranscriptItem, TranscriptSlot,
    TranscriptView, TuiView,
};
use super::wrap::wrap_all;
use rig_core::message::{CallId, ToolCall};
use rig_ecs::agent::{ActiveTurn, AgentId, Calls, Conversation, Partial};
use rig_ecs::inbox::{DeliveryMode, Incoming};
use rig_ecs::stage::{Effect, ForAgent, Holds, Outcome, Settled, Started, SupersededBy};
use rig_ecs::tools::ToolRequest;
use rig_ecs::turn::{ModelRequest, SentTo};
use rig_harness::note::Note;
use rig_harness::prelude::{CallOutput, ToolProgress};
use rig_harness::primary::Shown;

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
            Changed<ToolRenderer>,
            Changed<EntryRenderer>,
            Changed<ContentRenderer>,
            Changed<CallDecorator>,
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
        RemovedLooks,
    ),
    mut paced: Local<Paced>,
) -> bool {
    // Every reader is drained, so none redraws again for the same change.
    let requested = requests.read().count() > 0;
    let removed = removed.0.read().count()
        + removed.1.read().count()
        + removed.2.read().count()
        + removed.3.read().count()
        > 0
        || removed.4.any();
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

/// The looks taken out since the last frame.
#[derive(SystemParam)]
pub(crate) struct RemovedLooks<'w, 's> {
    tools: RemovedComponents<'w, 's, ToolRenderer>,
    entries: RemovedComponents<'w, 's, EntryRenderer>,
    contents: RemovedComponents<'w, 's, ContentRenderer>,
    decorators: RemovedComponents<'w, 's, CallDecorator>,
}

impl RemovedLooks<'_, '_> {
    /// Whether any was, each reader drained.
    fn any(&mut self) -> bool {
        self.tools.read().count()
            + self.entries.read().count()
            + self.contents.read().count()
            + self.decorators.read().count()
            > 0
    }
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
    screens: (Option<Res<Tui>>, Option<Res<TuiTestScreen>>),
    (input, view): (Res<TuiInput>, Res<TuiView>),
    mut frame: ResMut<FrameLayout>,
    mut panels: Query<(Entity, &TuiPanel, &mut PanelCanvas)>,
) {
    let size = match screens {
        (Some(tui), _) => tui.terminal.size().ok(),
        (None, Some(test)) => Some(test.size()),
        (None, None) => None,
    };
    let Some(size) = size else {
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

/// An agent, as its transcript draws it.
pub(crate) struct ShownAgent<'a> {
    pub(crate) entity: Entity,
    pub(crate) id: &'a AgentId,
    pub(crate) conversation: &'a Conversation,
    /// Whether its conversation changed since the last frame.
    pub(crate) changed: bool,
    /// The messages it accepted and has not sent yet, in the order they
    /// came.
    pub(crate) inbox: Vec<&'a Incoming>,
    /// Where its running turn's calls stand.
    pub(crate) calls: BTreeMap<CallId, CallState>,
    /// The output so far of its running calls that report it.
    pub(crate) outputs: BTreeMap<CallId, String>,
    /// The reply streaming in, if any.
    pub(crate) partial: Option<&'a Partial>,
    /// The tool calls it streams.
    pub(crate) streaming: Vec<ToolCall>,
    /// The notes in its conversation, in the order put.
    pub(crate) notes: Vec<ShownNote<'a>>,
    /// The calls under each of its tool calls that started any.
    pub(crate) children: ChildCalls,
}

/// An effect not settled yet, a tool or model call or any other, as
/// [`TranscriptLayout`] reads it for [`ToolCallView::children`].
type LiveEffect = (
    Entity,
    &'static Effect,
    (
        Option<&'static ToolRequest>,
        Option<&'static ModelRequest>,
        Option<&'static SentTo>,
    ),
    (Option<&'static ForAgent>, Option<&'static Outcome>),
    (Has<Started>, Has<Partial>),
);

/// What the transcript is drawn from besides the agent.
pub(crate) struct Sources<'a> {
    /// The transcript as plugins steer it: what is expanded, the notices.
    pub(crate) view: &'a TranscriptView,
    /// The looks plugins added.
    pub(crate) renderers: &'a Renderers<'a>,
    /// What settled calls published and how long they ran.
    pub(crate) facts: &'a CallFacts,
    /// Whether every item is drawn whole, as if expanded.
    pub(crate) whole: bool,
    /// Whether to note which items are drawn cut.
    pub(crate) note_cut: bool,
}

/// Lays out an agent's transcript as the terminal view draws it, without
/// drawing it: for any agent, shown or not, in a run with the terminal or
/// without, by the view's own layout (its order, its renderers, wrapping
/// and the matches it lights), so an export, a copy, a search or a screen
/// reader reads what the view shows. Each layout renders and wraps the
/// whole transcript, so lay one out when asked, not every frame. It is
/// there in any run with the [`TuiPlugin`](crate::TuiPlugin), and takes
/// the [`TranscriptView`] to lay out with, so a system that changes the
/// view lays out with it too.
///
/// ```no_run
/// use rig_harness::prelude::*;
/// use rig_tui::{Expanded, TranscriptLayout, TranscriptView};
///
/// /// `/export`: the agent's transcript as the view draws it, every item
/// /// whole, a line per row.
/// fn export(In(args): In<CommandArgs>, layout: TranscriptLayout, view: Res<TranscriptView>) {
///     let width = view.width().unwrap_or(100);
///     let Some(laid_out) = layout.lay_out(&view, args.agent, width, Expanded::All) else {
///         return;
///     };
///     let rows: Vec<&str> = laid_out.rows().iter().map(|row| row.text.as_str()).collect();
///     std::fs::write("transcript.txt", rows.join("\n")).ok();
/// }
/// ```
#[derive(SystemParam)]
pub struct TranscriptLayout<'w, 's> {
    agents: Query<'w, 's, (&'static AgentId, &'static Conversation)>,
    pub(crate) active: Query<'w, 's, &'static ActiveTurn>,
    turns: Query<'w, 's, &'static Calls>,
    /// The replies streaming in that nothing superseded, such as a race's
    /// running members.
    partials: Query<'w, 's, &'static Partial, Without<SupersededBy>>,
    accepted: Query<
        'w,
        's,
        (&'static Effect, &'static Incoming, &'static ForAgent),
        (With<Started>, Without<Outcome>),
    >,
    pub(crate) looks: Looks<'w, 's>,
    calls: CallStates<'w, 's>,
    live: Query<'w, 's, LiveEffect, Without<Settled>>,
    names: Query<'w, 's, &'static Name>,
    facts: Res<'w, CallFacts>,
    streaming: Res<'w, StreamingCalls>,
}

/// Which items a [`TranscriptLayout`] lays out whole.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Expanded {
    /// Those the view draws whole, its [`TranscriptView::expanded`]; the
    /// rest are cut as their look says.
    #[default]
    AsShown,
    /// Every item, as the view draws it once expanded.
    All,
}

impl TranscriptLayout<'_, '_> {
    /// `agent`'s transcript laid out in rows `width` columns wide (as
    /// [`TranscriptView::width`] says the view draws it; `u16::MAX` keeps
    /// a line in one row), its items drawn whole as `expanded` says and
    /// the matches of `view`'s highlights counted, as the view would draw
    /// it with `view`. It says which items are drawn cut, and has no rows
    /// on screen. [`DrawnTranscript::item`] takes one item's rows out of
    /// it. `None` when `agent` is no agent.
    pub fn lay_out(
        &self,
        view: &TranscriptView,
        agent: Entity,
        width: u16,
        expanded: Expanded,
    ) -> Option<DrawnTranscript> {
        let shown = self.agent(agent, true)?;
        let renderers = self.looks.by_kind();
        let sources = Sources {
            view,
            renderers: &renderers,
            facts: &self.facts,
            whole: expanded == Expanded::All,
            note_cut: true,
        };
        let mut transcript = Transcript::default();
        let parts = lay_out_transcript(&mut transcript, &sources, Some(&shown), width);
        Some(transcript.laid_out(Some(agent), &parts, &view.highlights, 0..0))
    }

    /// The agent `entity` as its transcript draws it, its conversation
    /// `changed` since the last frame or not; `None` when it is no agent.
    pub(crate) fn agent(&self, entity: Entity, changed: bool) -> Option<ShownAgent<'_>> {
        let (id, conversation) = self.agents.get(entity).ok()?;
        let turn = self.active.get(entity).ok();
        let calls = turn.and_then(|turn| self.turns.get(turn.turn()).ok());
        // The newest of the turn's replies streaming in that nothing
        // superseded: in a race, the member still running.
        let partial = calls.and_then(|calls| {
            let streams = calls.iter();
            let mut streams =
                streams.filter_map(|call| Some((call, self.partials.get(call).ok()?)));
            streams.next_back()
        });
        let streaming = partial.map(|(call, _)| self.streaming.of(call));
        let streaming = streaming.unwrap_or_default().iter();
        let mut inbox: Vec<(&Effect, &Incoming)> = self
            .accepted
            .iter()
            .filter(|(.., to)| to.0 == entity)
            .map(|(effect, message, _)| (effect, message))
            .collect();
        inbox.sort_by_key(|(effect, _)| effect.id());
        Some(ShownAgent {
            entity,
            id,
            conversation,
            changed,
            inbox: inbox.into_iter().map(|(_, message)| message).collect(),
            calls: self.calls.of(calls.into_iter().flat_map(Calls::iter)),
            outputs: self.calls.outputs(calls.into_iter().flat_map(Calls::iter)),
            partial: partial.map(|(_, partial)| partial),
            streaming: streaming.map(|(_, call)| call.clone()).collect(),
            notes: conversation
                .asides()
                .filter_map(|(id, aside)| {
                    let note = Note::of(aside)?;
                    Some(ShownNote { id, aside, note })
                })
                .collect(),
            children: self.facts.children(&self.live_calls(), &id.0),
        })
    }

    /// The effects not settled yet, by the effects that started them, the
    /// calls standing as they stand now.
    fn live_calls(&self) -> CallTree {
        let mut live = CallTree::default();
        for (entity, effect, requests, (agent, outcome), (started, streams)) in &self.live {
            let state = match requests.0 {
                Some(request) => self.calls.of([entity]).remove(&request.call.id),
                None if outcome.is_some() => Some(CallState::Done),
                None if streams => Some(CallState::Streaming),
                None if started => Some(CallState::Running(None)),
                None => None,
            };
            let agent = agent.map(|ForAgent(agent)| *agent);
            let name = agent.and_then(|agent| self.names.get(agent).ok());
            let state = state.unwrap_or(CallState::Waiting);
            let child = live_call(requests, (name, outcome), state);
            live.add(effect.parent, effect.id(), child, None);
            let id = agent.and_then(|agent| self.agents.get(agent).ok());
            if let (Some(request), Some((id, _))) = (requests.0, id) {
                live.tool_call(effect.id(), &id.0, request.call.id.clone());
            }
        }
        live
    }

    /// What settled calls published and how long they ran.
    pub(crate) fn facts(&self) -> &CallFacts {
        &self.facts
    }
}

/// Brings `transcript` up to date with the agent shown and returns its
/// parts at `width`: each message, with the notices written inside it,
/// after the notes and notices written before it; then those after the
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
        parts.push(Part::rows(&extra, &marks, Vec::new(), wrap_width));
        return parts;
    };
    let (conversation, count) = (shown.conversation, shown.conversation.messages().len());
    // Notes and notices before the first message, after each message,
    // and notices inside one.
    let mut after: Vec<Vec<(usize, &ShownNotice)>> = vec![Vec::new(); count + 1];
    let mut notes_after: Vec<Vec<&ShownNote<'_>>> = vec![Vec::new(); count + 1];
    let mut look = Look {
        renderers: sources.renderers,
        calls: &shown.calls,
        outputs: &shown.outputs,
        inside: HashMap::new(),
        expanded: &view.expanded,
        whole: sources.whole,
        note_cut: sources.note_cut,
        facts: sources.facts,
        children: &shown.children,
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
    for note in &shown.notes {
        let before = match TranscriptSlot::of_aside(note.aside, conversation) {
            TranscriptSlot::Top => 0,
            TranscriptSlot::After(index) => index + 1,
            TranscriptSlot::Inside(index, item) => {
                let inside = look.inside.entry(index).or_default();
                inside.push((item, Inside::Note(note)));
                continue;
            }
        };
        if let Some(at) = notes_after.get_mut(before) {
            at.push(note);
        }
    }
    transcript.update(shown.entity, conversation, shown.changed, &look, width);
    let mut after = after.into_iter().zip(notes_after);
    for index in 0..=count {
        let (notices, notes) = after.next().unwrap_or_default();
        for note in notes {
            let item = note.item();
            let (drawn, cut) = look.drawn(look.is_expanded(&item), |expanded| {
                sources.renderers.entry(&note.view(expanded))
            });
            let mut lines = vec![Line::default()];
            lines.extend(drawn);
            let marks = [(item.clone(), 0..lines.len())];
            let cut = if cut { vec![item] } else { Vec::new() };
            parts.push(Part::rows(&lines, &marks, cut, wrap_width));
        }
        let (mut extra, mut marks) = (Vec::new(), Marks::new());
        for notice in notices {
            notice_lines(notice, &mut extra, &mut marks);
        }
        if index == count {
            if let Some(partial) = shown.partial {
                extra.extend(plain_lines(&partial.reasoning, Style::new().dim().italic()));
                if !partial.text.is_empty() {
                    extra.push(Line::default());
                    extra.extend(markdown::render(&partial.text));
                }
            }
            for call in &shown.streaming {
                let view = ToolCallView::new(call, None, CallState::Streaming);
                extra.extend(sources.renderers.call(&view));
            }
            inbox_lines(&shown.inbox, &mut extra);
            parts.push(Part::rows(&extra, &marks, Vec::new(), wrap_width));
            break;
        }
        if !extra.is_empty() {
            parts.push(Part::rows(&extra, &marks, Vec::new(), wrap_width));
        }
        parts.push(Part::Message(index));
    }
    parts
}

/// `rows`, the first being row `first` of the transcript, with the rows of
/// the `selected` item marked.
pub(crate) fn decorate(
    rows: Vec<Line<'static>>,
    first: usize,
    marks: &Marks,
    selected: Option<&TranscriptItem>,
) -> Vec<Line<'static>> {
    let selected = selected.and_then(|item| marks.iter().find(|(marked, _)| marked == item));
    rows.into_iter()
        .enumerate()
        .map(|(index, row)| match selected {
            Some((_, rows)) if rows.contains(&(first + index)) => {
                row.patch_style(Style::new().on_dark_gray())
            }
            _ => row,
        })
        .collect()
}

/// The highlights to light, and where the spans of those with any are:
/// by the highlight's index and the first row of their line in the
/// transcript.
#[derive(Default)]
pub(crate) struct Lights<'a> {
    pub(crate) highlights: &'a [Highlight],
    pub(crate) placed: HashMap<(usize, usize), Vec<Range<usize>>>,
}

impl Lights<'_> {
    /// Whether any of them lights anything.
    pub(crate) fn any(&self) -> bool {
        let lights =
            |highlight: &Highlight| !highlight.text.is_empty() || !highlight.spans.is_empty();
        self.highlights.iter().any(lights)
    }
}

/// `rows`, the first being row `first` of the transcript, each continuing
/// the one before it as `continues` says, with each match of `lights` in
/// its style. A match is sought in each line the rows wrap, so one a wrap
/// splits lights on both rows.
pub(crate) fn light(
    rows: &[Line<'static>],
    continues: &[Option<Continuation>],
    first: usize,
    lights: &Lights<'_>,
) -> Vec<Line<'static>> {
    let joints = continues.iter().map(Option::as_ref);
    let lit_rows = lit(rows.iter().zip(joints), first, lights);
    rows.iter()
        .zip(lit_rows)
        .map(|(line, row)| {
            if !row.cells.iter().any(|(.., lit)| *lit) {
                return line.clone();
            }
            let mut spans: Vec<Span<'static>> = Vec::new();
            for (character, style, _) in row.cells {
                match spans.last_mut() {
                    Some(span) if span.style == style => span.content.to_mut().push(character),
                    _ => spans.push(Span::styled(character.to_string(), style)),
                }
            }
            let mut lit = Line::from(spans).style(line.style);
            lit.alignment = line.alignment;
            lit
        })
        .collect()
}

/// A row once highlights are lit: its characters, each with its style and
/// whether a match lit it, and how many matches of each highlight start
/// in it.
pub(crate) struct LitRow {
    pub(crate) cells: Vec<(char, Style, bool)>,
    pub(crate) matches: Vec<usize>,
}

/// A character of a line being searched: in which row it is drawn, at
/// which character of it (`None` for a blank the wrap dropped), and
/// whether a match lit it.
struct Slot {
    row: usize,
    at: Option<usize>,
    character: char,
    lit: bool,
}

/// `rows`, the first being row `first` of the transcript, each with how it
/// continues the row before it, with the matches of `lights` lit. A match
/// is sought in the line the rows wrap, so it may span rows; it counts in
/// the row it starts in. A match over a character an earlier match lit is
/// no match.
pub(crate) fn lit<'a, 'line: 'a>(
    rows: impl IntoIterator<Item = (&'a Line<'line>, Option<&'a Continuation>)>,
    first: usize,
    lights: &Lights<'_>,
) -> Vec<LitRow> {
    let mut lit = Vec::new();
    // The line being gathered, and its first row in the transcript.
    let mut line = Vec::new();
    let mut start = first;
    for (row, joint) in rows {
        if joint.is_none() {
            light_line(&mut lit, &mut line, start, lights);
            line.clear();
            start = first + lit.len();
        }
        let at = lit.len();
        let cells: Vec<(char, Style, bool)> = row
            .spans
            .iter()
            .flat_map(|span| span.content.chars().map(move |c| (c, span.style, false)))
            .collect();
        let indent = joint.map_or(0, |joint| joint.indent);
        let gap = joint.map(|joint| joint.gap.chars()).into_iter().flatten();
        line.extend(gap.map(|character| Slot {
            row: at,
            at: None,
            character,
            lit: false,
        }));
        let drawn = cells.iter().enumerate().skip(indent);
        line.extend(drawn.map(|(index, (character, ..))| Slot {
            row: at,
            at: Some(index),
            character: *character,
            lit: false,
        }));
        lit.push(LitRow {
            cells,
            matches: vec![0; lights.highlights.len()],
        });
    }
    light_line(&mut lit, &mut line, start, lights);
    lit
}

/// Lights the matches of `lights` in `line`, which starts at row `start`
/// of the transcript, in the `rows` it is drawn in.
fn light_line(rows: &mut [LitRow], line: &mut [Slot], start: usize, lights: &Lights<'_>) {
    for (index, highlight) in lights.highlights.iter().enumerate() {
        if !highlight.spans.is_empty() {
            let spans = lights.placed.get(&(index, start));
            for bytes in spans.into_iter().flatten() {
                if let Some(window) = slots_of(line, bytes) {
                    light_match(rows, line, window, (index, highlight.style));
                }
            }
            continue;
        }
        let needle: Vec<char> = highlight.text.chars().collect();
        let same = |a: char, b: char| {
            if highlight.match_case {
                a == b
            } else {
                a.to_lowercase().eq(b.to_lowercase())
            }
        };
        let mut at = 0;
        while !needle.is_empty() && at + needle.len() <= line.len() {
            let window = at..at + needle.len();
            let text = line.get(window.clone()).unwrap_or_default();
            let found = text
                .iter()
                .zip(&needle)
                .all(|(slot, n)| same(slot.character, *n));
            if found && light_match(rows, line, window, (index, highlight.style)) {
                at += needle.len();
            } else {
                at += 1;
            }
        }
    }
}

/// The characters of `line` that `bytes` of its text are; `None` for none,
/// or for bytes that cut a character or reach past the line.
fn slots_of(line: &[Slot], bytes: &Range<usize>) -> Option<Range<usize>> {
    let mut at = 0;
    let (mut from, mut to) = (None, None);
    for (index, slot) in line.iter().enumerate() {
        if at == bytes.start {
            from = Some(index);
        }
        if at == bytes.end {
            to = Some(index);
        }
        at += slot.character.len_utf8();
    }
    if at == bytes.end {
        to = Some(line.len());
    }
    Some(from?..to?).filter(|window| !window.is_empty())
}

/// Lights the characters `window` of `line` as a match of highlight
/// `index` in `style`, counted in the row it starts in, unless an earlier
/// match lit one of them; says whether it did.
fn light_match(
    rows: &mut [LitRow],
    line: &mut [Slot],
    window: Range<usize>,
    (index, style): (usize, Style),
) -> bool {
    let window = line.get_mut(window).unwrap_or_default();
    if window.iter().any(|slot| slot.lit) {
        return false;
    }
    let start = window.first().map(|slot| slot.row);
    let count = start.and_then(|row| rows.get_mut(row)?.matches.get_mut(index));
    if let Some(count) = count {
        *count += 1;
    }
    for slot in window {
        slot.lit = true;
        let row = rows.get_mut(slot.row);
        let cell = slot.at.and_then(|at| row?.cells.get_mut(at));
        if let Some((_, style_there, lit)) = cell {
            *style_there = style_there.patch(style);
            *lit = true;
        }
    }
    true
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
    (mut tui, mut test): (Option<ResMut<Tui>>, Option<ResMut<TuiTestScreen>>),
    mut drew: ResMut<TuiFrame>,
    (mut view, mut shown_view): (ResMut<TuiView>, ResMut<TranscriptView>),
    (typed, focus, choices, shown_agent): (Res<TuiInput>, Res<KeyFocus>, Choices, Res<Shown>),
    mut frame_layout: ResMut<FrameLayout>,
    (mut transcript, mut drawn): (Local<Transcript>, Option<ResMut<DrawnTranscript>>),
    (from, changed): (TranscriptLayout, Query<(), Changed<Conversation>>),
    panels: Query<(Entity, &TuiPanel, &PanelCanvas)>,
    status: (Query<&StatusItems>, Option<Res<AppStatus>>),
    mut removed_looks: RemovedLooks,
) -> Result {
    let frame_layout = std::mem::take(&mut *frame_layout);
    let Some(input_rows) = frame_layout.input_rows else {
        return Ok(());
    };
    // Placing the scroll is drawing's own bookkeeping, not a change to
    // redraw for; so is noting where it is, after drawing.
    let view = view.bypass_change_detection();
    let transcript_view = shown_view.bypass_change_detection();
    let mut noted = None;
    let shown_agent = shown_agent.0;
    let turn = shown_agent.and_then(|agent| from.active.get(agent).ok());
    let shown = shown_agent.and_then(|entity| from.agent(entity, changed.contains(entity)));
    // The panels at the sides, then the boxes over the screen.
    let mut panels: Vec<_> = panels.iter().collect();
    panels.sort_by_key(|(entity, panel, _)| {
        (matches!(panel.placement, Placement::Over { .. }), *entity)
    });
    // A renderer changed: what it drew is drawn again.
    if removed_looks.any() || from.looks.changed() {
        transcript.clear();
    }
    let renderers = from.looks.by_kind();
    let edit = view
        .choosing
        .as_ref()
        .filter(|choosing| choosing.draft.is_some())
        .and_then(|choosing| choices.shown(choosing.entity));
    let inline = view.inline.first().and_then(|&ask| choices.shown(ask));
    // The cursor is shown at the input once the frame is out, when the
    // input has the keys.
    let mut cursor = None;
    let mut areas = TuiFrame::default();
    let draw_frame = |frame: &mut Frame| {
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
            renderers: &renderers,
            facts: from.facts(),
            whole: false,
            note_cut: drawn.is_some(),
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
            let on_screen = visible.first..visible.first + visible.rows.len();
            drawn.set_if_neq(transcript.laid_out(agent, &parts, highlights, on_screen));
        }
        let rows = decorate(
            visible.rows,
            visible.first,
            &visible.marks,
            transcript_view.selected.as_ref(),
        );
        frame.render_widget(Paragraph::new(rows), transcript_area);
        draw_below(frame, visible.below, transcript_area);
        view.items = visible.marks.into_iter().map(|(item, _)| item).collect();
        noted = Some((visible.top, transcript_area.width));
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
        let mut body = None;
        if let Some(edit) = &edit {
            block = block.title(Line::from(format!(" {} ", edit.title)).bold());
            body = draw_body(frame, edit.body, input);
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
        let mut overlays = Vec::new();
        for (_, panel, canvas) in &panels {
            canvas.copy_to(frame.buffer_mut());
            if matches!(panel.placement, Placement::Over { .. }) {
                overlays.push(canvas.area());
            }
        }
        overlays.extend(body);
        let shown = view
            .choosing
            .as_mut()
            .filter(|choosing| choosing.draft.is_none())
            .and_then(|choosing| Some((choices.shown(choosing.entity)?, choosing)));
        match shown {
            Some((shown, choosing)) => overlays.push(draw_choice(frame, choosing, &shown)),
            None if focus.0.is_some() => {}
            None => {
                if let Some(completion) = &view.completion {
                    overlays.extend(draw_completion(frame, completion, input));
                }
                let column = u16::try_from(layout.cursor_column).unwrap_or(u16::MAX);
                let row = u16::try_from(layout.cursor_row - input_top).unwrap_or(0);
                cursor = Some(MoveTo(
                    input.x.saturating_add(1).saturating_add(column),
                    input.y.saturating_add(1).saturating_add(row),
                ));
            }
        }
        areas = TuiFrame {
            screen,
            transcript: transcript_area,
            status: status_area,
            strip,
            input,
            overlays,
        };
    };
    let drawn = match (tui.as_deref_mut(), test.as_deref_mut()) {
        (Some(tui), _) => {
            tui.begin_frame()?;
            let drawn = tui.terminal.draw(draw_frame).map(|_| ());
            tui.end_frame(cursor.filter(|_| drawn.is_ok()))?;
            drawn
        }
        (None, Some(test)) => {
            test.draw(draw_frame);
            Ok(())
        }
        (None, None) => Ok(()),
    };
    if let Some((top, width)) = noted {
        TranscriptView::note_drawn(&mut shown_view, top, width);
    }
    drew.set_if_neq(areas);
    drawn?;
    Ok(())
}

/// Where the last frame drew each part of the screen: for a plugin that
/// writes to the terminal itself after the frame ([`TerminalWrite`]),
/// such as an image over the rows its content renderer left blank,
/// which finds those rows in the [`DrawnTranscript`] (see
/// [`TuiFrame::screen_row`]) and leaves out what an overlay covers. The
/// view writes it after each frame it draws, in
/// [`TuiSystems::Render`](crate::TuiSystems::Render); read it after that.
/// All empty in a run without the terminal.
///
/// [`TerminalWrite`]: crate::TerminalWrite
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct TuiFrame {
    /// The whole screen.
    pub screen: Rect,
    /// The transcript.
    pub transcript: Rect,
    /// The status line.
    pub status: Rect,
    /// The strip above the input of a question shown beside the work;
    /// no rows without one.
    pub strip: Rect,
    /// The input box.
    pub input: Rect,
    /// What was drawn over the rest: the panels placed over the screen,
    /// the picker or question, the completion list, a question's text
    /// above the input.
    pub overlays: Vec<Rect>,
}

impl TuiFrame {
    /// The screen row the `row` of `drawn` (the transcript this frame
    /// drew) is on, when it is on screen.
    pub fn screen_row(&self, drawn: &DrawnTranscript, row: usize) -> Option<u16> {
        let on_screen = drawn.on_screen();
        if !on_screen.contains(&row) {
            return None;
        }
        let below_top = u16::try_from(row - on_screen.start).ok()?;
        Some(self.transcript.y.saturating_add(below_top))
    }

    /// Whether an overlay covers any of `area`.
    pub fn covered(&self, area: Rect) -> bool {
        self.overlays.iter().any(|over| over.intersects(area))
    }
}

/// What a question answered in the input is about, in a box just above
/// the input box; where, if it is drawn.
fn draw_body(frame: &mut Frame, body: &str, input: Rect) -> Option<Rect> {
    if body.trim().is_empty() {
        return None;
    }
    let lines = wrap_all(
        &plain_lines(body, Style::new()),
        usize::from(input.width.saturating_sub(2).max(1)),
    );
    let rows = u16::try_from(lines.len()).unwrap_or(u16::MAX);
    let height = rows.saturating_add(2).min(input.y);
    if height < 3 {
        return None;
    }
    let area = Rect::new(input.x, input.y - height, input.width, height);
    frame.render_widget(Clear, area);
    frame.render_widget(Paragraph::new(lines).block(Block::bordered().dim()), area);
    Some(area)
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

/// The completion list, just above the input box; where, if it fit.
fn draw_completion(frame: &mut Frame, completion: &Completion, input: Rect) -> Option<Rect> {
    let rows = completion.items.len().min(COMPLETION_ROWS);
    let height = u16::try_from(rows + 2).unwrap_or(3).min(input.y);
    if height < 3 {
        return None;
    }
    let area = Rect::new(input.x, input.y - height, input.width, height);
    frame.render_widget(Clear, area);
    let marker = completion
        .kind
        .marker()
        .map(String::from)
        .unwrap_or_default();
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
    Some(area)
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
/// agent's and its turn's, by `order`, then name, with `no agent` where an
/// agent's model would be when none is shown.
fn status_pieces(shown: bool, items: [Option<&StatusItems>; 3]) -> (Vec<Piece>, Vec<Piece>) {
    let no_agent = StatusItem::new("agent", Side::Left, 30, u8::MAX).says("no agent", Tone::Dim);
    let mut items: Vec<&StatusItem> = items
        .into_iter()
        .flatten()
        .flat_map(StatusItems::iter)
        .collect();
    if !shown {
        items.push(&no_agent);
    }
    items.sort_by_key(|item| (item.order, item.name));
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

/// The rows a picker's list keeps, at least, when a long body takes the
/// rest of its box.
const LEAST_LIST: usize = 3;

/// The picker or question in a centred box over the transcript: its body,
/// scrolled to `choosing.scroll` (kept within it), what is typed, then the
/// rows it leaves, each with its detail at the right and, in a choice of
/// several, its tick box; for free text, the typed answer.
fn draw_choice(frame: &mut Frame, choosing: &mut Choosing, shown: &Listing<'_>) -> Rect {
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
    let inner = Block::bordered().inner(popup);
    // The body takes what the typed line, a gap and the list's least
    // leave, and scrolls when that is not all of it.
    let list_least = rows.len().clamp(1, LEAST_LIST);
    let room = usize::from(inner.height).saturating_sub(2 + list_least);
    let body_height = body.len().min(room);
    choosing.scroll = choosing.scroll.min(body.len() - body_height);
    let mut keys = vec![if shown.free_text {
        "type to filter or answer"
    } else {
        "type to filter"
    }];
    if body_height < body.len() {
        keys.push("Shift+↑↓ scrolls");
    }
    if shown.multiple {
        keys.push("Tab ticks");
    }
    let block = Block::bordered().title(format!(
        " {} · {}, Enter picks, Esc closes ",
        shown.title,
        keys.join(", ")
    ));
    frame.render_widget(block, popup);
    let [body_area, _, typed, list] = Layout::vertical([
        Constraint::Length(u16::try_from(body_height).unwrap_or(u16::MAX)),
        Constraint::Length(u16::from(body_height > 0)),
        Constraint::Length(1),
        Constraint::Min(1),
    ])
    .areas(inner);
    let scroll = u16::try_from(choosing.scroll).unwrap_or(u16::MAX);
    frame.render_widget(Paragraph::new(body).scroll((scroll, 0)), body_area);
    frame.render_widget(Line::from(format!("› {}▏", choosing.filter)), typed);
    let width = usize::from(list.width);
    let items: Vec<Line<'static>> = rows
        .iter()
        .map(|row| match row {
            Row::Item(index) => {
                let Some(item) = shown.items.get(*index) else {
                    return Line::default();
                };
                if !shown.multiple {
                    return item.line(width);
                }
                let tick = match choosing.ticked.contains(index) {
                    true => "[x] ",
                    false => "[ ] ",
                };
                let mut line = item.line(width.saturating_sub(tick.len()));
                line.spans.insert(0, Span::from(tick));
                line
            }
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
    popup
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
        let text = message.text();
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
