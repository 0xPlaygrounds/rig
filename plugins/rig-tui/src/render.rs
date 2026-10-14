//! Draws the focused agent: transcript, status line, input editor with its
//! completion list, and the picker or question shown.

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::time::{Duration, Instant};

use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use crossterm::cursor::MoveTo;
use ratatui::Frame;
use ratatui::backend::Backend;
use ratatui::buffer::{Buffer, Cell};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, List, ListState, Paragraph};

use super::complete::Completion;
use super::editor::{Caret, Editor};
use super::input::TuiInputState;
use super::keys::KeyFocus;
use super::pane::{PaneState, TranscriptPane};
use super::panel::{self, PanelCanvas, Placement, Repaint, RequestRedraw, TuiPanel};
use super::pick::{Choices, Choosing, InlineStrip, Listing, Row};
use super::renderers::{
    AsideRenderer, CallDecorator, CallState, ContentRenderer, EntryRenderer, Looks,
    MessageDecorator, MessageRenderer, MessageView, Renderers, Role, ToolCallView, ToolRenderer,
    UNNAMED_HOLD,
};
use super::status::{AppStatus, StatusItems, StatusLine};
use super::terminal::{Tui, TuiTestScreen};
use super::transcript::{
    CallFacts, CallTree, ChildCalls, Inside, Look, Marks, Part, ShownAside, StreamingCalls,
    Transcript, live_call, notice_lines, plain_lines,
};
use super::view::{
    Continuation, DrawnTranscript, Highlight, MarkIn, Notices, ShownNotice, TranscriptItem,
    TranscriptSlot, TranscriptView, TuiView,
};
use super::wrap::wrap_all;
use rig_core::conversation;
use rig_core::message::{CallId, ToolCall};
use rig_ecs::agent::{ActiveTurn, AgentId, Calls, Conversation, Partial};
use rig_ecs::inbox::{DeliveryMode, Incoming};
use rig_ecs::stage::{Effect, ForAgent, Holds, Outcome, Settled, Started, SupersededBy};
use rig_ecs::tools::ToolRequest;
use rig_ecs::turn::{ModelRequest, SentTo};
use rig_harness::prelude::{CallOutput, ToolProgress};
use rig_harness::primary::Shown;

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
    (view, notices, facts, streaming): (
        Res<TuiView>,
        Res<Notices>,
        Res<CallFacts>,
        Res<StreamingCalls>,
    ),
    (focus, shown): (Res<KeyFocus>, Res<Shown>),
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
            Changed<AsideRenderer>,
            Changed<MessageRenderer>,
            Changed<MessageDecorator>,
        )>,
    >,
    widgets: Query<
        (),
        Or<(
            Changed<TranscriptPane>,
            Changed<TranscriptView>,
            Changed<PaneState>,
            Changed<Editor>,
            Changed<StatusLine>,
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
    (app_status, test): (Option<Res<AppStatus>>, Option<Res<TuiTestScreen>>),
    (mut requests, mut repaints): (MessageReader<RequestRedraw>, MessageReader<Repaint>),
    mut removed: (
        RemovedComponents<TuiPanel>,
        RemovedComponents<StatusItems>,
        RemovedComponents<Holds>,
        RemovedComponents<ToolProgress>,
        (RemovedComponents<TranscriptPane>, RemovedComponents<Editor>),
        RemovedLooks,
    ),
    mut paced: Local<Paced>,
) -> bool {
    // Every reader is drained, so none redraws again for the same change.
    let requested = requests.read().count() + repaints.read().count() > 0;
    let removed = removed.0.read().count()
        + removed.1.read().count()
        + removed.2.read().count()
        + removed.3.read().count()
        + removed.4.0.read().count()
        + removed.4.1.read().count()
        > 0
        || removed.5.any();
    let changed = requested
        || removed
        || view.is_changed()
        || notices.is_changed()
        || facts.is_changed()
        || shown.is_changed()
        || !widgets.is_empty()
        || focus.is_changed()
        || app_status.is_some_and(|status| status.is_changed())
        || test.is_some_and(|test| test.is_changed())
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

/// The looks taken out since the last frame: [`lay_out_panes`] draws
/// what they drew again.
///
/// [`lay_out_panes`]: crate::pane::lay_out_panes
#[derive(SystemParam)]
pub(crate) struct RemovedLooks<'w, 's> {
    tools: RemovedComponents<'w, 's, ToolRenderer>,
    entries: RemovedComponents<'w, 's, EntryRenderer>,
    contents: RemovedComponents<'w, 's, ContentRenderer>,
    asides: RemovedComponents<'w, 's, AsideRenderer>,
    messages: RemovedComponents<'w, 's, MessageRenderer>,
    decorators: RemovedComponents<'w, 's, CallDecorator>,
    message_decorators: RemovedComponents<'w, 's, MessageDecorator>,
}

impl RemovedLooks<'_, '_> {
    /// Whether any was, each reader drained.
    pub(crate) fn any(&mut self) -> bool {
        self.tools.read().count()
            + self.entries.read().count()
            + self.contents.read().count()
            + self.asides.read().count()
            + self.messages.read().count()
            + self.decorators.read().count()
            + self.message_decorators.read().count()
            > 0
    }
}

/// Whether [`layout`] laid out a frame that [`render`] did not draw yet.
#[derive(Resource, Default)]
pub(crate) struct FrameLayout {
    due: bool,
}

/// Whether [`layout`] laid out a frame that is not drawn yet: the
/// condition of [`TuiSystems::Draw`](super::TuiSystems::Draw) and
/// [`render`].
pub(crate) fn frame_due(frame: Res<FrameLayout>) -> bool {
    frame.due
}

/// Lays out a frame: every panel by its placement, in the order spawned
/// (the default screen's first), an editor as tall as its text and a strip
/// as tall as what it shows.
pub(crate) fn layout(
    screens: (Option<Res<Tui>>, Option<Res<TuiTestScreen>>),
    view: Res<TuiView>,
    mut frame: ResMut<FrameLayout>,
    mut panels: Query<(
        Entity,
        &TuiPanel,
        &mut PanelCanvas,
        Option<&Editor>,
        Has<InlineStrip>,
    )>,
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
    let inline = u16::from(!view.inline.is_empty());
    let mut panels: Vec<_> = panels.iter_mut().collect();
    panels.sort_by_key(|(entity, ..)| panel::spawned(*entity));
    let panels = panels.into_iter().map(|(_, panel, canvas, editor, strip)| {
        let fits = editor.map(|editor| editor.fit_rows(screen.width));
        (panel, fits.or(strip.then_some(inline)), canvas)
    });
    panel::lay_out(panels, screen);
    frame.due = true;
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
    /// Its id; `None` for a conversation no agent of this app holds,
    /// which shows no notices.
    pub(crate) id: Option<&'a AgentId>,
    pub(crate) conversation: &'a conversation::Conversation,
    /// Whether its conversation changed since the last frame.
    pub(crate) changed: bool,
    /// The messages it accepted and has not sent yet, in the order they
    /// came.
    pub(crate) inbox: Vec<&'a Incoming>,
    /// Where its running turn's calls stand.
    pub(crate) calls: BTreeMap<CallId, CallState>,
    /// The output so far of its running calls that report it.
    pub(crate) outputs: BTreeMap<CallId, String>,
    /// The reply streaming in, if any, and the call streaming it.
    pub(crate) partial: Option<(Entity, &'a Partial)>,
    /// The tool calls it streams.
    pub(crate) streaming: Vec<ToolCall>,
    /// The asides in its conversation, in the order put.
    pub(crate) asides: Vec<ShownAside<'a>>,
    /// The calls under each of its tool calls that started any.
    pub(crate) children: ChildCalls,
}

impl<'a> ShownAgent<'a> {
    /// `conversation`, which no agent of this app holds, drawn as
    /// `entity`'s: no live parts, no notices.
    pub(crate) fn value(conversation: &'a conversation::Conversation, entity: Entity) -> Self {
        Self {
            entity,
            id: None,
            conversation,
            changed: true,
            inbox: Vec::new(),
            calls: BTreeMap::new(),
            outputs: BTreeMap::new(),
            partial: None,
            streaming: Vec::new(),
            asides: asides(conversation),
            children: ChildCalls::default(),
        }
    }
}

/// The asides of `conversation`, in the order put.
fn asides(conversation: &conversation::Conversation) -> Vec<ShownAside<'_>> {
    let asides = conversation.asides();
    asides.map(|(id, aside)| ShownAside { id, aside }).collect()
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
    /// The pane's view as plugins steer it: what is expanded and marked.
    pub(crate) view: &'a TranscriptView,
    /// The notices shown.
    pub(crate) notices: &'a [ShownNotice],
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
    notices: Res<'w, Notices>,
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
        let renderers = self.looks.by_kind(None);
        let sources = Sources {
            view,
            notices: self.notices(),
            renderers: &renderers,
            facts: &self.facts,
            whole: expanded == Expanded::All,
            note_cut: true,
        };
        let mut transcript = Transcript::default();
        let parts = lay_out_transcript(&mut transcript, &sources, Some(&shown), width);
        Some(transcript.laid_out(Some(agent), &parts, &view.highlights, 0..0))
    }

    /// `conversation`, a value no agent of this app holds (such as a saved
    /// session read with `rig_memory::Session::read`), laid out as
    /// [`lay_out`](Self::lay_out) lays out an agent's: the same order,
    /// renderers and wrapping, with no live calls, notices or facts, and
    /// the matches of `view`'s highlights counted. Its marks and expanded
    /// items name messages by id, which another conversation's messages
    /// may share: lay out with a view of its own.
    pub fn lay_out_conversation(
        &self,
        view: &TranscriptView,
        conversation: &conversation::Conversation,
        width: u16,
        expanded: Expanded,
    ) -> DrawnTranscript {
        let shown = ShownAgent::value(conversation, Entity::PLACEHOLDER);
        let renderers = self.looks.by_kind(None);
        let facts = CallFacts::default();
        let sources = Sources {
            view,
            notices: &[],
            renderers: &renderers,
            facts: &facts,
            whole: expanded == Expanded::All,
            note_cut: true,
        };
        let mut transcript = Transcript::default();
        let parts = lay_out_transcript(&mut transcript, &sources, Some(&shown), width);
        transcript.laid_out(None, &parts, &view.highlights, 0..0)
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
            id: Some(id),
            conversation,
            changed,
            inbox: inbox.into_iter().map(|(_, message)| message).collect(),
            calls: self.calls.of(calls.into_iter().flat_map(Calls::iter)),
            outputs: self.calls.outputs(calls.into_iter().flat_map(Calls::iter)),
            partial,
            streaming: streaming.map(|(_, call)| call.clone()).collect(),
            asides: asides(conversation),
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

    /// The notices shown.
    pub(crate) fn notices(&self) -> &[ShownNotice] {
        self.notices.all()
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
        let notices = sources.notices.iter().enumerate();
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
    let mut asides_after: Vec<Vec<&ShownAside<'_>>> = vec![Vec::new(); count + 1];
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
        marks: &view.marks,
    };
    let notices = shown.id.map(|id| (id, sources.notices));
    let notices = notices.into_iter().flat_map(|(id, notices)| {
        let notices = notices.iter().enumerate();
        notices.map(move |(at, notice)| (id, at, notice))
    });
    for (id, at, notice) in notices {
        let before = match notice.slot(id, conversation) {
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
    let drawn = shown.asides.iter();
    for aside in drawn.filter(|shown| sources.renderers.draws(shown.aside)) {
        let before = match TranscriptSlot::of_aside(aside.aside, conversation) {
            TranscriptSlot::Top => 0,
            TranscriptSlot::After(index) => index + 1,
            TranscriptSlot::Inside(index, item) => {
                let inside = look.inside.entry(index).or_default();
                inside.push((item, Inside::Aside(aside)));
                continue;
            }
        };
        if let Some(at) = asides_after.get_mut(before) {
            at.push(aside);
        }
    }
    transcript.update(shown.entity, conversation, shown.changed, &look, width);
    let mut after = after.into_iter().zip(asides_after);
    for index in 0..=count {
        let (notices, asides) = after.next().unwrap_or_default();
        for aside in asides {
            let item = aside.item();
            let (drawn, cut) = look.drawn(look.is_expanded(&item), |expanded| {
                aside.lines(sources.renderers, expanded)
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
            if let Some((call, partial)) = shown.partial {
                let view = |role, text, marks| MessageView {
                    role,
                    message: None,
                    block: 0,
                    text,
                    marks,
                };
                let reasoning = view(Role::Reasoning, &partial.reasoning, &[]);
                extra.extend(sources.renderers.message(&reasoning));
                if !partial.text.is_empty() {
                    extra.push(Line::default());
                    let marks = look.marks_in(MarkIn::Streaming(call));
                    extra.extend(sources.renderers.message(&view(
                        Role::Assistant,
                        &partial.text,
                        &marks,
                    )));
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

/// Draws the frame [`layout`] laid out: a compositor. It copies every
/// panel's canvas in z-order (the sides and the rest, then the boxes over
/// the screen), then draws the view's own overlays over everything: the
/// text of a question answered in the input above the editor with the
/// keys, the picker or question shown, the completion list, and the caret
/// of the editor with the keys.
pub(crate) fn render(
    (mut tui, mut test): (Option<ResMut<Tui>>, Option<ResMut<TuiTestScreen>>),
    mut drew: ResMut<TuiFrame>,
    mut view: ResMut<TuiView>,
    (state, focus, choices): (Res<TuiInputState>, Res<KeyFocus>, Choices),
    mut frame_layout: ResMut<FrameLayout>,
    (panels, carets): (Query<(Entity, &TuiPanel, &PanelCanvas)>, Query<&Caret>),
    mut repaints: MessageReader<Repaint>,
) -> Result {
    if !std::mem::take(&mut frame_layout.due) {
        return Ok(());
    }
    let repaint: Vec<Rect> = repaints.read().map(|repaint| repaint.0).collect();
    // Drawing the picker keeps its scroll within its body: drawing's own
    // bookkeeping, not a change to redraw for.
    let view = view.bypass_change_detection();
    // The panels at the sides, then the boxes over the screen.
    let mut panels: Vec<_> = panels.iter().collect();
    panels.sort_by_key(|(entity, panel, _)| {
        let over = matches!(panel.placement, Placement::Over { .. });
        (over, panel::spawned(*entity))
    });
    let keyed = state.editor().and_then(|editor| {
        let found = panels.iter().find(|(entity, ..)| *entity == editor);
        found.map(|(.., canvas)| canvas.area())
    });
    let edit = view
        .choosing
        .as_ref()
        .filter(|choosing| choosing.draft.is_some())
        .and_then(|choosing| choices.shown(choosing.entity));
    let cursor = carets.iter().find_map(|caret| caret.0);
    let cursor = cursor.map(|at| MoveTo(at.x, at.y));
    let mut areas = TuiFrame::default();
    let draw_frame = |frame: &mut Frame| {
        let screen = frame.area();
        let mut overlays = Vec::new();
        let mut drawn = Vec::new();
        for (entity, panel, canvas) in &panels {
            canvas.copy_to(frame.buffer_mut());
            drawn.push((*entity, canvas.area()));
            if matches!(panel.placement, Placement::Over { .. }) {
                overlays.push(canvas.area());
            }
        }
        let body = edit.as_ref().zip(keyed);
        overlays.extend(body.and_then(|(edit, at)| draw_body(frame, edit.body, at)));
        let shown = view
            .choosing
            .as_mut()
            .filter(|choosing| choosing.draft.is_none())
            .and_then(|choosing| Some((choices.shown(choosing.entity)?, choosing)));
        match (shown, &view.completion, keyed) {
            (Some((shown, choosing)), ..) => overlays.push(draw_choice(frame, choosing, &shown)),
            (None, Some(completion), Some(at)) if focus.0.is_none() => {
                overlays.extend(draw_completion(frame, completion, at));
            }
            _ => {}
        }
        areas = TuiFrame {
            screen,
            areas: drawn,
            overlays,
        };
    };
    let drawn = match (tui.as_deref_mut(), test.as_mut()) {
        (Some(tui), _) => {
            tui.begin_frame()?;
            let drawn = tui.terminal.draw(draw_frame);
            let drawn = drawn.map(|completed| cells_in(completed.buffer, &repaint));
            let drawn = drawn.and_then(|cells| {
                let cells = cells.iter().map(|(x, y, cell)| (*x, *y, cell));
                tui.terminal.backend_mut().draw(cells)
            });
            tui.end_frame(cursor.filter(|_| drawn.is_ok()))?;
            drawn
        }
        (None, Some(test)) => {
            // Drawing is not a change to redraw for; a resize is.
            test.bypass_change_detection().draw(draw_frame, &repaint);
            Ok(())
        }
        (None, None) => Ok(()),
    };
    drew.set_if_neq(areas);
    drawn?;
    Ok(())
}

/// The cells of `buffer` in `areas`, clipped to it, to write again.
fn cells_in(buffer: &Buffer, areas: &[Rect]) -> Vec<(u16, u16, Cell)> {
    let areas = areas.iter().map(|area| area.intersection(buffer.area));
    let positions = areas.flat_map(|area| area.positions());
    positions
        .filter_map(|at| Some((at.x, at.y, buffer.cell(at)?.clone())))
        .collect()
}

/// The items a selection moves through, of the marks drawn: all but a
/// decorator's rows, which belong to their call.
pub(crate) fn selectable(marks: Marks) -> Vec<TranscriptItem> {
    let items = marks.into_iter().map(|(item, _)| item);
    items
        .filter(|item| {
            !matches!(
                item,
                TranscriptItem::Decoration(..) | TranscriptItem::MessageDecoration(..)
            )
        })
        .collect()
}

/// Where the last frame drew each part of the screen: for a plugin that
/// writes to the terminal itself after the frame ([`TerminalWrite`]),
/// such as an image over the rows its content renderer left blank,
/// which finds those rows in a pane's [`DrawnTranscript`] (see
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
    /// Each panel drawn, the default screen's transcript, input and status
    /// line among them, with its area, in the order drawn.
    pub areas: Vec<(Entity, Rect)>,
    /// What was drawn over the rest: the panels placed over the screen,
    /// the picker or question, the completion list, a question's text
    /// above the input.
    pub overlays: Vec<Rect>,
}

impl TuiFrame {
    /// The screen row the `row` of `drawn` (a pane's transcript this frame
    /// drew) is on, when it is on screen.
    pub fn screen_row(&self, drawn: &DrawnTranscript, row: usize) -> Option<u16> {
        let on_screen = drawn.on_screen();
        if !on_screen.contains(&row) {
            return None;
        }
        let below_top = u16::try_from(row - on_screen.start).ok()?;
        Some(drawn.area().y.saturating_add(below_top))
    }

    /// Where the panel `entity` was drawn, if it was.
    pub fn area(&self, entity: Entity) -> Option<Rect> {
        let drawn = self.areas.iter().find(|(drawn, _)| *drawn == entity);
        drawn.map(|(_, area)| *area)
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
    let more = shown.more(choosing);
    let body = wrap_all(
        &plain_lines(more.unwrap_or(shown.body), Style::new()),
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
    if shown.ticks {
        keys.push("Tab ticks");
    }
    if more.is_some() {
        keys.push("← hides more");
    } else if shown.items.iter().any(|item| item.more.is_some()) {
        keys.push("→ shows more");
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
                if !shown.ticks {
                    return item.line(width);
                }
                let tick = match (shown.multiple, choosing.ticked.contains(index)) {
                    (true, true) => "[x] ",
                    (true, false) => "[ ] ",
                    (false, true) => "(x) ",
                    (false, false) => "( ) ",
                };
                let mut line = item.line(width.saturating_sub(tick.len()));
                line.spans.insert(0, Span::from(tick));
                line
            }
            Row::Typed if choosing.filter.trim().is_empty() => {
                Line::from("✎ or type an answer").dim()
            }
            Row::Typed if choosing.ticked.is_empty() => {
                Line::from(format!("✎ {}", choosing.filter.trim()))
            }
            Row::Typed => Line::from(format!("✎ {} (with the ticked)", choosing.filter.trim())),
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
