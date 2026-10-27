//! The frame: when one is due ([`needs_redraw`]), where each panel goes
//! ([`layout`]), how a transcript is laid out and lit
//! ([`TranscriptLayout`]), and the compositor that copies every panel's
//! canvas to the terminal under the view's overlays ([`render`]).

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::time::{Duration, Instant};

use bevy_ecs::change_detection::Tick;
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::ScheduleLabel;
use bevy_ecs::system::{SystemChangeTick, SystemParam};
use crossterm::cursor::MoveTo;
use ratatui::Frame;
use ratatui::backend::Backend;
use ratatui::buffer::{Buffer, Cell};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, List, ListState, Padding, Paragraph};
use unicode_width::UnicodeWidthStr;

use super::complete::Completion;
use super::editor::{Caret, Editor};
use super::input::InputContext;
use super::keys::KeyFocus;
use super::pane::{PaneState, TranscriptPane};
use super::panel::{
    self, PanelCanvas, PanelSpawns, Placement, RedrawArea, RequestRedraw, TerminalSize, TuiPanel,
};
use super::pick::{Choices, InlineStrip, Item, Listing, PickerInput, Row, detail_column};
use super::renderers::{
    CallState, MessageView, Place, Renderers, Role, ToolCallView, UNNAMED_HOLD,
};
use super::status::{AppStatus, StatusItems, StatusLine};
use super::terminal::{Tui, TuiTestScreen};
use super::theme::Theme;
use super::transcript::{
    CallFacts, CallTree, ChildCalls, Inside, Look, Marks, Part, Replaced, ShownAside, ShownEntries,
    Stops, StreamingCalls, Transcript, live_call, notice_lines, on_band, plain_lines,
};
use super::view::{
    DrawnTranscript, Highlight, MarkIn, Notices, RowWrap, ShownNotice, TranscriptItem,
    TranscriptSlot, TranscriptView, TuiView, message_items,
};
use super::wrap::wrap_all;
use rig_core::conversation;
use rig_core::message::{CallId, ToolCall};
use rig_ecs::agent::{ActiveTurn, AgentId, Calls, Conversation, Partial};
use rig_ecs::inbox::{DeliveryMode, Incoming};
use rig_ecs::stage::{Effect, ForAgent, Holds, Outcome, Settled, Started, SupersededBy};
use rig_ecs::tools::ToolRequest;
use rig_ecs::turn::{ModelRequest, SentTo};
use rig_harness::main_agent::ShownAgent;
use rig_harness::pick::{PickPreview, Picker};
use rig_harness::prelude::{CallOutput, ToolProgress};

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

/// Whether anything drawn changed since the last frame; the parameters
/// are the list. A streaming reply alone draws at most every
/// [`STREAM_FRAME`]; its waiting text is drawn by one of the frames the
/// kernel runs after the wake that brought it ([`Wake`](rig_ecs::tasks::Wake)).
pub(crate) fn needs_redraw(
    (view, notices, facts): (Res<TuiView>, Res<Notices>, Res<CallFacts>),
    (focus, shown, screen): (Res<KeyFocus>, Res<ShownAgent>, Res<TerminalSize>),
    changed: Query<
        (),
        Or<(
            Changed<Conversation>,
            Changed<Replaced>,
            Changed<ActiveTurn>,
            Changed<Calls>,
            Changed<StatusItems>,
            Changed<TuiPanel>,
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
            Changed<PickerInput>,
            Changed<PickPreview>,
            Changed<Picker>,
        )>,
    >,
    partials: Query<(), Or<(Changed<Partial>, Changed<StreamingCalls>)>>,
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
    (mut requests, mut repaints): (MessageReader<RequestRedraw>, MessageReader<RedrawArea>),
    revision: Res<crate::RenderRevision>,
    mut paced: Local<Paced>,
) -> bool {
    // Both readers are drained, so neither redraws again for the same change.
    let requested = requests.read().count() + repaints.read().count() > 0;
    let changed = requested
        || revision.is_changed()
        || view.is_changed()
        || notices.is_changed()
        || facts.is_changed()
        || shown.is_changed()
        || !widgets.is_empty()
        || focus.is_changed()
        || screen.is_changed()
        || app_status.is_some_and(|status| status.is_changed())
        || test.is_some_and(|test| test.is_changed())
        || !changed.is_empty()
        || !messages.is_empty()
        || !calls.is_empty();
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

/// Redraws once something drawn went away.
pub(crate) fn redraw_on_remove(
    _: On<
        Remove<(
            TuiPanel,
            StatusItems,
            Holds,
            ToolProgress,
            TranscriptPane,
            Editor,
            PickerInput,
        )>,
    >,
    mut redraw: MessageWriter<RequestRedraw>,
) {
    redraw.write(RequestRedraw);
}

/// Whether [`layout`] laid out a frame that [`render`] did not draw yet.
#[derive(Resource, Default)]
pub(crate) struct FrameLayout {
    due: bool,
    /// The area of the picker shown with a preview, the screen: its list
    /// at the left and its preview at the right, over its keys row.
    picker: Option<Rect>,
}

/// Whether [`layout`] laid out a frame that is not drawn yet: the
/// condition of [`TuiSystems::Draw`](super::TuiSystems::Draw) and
/// [`render`].
pub(crate) fn frame_due(frame: Res<FrameLayout>) -> bool {
    frame.due
}

/// The view's layout of a frame: the panels ([`layout`]), then the panes
/// in what they leave, each scroll placed
/// ([`lay_out_panes`](crate::pane::lay_out_panes)). Run in
/// [`TuiSystems::Layout`](crate::TuiSystems::Layout) ([`lay_out_frame`]),
/// and again after [`TuiSystems::Fit`](crate::TuiSystems::Fit) when a
/// panel changed there ([`lay_out_fitted`]).
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct LayOut;

/// When [`lay_out_frame`] last ran [`LayOut`].
#[derive(Resource, Default)]
pub(crate) struct LaidOutAt(Option<Tick>);

/// Lays out the frame ([`LayOut`]), in
/// [`TuiSystems::Layout`](crate::TuiSystems::Layout).
pub(crate) fn lay_out_frame(world: &mut World) {
    if world.try_run_schedule(LayOut).is_ok() {
        let now = world.change_tick();
        world.insert_resource(LaidOutAt(Some(now)));
    }
}

/// Lays out the frame again when a panel was spawned or changed since
/// [`lay_out_frame`], in [`TuiSystems::Fit`](crate::TuiSystems::Fit), so
/// a panel fitted to a pane's scroll is drawn in the frame that placed it.
pub(crate) fn lay_out_fitted(world: &mut World, panels: &mut QueryState<Ref<TuiPanel>>) {
    let Some(at) = world.get_resource::<LaidOutAt>().and_then(|laid| laid.0) else {
        return;
    };
    let now = world.read_change_tick();
    let fitted = panels
        .iter(world)
        .any(|panel| panel.last_changed().is_newer_than(at, now));
    if fitted {
        world.try_run_schedule(LayOut).ok();
    }
}

/// Lays out a frame: every panel by its placement, in the order spawned
/// (the default screen's first), an editor as tall as its text and a strip
/// as tall as what it shows.
pub(crate) fn layout(
    screens: (Option<Res<Tui>>, Option<Res<TuiTestScreen>>),
    (view, previewed): (
        Res<TuiView>,
        Query<(Entity, &PickerInput), With<PickPreview>>,
    ),
    mut frame: ResMut<FrameLayout>,
    mut panels: Query<(
        Entity,
        &TuiPanel,
        &mut PanelCanvas,
        Option<&Editor>,
        Has<InlineStrip>,
    )>,
    (spawns, ticks): (PanelSpawns, SystemChangeTick),
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
    panels.sort_by_key(|(entity, ..)| panel::spawn_order(ticks.this_run(), &spawns, *entity));
    let panels = panels
        .into_iter()
        .map(|(entity, panel, canvas, editor, strip)| {
            let fits = editor.map(|editor| editor.fit_rows(screen.width));
            (entity, panel, fits.or(strip.then_some(inline)), canvas)
        });
    // A picker with a preview takes the screen; what is laid out within it
    // stays above its keys row.
    let previewed = previewed.iter().find(|(_, state)| !state.in_input());
    let above_keys = Rect {
        height: screen.height.saturating_sub(1),
        ..screen
    };
    let within = previewed.map(|(entity, _)| (entity, above_keys));
    panel::lay_out(panels, screen, within.as_slice());
    *frame = FrameLayout {
        due: true,
        picker: previewed.map(|_| screen),
    };
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
pub(crate) struct ShownConversation<'a> {
    pub(crate) entity: Entity,
    /// Its id; `None` for a conversation no agent of this app holds,
    /// which shows no notices.
    pub(crate) id: Option<&'a AgentId>,
    pub(crate) conversation: &'a conversation::Conversation,
    /// What entries of it stand for, drawn above them.
    pub(crate) replaced: Option<&'a Replaced>,
    /// Whether its conversation, or what its entries stand for, changed
    /// since the last frame.
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

impl<'a> ShownConversation<'a> {
    /// `conversation`, which no agent of this app holds, drawn as
    /// `entity`'s: no live parts, no notices.
    pub(crate) fn value(conversation: &'a conversation::Conversation, entity: Entity) -> Self {
        Self {
            entity,
            id: None,
            conversation,
            replaced: None,
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
    /// The pane's view as plugins steer it: what is marked.
    pub(crate) view: &'a TranscriptView,
    /// The notices shown.
    pub(crate) notices: &'a Notices,
    /// The looks plugins added.
    pub(crate) renderers: &'a Renderers,
    /// What settled calls published and how long they ran.
    pub(crate) facts: &'a CallFacts,
    /// The transcript's colors.
    pub(crate) theme: &'a Theme,
}

/// Lays out an agent's transcript through [`crate::TranscriptRender`],
/// including pane-less exports. Call from an exclusive system so render
/// systems can access the world synchronously.
///
/// ```no_run
/// use rig_harness::prelude::*;
/// use rig_tui::{TranscriptLayout, TranscriptView};
/// fn export(world: &mut World, agent: Entity) {
///     let rows = TranscriptLayout::lay_out(world, &TranscriptView::default(), agent, 80);
/// }
/// ```
pub struct TranscriptLayout;

impl TranscriptLayout {
    /// Draw `agent` at `width` using global render systems (`pane = None`).
    /// Returns `None` if the entity has no conversation.
    pub fn lay_out(
        world: &mut World,
        view: &TranscriptView,
        agent: Entity,
        width: u16,
    ) -> Option<DrawnTranscript> {
        world.get::<AgentId>(agent)?;
        world.get::<Conversation>(agent)?;
        let mut transcript = Transcript::default();
        let mut renderers = Renderers::new((None, Some(agent), width), HashMap::new());
        let showing = Showing::Agent(agent, true);
        let parts = lay_out_rendered(
            world,
            (&mut transcript, &mut renderers),
            (view, &showing, width),
        )?;
        Some(transcript.laid_out(Some(agent), &parts, &view.highlights, 0..0))
    }
}

/// What a layout draws.
pub(crate) enum Showing<'v> {
    /// This agent, live; its conversation changed since the last frame or
    /// not.
    Agent(Entity, bool),
    /// A conversation no agent holds, in this pane; changed or not.
    Value(&'v conversation::Conversation, Entity, bool),
    /// No agent: the notices about every agent.
    Nothing,
}

/// The [`LayoutSources`] state, kept between layouts.
#[derive(Resource)]
struct SourcesState(bevy_ecs::system::SystemState<LayoutSources<'static, 'static>>);

/// Brings `transcript` up to date with `showing` through
/// [`crate::TranscriptRender`] and returns its parts: one pass draws the
/// default lines and collects the requests of what is not cached, the
/// render systems run, and a second pass lays out only what they changed.
/// `None` when the sources cannot be read.
pub(crate) fn lay_out_rendered(
    world: &mut World,
    (transcript, renderers): (&mut Transcript, &mut Renderers),
    (view, showing, width): (&TranscriptView, &Showing<'_>, u16),
) -> Option<Vec<Part>> {
    let mut state = match world.remove_resource::<SourcesState>() {
        Some(state) => state,
        None => SourcesState(bevy_ecs::system::SystemState::new(world)),
    };
    let pass = |world: &World,
                state: &mut SourcesState,
                transcript: &mut Transcript,
                renderers: &Renderers| {
        let from = state.0.get(world).ok()?;
        let shown = match *showing {
            Showing::Agent(agent, changed) => from.agent(agent, changed),
            Showing::Value(conversation, pane, changed) => {
                let mut value = ShownConversation::value(conversation, pane);
                value.changed = changed;
                Some(value)
            }
            Showing::Nothing => None,
        };
        let sources = Sources {
            view,
            notices: from.notices(),
            renderers,
            facts: from.facts(),
            theme: from.theme(),
        };
        Some(lay_out_transcript(
            transcript,
            &sources,
            shown.as_ref(),
            width,
        ))
    };
    let mut parts = pass(world, &mut state, transcript, renderers);
    if parts.is_some() && renderers.run(world) {
        parts = pass(world, &mut state, transcript, renderers);
    }
    world.insert_resource(state);
    parts
}

#[derive(SystemParam)]
pub(crate) struct LayoutSources<'w, 's> {
    agents: Query<
        'w,
        's,
        (
            &'static AgentId,
            &'static Conversation,
            Option<&'static Replaced>,
        ),
    >,
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
    calls: CallStates<'w, 's>,
    live: Query<'w, 's, LiveEffect, Without<Settled>>,
    names: Query<'w, 's, &'static Name>,
    facts: Res<'w, CallFacts>,
    streaming: Query<'w, 's, &'static StreamingCalls>,
    notices: Res<'w, Notices>,
    theme: Res<'w, Theme>,
}

impl LayoutSources<'_, '_> {
    /// The agent `entity` as its transcript draws it, its conversation
    /// `changed` since the last frame or not; `None` when it is no agent.
    pub(crate) fn agent(&self, entity: Entity, changed: bool) -> Option<ShownConversation<'_>> {
        let (id, conversation, replaced) = self.agents.get(entity).ok()?;
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
        let streaming = partial.and_then(|(call, _)| self.streaming.get(call).ok());
        let mut inbox: Vec<(&Effect, &Incoming)> = self
            .accepted
            .iter()
            .filter(|(.., to)| to.0 == entity)
            .map(|(effect, message, _)| (effect, message))
            .collect();
        inbox.sort_by_key(|(effect, _)| effect.id());
        Some(ShownConversation {
            entity,
            id: Some(id),
            conversation,
            replaced,
            changed,
            inbox: inbox.into_iter().map(|(_, message)| message).collect(),
            calls: self.calls.of(calls.into_iter().flat_map(Calls::iter)),
            outputs: self.calls.outputs(calls.into_iter().flat_map(Calls::iter)),
            partial,
            streaming: streaming
                .into_iter()
                .flat_map(StreamingCalls::calls)
                .collect(),
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
            if let (Some(request), Some((id, ..))) = (requests.0, id) {
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
    pub(crate) fn notices(&self) -> &Notices {
        &self.notices
    }

    /// The transcript's colors.
    pub(crate) fn theme(&self) -> &Theme {
        &self.theme
    }
}

/// Brings `transcript` up to date with the shown agent and returns its
/// parts at `width`: each message, with the notices written inside it,
/// after the notes and notices written before it, and after the archived
/// entries it stands for and a line saying how many; then those after the
/// last message, the reply streaming in with the tool calls it streams,
/// and the messages the agent accepted and has not sent yet. Without an
/// agent, the notices about every agent.
pub(crate) fn lay_out_transcript(
    transcript: &mut Transcript,
    sources: &Sources<'_>,
    shown: Option<&ShownConversation<'_>>,
    width: u16,
) -> Vec<Part> {
    sources.renderers.at(Place::default());
    let wrap_width = usize::from(width.max(1));
    let mut parts = Vec::new();
    let view = sources.view;
    let Some(shown) = shown else {
        let (mut extra, mut marks) = (Vec::new(), Marks::new());
        let notices = sources.notices.numbered();
        for notice in notices.filter(|(_, notice)| notice.is_for_every_agent()) {
            notice_lines(notice, &mut extra, &mut marks);
        }
        parts.push(Part::rows(&extra, &marks, wrap_width));
        return parts;
    };
    let conversation = shown.conversation;
    let entries = ShownEntries::new(conversation, shown.replaced);
    let count = entries.len();
    // Notes and notices before the first message, after each message,
    // and notices inside one, by where it is among the entries drawn.
    let mut after: Vec<Vec<(usize, &ShownNotice)>> = vec![Vec::new(); count + 1];
    let mut asides_after: Vec<Vec<&ShownAside<'_>>> = vec![Vec::new(); count + 1];
    let stops = Stops::of(conversation, &entries);
    let mut look = Look {
        renderers: sources.renderers,
        calls: &shown.calls,
        outputs: &shown.outputs,
        inside: HashMap::new(),
        facts: sources.facts,
        theme: sources.theme,
        children: &shown.children,
        marks: &view.marks,
        stops: &stops,
    };
    let notices = shown.id.map(|id| (id, sources.notices));
    let notices = notices
        .into_iter()
        .flat_map(|(id, notices)| notices.numbered().map(move |(at, notice)| (id, at, notice)));
    for (id, at, notice) in notices {
        let Some(slot) = notice.slot(id, conversation) else {
            continue;
        };
        // Where it goes among the entries drawn: a notice about an archived
        // entry keeps its place there.
        let archived = notice.message_in(id).and_then(|(message, items)| {
            let shown = entries.of_archived(message)?;
            let count = entries.get(shown).map(|entry| message_items(entry.message));
            Some(match count.is_some_and(|count| count > items) {
                true => TranscriptSlot::Inside(shown, items),
                false => TranscriptSlot::After(shown),
            })
        });
        let shown = archived.or_else(|| match slot {
            TranscriptSlot::Top => Some(TranscriptSlot::Top),
            TranscriptSlot::After(index) => entries.of_live(index).map(TranscriptSlot::After),
            TranscriptSlot::Inside(index, item) => entries
                .of_live(index)
                .map(|index| TranscriptSlot::Inside(index, item)),
        });
        let before = match shown {
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
    for aside in drawn {
        let archived = aside
            .aside
            .after
            .and_then(|after| entries.of_archived(after));
        let before = match TranscriptSlot::of_aside(aside.aside, conversation) {
            _ if archived.is_some() => archived.map(|at| at + 1),
            TranscriptSlot::Top => Some(0),
            TranscriptSlot::After(index) => entries.of_live(index).map(|at| at + 1),
            TranscriptSlot::Inside(index, item) => {
                if let Some(index) = entries.of_live(index) {
                    let inside = look.inside.entry(index).or_default();
                    inside.push((item, Inside::Aside(aside)));
                }
                continue;
            }
        };
        if let Some(at) = before.and_then(|before| asides_after.get_mut(before)) {
            at.push(aside);
        }
    }
    transcript.update(shown.entity, &entries, shown.changed, &look, width);
    sources.renderers.at(Place::default());
    let mut after = after.into_iter().zip(asides_after);
    for index in 0..=count {
        let (notices, asides) = after.next().unwrap_or_default();
        for aside in asides {
            let drawn = aside.lines(sources.renderers, sources.theme);
            if drawn.is_empty() {
                continue;
            }
            let mut lines = vec![Line::default()];
            lines.extend(on_band(drawn, sources.theme.tool));
            let marks = [(aside.item(), 0..lines.len())];
            parts.push(Part::rows(&lines, &marks, wrap_width));
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
                    width: wrap_width,
                    theme: *sources.theme,
                };
                let reasoning = view(Role::Reasoning, &partial.reasoning, &[]);
                extra.extend(reasoning.default_lines());
                if !partial.text.is_empty() {
                    extra.push(Line::default());
                    let marks = look.marks_in(MarkIn::Streaming(call));
                    extra.extend(view(Role::Assistant, &partial.text, &marks).default_lines());
                }
            }
            for (at, call) in shown.streaming.iter().enumerate() {
                sources.renderers.at(Place::Outside(at));
                let mut view = ToolCallView::new(call, None, CallState::Streaming);
                view.theme = *sources.theme;
                let start = extra.len();
                let (drawn, decorated) = sources.renderers.call_decorated(&view);
                extra.extend(on_band(drawn, sources.theme.tool));
                marks.push((TranscriptItem::Call(call.id.clone()), start..extra.len()));
                for (name, rows) in decorated {
                    marks.push((
                        TranscriptItem::Decoration(call.id.clone(), name),
                        start + rows.start..start + rows.end,
                    ));
                }
            }
            inbox_lines(&shown.inbox, sources.theme, &mut extra);
            parts.push(Part::rows(&extra, &marks, wrap_width));
            break;
        }
        let stands_for = entries.get(index).map_or(0, |entry| entry.stands_for);
        if stands_for > 0 {
            extra.extend([Line::default(), compacted_line(stands_for)]);
        }
        if !extra.is_empty() {
            parts.push(Part::rows(&extra, &marks, wrap_width));
        }
        parts.push(Part::Message(index));
    }
    parts
}

/// The dim line between the `count` archived entries an entry stands for
/// and that entry.
fn compacted_line(count: usize) -> Line<'static> {
    let text = match count {
        1 => "── 1 message above was compacted ──".to_owned(),
        count => format!("── {count} messages above were compacted ──"),
    };
    Line::styled(text, Style::new().dim())
}

/// `rows`, the first being row `first` of the transcript, with the rows of
/// the `selected` item on `theme`'s selected color, over their bands.
pub(crate) fn decorate(
    rows: Vec<Line<'static>>,
    first: usize,
    selected: Option<Range<usize>>,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let on = theme.selection();
    rows.into_iter()
        .enumerate()
        .map(|(index, row)| match &selected {
            Some(rows) if rows.contains(&(first + index)) => {
                let spans = row.spans.into_iter().map(|span| span.patch_style(on));
                Line::from(spans.collect::<Vec<_>>()).style(row.style.patch(on))
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
    continues: &[Option<RowWrap>],
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
    rows: impl IntoIterator<Item = (&'a Line<'line>, Option<&'a RowWrap>)>,
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
    view: Res<TuiView>,
    (state, focus, choices): (Res<InputContext>, Res<KeyFocus>, Choices),
    mut frame_layout: ResMut<FrameLayout>,
    (panels, carets): (Query<(Entity, &TuiPanel, &PanelCanvas)>, Query<&Caret>),
    (spawns, ticks): (PanelSpawns, SystemChangeTick),
    mut showing: Query<(Entity, &mut PickerInput, Option<&PickPreview>)>,
    mut repaints: MessageReader<RedrawArea>,
    theme: Res<Theme>,
) -> Result {
    if !std::mem::take(&mut frame_layout.due) {
        return Ok(());
    }
    let repaint: Vec<Rect> = repaints.read().map(|repaint| repaint.0).collect();
    let picker = showing.iter().find(|(_, choosing, _)| !choosing.in_input());
    let picker = picker.map(|(entity, ..)| entity);
    // The panels at the sides, then the boxes over the screen, then those
    // within another, those within the picker over it.
    let layer = |panel: &TuiPanel| match panel.placement {
        Placement::Over { .. } => 1,
        Placement::Within { panel, .. } if Some(panel) == picker => 3,
        Placement::Within { .. } => 2,
        _ => 0,
    };
    let keyed = state.editor().and_then(|editor| {
        let found = panels.get(editor).ok();
        found.map(|(.., canvas)| canvas.area())
    });
    let mut panels: Vec<_> = panels.iter().collect();
    let order = |entity| panel::spawn_order(ticks.this_run(), &spawns, entity);
    panels.sort_by_key(|(entity, panel, _)| (layer(panel), order(*entity)));
    let (under, over_picker): (Vec<_>, Vec<_>) = panels
        .into_iter()
        .partition(|(_, panel, _)| layer(panel) < 3);
    let edit = showing.iter().find(|(_, choosing, _)| choosing.in_input());
    let edit = edit.and_then(|(entity, ..)| choices.shown(entity));
    let cursor = carets.iter().find_map(|caret| caret.0);
    let cursor = cursor.map(|at| MoveTo(at.x, at.y));
    let mut areas = TuiFrame::default();
    let draw_frame = |frame: &mut Frame| {
        let screen = frame.area();
        let mut overlays = Vec::new();
        let mut drawn = Vec::new();
        let mut copy =
            |frame: &mut Frame, (entity, panel, canvas): &(Entity, &TuiPanel, &PanelCanvas)| {
                canvas.copy_to(frame.buffer_mut());
                drawn.push((*entity, canvas.area()));
                if layer(panel) > 0 {
                    overlays.push(canvas.area());
                }
            };
        for panel in &under {
            copy(frame, panel);
        }
        let body = edit.as_ref().zip(keyed);
        let body = body.and_then(|(edit, at)| draw_body(frame, &theme, edit.body, at));
        // Drawing the picker keeps its scroll within its body: drawing's
        // own bookkeeping, not a change to redraw for.
        let shown = picker.and_then(|picker| {
            let (_, choosing, preview) = showing.get_mut(picker).ok()?;
            Some((choices.shown(picker)?, choosing, preview))
        });
        let mut over = body.into_iter().collect::<Vec<_>>();
        match (shown, &view.completion, keyed) {
            (Some((shown, mut choosing, preview)), ..) => {
                let preview = preview.zip(frame_layout.picker);
                let choosing = choosing.bypass_change_detection();
                over.push(draw_choice(frame, &theme, choosing, &shown, preview));
                for panel in &over_picker {
                    copy(frame, panel);
                }
            }
            (None, Some(completion), Some(at)) if focus.0.is_none() => {
                over.extend(draw_completion(frame, &theme, completion, at));
            }
            _ => {}
        }
        overlays.extend(over);
        if theme.plain {
            strip_colors(frame.buffer_mut());
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

/// Whether a selection moves through `item`: all but a decoration's rows,
/// which belong to their call.
pub(crate) fn selectable(item: &TranscriptItem) -> bool {
    !matches!(
        item,
        TranscriptItem::Decoration(..) | TranscriptItem::MessageDecoration(..)
    )
}

/// Where the last frame drew each part of the screen: for a plugin that
/// writes to the terminal itself after the frame ([`TerminalWrite`]),
/// such as an image over the rows its render system left blank,
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

/// What a question answered in the input is about, on a raised surface
/// just above the input; where, if it is drawn.
fn draw_body(frame: &mut Frame, theme: &Theme, body: &str, input: Rect) -> Option<Rect> {
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
    frame.render_widget(Paragraph::new(lines).block(theme.raised("")), area);
    Some(area)
}

/// The completion list on a raised surface just above the input, its keys
/// on its last row; where, if it fit.
fn draw_completion(
    frame: &mut Frame,
    theme: &Theme,
    completion: &Completion,
    input: Rect,
) -> Option<Rect> {
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
            .block(with_keys(
                theme.raised(""),
                "Tab or Enter completes · Esc closes",
            ))
            .highlight_style(theme.selection()),
        area,
        &mut state,
    );
    Some(area)
}

/// The rows a picker's list keeps, at least, when a long body takes the
/// rest of its box.
const LEAST_LIST: usize = 3;

/// The picker or question on a raised surface centred over the
/// transcript, titled, its keys on its last row: its body,
/// scrolled to `choosing.scroll` (kept within it), what is typed, then the
/// rows it leaves, each with its detail at the right and, in a choice of
/// several, its tick box; for free text, the typed answer. With a preview,
/// it takes the whole of the preview's area instead, its keys on the last
/// row under both columns.
fn draw_choice(
    frame: &mut Frame,
    theme: &Theme,
    choosing: &mut PickerInput,
    shown: &Listing<'_>,
    preview: Option<(&PickPreview, Rect)>,
) -> Rect {
    let area = frame.area();
    // With a preview, the list takes the left of its area, the preview the
    // right: text drawn here, a conversation by its pane.
    let boxed = preview.map(|(preview, at)| {
        let [columns, keys] =
            Layout::vertical([Constraint::Fill(1), Constraint::Length(1)]).areas(at);
        let [list, beside] =
            Layout::horizontal([Constraint::Fill(1), Constraint::Fill(1)]).areas(columns);
        (at, list, preview, beside, keys)
    });
    let width = boxed.map_or(area.width.saturating_mul(4) / 5, |(_, list, ..)| list.width);
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
    let popup = match boxed {
        Some((_, list, ..)) => list,
        None => area.centered(Constraint::Length(width), Constraint::Length(height)),
    };
    frame.render_widget(Clear, boxed.map_or(popup, |(whole, ..)| whole));
    if let Some((_, _, PickPreview::Text(text), beside, _)) = boxed {
        let wide = usize::from(beside.width.saturating_sub(2).max(1));
        let lines = wrap_all(&plain_lines(text, Style::new()), wide);
        frame.render_widget(Clear, beside);
        let block = theme.raised("preview");
        frame.render_widget(Paragraph::new(lines).block(block), beside);
    }
    let inner = theme.raised("").inner(popup);
    // The body takes what the typed line, a gap and the list's least
    // leave, and scrolls when that is not all of it.
    let list_least = rows.len().clamp(1, LEAST_LIST);
    let room = usize::from(inner.height).saturating_sub(2 + list_least);
    let body_height = body.len().min(room);
    choosing.scroll = choosing.scroll.min(body.len() - body_height);
    let mut keys = vec![if shown.secret {
        "type or paste it"
    } else if shown.free_text {
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
    let width = usize::from(inner.width);
    let ticks = if shown.ticks { "[ ] ".len() } else { 0 };
    let column = detail_column(&shown.items, width.saturating_sub(ticks));
    // A detail standing in for more says more only where its row cuts it.
    let room = width.saturating_sub(ticks + column);
    let says_more = |item: &Item<'_>| {
        item.more
            .is_some_and(|more| Some(more) != item.detail || more.width() > room)
    };
    if more.is_some() {
        keys.push("← hides more");
    } else if shown.items.iter().any(says_more) {
        keys.push("→ shows more");
    }
    let title = theme.raised(shown.title);
    match boxed {
        Some((.., row)) => {
            let keys = keys_hint(&keys, usize::from(row.width.saturating_sub(2)));
            let keys = Line::from(format!("{keys} ")).dim().right_aligned();
            let surface = Style::new().bg(theme.raised_surface);
            frame.render_widget(Paragraph::new(keys).style(surface), row);
            frame.render_widget(title, popup);
        }
        None => {
            let keys = keys_hint(&keys, usize::from(popup.width.saturating_sub(2)));
            frame.render_widget(with_keys(title, &keys), popup);
        }
    }
    let [body_area, _, typed, list] = Layout::vertical([
        Constraint::Length(u16::try_from(body_height).unwrap_or(u16::MAX)),
        Constraint::Length(u16::from(body_height > 0)),
        Constraint::Length(1),
        Constraint::Min(1),
    ])
    .areas(inner);
    let scroll = u16::try_from(choosing.scroll).unwrap_or(u16::MAX);
    frame.render_widget(Paragraph::new(body).scroll((scroll, 0)), body_area);
    let typed_text = shown.typed(&choosing.filter);
    frame.render_widget(Line::from(format!("› {typed_text}▏")), typed);
    let items: Vec<Line<'static>> = rows
        .iter()
        .map(|row| match row {
            Row::Item(index) => {
                let Some(item) = shown.items.get(*index) else {
                    return Line::default();
                };
                if !shown.ticks {
                    return item.line(width, column);
                }
                let tick = match (shown.multiple, choosing.ticked.contains(index)) {
                    (true, true) => "[x] ",
                    (true, false) => "[ ] ",
                    (false, true) => "(x) ",
                    (false, false) => "( ) ",
                };
                let mut line = item.line(width.saturating_sub(tick.len()), column);
                line.spans.insert(0, Span::from(tick));
                line
            }
            Row::Typed if choosing.filter.trim().is_empty() => {
                Line::from("✎ or type an answer").dim()
            }
            Row::Typed if choosing.ticked.is_empty() => {
                Line::from(format!("✎ {}", shown.typed(choosing.filter.trim())))
            }
            Row::Typed => Line::from(format!(
                "✎ {} (with the ticked)",
                shown.typed(choosing.filter.trim())
            )),
        })
        .collect();
    let mut state = ListState::default().with_selected(Some(choosing.selected));
    frame.render_stateful_widget(
        List::new(items).highlight_style(theme.selection()),
        list,
        &mut state,
    );
    boxed.map_or(popup, |(whole, ..)| whole)
}

/// Every cell of `buffer` in the terminal's own colors, keeping its bold,
/// underline and the like, for a [`plain`](Theme::plain) theme.
fn strip_colors(buffer: &mut Buffer) {
    for cell in &mut buffer.content {
        cell.fg = Color::Reset;
        cell.bg = Color::Reset;
        cell.underline_color = Color::Reset;
    }
}

/// A picker's keys, `keys` then Enter's and Esc's, in at most `width`
/// columns: the first of `keys` are left out while it is wider.
fn keys_hint(keys: &[&str], width: usize) -> String {
    let mut keys = keys;
    loop {
        let all = keys.iter().copied().chain(["Enter picks", "Esc closes"]);
        let hint = all.collect::<Vec<_>>().join(", ");
        match keys.split_first() {
            Some((_, rest)) if hint.width() > width => keys = rest,
            _ => return hint,
        }
    }
}

/// `surface` with `keys` dim at the right of its last row, where its
/// padding was.
fn with_keys<'a>(surface: Block<'a>, keys: &str) -> Block<'a> {
    surface
        .padding(Padding::horizontal(1))
        .title_bottom(Line::from(format!("{keys} ")).dim().right_aligned())
}

/// The messages the agent accepted and has not sent yet, under the
/// transcript, in the order they came.
fn inbox_lines(inbox: &[&Incoming], theme: &Theme, lines: &mut Vec<Line<'static>>) {
    if inbox.is_empty() {
        return;
    }
    lines.push(Line::default());
    for message in inbox {
        let when = match message.mode {
            DeliveryMode::Steer => "steering",
            DeliveryMode::Queue => "after this turn",
            DeliveryMode::Inform => "noted",
        };
        let text = message.text();
        let mut rows = text.lines();
        let first = Line::from(vec![
            Span::from(format!("  ⏵ {when}: ")).dim(),
            Span::from(rows.next().unwrap_or_default().to_owned()).dim(),
        ]);
        let rest = rows.map(|row| Line::from(format!("    {row}")).dim());
        lines.extend(on_band(std::iter::once(first).chain(rest), theme.user));
    }
}

#[cfg(test)]
mod tests;
