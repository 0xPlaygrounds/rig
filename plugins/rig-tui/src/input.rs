//! Terminal input, read on a thread that wakes the loop, as in Bevy's
//! `examples/async_tasks/external_source_external_thread.rs`.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use bevy_ecs::prelude::*;
use bevy_ecs::schedule::ScheduleLabel;
use bevy_ecs::system::SystemParam;
use bevy_log::error;
use crossbeam_channel::Receiver;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use ratatui::layout::Rect;

use super::clipboard;
use super::complete::{self, Completion, CompletionKind, CompletionSource};
use super::editor::{Editor, MainEditor};
use super::history::{PromptHistory, Sent};
use super::keys::{self, Found, InputAction, KeyBinding, KeyFocus, KeyPress, KeyPressed, KeyScope};
use super::pane::{MainPane, PaneState};
use super::panel::{RequestRedraw, TerminalSize, TuiPanel};
use super::pick::{self, Choices, PickerInput, Showing};
use super::render::selectable;
use super::terminal::TerminalFocus;
use super::view::{ItemRow, TranscriptItem, TranscriptView, TuiView};
use rig_ecs::agent::{ActiveTurn, Notice, StopTurn};
use rig_ecs::inbox::{Deliver, DeliveryMode};
use rig_ecs::tasks::Wake;
use rig_harness::main_agent::ShownAgent;
use rig_harness::pick::{Chosen, Pick};
use rig_harness::prelude::{Due, Exit, ExitReason, Processes, Scratch, WakeAt};

/// Lines a page key scrolls.
const PAGE: usize = 10;

/// How long the input thread waits for an event before it checks whether
/// the view is gone.
const POLL: Duration = Duration::from_millis(100);

/// The items of [`PendingInput`] a frame does at most: the rest, such as
/// those of a macro that never ends, wait for the next frame, so the view
/// keeps drawing.
const KEYS_PER_FRAME: usize = 1024;

/// The frame's input, in order: what the terminal reported and the
/// actions plugins pushed. The view does one item at a time, running
/// [`TuiKeys`] for each: [`InputSystems::Filter`] systems may change, drop
/// or replace it (with the main [`Editor`](crate::Editor) and
/// [`InputContext`] as the keys before it left them), and
/// [`InputSystems::Dispatch`] does it. The picker or question shown takes
/// it first, then a [`KeyBinding`](crate::KeyBinding), then the input.
/// Items left by a run are done next; a push outside [`TuiKeys`] waits for
/// the next frame.
///
/// ```no_run
/// use rig_harness::prelude::*;
/// use rig_tui::crossterm::event::{Event, KeyCode};
/// use rig_tui::{Editor, InputSystems, KeyInput, MainEditor, PendingInput, InputContext, TuiKeys};
///
/// /// `/` on an empty input opens a search instead of a slash command.
/// fn slash_searches(
///     mut events: ResMut<PendingInput>,
///     input: Single<&Editor, With<MainEditor>>,
///     state: Res<InputContext>,
/// ) {
///     let slash = |item: &KeyInput| {
///         matches!(item, KeyInput::Event(Event::Key(key)) if key.code == KeyCode::Char('/'))
///     };
///     if input.is_empty() && state.picking().is_none() && events.0.first().is_some_and(slash) {
///         events.0.remove(0);
///         // open the search
///     }
/// }
///
/// fn build(app: &mut App) {
///     app.add_systems(TuiKeys, slash_searches.in_set(InputSystems::Filter));
/// }
/// ```
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct PendingInput(pub Vec<KeyInput>);

/// An item of the [`PendingInput`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeyInput {
    /// What the terminal reported: a key, a paste, a resize, a change of
    /// focus.
    Event(Event),
    /// A built-in action of the input, done where it stands among the keys.
    Action(InputAction),
}

/// The schedule the view runs once for each item of the frame's
/// [`PendingInput`], in order, in `PreUpdate`, with that one item in
/// them: filters see one key (and what earlier filters made of it) with
/// the input and the [`InputContext`] as the keys before it left them.
/// Its sets, in this order, are [`InputSystems`]; the commands of a run
/// are applied before the next, so a panel a key opens takes the next.
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TuiKeys;

/// Where the terminal view handles each item of the [`PendingInput`],
/// in [`TuiKeys`], in this order.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InputSystems {
    /// Plugins read and change the item in the [`PendingInput`], with
    /// the [`InputContext`] and the input as the keys before it left them.
    Filter,
    /// The view does what is left in the [`PendingInput`].
    Dispatch,
}

/// What the view does with the keys, as the keys before the one being done
/// left it, in [`TuiKeys`], and as the frame's keys left it, after them:
/// for a filter that acts only in some states, such as not while a
/// completion is open. The view writes it; it is read-only to plugins.
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct InputContext {
    scope: KeyScope,
    editor: Option<Entity>,
    completion: Option<Completion>,
    typed: Option<(CompletionKind, String)>,
    answering: Option<Entity>,
    picking: Option<Entity>,
    selection: Option<usize>,
}

impl InputContext {
    /// Where the keys go that no picker, question or global binding takes:
    /// the input, the transcript while the person reads it
    /// ([`KeyScope::Transcript`], whose keys the input's back), or the panel
    /// with the [`KeyFocus`].
    pub fn scope(&self) -> KeyScope {
        self.scope
    }

    /// The [`Editor`] the keys type into: the one of the panel with the
    /// [`KeyFocus`], else the [`MainEditor`]; `None` without either.
    pub fn editor(&self) -> Option<Entity> {
        self.editor
    }

    /// The open completion, which Up, Down, Tab, Enter and Esc act on.
    pub fn completion(&self) -> Option<&Completion> {
        self.completion.as_ref()
    }

    /// The token at the cursor that completion completes, and what is typed
    /// of it after its marker, also while nothing matches: for a
    /// [`CompletionSource`](crate::CompletionSource) that looks its
    /// candidates up, such as issues matching `#14`.
    pub fn typed(&self) -> Option<(CompletionKind, &str)> {
        let (kind, text) = self.typed.as_ref()?;
        Some((*kind, text))
    }

    /// The question answered in the input: Enter answers it, Esc refuses
    /// it.
    pub fn answering(&self) -> Option<Entity> {
        self.answering
    }

    /// The picker or question shown over the transcript, which takes every
    /// key.
    pub fn picking(&self) -> Option<Entity> {
        self.picking
    }

    /// The item highlighted in the picker or question shown over the
    /// transcript ([`picking`](Self::picking)), as an index into all of its
    /// items, whatever typing filtered out; `None` while none is, such as
    /// when the typed answer is.
    pub fn selection(&self) -> Option<usize> {
        self.selection
    }
}

/// Terminal events read by the input thread. Dropping it stops the thread.
#[derive(Resource)]
pub(crate) struct TerminalInput {
    events: Receiver<Event>,
    /// Tells the input thread to stop.
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl TerminalInput {
    /// Starts the input thread.
    pub(crate) fn start(wake: Wake) -> std::io::Result<Self> {
        let (sender, events) = crossbeam_channel::unbounded();
        let stop = Arc::new(AtomicBool::new(false));
        let shared = Arc::clone(&stop);
        let thread = std::thread::Builder::new()
            .name("rig-harness-input".to_owned())
            .spawn(move || {
                while !shared.load(Ordering::Relaxed) {
                    let event = match event::poll(POLL) {
                        Ok(false) => continue,
                        Ok(true) => event::read(),
                        Err(failure) => Err(failure),
                    };
                    match event {
                        Ok(event) => {
                            if sender.send(event).is_err() {
                                return;
                            }
                            wake.wake();
                        }
                        Err(failure) => {
                            error!("could not read the terminal: {failure}");
                            return;
                        }
                    }
                }
            })?;
        Ok(Self {
            events,
            stop,
            thread: Some(thread),
        })
    }

    /// Stops the input thread and waits until it no longer reads the
    /// terminal, for a program to have it.
    pub(crate) fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            thread.join().ok();
        }
    }
}

impl Drop for TerminalInput {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// Passes on what the input thread read.
pub(crate) fn receive_events(input: Res<TerminalInput>, mut events: ResMut<PendingInput>) {
    let mut read = input.events.try_iter().peekable();
    if read.peek().is_some() {
        events.0.extend(read.map(KeyInput::Event));
    }
}

/// Drops the [`PendingInput`] in a run without the view, which nothing
/// does.
pub(crate) fn clear_events(mut events: ResMut<PendingInput>) {
    if !events.0.is_empty() {
        events.0.clear();
    }
}

/// Runs [`TuiKeys`] for each item of the frame's [`PendingInput`], in
/// order, with that item alone in them; what a run leaves goes next. A run
/// that changed the input's text ends the reading of the transcript: the
/// selection goes. Past [`KEYS_PER_FRAME`] the rest waits for the next
/// frame, which it wakes.
pub(crate) fn run_keys(world: &mut World) {
    if world
        .get_resource::<PendingInput>()
        .is_none_or(|events| events.0.is_empty())
    {
        return;
    }
    let mut events = world.resource_mut::<PendingInput>();
    let mut queue: VecDeque<KeyInput> = std::mem::take(&mut events.0).into();
    let mut done = 0;
    while let Some(item) = queue.pop_front() {
        if done == KEYS_PER_FRAME {
            queue.push_front(item);
            world.resource_mut::<PendingInput>().0 = queue.into();
            if let Some(wake) = world.get_resource::<Wake>() {
                wake.wake();
            }
            return;
        }
        done += 1;
        world.resource_mut::<PendingInput>().0.push(item);
        let text = main_text(world);
        world.run_schedule(TuiKeys);
        let typed = main_text(world) != text;
        if typed {
            let mut main = world.query_filtered::<&mut TranscriptView, With<MainPane>>();
            for mut transcript in main.iter_mut(world) {
                if transcript.selected.is_some() {
                    transcript.selected = None;
                }
            }
        }
        let left = std::mem::take(&mut world.resource_mut::<PendingInput>().0);
        for item in left.into_iter().rev() {
            queue.push_front(item);
        }
    }
}

/// The main editor's text.
fn main_text(world: &mut World) -> Option<String> {
    let mut main = world.query_filtered::<&Editor, With<MainEditor>>();
    let text = main.single(world).ok().map(Editor::text);
    text.map(str::to_owned)
}

/// Notes the [`InputContext`] for the next key's filters, and after the
/// frame's keys.
pub(crate) fn note_input_context(
    (view, main): (Res<TuiView>, Query<&TranscriptView, With<MainPane>>),
    showing: Query<(Entity, &PickerInput)>,
    focus: Res<KeyFocus>,
    panels: Query<(), With<TuiPanel>>,
    choices: Choices,
    (editors, sources): (
        Query<(Entity, &Editor, Has<MainEditor>)>,
        Query<&CompletionSource>,
    ),
    mut state: ResMut<InputContext>,
) {
    let main_editor = editors.iter().find(|(.., main)| *main);
    let typed = main_editor.and_then(|(_, input, _)| complete::typed(input, &sources));
    let focused = focus.0.filter(|panel| editors.contains(*panel));
    let editor = focused.or(main_editor.map(|(entity, ..)| entity));
    let scope = match focus.0 {
        Some(panel) if panels.contains(panel) => KeyScope::Panel(panel),
        _ if reading(&main) => KeyScope::Transcript,
        _ => KeyScope::Input,
    };
    let choosing = showing.iter().next();
    let answering = choosing.filter(|(_, c)| c.draft.is_some()).map(|(e, _)| e);
    let picker = choosing.filter(|(_, c)| c.draft.is_none());
    let picking = picker.map(|(e, _)| e);
    let selection = picker.and_then(|(entity, choosing)| {
        let shown = choices.shown(entity)?;
        match shown.rows(&choosing.filter).get(choosing.selected) {
            Some(&pick::Row::Item(index)) => Some(index),
            _ => None,
        }
    });
    state.set_if_neq(InputContext {
        scope,
        editor,
        completion: view.completion.clone(),
        typed,
        answering,
        picking,
        selection,
    });
}

/// Whether the person reads the main transcript: an item of it is
/// selected.
fn reading(main: &Query<&TranscriptView, With<MainPane>>) -> bool {
    main.iter().any(|view| view.selected.is_some())
}

/// The key bindings and who has the keys.
#[derive(SystemParam)]
pub(crate) struct Keymap<'w, 's> {
    bindings: Query<'w, 's, (Entity, &'static KeyBinding, Option<&'static InputAction>)>,
    focus: ResMut<'w, KeyFocus>,
    panels: Query<'w, 's, (), With<TuiPanel>>,
}

impl Keymap<'_, '_> {
    /// Where the keys go besides the global bindings: the focused panel
    /// while it is one, else the transcript while the person is `reading`
    /// it, else the input.
    fn scope(&mut self, reading: bool) -> KeyScope {
        let unfocused = if reading {
            KeyScope::Transcript
        } else {
            KeyScope::Input
        };
        match self.focus.0 {
            Some(panel) if self.panels.contains(panel) => KeyScope::Panel(panel),
            Some(_) => {
                self.focus.0 = None;
                unfocused
            }
            None => unfocused,
        }
    }
}

/// What a built-in key needs besides the view and the input.
pub(crate) struct Context<'a, 'w, 's> {
    /// The shown agent.
    agent: Option<Entity>,
    busy: bool,
    scratch: Option<&'a Scratch>,
    processes: Option<&'a Processes>,
    wake: &'a Wake,
    /// The prompts Up and Down browse.
    history: &'a [String],
    /// The pickers and questions open.
    choices: &'a Choices<'w, 's>,
    /// Whether the exit is armed ([`ExitArmed`]).
    exit_armed: bool,
}

/// How long a first Ctrl+D on an empty input keeps the exit armed.
const EXIT_WINDOW: Duration = Duration::from_millis(500);

/// On the entity that arms the exit: a second [`InputAction::Exit`]
/// before it is [`Due`] quits. It is despawned once due.
#[derive(Component)]
pub(crate) struct ExitArmed;

/// Disarms the exit once its moment passed, and redraws the input's key
/// row without the hint.
pub(crate) fn disarm_exit(
    due: On<Add<Due>>,
    armed: Query<(), With<ExitArmed>>,
    mut redraw: MessageWriter<RequestRedraw>,
    mut commands: Commands,
) {
    if armed.contains(due.entity) {
        commands.entity(due.entity).despawn();
        redraw.write(RequestRedraw);
    }
}

/// What a built-in key changes: the view, the input, and the transcript
/// pane the keys act on, the one with the [`KeyFocus`] or else the main
/// one. Its view is marked changed only when a key really changes it, as
/// plugins rerun on its changes.
pub(crate) struct Target<'a> {
    view: &'a mut TuiView,
    input: &'a mut Editor,
    pane: Option<(Mut<'a, TranscriptView>, Mut<'a, PaneState>)>,
}

/// The transcript panes, which the keys scroll and select in.
type Panes<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static mut TranscriptView,
        &'static mut PaneState,
        Has<MainPane>,
    ),
>;

/// The editors, which the keys type into.
type Editors<'w, 's> = Query<'w, 's, (Entity, &'static mut Editor, Has<MainEditor>)>;

/// The editor the keys type into: the panel's with the keys, else the main
/// one.
fn keyed_editor(editors: &Editors, focus: Option<Entity>) -> Option<Entity> {
    let focused = focus.filter(|entity| editors.contains(*entity));
    let main = || editors.iter().find(|(.., main)| *main);
    focused.or_else(|| main().map(|(entity, ..)| entity))
}

/// The pane the keys act on: the one with the keys, else the main one.
fn keyed_pane(panes: &Panes, focus: Option<Entity>) -> Option<Entity> {
    let focused = focus.filter(|entity| panes.contains(*entity));
    let main = || {
        panes
            .iter()
            .find(|(.., main)| *main)
            .map(|(entity, ..)| entity)
    };
    focused.or_else(main)
}

/// Does the [`PendingInput`] left of the key being done, in order.
pub(crate) fn read_input(
    mut events: ResMut<PendingInput>,
    (mut view, mut panes, shown): (ResMut<TuiView>, Panes, Res<ShownAgent>),
    (mut editors, mut showing): (Editors, Showing),
    mut keymap: Keymap,
    choices: Choices,
    agents: Query<Has<ActiveTurn>>,
    (scratch, processes, wake, history, armed): (
        Option<Res<Scratch>>,
        Option<Res<Processes>>,
        Res<Wake>,
        Res<PromptHistory>,
        Query<(), (With<ExitArmed>, Without<Due>)>,
    ),
    (mut screen, mut focus): (ResMut<TerminalSize>, ResMut<TerminalFocus>),
    mut commands: Commands,
) {
    // Taking the resources mutably marks them changed, and a frame with no
    // input must leave them as they are, or the view redraws every frame.
    if events.0.is_empty() {
        return;
    }
    let busy = shown
        .0
        .and_then(|agent| agents.get(agent).ok())
        .unwrap_or(false);
    let context = Context {
        agent: shown.0,
        busy,
        scratch: scratch.as_deref(),
        processes: processes.as_deref(),
        wake: &wake,
        history: &history.entries,
        choices: &choices,
        exit_armed: !armed.is_empty(),
    };
    // Keys typed with no editor to take them go nowhere.
    let mut none = Editor::default();
    for item in std::mem::take(&mut events.0) {
        let editor = keyed_editor(&editors, keymap.focus.0);
        let in_panel = editor.is_some() && editor == keymap.focus.0;
        let mut found = editor.and_then(|editor| editors.get_mut(editor).ok());
        let input: &mut Editor = match &mut found {
            Some((_, editor, _)) => editor,
            None => &mut none,
        };
        let pane = keyed_pane(&panes, keymap.focus.0);
        let reading = panes
            .iter()
            .any(|(_, view, _, main)| main && view.selected.is_some());
        let event = match item {
            KeyInput::Event(event) => event,
            KeyInput::Action(action) => {
                let pane = pane.and_then(|pane| panes.get_mut(pane).ok());
                let mut target = Target {
                    view: &mut view,
                    input,
                    pane: pane.map(|(_, view, state, _)| (view, state)),
                };
                act(action, &mut target, (&context, &showing), &mut commands);
                continue;
            }
        };
        match event {
            Event::Key(key) if key.kind == KeyEventKind::Release => {
                released(key, (shown.0, reading), &mut keymap, &mut commands);
            }
            Event::Key(key) => {
                // A question answered in the input leaves the keys to it.
                let picking = showing.iter_mut().find(|(_, c)| c.draft.is_none());
                if let Some((entity, mut choosing)) = picking {
                    let shown = choices.shown(entity);
                    let chosen =
                        shown.and_then(|shown| pick::choice_key(key, &mut choosing, &shown));
                    if let Some(chosen) = chosen {
                        commands.trigger(Pick::person(entity, chosen));
                        // The next one takes the keys that follow.
                        view.queue.retain(|&queued| queued != entity);
                        pick::show_first(&view, &choices, (&showing, input), &mut commands);
                    }
                    continue;
                }
                let pane = pane.and_then(|pane| panes.get_mut(pane).ok());
                let mut target = Target {
                    view: &mut view,
                    input,
                    pane: pane.map(|(_, view, state, _)| (view, state)),
                };
                pressed(
                    key,
                    (reading, in_panel),
                    &mut target,
                    &mut keymap,
                    (&context, &showing),
                    &mut commands,
                );
            }
            // A paste arrives whole, newlines included, so it is not sent
            // line by line. A panel with the keys reads it from the event.
            Event::Paste(text) => match showing.iter_mut().find(|(_, c)| c.draft.is_none()) {
                Some((_, mut choosing)) => {
                    choosing
                        .filter
                        .push_str(text.lines().next().unwrap_or_default());
                    choosing.selected = 0;
                }
                None if !in_panel && matches!(keymap.scope(reading), KeyScope::Panel(_)) => {}
                // A dropped image file becomes `@path`, which attaches it.
                None => match clipboard::dropped_image(&text) {
                    Some(path) => clipboard::type_path(input, &path),
                    None => input.insert(&text),
                },
            },
            Event::Resize(width, height) => {
                screen.set_if_neq(TerminalSize(Rect::new(0, 0, width, height)));
            }
            Event::FocusGained => {
                focus.set_if_neq(TerminalFocus(Some(true)));
            }
            Event::FocusLost => {
                focus.set_if_neq(TerminalFocus(Some(false)));
            }
            Event::Mouse(_) => {}
        }
    }
}

/// Handles a key pressed with no picker or question shown: a binding's,
/// the next of a binding of several, else a character typed into the
/// input, or into the editor of the panel with the keys (`in_panel`). With
/// a panel focused, Esc that it does not bind gives the keys back to the
/// input.
fn pressed(
    key: KeyEvent,
    (reading, in_panel): (bool, bool),
    target: &mut Target,
    keymap: &mut Keymap,
    (context, showing): (&Context, &Showing),
    commands: &mut Commands,
) {
    let Target { view, input, .. } = target;
    let press = KeyPress::of(&key);
    let scope = keymap.scope(reading);
    let bindings: Vec<(Entity, &KeyBinding)> = keymap
        .bindings
        .iter()
        .map(|(entity, binding, _)| (entity, binding))
        .collect();
    view.pending.push(press);
    let found = keys::find(&bindings, &view.pending, scope);
    if found == Found::Partial {
        return;
    }
    let sequence = std::mem::take(&mut view.pending).len() > 1;
    let Found::Binding(entity) = found else {
        if sequence {
            return;
        }
        match scope {
            KeyScope::Panel(_) if press.code == KeyCode::Esc => keymap.focus.0 = None,
            KeyScope::Panel(_) if !in_panel => {}
            _ => {
                if let Some(character) = press.typed() {
                    input.insert_char(character);
                }
            }
        }
        return;
    };
    let Ok((_, binding, action)) = keymap.bindings.get(entity) else {
        return;
    };
    // Holding a key that also reports its release repeats it: one press.
    if binding.releases && key.kind == KeyEventKind::Repeat {
        return;
    }
    match action {
        Some(action) => act(*action, target, (context, showing), commands),
        None => commands.trigger(KeyPressed {
            entity,
            agent: context.agent,
            released: false,
        }),
    }
}

/// Tells a binding that asked for releases that its key was let go, the
/// person `reading` the transcript or not.
fn released(
    key: KeyEvent,
    (agent, reading): (Option<Entity>, bool),
    keymap: &mut Keymap,
    commands: &mut Commands,
) {
    let press = KeyPress::of(&key);
    let scope = keymap.scope(reading);
    let released = keymap.bindings.iter().find(|(_, binding, _)| {
        binding.releases
            && binding.keys.presses() == [press]
            && scope.applying().any(|applying| applying == binding.scope)
    });
    if let Some((entity, ..)) = released {
        commands.trigger(KeyPressed {
            entity,
            agent,
            released: true,
        });
    }
}

/// Does what a built-in key does. With a turn running, Enter steers it and
/// Tab queues a follow-up. Esc interrupts the agent: it stops its turn,
/// or, idle, what a plugin stops on it, such as a rebuild.
fn act(
    action: InputAction,
    target: &mut Target,
    (context, showing): (&Context, &Showing),
    commands: &mut Commands,
) {
    use InputAction as Do;
    let Target { view, input, pane } = target;
    // A question answered in the input: Enter answers it, Esc refuses it.
    let editing = showing
        .iter()
        .find(|(_, choosing)| choosing.draft.is_some());
    let editing = editing.map(|(entity, _)| entity);
    match (action, editing) {
        (Do::Send, Some(ask)) if view.completion.as_ref().is_none_or(|c| c.is_exact()) => {
            let answer = Chosen::text(input.take());
            return close(ask, Some(answer), target, (context, showing), commands);
        }
        (Do::Stop, Some(ask)) if view.completion.is_none() => {
            return close(ask, None, target, (context, showing), commands);
        }
        _ => {}
    }
    match action {
        Do::Clear => input.clear(),
        Do::Exit if !input.is_empty() => {}
        Do::Exit if context.exit_armed => commands.trigger(Exit(ExitReason::Quit)),
        Do::Exit => {
            commands.spawn((ExitArmed, WakeAt::after(EXIT_WINDOW)));
            commands.write_message(RequestRedraw);
        }
        Do::NewLine => input.insert_char('\n'),
        Do::LineStart => input.home(),
        Do::LineEnd => input.end(),
        Do::Left => input.left(),
        Do::Right => input.right(),
        Do::WordLeft => input.word_left(),
        Do::WordRight => input.word_right(),
        Do::Backspace => input.backspace(),
        Do::Delete => input.delete(),
        Do::DeleteWordBefore => input.delete_word_before(),
        Do::DeleteWordAfter => input.delete_word_after(),
        Do::DeleteToLineStart => input.delete_to_line_start(),
        Do::DeleteToLineEnd => input.delete_to_line_end(),
        Do::LineUp => input.up(context.history),
        Do::LineDown => input.down(context.history),
        Do::Up => match &mut view.completion {
            Some(completion) => completion.select(-1),
            None => input.up(context.history),
        },
        Do::Down => match &mut view.completion {
            Some(completion) => completion.select(1),
            None => input.down(context.history),
        },
        Do::ScrollUp | Do::ScrollDown | Do::PageUp | Do::PageDown => {
            if let Some((_, state)) = pane {
                let page = PAGE.min(state.page());
                match action {
                    Do::ScrollUp => state.scroll.up(1),
                    Do::ScrollDown => state.scroll.down(1),
                    Do::PageUp => state.scroll.up(page),
                    _ => state.scroll.down(page),
                }
            }
        }
        Do::Tab if view.completion.is_none() && context.busy && editing.is_none() => {
            let state = pane.as_mut().map(|(_, state)| &mut **state);
            send(view, input, state, context.agent, commands, true);
        }
        Do::Tab => accept_completion(view, input),
        Do::Send => {
            // Enter completes a partly typed command or path first.
            if view
                .completion
                .as_ref()
                .is_some_and(|completion| !completion.is_exact())
            {
                accept_completion(view, input);
                return;
            }
            if input.continue_line() {
                return;
            }
            let state = pane.as_mut().map(|(_, state)| &mut **state);
            send(view, input, state, context.agent, commands, false);
        }
        Do::Stop => match view.completion.take() {
            Some(completion) => view.dismissed = Some(completion.start),
            None => {
                if let Some(entity) = context.agent {
                    commands.trigger(StopTurn::new(entity));
                }
            }
        },
        Do::PasteImage => {
            let Some(images) = context.scratch.and_then(Scratch::images) else {
                let why = "This session keeps no files to paste an image into.";
                commands.write_message(Notice::info(context.agent, why));
                return;
            };
            let launcher = context
                .processes
                .map(Processes::launcher)
                .unwrap_or_default();
            clipboard::paste_image(images, launcher, context.wake, commands);
        }
        Do::SelectUp | Do::SelectDown => {
            if let Some((transcript, state)) = pane {
                select(action == Do::SelectUp, transcript, state);
            }
        }
        Do::Deselect => {
            if let Some((transcript, _)) = pane
                && transcript.selected.is_some()
            {
                transcript.selected = None;
            }
        }
        Do::AnswerInline(number) => {
            answer_inline(usize::from(number), target, (context, showing), commands);
        }
        Do::DismissInline => {
            if let Some(&ask) = view.inline.first() {
                view.inline.remove(0);
                commands.trigger(Pick::person(ask, None));
            }
        }
    }
}

/// Answers the question taking the input with `chosen`, `None` refusing
/// it, and puts the draft back, or shows the next question.
fn close(
    ask: Entity,
    chosen: Option<Chosen>,
    target: &mut Target,
    (context, showing): (&Context, &Showing),
    commands: &mut Commands,
) {
    commands.trigger(Pick::person(ask, chosen));
    target.view.queue.retain(|&queued| queued != ask);
    pick::show_first(
        target.view,
        context.choices,
        (showing, target.input),
        commands,
    );
}

/// Moves the selection up or down the items of the pane laid out last:
/// up from none selects the newest, down past the newest selects none. The
/// item selected is brought on screen.
fn select(up: bool, transcript: &mut Mut<TranscriptView>, state: &mut Mut<PaneState>) {
    let items = state
        .transcript
        .marks_in(&state.parts)
        .map(|(item, _)| item);
    let items: Vec<&TranscriptItem> = items.filter(|item| selectable(item)).collect();
    let at = (transcript.selected.as_ref())
        .and_then(|selected| items.iter().position(|item| *item == selected));
    let next = match (at, up) {
        (None, true) => items.len().checked_sub(1),
        (None, false) => None,
        (Some(at), true) => Some(at.saturating_sub(1)),
        (Some(at), false) => Some(at + 1),
    };
    let selected = next
        .and_then(|next| items.get(next))
        .map(|item| (*item).clone());
    if let Some(item) = &selected {
        state.scroll.to(item.clone(), ItemRow::Reveal);
    }
    if transcript.selected != selected {
        transcript.selected = selected;
    }
}

/// Answers the question shown beside the work with its choice `number`,
/// counted from 1, or after its choices, when it takes text, takes the
/// input for its answer.
fn answer_inline(
    number: usize,
    target: &mut Target,
    (context, showing): (&Context, &Showing),
    commands: &mut Commands,
) {
    let Some(&ask) = target.view.inline.first() else {
        return;
    };
    let Some(shown) = context.choices.shown(ask) else {
        return;
    };
    let Some(index) = number.checked_sub(1) else {
        return;
    };
    if index < shown.items.len() {
        target.view.inline.remove(0);
        commands.trigger(Pick::person(ask, Some(Chosen::item(index))));
    } else if index == shown.items.len() && shown.free_text && showing.is_empty() {
        target.view.inline.remove(0);
        target.view.queue.insert(0, ask);
        pick::show_first(
            target.view,
            context.choices,
            (showing, target.input),
            commands,
        );
    }
}

/// Sends the input to `agent`, the one shown, as the user's [`Deliver`]: a
/// slash command, or a message that starts a turn, or steers the running
/// one, or is queued for after it with `queue`. A refused command sent
/// again unchanged is a message. The prompt history records it, and the
/// pane the keys act on follows the newest output.
pub(crate) fn send(
    view: &mut TuiView,
    input: &mut Editor,
    pane: Option<&mut PaneState>,
    agent: Option<Entity>,
    commands: &mut Commands,
    queue: bool,
) {
    let Some(entity) = agent else {
        return;
    };
    if input.text().trim().is_empty() {
        return;
    }
    let text = input.take();
    commands.write_message(Sent(text.clone()));
    if let Some(pane) = pane {
        pane.scroll.follow();
    }
    let mode = if queue {
        DeliveryMode::Queue
    } else {
        DeliveryMode::Steer
    };
    let refused = view.refused.take().as_ref() == Some(&text);
    let typed = Deliver::new(entity, text, mode);
    commands.trigger(if refused { typed.literal() } else { typed });
}

/// Puts the selected completion in place of the token being completed.
fn accept_completion(view: &mut TuiView, input: &mut Editor) {
    let Some(completion) = view.completion.take() else {
        return;
    };
    if let Some(text) = completion.chosen() {
        input.replace_before_cursor(completion.start, &text);
    }
}

#[cfg(test)]
mod tests;
