//! Terminal input, read on a thread of its own that wakes the loop for
//! each event, as in Bevy's `examples/async_tasks/external_source_external_thread.rs`.
//! Each event becomes a [`TerminalEvent`] message, which the view handles
//! in `PreUpdate`: the shown picker or question takes the keys, else a
//! [`KeyBinding`](crate::KeyBinding) (the view's own editing keys are
//! bindings too), else a character is typed into the input.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use bevy_log::error;
use crossbeam_channel::Receiver;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use ratatui::layout::Rect;

use super::clipboard::{self, Clipboard};
use super::editor::TuiInput;
use super::keys::{self, Found, InputAction, KeyBinding, KeyFocus, KeyPress, KeyPressed, KeyScope};
use super::panel::TuiScreen;
use super::pick::{self, Choices};
use super::view::TuiView;
use rig_ecs::agent::{ActiveTurn, Interrupt};
use rig_ecs::calls::Wake;
use rig_ecs::inbox::{Deliver, DeliveryMode};
use rig_harness::prelude::SessionPaths;

/// Lines a page key scrolls.
const PAGE: usize = 10;

/// How long the input thread waits for an event before it checks whether
/// the view is gone.
const POLL: Duration = Duration::from_millis(100);

/// An event the terminal reported: a key, a paste, a resize. The view
/// handles each one written, so a plugin may also write some, such as keys
/// of a macro, and read them, such as to know that the user types.
#[derive(Message, Clone, Debug, PartialEq, Eq)]
pub struct TerminalEvent(pub Event);

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
pub(crate) fn receive_events(input: Res<TerminalInput>, mut events: MessageWriter<TerminalEvent>) {
    events.write_batch(input.events.try_iter().map(TerminalEvent));
}

/// The key bindings and who has the keys.
#[derive(SystemParam)]
pub(crate) struct Keymap<'w, 's> {
    bindings: Query<'w, 's, (Entity, &'static KeyBinding, Option<&'static InputAction>)>,
    focus: ResMut<'w, KeyFocus>,
    panels: Query<'w, 's, ()>,
}

impl Keymap<'_, '_> {
    /// Where the keys go besides the global bindings: the focused panel
    /// while it exists, else the input.
    fn scope(&mut self) -> KeyScope {
        match self.focus.0 {
            Some(panel) if self.panels.contains(panel) => KeyScope::Panel(panel),
            Some(_) => {
                self.focus.0 = None;
                KeyScope::Input
            }
            None => KeyScope::Input,
        }
    }
}

/// What a built-in key needs besides the view and the input.
pub(crate) struct Context<'a> {
    busy: bool,
    clipboard: &'a Clipboard,
    paths: Option<&'a SessionPaths>,
    wake: &'a Wake,
}

/// Handles every terminal event written since the last frame.
pub(crate) fn read_input(
    mut events: MessageReader<TerminalEvent>,
    mut view: ResMut<TuiView>,
    mut input: ResMut<TuiInput>,
    mut keymap: Keymap,
    choices: Choices,
    agents: Query<Has<ActiveTurn>>,
    (clipboard, paths, wake): (Res<Clipboard>, Option<Res<SessionPaths>>, Res<Wake>),
    mut screen: ResMut<TuiScreen>,
    mut commands: Commands,
) {
    let busy = view
        .agent
        .and_then(|agent| agents.get(agent).ok())
        .unwrap_or(false);
    let context = Context {
        busy,
        clipboard: &clipboard,
        paths: paths.as_deref(),
        wake: &wake,
    };
    for TerminalEvent(event) in events.read() {
        match event {
            Event::Key(key) if key.kind == KeyEventKind::Release => {
                released(*key, &view, &mut keymap, &mut commands);
            }
            Event::Key(key) => {
                if let Some(choosing) = &mut view.choosing {
                    let Some(shown) = choices.shown(choosing.entity) else {
                        continue;
                    };
                    if let Some(chosen) = pick::choice_key(*key, choosing, &shown) {
                        let entity = choosing.entity;
                        choices.close(entity, chosen, view.agent, &mut commands);
                        // The next one takes the keys that follow.
                        view.queue.retain(|&queued| queued != entity);
                        view.choosing = pick::first(&view.queue, &choices);
                    }
                    continue;
                }
                pressed(
                    *key,
                    &mut view,
                    &mut input,
                    &mut keymap,
                    &context,
                    &mut commands,
                );
            }
            // A paste arrives whole, newlines included, so it is not sent
            // line by line.
            Event::Paste(text) => match &mut view.choosing {
                Some(choosing) => {
                    choosing
                        .filter
                        .push_str(text.lines().next().unwrap_or_default());
                    choosing.selected = 0;
                }
                // A dropped image file becomes `@path`, which attaches it.
                None => match clipboard::dropped_image(text) {
                    Some(path) => clipboard::type_path(&mut input, &path),
                    None => input.insert(text),
                },
            },
            Event::Resize(width, height) => {
                view.set_changed();
                screen.set_if_neq(TuiScreen(Rect::new(0, 0, *width, *height)));
            }
            _ => {}
        }
    }
}

/// Handles a key pressed with no picker or question shown: a binding's,
/// the next of a binding of several, else a character typed into the
/// input. With a panel focused, Esc that it does not bind gives the keys
/// back to the input.
fn pressed(
    key: KeyEvent,
    view: &mut TuiView,
    input: &mut TuiInput,
    keymap: &mut Keymap,
    context: &Context,
    commands: &mut Commands,
) {
    let press = KeyPress::of(&key);
    let scope = keymap.scope();
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
            KeyScope::Panel(_) => {}
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
        Some(action) => act(*action, view, input, context, commands),
        None => commands.trigger(KeyPressed {
            entity,
            agent: view.agent,
            released: false,
        }),
    }
}

/// Tells a binding that asked for releases that its key was let go.
fn released(key: KeyEvent, view: &TuiView, keymap: &mut Keymap, commands: &mut Commands) {
    let press = KeyPress::of(&key);
    let scope = keymap.scope();
    let released = keymap.bindings.iter().find(|(_, binding, _)| {
        binding.releases
            && binding.keys.presses() == [press]
            && (binding.scope == KeyScope::Global || binding.scope == scope)
    });
    if let Some((entity, ..)) = released {
        commands.trigger(KeyPressed {
            entity,
            agent: view.agent,
            released: true,
        });
    }
}

/// Does what a built-in key does. With a turn running, Enter steers it and
/// Tab queues a follow-up. Esc interrupts the agent: it stops its turn,
/// or, idle, what a plugin stops on it, such as a rebuild.
fn act(
    action: InputAction,
    view: &mut TuiView,
    input: &mut TuiInput,
    context: &Context,
    commands: &mut Commands,
) {
    use InputAction as Do;
    match action {
        Do::Clear => input.clear(),
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
        Do::LineUp => input.up(),
        Do::LineDown => input.down(),
        Do::Up => match &mut view.completion {
            Some(completion) => completion.select(-1),
            None => input.up(),
        },
        Do::Down => match &mut view.completion {
            Some(completion) => completion.select(1),
            None => input.down(),
        },
        Do::ScrollUp => view.scroll.up(1),
        Do::ScrollDown => view.scroll.down(1),
        Do::PageUp => view.scroll.up(PAGE),
        Do::PageDown => view.scroll.down(PAGE),
        Do::Tab if view.completion.is_none() && context.busy => send(view, input, commands, true),
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
            send(view, input, commands, false);
        }
        Do::Stop => match view.completion.take() {
            Some(completion) => view.dismissed = Some(completion.start),
            None => {
                if let Some(entity) = view.agent {
                    commands.trigger(Interrupt::new(entity));
                }
            }
        },
        Do::PasteImage => {
            let images = context.paths.map_or_else(
                || std::env::temp_dir().join("rig-images"),
                |paths| paths.images(),
            );
            context.clipboard.paste_image(images, context.wake.clone());
        }
    }
}

/// Sends the input to the focused agent as the user's [`Deliver`]: a slash
/// command, or a message that starts a turn, or steers the running one, or
/// is queued for after it with `queue`. A refused command sent again
/// unchanged is a message.
pub(crate) fn send(view: &mut TuiView, input: &mut TuiInput, commands: &mut Commands, queue: bool) {
    let Some(entity) = view.agent else {
        return;
    };
    if input.text().trim().is_empty() {
        return;
    }
    let text = input.take();
    input.remember(&text);
    view.scroll.follow();
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
fn accept_completion(view: &mut TuiView, input: &mut TuiInput) {
    let Some(completion) = view.completion.take() else {
        return;
    };
    if let Some(text) = completion.chosen() {
        input.replace_before_cursor(completion.start, &text);
    }
}

#[cfg(test)]
mod tests;
