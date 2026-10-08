//! Terminal input, read on a thread of its own that wakes the loop for
//! each event, as in Bevy's `examples/async_tasks/external_source_external_thread.rs`.
//! Keys edit the input or the view, or become the same requests any other
//! view sends.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use bevy_app::AppExit;
use bevy_ecs::prelude::*;
use bevy_log::error;
use crossbeam_channel::Receiver;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use super::complete::{self, FileIndex};
use super::external::EditRequested;
use super::view::{Overlay, PickValue, Picker, TuiView};
use crate::core::agent::{ActiveTurn, Effort, Interrupt, SetEffort, SetModel, Submit};
use crate::core::calls::Wake;
use crate::core::commands::SlashCommand;
use crate::host::reload::{CancelReload, ReloadBuild};

/// Lines a page key scrolls.
const PAGE: usize = 10;

/// How long the input thread waits for an event before it checks whether
/// the view is gone.
const POLL: Duration = Duration::from_millis(100);

/// How long [`TerminalInput::pause`] waits for the thread to stop reading.
const PAUSE_WAIT: Duration = Duration::from_millis(500);

/// Terminal events read by the input thread. Dropping it stops the thread.
#[derive(Resource)]
pub(crate) struct TerminalInput {
    events: Receiver<Event>,
    flags: Arc<Flags>,
}

/// What the input thread is told and tells back.
#[derive(Default)]
struct Flags {
    stop: AtomicBool,
    /// Asked to stop reading for now.
    pause: AtomicBool,
    /// Set by the thread while it is paused and not reading.
    paused: AtomicBool,
}

impl TerminalInput {
    /// Starts the input thread.
    pub(crate) fn start(wake: Wake) -> std::io::Result<Self> {
        let (sender, events) = crossbeam_channel::unbounded();
        let flags = Arc::new(Flags::default());
        let shared = Arc::clone(&flags);
        std::thread::Builder::new()
            .name("rig-code-input".to_owned())
            .spawn(move || {
                while !shared.stop.load(Ordering::Relaxed) {
                    if shared.pause.load(Ordering::Acquire) {
                        shared.paused.store(true, Ordering::Release);
                        std::thread::sleep(Duration::from_millis(20));
                        continue;
                    }
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
        Ok(Self { events, flags })
    }

    /// Stops reading the terminal, so another program gets the keys. Waits
    /// for a read in progress, up to half a second.
    pub(crate) fn pause(&self) {
        self.flags.pause.store(true, Ordering::Release);
        let deadline = Instant::now() + PAUSE_WAIT;
        while !self.flags.paused.load(Ordering::Acquire) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// Reads the terminal again after [`TerminalInput::pause`].
    pub(crate) fn resume(&self) {
        self.flags.paused.store(false, Ordering::Release);
        self.flags.pause.store(false, Ordering::Release);
    }
}

impl Drop for TerminalInput {
    fn drop(&mut self) {
        self.flags.stop.store(true, Ordering::Relaxed);
    }
}

/// Handles every terminal event read since the last frame, then brings the
/// completion list up to date with the input.
pub(crate) fn read_input(
    input: Res<TerminalInput>,
    mut view: ResMut<TuiView>,
    agents: Query<Has<ActiveTurn>>,
    build: Option<Res<ReloadBuild>>,
    slash: Query<&SlashCommand>,
    mut index: ResMut<FileIndex>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut exit: MessageWriter<AppExit>,
) {
    // Esc stops a running turn first, and a running rebuild only when the
    // agent is idle.
    let esc_cancels_reload = build.is_some_and(|build| !build.is_ready())
        && view
            .agent
            .and_then(|agent| agents.get(agent).ok())
            .is_some_and(|busy| !busy);
    let mut edited = false;
    for event in input.events.try_iter() {
        match event {
            Event::Key(key) if key.kind != KeyEventKind::Release => match &mut view.overlay {
                // The report is modal: Esc or Enter closes it.
                Some(Overlay::ReloadFailure(_)) => {
                    if matches!(key.code, KeyCode::Esc | KeyCode::Enter) {
                        view.overlay = None;
                    }
                }
                Some(Overlay::Picker(picker)) => {
                    if picker_key(key, picker, &mut commands) {
                        view.overlay = None;
                    }
                }
                None => {
                    edited = true;
                    input_key(key, &mut view, &mut commands, &mut exit, esc_cancels_reload);
                }
            },
            // A paste arrives whole, newlines included, so it is not sent
            // line by line.
            Event::Paste(text) => match &mut view.overlay {
                Some(Overlay::Picker(picker)) => {
                    picker
                        .filter
                        .push_str(text.lines().next().unwrap_or_default());
                    picker.selected = 0;
                }
                Some(Overlay::ReloadFailure(_)) => {}
                None => {
                    edited = true;
                    view.editor.insert(&text);
                }
            },
            Event::Resize(..) => view.set_changed(),
            _ => {}
        }
    }
    if edited {
        complete::update(&mut view, &mut index, &slash, &wake);
    }
}

fn input_key(
    key: KeyEvent,
    view: &mut TuiView,
    commands: &mut Commands,
    exit: &mut MessageWriter<AppExit>,
    esc_cancels_reload: bool,
) {
    let control = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);
    let editor = &mut view.editor;
    match key.code {
        KeyCode::Char('c') if control => {
            if editor.is_empty() {
                exit.write(AppExit::Success);
            } else {
                editor.clear();
            }
        }
        KeyCode::Char('d') if control => {
            if editor.is_empty() {
                exit.write(AppExit::Success);
            } else {
                editor.delete();
            }
        }
        KeyCode::Char('g') if control => commands.insert_resource(EditRequested),
        KeyCode::Char('j') if control => editor.insert_char('\n'),
        KeyCode::Char('a') if control => editor.home(),
        KeyCode::Char('e') if control => editor.end(),
        KeyCode::Char('b') if control => editor.left(),
        KeyCode::Char('f') if control => editor.right(),
        KeyCode::Char('k') if control => editor.delete_to_line_end(),
        KeyCode::Char('u') if control => editor.delete_to_line_start(),
        KeyCode::Char('w') if control => editor.delete_word_before(),
        KeyCode::Char('p') if control => editor.up(),
        KeyCode::Char('n') if control => editor.down(),
        KeyCode::Char('b') if alt => editor.word_left(),
        KeyCode::Char('f') if alt => editor.word_right(),
        KeyCode::Char('d') if alt => editor.delete_word_after(),
        KeyCode::Char(_) if control || alt => {}
        KeyCode::Char(character) => editor.insert_char(character),
        KeyCode::Backspace if control || alt => editor.delete_word_before(),
        KeyCode::Backspace => editor.backspace(),
        KeyCode::Delete if control || alt => editor.delete_word_after(),
        KeyCode::Delete => editor.delete(),
        KeyCode::Left if control || alt => editor.word_left(),
        KeyCode::Left => editor.left(),
        KeyCode::Right if control || alt => editor.word_right(),
        KeyCode::Right => editor.right(),
        KeyCode::Home => editor.home(),
        KeyCode::End => editor.end(),
        KeyCode::Tab => accept_completion(view),
        KeyCode::Enter if shift || alt => editor.insert_char('\n'),
        KeyCode::Enter => {
            // Enter completes a partly typed command or path first.
            if view
                .completion
                .as_ref()
                .is_some_and(|completion| !completion.is_exact())
            {
                accept_completion(view);
                return;
            }
            if view.editor.continue_line() {
                return;
            }
            if let Some(entity) = view.agent
                && !view.editor.text().trim().is_empty()
            {
                let text = view.editor.take();
                view.editor.remember(&text);
                view.scroll = 0;
                commands.trigger(Submit { entity, text });
            }
        }
        KeyCode::Esc if view.completion.is_some() => {
            view.dismissed = view.completion.take().map(|completion| completion.start);
        }
        KeyCode::Esc if esc_cancels_reload => commands.trigger(CancelReload),
        KeyCode::Esc => {
            if let Some(entity) = view.agent {
                commands.trigger(Interrupt { entity });
            }
        }
        KeyCode::Up if shift => view.scroll += 1,
        KeyCode::Down if shift => view.scroll = view.scroll.saturating_sub(1),
        KeyCode::Up => match &mut view.completion {
            Some(completion) => completion.select(-1),
            None => view.editor.up(),
        },
        KeyCode::Down => match &mut view.completion {
            Some(completion) => completion.select(1),
            None => view.editor.down(),
        },
        KeyCode::PageUp => view.scroll += PAGE,
        KeyCode::PageDown => view.scroll = view.scroll.saturating_sub(PAGE),
        _ => {}
    }
}

/// Puts the selected completion in place of the token being completed.
fn accept_completion(view: &mut TuiView) {
    let Some(completion) = view.completion.take() else {
        return;
    };
    if let Some(text) = completion.chosen() {
        view.editor.replace_before_cursor(completion.start, &text);
    }
}

/// Handles a key in the open picker; returns whether the picker closes.
fn picker_key(key: KeyEvent, picker: &mut Picker, commands: &mut Commands) -> bool {
    let control = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        KeyCode::Esc => return true,
        KeyCode::Char('c') if control => return true,
        KeyCode::Char(_) if control || key.modifiers.contains(KeyModifiers::ALT) => {}
        KeyCode::Char(character) => {
            picker.filter.push(character);
            picker.selected = 0;
        }
        KeyCode::Backspace => {
            picker.filter.pop();
            picker.selected = 0;
        }
        KeyCode::Up => picker.selected = picker.selected.saturating_sub(1),
        KeyCode::Down => {
            let last = picker.visible().len().saturating_sub(1);
            picker.selected = (picker.selected + 1).min(last);
        }
        KeyCode::PageUp => picker.selected = picker.selected.saturating_sub(PAGE),
        KeyCode::PageDown => {
            let last = picker.visible().len().saturating_sub(1);
            picker.selected = (picker.selected + PAGE).min(last);
        }
        KeyCode::Enter => {
            let entity = picker.agent;
            match picker.chosen() {
                Some(PickValue::Model(model)) => commands.trigger(SetModel { entity, model }),
                Some(PickValue::Effort(effort)) => commands.trigger(SetEffort {
                    entity,
                    effort: Effort(effort),
                }),
                None => return false,
            }
            return true;
        }
        _ => {}
    }
    false
}
