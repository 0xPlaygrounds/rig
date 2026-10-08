//! Terminal input, read on a thread of its own that wakes the loop for
//! each event, as in Bevy's `examples/async_tasks/external_source_external_thread.rs`.
//! Keys edit the input or the view, or become the same requests any other
//! view sends.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use bevy_ecs::prelude::*;
use bevy_log::error;
use crossbeam_channel::Receiver;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use super::clipboard::{self, Clipboard};
use super::complete::{self, FileIndex};
use super::view::{Overlay, PickValue, Picker, TuiView};
use crate::core::agent::{ActiveTurn, Effort, Focus, Interrupt, SetEffort, SetModel, Submit};
use crate::core::calls::Wake;
use crate::core::commands::SlashCommand;
use crate::core::inbox::FollowUp;
use crate::core::journal::SessionPaths;
use crate::host::reload::{CancelReload, ReloadBuild};
use crate::host::sessions::SwitchSession;

/// Lines a page key scrolls.
const PAGE: usize = 10;

/// How long the input thread waits for an event before it checks whether
/// the view is gone.
const POLL: Duration = Duration::from_millis(100);

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
}

impl TerminalInput {
    /// Starts the input thread.
    pub(crate) fn start(wake: Wake) -> std::io::Result<Self> {
        let (sender, events) = crossbeam_channel::unbounded();
        let flags = Arc::new(Flags::default());
        let shared = Arc::clone(&flags);
        std::thread::Builder::new()
            .name("rig-harness-input".to_owned())
            .spawn(move || {
                while !shared.stop.load(Ordering::Relaxed) {
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
    clipboard: Res<Clipboard>,
    paths: Option<Res<SessionPaths>>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let busy = view
        .agent
        .and_then(|agent| agents.get(agent).ok())
        .unwrap_or(false);
    // Esc stops a running turn first, and a running rebuild only when the
    // agent is idle.
    let esc_cancels_reload = build.is_some_and(|build| !build.is_ready()) && !busy;
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
                None if key.code == KeyCode::Char('v')
                    && key.modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    let images = paths.as_ref().map_or_else(
                        || std::env::temp_dir().join("rig-images"),
                        |paths| paths.images(),
                    );
                    clipboard.paste_image(images, wake.clone());
                }
                None => {
                    edited = true;
                    let keys = Keys {
                        busy,
                        esc_cancels_reload,
                    };
                    input_key(key, &mut view, &mut commands, keys);
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
                    // A dropped image file becomes `@path`, which attaches it.
                    match clipboard::dropped_image(&text) {
                        Some(path) => {
                            let before = view
                                .editor
                                .text()
                                .get(..view.editor.cursor())
                                .unwrap_or_default();
                            let space = clipboard::separator(before);
                            view.editor.insert(&format!("{space}@{} ", path.display()));
                        }
                        None => view.editor.insert(&text),
                    }
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

/// What a key does depends on.
#[derive(Clone, Copy)]
struct Keys {
    /// The focused agent runs a turn: Enter steers it and Tab queues a
    /// follow-up.
    busy: bool,
    /// Esc cancels the running rebuild.
    esc_cancels_reload: bool,
}

fn input_key(
    key: KeyEvent,
    view: &mut TuiView,
    commands: &mut Commands,
    Keys {
        busy,
        esc_cancels_reload,
    }: Keys,
) {
    let control = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);
    let editor = &mut view.editor;
    match key.code {
        KeyCode::Char('c') if control => editor.clear(),
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
        KeyCode::Tab if view.completion.is_none() && busy => send(view, commands, true),
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
            send(view, commands, false);
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

/// Sends the input to the focused agent: as a follow-up for after its
/// turn, or as a message that starts a turn or steers the running one.
fn send(view: &mut TuiView, commands: &mut Commands, follow_up: bool) {
    let Some(entity) = view.agent else {
        return;
    };
    if view.editor.text().trim().is_empty() {
        return;
    }
    let text = view.editor.take();
    view.editor.remember(&text);
    view.scroll = 0;
    if follow_up {
        commands.trigger(FollowUp { entity, text });
    } else {
        commands.trigger(Submit { entity, text });
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
                Some(PickValue::Session(session)) => commands.trigger(SwitchSession {
                    session: Some(session),
                }),
                Some(PickValue::Agent(entity)) => commands.trigger(Focus { entity }),
                None => return false,
            }
            return true;
        }
        _ => {}
    }
    false
}
