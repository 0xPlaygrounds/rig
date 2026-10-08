//! Terminal input, read on a thread of its own that wakes the loop for
//! each event, as in Bevy's `examples/async_tasks/external_source_external_thread.rs`.
//! Keys edit the view or become the same requests any other view sends.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use bevy_app::AppExit;
use bevy_ecs::prelude::*;
use bevy_log::error;
use crossbeam_channel::Receiver;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use super::view::{PickValue, TuiView};
use crate::core::agent::{ActiveTurn, Effort, Interrupt, SetEffort, SetModel, Submit};
use crate::core::calls::Wake;
use crate::host::reload::{CancelReload, ReloadBuild};

/// Lines a page key scrolls.
const PAGE: usize = 10;

/// How long the input thread waits for an event before it checks whether
/// the view is gone.
const POLL: Duration = Duration::from_millis(100);

/// Terminal events read by the input thread. Dropping it stops the thread.
#[derive(Resource)]
pub struct TerminalInput {
    events: Receiver<Event>,
    stop: Arc<AtomicBool>,
}

impl TerminalInput {
    /// Starts the input thread.
    pub fn start(wake: Wake) -> std::io::Result<Self> {
        let (sender, events) = crossbeam_channel::unbounded();
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        std::thread::Builder::new()
            .name("rig-code-input".to_owned())
            .spawn(move || {
                while !stopped.load(Ordering::Relaxed) {
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
        Ok(Self { events, stop })
    }
}

impl Drop for TerminalInput {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// Handles every terminal event read since the last frame.
pub fn read_input(
    input: Res<TerminalInput>,
    mut view: ResMut<TuiView>,
    agents: Query<Has<ActiveTurn>>,
    build: Option<Res<ReloadBuild>>,
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
    for event in input.events.try_iter() {
        match event {
            Event::Key(key) if key.kind != KeyEventKind::Release => {
                if view.reload_failure.is_some() {
                    // The report is modal: Esc or Enter closes it.
                    if matches!(key.code, KeyCode::Esc | KeyCode::Enter) {
                        view.reload_failure = None;
                    }
                } else if view.picker.is_some() {
                    picker_key(key, &mut view, &mut commands);
                } else {
                    input_key(key, &mut view, &mut commands, &mut exit, esc_cancels_reload);
                }
            }
            // A paste arrives whole, newlines included, so it is not sent
            // line by line.
            Event::Paste(text) => {
                let text = text.replace("\r\n", "\n").replace('\r', "\n");
                match view.picker.as_mut() {
                    Some(picker) => {
                        picker
                            .filter
                            .push_str(text.lines().next().unwrap_or_default());
                        picker.selected = 0;
                    }
                    None => view.input.push_str(&text),
                }
            }
            Event::Resize(..) => view.set_changed(),
            _ => {}
        }
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
    match key.code {
        KeyCode::Char('c' | 'd') if control => {
            if view.input.is_empty() {
                exit.write(AppExit::Success);
            } else {
                view.input.clear();
            }
        }
        KeyCode::Char(_) if control || key.modifiers.contains(KeyModifiers::ALT) => {}
        KeyCode::Char(character) => view.input.push(character),
        KeyCode::Backspace => {
            view.input.pop();
        }
        KeyCode::Enter => {
            let text = std::mem::take(&mut view.input);
            if let Some(entity) = view.agent
                && !text.trim().is_empty()
            {
                view.scroll = 0;
                commands.trigger(Submit { entity, text });
            }
        }
        KeyCode::Esc if esc_cancels_reload => commands.trigger(CancelReload),
        KeyCode::Esc => {
            if let Some(entity) = view.agent {
                commands.trigger(Interrupt { entity });
            }
        }
        KeyCode::Up => view.scroll += 1,
        KeyCode::Down => view.scroll = view.scroll.saturating_sub(1),
        KeyCode::PageUp => view.scroll += PAGE,
        KeyCode::PageDown => view.scroll = view.scroll.saturating_sub(PAGE),
        _ => {}
    }
}

fn picker_key(key: KeyEvent, view: &mut TuiView, commands: &mut Commands) {
    let Some(picker) = view.picker.as_mut() else {
        return;
    };
    let control = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        KeyCode::Esc => view.picker = None,
        KeyCode::Char('c') if control => view.picker = None,
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
                None => return,
            }
            view.picker = None;
        }
        _ => {}
    }
}
