//! Terminal input: a thread blocks on terminal events and a system drains
//! them each frame without blocking.

use async_channel::Receiver;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::error;
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use super::view::View;
use crate::core::agent::{Agent, Status};
use crate::core::registry::RunCommand;
use crate::core::turn::{Stop, Submit};

/// Terminal events read on the input thread.
#[derive(Resource)]
pub(crate) struct Input(Receiver<Event>);

impl Input {
    /// Starts the input thread. If it cannot start, no input arrives and
    /// the reason is logged.
    pub(crate) fn spawn() -> Self {
        let (sender, receiver) = async_channel::unbounded();
        let started = std::thread::Builder::new()
            .name("rig-code-input".to_owned())
            .spawn(move || {
                while let Ok(event) = crossterm::event::read() {
                    if sender.send_blocking(event).is_err() {
                        break;
                    }
                }
            });
        if let Err(error) = started {
            error!("cannot start the terminal input thread: {error}");
        }
        Self(receiver)
    }
}

/// Applies the terminal events that arrived since the last frame.
pub(crate) fn read_input(
    input: Res<Input>,
    mut view: ResMut<View>,
    agents: Query<Entity, With<Agent>>,
    statuses: Query<&Status>,
    mut commands: Commands,
    mut exit: MessageWriter<AppExit>,
) {
    if view.agent.is_none_or(|agent| !agents.contains(agent)) {
        view.agent = agents.iter().next();
    }
    let Some(agent) = view.agent else {
        return;
    };
    while let Ok(event) = input.0.try_recv() {
        match event {
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                    if view.composer.is_empty() {
                        exit.write(AppExit::Success);
                    } else {
                        view.composer.clear();
                    }
                } else if view.picker.is_some() {
                    picker_key(key, &mut view, &mut commands);
                } else {
                    let busy = statuses
                        .get(agent)
                        .is_ok_and(|status| *status != Status::Idle);
                    composer_key(key, agent, busy, &mut view, &mut commands);
                }
            }
            Event::Paste(text) => match &mut view.picker {
                Some(picker) => picker.filter.push_str(&text),
                None => view.composer.push_str(&text),
            },
            Event::Resize(..) => view.set_changed(),
            _ => {}
        }
    }
}

fn picker_key(key: KeyEvent, view: &mut View, commands: &mut Commands) {
    let Some(picker) = &mut view.picker else {
        return;
    };
    match key.code {
        KeyCode::Esc => view.picker = None,
        KeyCode::Enter => {
            if let Some(option) = picker.matches().get(picker.selected) {
                commands.trigger(RunCommand {
                    entity: picker.agent,
                    line: format!("/{} {}", picker.command, option.value),
                });
            }
            view.picker = None;
        }
        KeyCode::Up => picker.selected = picker.selected.saturating_sub(1),
        KeyCode::Down => {
            picker.selected = (picker.selected + 1).min(picker.matches().len().saturating_sub(1));
        }
        KeyCode::Backspace => {
            picker.filter.pop();
            picker.selected = 0;
        }
        KeyCode::Char(character) => {
            picker.filter.push(character);
            picker.selected = 0;
        }
        _ => {}
    }
}

/// Applies a key to the composer. A message to a `busy` agent is refused
/// with a notice and stays in the composer.
fn composer_key(
    key: KeyEvent,
    agent: Entity,
    busy: bool,
    view: &mut View,
    commands: &mut Commands,
) {
    match key.code {
        KeyCode::Esc => commands.trigger(Stop { entity: agent }),
        KeyCode::Enter => {
            view.scroll = 0;
            if view.composer.trim_start().starts_with('/') {
                commands.trigger(RunCommand {
                    entity: agent,
                    line: std::mem::take(&mut view.composer),
                });
            } else {
                let text = if busy {
                    view.composer.clone()
                } else {
                    std::mem::take(&mut view.composer)
                };
                commands.trigger(Submit {
                    entity: agent,
                    text,
                });
            }
        }
        KeyCode::Backspace => {
            view.composer.pop();
        }
        KeyCode::Char(character) => view.composer.push(character),
        KeyCode::Up => view.scroll += 1,
        KeyCode::Down => view.scroll = view.scroll.saturating_sub(1),
        KeyCode::PageUp => view.scroll += 10,
        KeyCode::PageDown => view.scroll = view.scroll.saturating_sub(10),
        _ => {}
    }
}
