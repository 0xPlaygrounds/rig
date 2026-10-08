//! Terminal input, polled without blocking once per frame. Keys edit the
//! view or become the same requests any other view sends.

use std::time::Duration;

use bevy_app::AppExit;
use bevy_ecs::prelude::*;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use super::view::{PickValue, TuiView};
use crate::core::agent::{AgentStatus, Interrupt, SetEffort, SetModel, Submit};
use crate::host::reload::{CancelReload, ReloadBuild};

/// Lines a page key scrolls.
const PAGE: usize = 10;

/// Reads every pending terminal event.
pub fn read_input(
    mut view: ResMut<TuiView>,
    agents: Query<&AgentStatus>,
    build: Option<Res<ReloadBuild>>,
    mut commands: Commands,
    mut exit: MessageWriter<AppExit>,
) -> Result {
    // Esc stops a running turn first, and a running rebuild only when the
    // agent is idle.
    let esc_cancels_reload = build.is_some_and(|build| !build.is_ready())
        && view
            .agent
            .and_then(|agent| agents.get(agent).ok())
            .is_some_and(|status| *status == AgentStatus::Idle);
    while event::poll(Duration::ZERO)? {
        match event::read()? {
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
    Ok(())
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
                Some(PickValue::Effort(effort)) => commands.trigger(SetEffort { entity, effort }),
                None => return,
            }
            view.picker = None;
        }
        _ => {}
    }
}
