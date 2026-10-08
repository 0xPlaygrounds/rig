//! Keyboard input, read without blocking: every frame drains the events
//! that are ready and turns them into messages.

use std::time::Duration;

use bevy::prelude::*;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use crate::core::{Agent, AgentStatus, Interrupt, Submit};

use super::view::TuiView;

/// Rows a page key scrolls.
const PAGE: usize = 10;

/// What the keys of one frame asked for, besides view changes.
#[derive(Default)]
struct Requests {
    submits: Vec<Submit>,
    interrupt: bool,
    exit: bool,
}

/// Drains the terminal's pending events.
pub(super) fn read_input(
    mut view: ResMut<TuiView>,
    agents: Query<(Entity, &AgentStatus), With<Agent>>,
    mut submits: MessageWriter<Submit>,
    mut interrupts: MessageWriter<Interrupt>,
    mut exit: MessageWriter<AppExit>,
) {
    let focus = agents.iter().next();
    let mut requests = Requests::default();
    while matches!(event::poll(Duration::ZERO), Ok(true)) {
        let Ok(event) = event::read() else {
            break;
        };
        match event {
            Event::Key(key) if key.kind != KeyEventKind::Release => {
                on_key(&mut view, key, focus.map(|(agent, _)| agent), &mut requests);
            }
            Event::Paste(text) => {
                let text = text.replace(['\r', '\n'], " ");
                match &mut view.picker {
                    Some(picker) => picker.filter.push_str(&text),
                    None => view.input.push_str(&text),
                }
            }
            Event::Resize(..) => view.set_changed(),
            _ => {}
        }
    }
    submits.write_batch(requests.submits);
    if let (true, Some((agent, status))) = (requests.interrupt, focus)
        && status.is_busy()
    {
        interrupts.write(Interrupt { agent });
    }
    if requests.exit {
        exit.write(AppExit::Success);
    }
}

fn on_key(view: &mut TuiView, key: KeyEvent, focus: Option<Entity>, requests: &mut Requests) {
    let control = key.modifiers.contains(KeyModifiers::CONTROL);
    if let Some(picker) = &mut view.picker {
        let count = picker.visible().len();
        let close = match key.code {
            KeyCode::Esc => true,
            KeyCode::Char('c') if control => true,
            KeyCode::Up => {
                picker.selected = picker.selected.saturating_sub(1);
                false
            }
            KeyCode::Down => {
                picker.selected = (picker.selected + 1).min(count.saturating_sub(1));
                false
            }
            KeyCode::Backspace => {
                picker.filter.pop();
                picker.selected = 0;
                false
            }
            KeyCode::Enter => {
                if let Some(choice) = picker.visible().get(picker.selected) {
                    requests.submits.push(Submit {
                        agent: picker.agent,
                        text: format!("/{} {}", picker.command, choice.value),
                    });
                }
                true
            }
            KeyCode::Char(c) if !control => {
                picker.filter.push(c);
                picker.selected = 0;
                false
            }
            _ => false,
        };
        if close {
            view.picker = None;
        }
        return;
    }
    match key.code {
        KeyCode::Enter => {
            let text = std::mem::take(&mut view.input);
            if let (false, Some(agent)) = (text.trim().is_empty(), focus) {
                view.notices.clear();
                view.scroll = 0;
                requests.submits.push(Submit { agent, text });
            }
        }
        KeyCode::Esc => {
            requests.interrupt = true;
            view.notices.clear();
        }
        KeyCode::Char('c') if control => {
            if view.input.is_empty() {
                requests.exit = true;
            }
            view.input.clear();
        }
        KeyCode::Char('d') if control && view.input.is_empty() => requests.exit = true,
        KeyCode::Char('u') if control => view.input.clear(),
        KeyCode::Char(c) if !control => view.input.push(c),
        KeyCode::Backspace => {
            view.input.pop();
        }
        KeyCode::PageUp => view.scroll = view.scroll.saturating_add(PAGE),
        KeyCode::PageDown => view.scroll = view.scroll.saturating_sub(PAGE),
        KeyCode::Up => view.scroll = view.scroll.saturating_add(1),
        KeyCode::Down => view.scroll = view.scroll.saturating_sub(1),
        _ => {}
    }
}
