//! Terminal input, read without blocking, turned into composer edits,
//! picker moves and the core's messages.

use std::time::Duration;

use bevy::prelude::*;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use super::view::{Composer, Focus, Picker, Scroll};
use crate::ecs::{Interrupt, Submit};

/// Lines one PageUp or PageDown scrolls.
const PAGE: usize = 10;

/// Read every pending terminal event. The poll has a zero timeout, so the
/// app's loop never waits on the terminal.
pub(super) fn read_terminal(
    focus: Res<Focus>,
    mut composer: ResMut<Composer>,
    mut picker: ResMut<Picker>,
    mut scroll: ResMut<Scroll>,
    mut submits: MessageWriter<Submit>,
    mut interrupts: MessageWriter<Interrupt>,
    mut exit: MessageWriter<AppExit>,
) {
    while event::poll(Duration::ZERO).unwrap_or(false) {
        let Ok(event) = event::read() else {
            return;
        };
        let Some(agent) = focus.0 else {
            continue;
        };
        match event {
            Event::Key(key) if key.kind == KeyEventKind::Press => {
                if picker.0.is_some() {
                    picker_key(key, &mut picker, agent, &mut submits);
                    continue;
                }
                match (key.code, key.modifiers) {
                    (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                        if composer.text.is_empty() {
                            exit.write(AppExit::Success);
                        } else {
                            composer.take();
                        }
                    }
                    (KeyCode::Esc, _) => {
                        interrupts.write(Interrupt { agent });
                    }
                    (KeyCode::Enter, KeyModifiers::ALT | KeyModifiers::SHIFT) => {
                        composer.insert("\n");
                    }
                    (KeyCode::Enter, _) => {
                        let text = composer.take();
                        if !text.trim().is_empty() {
                            scroll.0 = 0;
                            submits.write(Submit { agent, text });
                        }
                    }
                    (KeyCode::Char(c), modifiers) if !modifiers.contains(KeyModifiers::CONTROL) => {
                        composer.insert(c.encode_utf8(&mut [0; 4]));
                    }
                    (KeyCode::Backspace, _) => composer.backspace(),
                    (KeyCode::Delete, _) => composer.delete(),
                    (KeyCode::Left, _) => composer.move_by(-1),
                    (KeyCode::Right, _) => composer.move_by(1),
                    (KeyCode::Home, _) => composer.cursor = 0,
                    (KeyCode::End, _) => composer.end(),
                    (KeyCode::PageUp, _) => scroll.0 += PAGE,
                    (KeyCode::PageDown, _) => scroll.0 = scroll.0.saturating_sub(PAGE),
                    _ => {}
                }
            }
            Event::Paste(text) => {
                if let Some(state) = &mut picker.0 {
                    state.filter.push_str(&text);
                    state.selected = 0;
                } else {
                    composer.insert(&text.replace("\r\n", "\n").replace('\r', "\n"));
                }
            }
            _ => {}
        }
    }
}

/// Keys while a picker is open: type to filter, arrows to move, Enter to
/// submit the choice as a command, Esc to close.
fn picker_key(
    key: KeyEvent,
    picker: &mut Picker,
    agent: Entity,
    submits: &mut MessageWriter<Submit>,
) {
    let Some(state) = &mut picker.0 else {
        return;
    };
    let count = state.filtered().len();
    match key.code {
        KeyCode::Esc => picker.0 = None,
        KeyCode::Enter => {
            if let Some((_, value)) = state.filtered().get(state.selected) {
                submits.write(Submit {
                    agent,
                    text: format!("/{} {value}", state.command),
                });
            }
            picker.0 = None;
        }
        KeyCode::Up => state.selected = state.selected.saturating_sub(1),
        KeyCode::Down => state.selected = (state.selected + 1).min(count.saturating_sub(1)),
        KeyCode::Backspace => {
            state.filter.pop();
            state.selected = 0;
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            state.filter.push(c);
            state.selected = 0;
        }
        _ => {}
    }
}
