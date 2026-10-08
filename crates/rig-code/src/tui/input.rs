use std::sync::{
    Mutex,
    mpsc::{self, Receiver},
};

use bevy_ecs::prelude::*;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use super::view::TuiView;
use crate::agent::{Interrupt, Quit, Submit};

/// Rows one PageUp or PageDown scrolls.
const PAGE: usize = 10;

/// Terminal events from the reader thread.
#[derive(Resource)]
pub(super) struct TuiInput(Mutex<Receiver<Event>>);

impl TuiInput {
    /// Start the thread that blocks on terminal events.
    pub(super) fn spawn() -> Self {
        let (sender, events) = mpsc::channel();
        std::thread::spawn(move || {
            while let Ok(event) = event::read() {
                if sender.send(event).is_err() {
                    break;
                }
            }
        });
        Self(Mutex::new(events))
    }
}

/// What the view asks the app for after a key.
enum Request {
    Submit(String),
    Interrupt,
    Quit,
}

/// Apply the terminal events that arrived since the last frame.
pub(super) fn read_input(
    mut input: ResMut<TuiInput>,
    mut view: ResMut<TuiView>,
    mut submits: MessageWriter<Submit>,
    mut interrupts: MessageWriter<Interrupt>,
    mut quits: MessageWriter<Quit>,
) {
    let events = input
        .0
        .get_mut()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    while let Ok(event) = events.try_recv() {
        view.dirty = true;
        let request = match event {
            Event::Key(key) if key.kind != KeyEventKind::Release => key_pressed(&mut view, key),
            Event::Paste(text) => {
                insert(&mut view, &text.replace(['\r', '\n'], " "));
                None
            }
            _ => None,
        };
        let Some(agent) = view.agent else {
            continue;
        };
        match request {
            Some(Request::Submit(text)) => {
                submits.write(Submit { agent, text });
            }
            Some(Request::Interrupt) => {
                interrupts.write(Interrupt { agent });
            }
            Some(Request::Quit) => {
                quits.write(Quit);
            }
            None => {}
        }
    }
}

fn key_pressed(view: &mut TuiView, key: KeyEvent) -> Option<Request> {
    let control = key.modifiers.contains(KeyModifiers::CONTROL);
    if control && key.code == KeyCode::Char('c') {
        if view.picker.take().is_some() || !view.input.is_empty() {
            view.input.clear();
            view.cursor = 0;
            return None;
        }
        return Some(Request::Quit);
    }
    if control && key.code == KeyCode::Char('d') && view.input.is_empty() {
        return Some(Request::Quit);
    }
    if view.picker.is_some() {
        return picker_key(view, key);
    }
    match key.code {
        KeyCode::Enter => {
            let text = std::mem::take(&mut view.input);
            view.cursor = 0;
            view.scroll = 0;
            view.notices.clear();
            return (!text.trim().is_empty()).then_some(Request::Submit(text));
        }
        KeyCode::Esc => return Some(Request::Interrupt),
        KeyCode::Char(c) if !control => insert(view, &c.to_string()),
        KeyCode::Backspace if view.cursor > 0 => {
            view.cursor -= 1;
            let at = byte_at(&view.input, view.cursor);
            view.input.remove(at);
        }
        KeyCode::Delete if view.cursor < view.input.chars().count() => {
            let at = byte_at(&view.input, view.cursor);
            view.input.remove(at);
        }
        KeyCode::Left => view.cursor = view.cursor.saturating_sub(1),
        KeyCode::Right => view.cursor = (view.cursor + 1).min(view.input.chars().count()),
        KeyCode::Home => view.cursor = 0,
        KeyCode::End => view.cursor = view.input.chars().count(),
        KeyCode::PageUp => view.scroll += PAGE,
        KeyCode::PageDown => view.scroll = view.scroll.saturating_sub(PAGE),
        _ => {}
    }
    None
}

fn picker_key(view: &mut TuiView, key: KeyEvent) -> Option<Request> {
    let picker = view.picker.as_mut()?;
    match key.code {
        KeyCode::Esc => view.picker = None,
        KeyCode::Enter => {
            let command = picker
                .filtered()
                .into_iter()
                .nth(picker.selected)
                .map(|choice| choice.command.clone());
            view.picker = None;
            return command.map(Request::Submit);
        }
        KeyCode::Up => picker.selected = picker.selected.saturating_sub(1),
        KeyCode::Down => {
            let last = picker.filtered().len().saturating_sub(1);
            picker.selected = (picker.selected + 1).min(last);
        }
        KeyCode::Backspace => {
            picker.filter.pop();
            picker.selected = 0;
        }
        KeyCode::Char(c) => {
            picker.filter.push(c);
            picker.selected = 0;
        }
        _ => {}
    }
    None
}

/// Insert `text` at the cursor.
fn insert(view: &mut TuiView, text: &str) {
    let at = byte_at(&view.input, view.cursor);
    view.input.insert_str(at, text);
    view.cursor += text.chars().count();
}

/// The byte offset of character `index`, or the end.
fn byte_at(text: &str, index: usize) -> usize {
    text.char_indices()
        .nth(index)
        .map_or(text.len(), |(at, _)| at)
}
