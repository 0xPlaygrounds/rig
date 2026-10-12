//! Choices over the transcript: the pickers of [`rig_harness::pick`] and
//! the questions ([`Ask`]) of any plugin, which the view answers for
//! every agent. One is shown at a time, in the order they came; the next
//! waits until it is closed. A new [`Picker`] on the same entity starts it
//! over. The user's choice is a [`Pick`](rig_harness::pick::Pick), which the harness
//! carries out.
//!
//! A question with initial text or a multi-line answer is answered in the
//! input instead: its text replaces the draft until Enter answers or Esc
//! refuses, and the draft comes back. A question shown beside the work
//! ([`Ask::inline`]) waits in a strip above the input, which keeps the
//! keys; Alt+1 to Alt+9 pick its choices, or answer it in the input, and
//! Alt+0 dismisses it.

use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::style::Stylize;
use ratatui::text::{Line, Span};
use rig_ecs::stage::{Effect, Outcome, Settled};
use rig_harness::ask::Ask;
use rig_harness::pick::{Chosen, Picker};
use rig_tools::shorten;
use unicode_width::UnicodeWidthStr;

use super::editor::TuiInput;
use super::view::TuiView;

/// Lines a page key moves.
const PAGE: usize = 10;

/// The choice shown: a [`Picker`] or an [`Ask`], what is typed to filter
/// or answer it, and the selected row.
pub(crate) struct Choosing {
    /// The picker or question.
    pub(crate) entity: Entity,
    /// What is typed.
    pub(crate) filter: String,
    /// The selected position among the visible rows.
    pub(crate) selected: usize,
    /// For a question answered in the input, what was typed there before,
    /// which comes back once it is answered.
    pub(crate) draft: Option<String>,
}

/// A picker or a question, as the view lists it.
pub(crate) struct Listing<'a> {
    pub(crate) title: &'a str,
    pub(crate) body: &'a str,
    pub(crate) items: Vec<Item<'a>>,
    pub(crate) free_text: bool,
    pub(crate) selected: usize,
    /// For a question answered in the input, the text it starts from.
    pub(crate) edit: Option<&'a str>,
}

/// An item or choice of a [`Listing`].
pub(crate) struct Item<'a> {
    pub(crate) label: &'a str,
    /// What is shown beside the label, such as a session's age.
    pub(crate) detail: Option<&'a str>,
}

impl Item<'_> {
    /// Its row, `width` wide: the label, shortened with `…` to fit, and
    /// the detail whole and dimmed at the right edge.
    pub(crate) fn line(&self, width: usize) -> Line<'static> {
        let Some(detail) = self.detail else {
            return Line::from(shorten(self.label, width));
        };
        let label = shorten(self.label, width.saturating_sub(detail.width() + 2));
        let gap = width.saturating_sub(label.width() + detail.width());
        Line::from(vec![
            Span::from(label),
            Span::from(" ".repeat(gap)),
            Span::from(detail.to_owned()).dim(),
        ])
    }
}

impl Listing<'_> {
    /// The rows that `filter` leaves: the items whose label holds every
    /// word of it, ignoring case, then, for free text, the typed answer.
    pub(crate) fn rows(&self, filter: &str) -> Vec<Row> {
        let lower = filter.to_lowercase();
        let mut rows: Vec<Row> = (self.items.iter().enumerate())
            .filter(|(_, item)| {
                let label = item.label.to_lowercase();
                lower.split_whitespace().all(|word| label.contains(word))
            })
            .map(|(index, _)| Row::Item(index))
            .collect();
        if self.free_text {
            rows.push(Row::Typed);
        }
        rows
    }
}

/// A row of a shown choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Row {
    /// The item or choice at this position.
    Item(usize),
    /// What is typed, as the answer.
    Typed,
}

/// The pickers and open questions the view can show.
#[derive(SystemParam)]
pub(crate) struct Choices<'w, 's> {
    pickers: Query<'w, 's, &'static Picker>,
    asks: Query<'w, 's, &'static Ask, (Without<Outcome>, Without<Settled>)>,
}

impl Choices<'_, '_> {
    /// How `entity` is shown, while it is an open picker or question.
    pub(crate) fn shown(&self, entity: Entity) -> Option<Listing<'_>> {
        if let Ok(picker) = self.pickers.get(entity) {
            return Some(Listing {
                title: &picker.title,
                body: &picker.body,
                items: picker
                    .items
                    .iter()
                    .map(|item| Item {
                        label: &item.label,
                        detail: item.detail.as_deref(),
                    })
                    .collect(),
                free_text: picker.free_text,
                selected: picker.selected,
                edit: None,
            });
        }
        let ask = self.asks.get(entity).ok()?;
        Some(Listing {
            title: &ask.title,
            body: &ask.body,
            items: ask
                .choices
                .iter()
                .map(|choice| Item {
                    label: &choice.label,
                    detail: None,
                })
                .collect(),
            free_text: ask.free_text,
            selected: 0,
            // One beside the work is answered in the input once the user
            // chooses to type the answer.
            edit: ask
                .initial
                .as_deref()
                .or((ask.multiline || ask.inline).then_some("")),
        })
    }
}

/// Keeps the queue of pickers and questions in the order they came, and
/// the first one shown: a new one waits, one closed or answered elsewhere
/// leaves, and a replaced picker starts over. A question shown beside the
/// work waits in its own list.
pub(crate) fn queue_choices(
    mut view: ResMut<TuiView>,
    mut input: ResMut<TuiInput>,
    pickers: Query<Entity, Added<Picker>>,
    replaced: Query<(), Changed<Picker>>,
    asks: Query<(Entity, &Effect, &Ask), Added<Ask>>,
    choices: Choices,
) {
    let state = view.bypass_change_detection();
    let before = state.choosing.as_ref().map(|choosing| choosing.entity);
    let inline_before = state.inline.clone();
    let mut asked: Vec<(Entity, &Effect, &Ask)> = asks.iter().collect();
    asked.sort_by_key(|(_, effect, _)| effect.id());
    let open = |entity: &Entity| choices.shown(*entity).is_some();
    state.queue.retain(open);
    state.inline.retain(open);
    let (inline, asked): (Vec<_>, Vec<_>) = asked.into_iter().partition(|(.., ask)| ask.inline);
    for (entity, ..) in inline {
        if !state.inline.contains(&entity) && open(&entity) {
            state.inline.push(entity);
        }
    }
    for entity in pickers.iter().chain(asked.into_iter().map(|(ask, ..)| ask)) {
        if !state.queue.contains(&entity) && open(&entity) {
            state.queue.push(entity);
        }
    }
    if state.inline != inline_before {
        view.set_changed();
    }
    let front = view.queue.first().copied();
    let restart = front.is_some_and(|front| replaced.contains(front));
    if front == before && !restart {
        return;
    }
    show_first(&mut view, &choices, &mut input);
}

/// Shows the first of the queue still open, from the start. A question
/// answered in the input takes the input, keeping the draft, which comes
/// back when it closes.
pub(crate) fn show_first(view: &mut TuiView, choices: &Choices, input: &mut TuiInput) {
    if let Some(Choosing {
        draft: Some(draft), ..
    }) = view.choosing.take()
    {
        input.set(draft);
    }
    view.choosing = view.queue.iter().find_map(|&entity| {
        let shown = choices.shown(entity)?;
        let draft = shown.edit.map(|initial| {
            let draft = input.take();
            input.set(initial);
            draft
        });
        Some(Choosing {
            entity,
            filter: String::new(),
            selected: shown.selected.min(shown.items.len().saturating_sub(1)),
            draft,
        })
    });
}

/// Handles a key in the shown choice; returns what closes it: `Some(None)`
/// for closing without a choice.
pub(crate) fn choice_key(
    key: KeyEvent,
    choosing: &mut Choosing,
    shown: &Listing<'_>,
) -> Option<Option<Chosen>> {
    let control = key.modifiers.contains(KeyModifiers::CONTROL);
    let rows = shown.rows(&choosing.filter);
    let last = rows.len().saturating_sub(1);
    match key.code {
        KeyCode::Esc => return Some(None),
        KeyCode::Char('c') if control => return Some(None),
        KeyCode::Char(_) if control || key.modifiers.contains(KeyModifiers::ALT) => {}
        KeyCode::Char(character) => {
            choosing.filter.push(character);
            choosing.selected = 0;
        }
        KeyCode::Backspace => {
            choosing.filter.pop();
            choosing.selected = 0;
        }
        KeyCode::Up => choosing.selected = choosing.selected.saturating_sub(1),
        KeyCode::Down => choosing.selected = (choosing.selected + 1).min(last),
        KeyCode::PageUp => choosing.selected = choosing.selected.saturating_sub(PAGE),
        KeyCode::PageDown => choosing.selected = (choosing.selected + PAGE).min(last),
        KeyCode::Enter => {
            return match rows.get(choosing.selected)? {
                Row::Item(index) => Some(Some(Chosen::Item(*index))),
                Row::Typed if choosing.filter.trim().is_empty() => None,
                Row::Typed => Some(Some(Chosen::Text(choosing.filter.trim().to_owned()))),
            };
        }
        _ => {}
    }
    None
}

#[cfg(test)]
mod tests;
