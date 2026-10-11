//! Choices over the transcript: the pickers of [`rig_harness::pick`] and
//! the questions ([`Ask`]) of any plugin, which the view answers for
//! every agent. One is shown at a time, in the order they came; the next
//! waits until it is closed. A new [`Picker`] on the same entity starts it
//! over. The user's choice is a [`Pick`](rig_harness::pick::Pick), which the harness
//! carries out.

use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use rig_ecs::stage::{Effect, Outcome, Settled};
use rig_harness::ask::Ask;
use rig_harness::pick::{Chosen, Picker};

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
}

/// A picker or a question, as the view lists it.
pub(crate) struct Listing<'a> {
    pub(crate) title: &'a str,
    pub(crate) body: &'a str,
    pub(crate) labels: Vec<&'a str>,
    pub(crate) free_text: bool,
    pub(crate) selected: usize,
}

impl Listing<'_> {
    /// The rows that `filter` leaves: the items whose label holds every
    /// word of it, ignoring case, then, for free text, the typed answer.
    pub(crate) fn rows(&self, filter: &str) -> Vec<Row> {
        let lower = filter.to_lowercase();
        let mut rows: Vec<Row> = (0..self.labels.len())
            .filter(|&index| {
                let label = self.labels.get(index).map(|label| label.to_lowercase());
                let label = label.unwrap_or_default();
                lower.split_whitespace().all(|word| label.contains(word))
            })
            .map(Row::Item)
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
                labels: picker
                    .items
                    .iter()
                    .map(|item| item.label.as_str())
                    .collect(),
                free_text: picker.free_text,
                selected: picker.selected,
            });
        }
        let ask = self.asks.get(entity).ok()?;
        Some(Listing {
            title: &ask.title,
            body: &ask.body,
            labels: ask
                .choices
                .iter()
                .map(|choice| choice.label.as_str())
                .collect(),
            free_text: ask.free_text,
            selected: 0,
        })
    }
}

/// Keeps the queue of pickers and questions in the order they came, and
/// the first one shown: a new one waits, one closed or answered elsewhere
/// leaves, and a replaced picker starts over.
pub(crate) fn queue_choices(
    mut view: ResMut<TuiView>,
    pickers: Query<Entity, Added<Picker>>,
    replaced: Query<(), Changed<Picker>>,
    asks: Query<(Entity, &Effect), Added<Ask>>,
    choices: Choices,
) {
    let state = view.bypass_change_detection();
    let before = state.choosing.as_ref().map(|choosing| choosing.entity);
    let mut asked: Vec<(Entity, &Effect)> = asks.iter().collect();
    asked.sort_by_key(|(_, effect)| effect.id());
    state
        .queue
        .retain(|&entity| choices.shown(entity).is_some());
    for entity in pickers.iter().chain(asked.into_iter().map(|(ask, _)| ask)) {
        if !state.queue.contains(&entity) && choices.shown(entity).is_some() {
            state.queue.push(entity);
        }
    }
    let front = state.queue.first().copied();
    let restart = front.is_some_and(|front| replaced.contains(front));
    if front == before && !restart {
        return;
    }
    view.choosing = first(&view.queue, &choices);
}

/// The first of `queue` still open, shown from the start.
pub(crate) fn first(queue: &[Entity], choices: &Choices) -> Option<Choosing> {
    queue.iter().find_map(|&entity| {
        let shown = choices.shown(entity)?;
        Some(Choosing {
            entity,
            filter: String::new(),
            selected: shown.selected.min(shown.labels.len().saturating_sub(1)),
        })
    })
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
