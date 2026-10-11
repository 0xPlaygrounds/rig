//! Choices over the transcript: a plugin's [`Picker`], and the questions
//! ([`Ask`]) of any plugin, which the view answers for every agent. One is
//! shown at a time, in the order they came; the next waits until it is
//! closed. A picker is an entity of its plugin's: inserting a new
//! [`Picker`] replaces what it shows, and despawning it closes it. The view
//! despawns it once the user picks or closes it, after triggering
//! [`Picked`] on it, and runs the item's command, if it has one.
//!
//! ```no_run
//! use rig_harness::prelude::*;
//! use rig_tui::{Chosen, Picked, Picker};
//!
//! /// Asks which branch to check out, or a new one's name.
//! fn pick_branch(mut commands: Commands) {
//!     let picker = Picker::new("Branch").item("main", "").item("dev", "").free_text();
//!     commands.spawn(picker).observe(|picked: On<Picked>, mut notices: MessageWriter<Notice>| {
//!         let branch = match &picked.chosen {
//!             Some(Chosen::Item(0)) => "main".to_owned(),
//!             Some(Chosen::Item(_)) => "dev".to_owned(),
//!             Some(Chosen::Text(name)) => name.clone(),
//!             None => return,
//!         };
//!         notices.write(Notice::info(None, format!("checking out {branch}")));
//!     });
//! }
//! ```

use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use bevy_reflect::prelude::*;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use rig_ecs::commands::RunCommand;
use rig_ecs::stage::{Effect, Outcome, Settled};
use rig_harness::ask::{Answer, Ask};

use super::view::TuiView;

/// Lines a page key moves.
const PAGE: usize = 10;

/// One choice of a picker: what it shows, and the command line, without
/// its `/`, that choosing it runs for the agent, such as
/// `model openai/gpt-5`; an empty one runs nothing.
#[derive(Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Clone, Debug, Default, PartialEq)]
pub struct PickItem {
    /// What it shows.
    pub label: String,
    /// The command line it runs.
    pub command: String,
}

/// A filterable list for the user to choose one item from, on an entity of
/// the plugin's. See [`pick`](crate::pick).
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Clone, Debug, Default, PartialEq)]
pub struct Picker {
    /// The agent an item's command runs for; `None` for the agent shown.
    pub agent: Option<Entity>,
    /// What is picked.
    pub title: String,
    /// More about it, shown above the list; may be empty.
    pub body: String,
    /// The choices.
    pub items: Vec<PickItem>,
    /// The position of the choice selected at first.
    pub selected: usize,
    /// Whether what the user types may be the answer itself
    /// ([`Chosen::Text`]), besides filtering the items.
    pub free_text: bool,
}

impl Picker {
    /// An empty picker titled `title`.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            ..Self::default()
        }
    }

    /// The same picker, its items' commands run for `agent`.
    pub fn for_agent(mut self, agent: Entity) -> Self {
        self.agent = Some(agent);
        self
    }

    /// The same picker, about `body`.
    pub fn body(mut self, body: impl Into<String>) -> Self {
        self.body = body.into();
        self
    }

    /// The same picker with an item showing `label` that runs `command`
    /// (empty: nothing).
    pub fn item(mut self, label: impl Into<String>, command: impl Into<String>) -> Self {
        self.items.push(PickItem {
            label: label.into(),
            command: command.into(),
        });
        self
    }

    /// The same picker, also answered by what the user types.
    pub fn free_text(mut self) -> Self {
        self.free_text = true;
        self
    }
}

/// What the user chose.
#[derive(Reflect, Clone, Debug, PartialEq, Eq)]
#[reflect(Clone, Debug, PartialEq)]
pub enum Chosen {
    /// The item at this position of the [`Picker`]'s items.
    Item(usize),
    /// What the user typed, in a picker that takes free text.
    Text(String),
}

/// Triggered on a [`Picker`]'s entity when the user picks, or with `None`
/// closes it; the view then despawns it.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct Picked {
    /// The picker.
    pub entity: Entity,
    /// What was chosen, or `None` when the picker was closed.
    pub chosen: Option<Chosen>,
}

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

/// A picker or a question, as the view shows it.
pub(crate) struct Shown<'a> {
    pub(crate) title: &'a str,
    pub(crate) body: &'a str,
    pub(crate) labels: Vec<&'a str>,
    pub(crate) free_text: bool,
    pub(crate) selected: usize,
}

impl Shown<'_> {
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
    pub(crate) fn shown(&self, entity: Entity) -> Option<Shown<'_>> {
        if let Ok(picker) = self.pickers.get(entity) {
            return Some(Shown {
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
        Some(Shown {
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

    /// Carries out what the user chose for `entity`, `None` for closing
    /// it: runs a picker item's command for the agent, else the one shown,
    /// and triggers [`Picked`] before despawning the picker; answers a
    /// question, a closed one refused.
    pub(crate) fn close(
        &self,
        entity: Entity,
        chosen: Option<Chosen>,
        shown_agent: Option<Entity>,
        commands: &mut Commands,
    ) {
        if let Ok(picker) = self.pickers.get(entity) {
            let command = match &chosen {
                Some(Chosen::Item(index)) => picker.items.get(*index),
                _ => None,
            };
            if let (Some(item), Some(agent)) = (command, picker.agent.or(shown_agent))
                && !item.command.is_empty()
            {
                commands.trigger(RunCommand::new(agent, item.command.clone()));
            }
            commands.trigger(Picked { entity, chosen });
            commands.entity(entity).try_despawn();
            return;
        }
        let Ok(ask) = self.asks.get(entity) else {
            return;
        };
        let answer = match chosen {
            Some(Chosen::Item(index)) => ask
                .choices
                .get(index)
                .map(|choice| Answer::Choice(choice.id.clone()).outcome()),
            Some(Chosen::Text(text)) => Some(Answer::Text(text).outcome()),
            None => Some(Outcome::refused("The user closed the question.")),
        };
        if let Some(answer) = answer {
            commands.entity(entity).insert_if_new(answer);
        }
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
    shown: &Shown<'_>,
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
