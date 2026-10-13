//! Choices for the user that any front shows and answers: a plugin's
//! [`Picker`], a filterable list on an entity of the plugin's. Inserting a
//! new `Picker` replaces what it shows, and despawning it closes it. A
//! front that shows pickers (`Query<(Entity, &Picker)>`) triggers [`Pick`]
//! on the one the user picked from, or closed: the harness then runs the
//! item's command, for the picker's agent or else the agent the user is
//! looking at ([`Shown`]), triggers [`Picked`] on
//! it, and despawns it. A picker may take several items
//! ([`Picker::multiple`]), answered with [`Chosen::Items`]: the harness
//! runs each one's command in list order. A front answers an open [`Ask`]
//! the same way: a `Pick` on it decides the answer.
//!
//! ```no_run
//! use rig_harness::pick::{Chosen, Picked, Picker};
//! use rig_harness::prelude::*;
//!
//! /// Asks which branch to check out, or a new one's name.
//! fn pick_branch(mut commands: Commands) {
//!     let picker = Picker::new("Branch").item("main", "").item("dev", "").free_text();
//!     commands.spawn(picker).observe(|picked: On<Picked>, mut notices: MessageWriter<Notice>| {
//!         let branch = match &picked.chosen {
//!             Some(Chosen::Item(0)) => "main".to_owned(),
//!             Some(Chosen::Item(_)) => "dev".to_owned(),
//!             Some(Chosen::Text(name)) => name.clone(),
//!             _ => return,
//!         };
//!         notices.write(Notice::info(None, format!("checking out {branch}")));
//!     });
//! }
//! ```

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig_core::effect::Decider;
use rig_ecs::stage::{Decision, Outcome, Settled};

use crate::ask::{Answer, Ask};
use crate::commands::RunCommand;
use crate::primary::{Shown, keep_shown};

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
    /// What a front shows beside the label, kept whole where the label is
    /// shortened to fit, such as a session's age or why a model cannot be
    /// used.
    pub detail: Option<String>,
    /// Whether it is ticked at first, in a picker of several items.
    pub checked: bool,
}

/// A filterable list for the user to choose one item, or several, from, on
/// an entity of the plugin's. See [`pick`](crate::pick).
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
    /// Whether the user may tick several items ([`Chosen::Items`]).
    pub multiple: bool,
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
            detail: None,
            checked: false,
        });
        self
    }

    /// The same picker, its last item ticked at first, or not.
    pub fn checked(mut self, checked: bool) -> Self {
        if let Some(item) = self.items.last_mut() {
            item.checked = checked;
        }
        self
    }

    /// The same picker, taking several items.
    pub fn multiple(mut self) -> Self {
        self.multiple = true;
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
    /// The item at this position of the [`Picker`]'s items, or the choice
    /// at this position of the [`Ask`]'s choices.
    Item(usize),
    /// The items, or choices, at these positions, of a picker or question
    /// that takes several; a front that cannot tick items answers one.
    Items(Vec<usize>),
    /// What the user typed, where free text is taken.
    Text(String),
}

impl Chosen {
    /// The positions chosen, in list order, each once; none for text.
    fn positions(&self) -> Vec<usize> {
        let mut positions = match self {
            Self::Item(index) => vec![*index],
            Self::Items(indices) => indices.clone(),
            Self::Text(_) => Vec::new(),
        };
        positions.sort_unstable();
        positions.dedup();
        positions
    }
}

/// A front triggers it on a [`Picker`] or an open [`Ask`] when the user
/// picks, or with `None` closes it, with [`Pick::person`]; a plugin that
/// answers a question itself names itself in `by`. The harness carries out
/// the choice, and the question's record names who answered it.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct Pick {
    /// The picker or question.
    pub entity: Entity,
    /// What was chosen, or `None` when it was closed.
    pub chosen: Option<Chosen>,
    /// Who picked: [`Decider::Person`] from a front, else a plugin by name.
    /// Reflection leaves it out and reads it as a person.
    #[reflect(ignore, default = "person")]
    pub by: Decider,
}

/// [`Decider::Person`], for reflection.
fn person() -> Decider {
    Decider::Person
}

impl Pick {
    /// What the user chose on `entity` through a front.
    pub fn person(entity: Entity, chosen: Option<Chosen>) -> Self {
        Self {
            entity,
            chosen,
            by: Decider::Person,
        }
    }
}

/// Triggered on a [`Picker`]'s entity when the user picked, or with `None`
/// closed it, after its item's command was asked for and before it is
/// despawned.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct Picked {
    /// The picker.
    pub entity: Entity,
    /// What was chosen, or `None` when the picker was closed.
    pub chosen: Option<Chosen>,
}

/// Carries out [`Pick`]s.
pub struct PickPlugin;

impl Plugin for PickPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Shown>()
            .add_systems(First, keep_shown)
            .add_observer(pick);
    }
}

/// Runs the chosen items' commands, in list order, for the picker's agent,
/// else the one shown, and triggers [`Picked`] before despawning the
/// picker; answers a question, a closed one refused.
fn pick(
    pick: On<Pick>,
    pickers: Query<&Picker>,
    mut asks: Query<(&Ask, &mut Decision), (Without<Outcome>, Without<Settled>)>,
    shown: Res<Shown>,
    mut commands: Commands,
) {
    let (entity, chosen) = (pick.entity, pick.chosen.clone());
    if let Ok(picker) = pickers.get(entity) {
        let items = chosen.as_ref().map(Chosen::positions).unwrap_or_default();
        let items = items.iter().filter_map(|index| picker.items.get(*index));
        if let Some(agent) = picker.agent.or(shown.0) {
            for item in items.filter(|item| !item.command.is_empty()) {
                commands.trigger(RunCommand::new(agent, item.command.clone()));
            }
        }
        commands.trigger(Picked { entity, chosen });
        commands.entity(entity).try_despawn();
        return;
    }
    let Ok((ask, mut decision)) = asks.get_mut(entity) else {
        return;
    };
    let answer = match chosen {
        Some(Chosen::Text(text)) => Some(Answer::Text(text).outcome()),
        Some(chosen) => {
            let ids = chosen
                .positions()
                .into_iter()
                .map(|index| ask.choices.get(index));
            let ids: Option<Vec<String>> = ids.map(|c| c.map(|c| c.id.clone())).collect();
            match (ids, ask.multiple) {
                (Some(ids), true) => Some(Answer::Choices(ids).outcome()),
                (Some(mut ids), false) if ids.len() == 1 => {
                    ids.pop().map(|id| Answer::Choice(id).outcome())
                }
                _ => None,
            }
        }
        None => Some(Outcome::refused("The user closed the question.")),
    };
    if let Some(answer) = answer {
        decision.decide(pick.by.clone(), answer);
    }
}

#[cfg(test)]
mod tests;
