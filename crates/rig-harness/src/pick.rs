//! Choices for the user that any front shows and answers: a plugin's
//! [`Picker`], a filterable list on an entity of the plugin's. Inserting a
//! new `Picker` replaces what it shows, and despawning it closes it. A
//! front that shows pickers (`Query<(Entity, &Picker)>`) triggers [`Pick`]
//! on the one the user picked from, or closed: the harness then runs the
//! item's command, for the picker's agent or else the agent the user is
//! looking at ([`ShownAgent`]), triggers [`Picked`] on
//! it, and despawns it. A picker may take several items
//! ([`Picker::multiple`]): the harness runs each one's command in list
//! order. A [`Chosen`] holds the items picked and the text typed, either
//! or both. A front answers an open [`Ask`] the same way: a `Pick` on it
//! decides the answer, a question's choices with the note typed on them.
//!
//! ```no_run
//! use rig_harness::pick::{Chosen, Picked, Picker};
//! use rig_harness::prelude::*;
//!
//! /// Asks which branch to check out, or a new one's name.
//! fn pick_branch(mut commands: Commands) {
//!     let picker = Picker::new("Branch").item("main", "").item("dev", "").free_text();
//!     commands.spawn(picker).observe(|picked: On<Picked>, mut notices: MessageWriter<Notice>| {
//!         let Some(chosen) = &picked.chosen else {
//!             return;
//!         };
//!         let branch = match (chosen.items.first(), &chosen.text) {
//!             (Some(0), _) => "main".to_owned(),
//!             (Some(_), _) => "dev".to_owned(),
//!             (None, Some(name)) => name.clone(),
//!             (None, None) => return,
//!         };
//!         notices.write(Notice::info(None, format!("checking out {branch}")));
//!     });
//! }
//! ```

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig_core::conversation::{Conversation, MessageId};
use rig_core::effect::Decider;
use rig_ecs::stage::{Decision, Outcome, Settled};

use crate::ask::{Answer, Ask};
use crate::commands::RunCommand;
use crate::main_agent::{CurrentAgentPlugin, ShownAgent};

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
    /// What a front shows when the user asks for more, such as a session's
    /// full folder; a front falls back to the detail.
    pub more: Option<String>,
    /// Whether it is ticked at first, in a picker of several items.
    pub checked: bool,
}

/// A filterable list for the user to choose one item, or several, from, on
/// an entity of the plugin's. See [`pick`](crate::pick).
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Clone, Debug, Default, PartialEq)]
pub struct Picker {
    /// The agent an item's command runs for; `None` for the shown agent.
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
    /// ([`Chosen`]'s `text`), besides filtering the items.
    pub free_text: bool,
    /// Whether the user may tick several items.
    pub multiple: bool,
    /// Whether what the user types is a secret, such as an API key: a front
    /// hides it as it is typed. It reaches the plugin only in [`Picked`].
    pub secret: bool,
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
            more: None,
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

    /// The same picker, answered by a secret the user types, which a front
    /// hides as it is typed.
    pub fn secret(mut self) -> Self {
        self.secret = true;
        self.free_text = true;
        self
    }
}

/// What the user chose: the items or choices picked and the text typed,
/// either or both, such as a question's choice with a note on it.
#[derive(Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Clone, Debug, Default, PartialEq)]
pub struct Chosen {
    /// The positions of the [`Picker`]'s items, or the [`Ask`]'s choices,
    /// picked: one, or any number for one that takes several; none for an
    /// answer only typed.
    pub items: Vec<usize>,
    /// What the user typed, where free text is taken: alone, or a note on
    /// the items.
    pub text: Option<String>,
}

impl Chosen {
    /// The item, or choice, at `index`.
    pub fn item(index: usize) -> Self {
        Self::items([index])
    }

    /// The items, or choices, at `indices`.
    pub fn items(indices: impl IntoIterator<Item = usize>) -> Self {
        Self {
            items: indices.into_iter().collect(),
            text: None,
        }
    }

    /// What the user typed, `text`.
    pub fn text(text: impl Into<String>) -> Self {
        Self::default().with_text(text)
    }

    /// The same choice, with `text` typed as its note.
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// The positions chosen, in list order, each once.
    fn positions(&self) -> Vec<usize> {
        let mut positions = self.items.clone();
        positions.sort_unstable();
        positions.dedup();
        positions
    }
}

/// The position of the item a front highlights in a [`Picker`], set by the
/// front on the picker's entity as the highlight moves, so a plugin can
/// preview that item ([`PickPreview`]).
#[derive(Component, Reflect, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Clone, Debug, Default, PartialEq)]
pub struct Highlighted(pub usize);

/// What a front shows beside a [`Picker`] of the plugin's, previewing the
/// highlighted item ([`Highlighted`]): text, or a conversation drawn as a
/// transcript, scrolled to entry `at` when given. Any plugin sets it on its
/// picker and replaces it as the highlight moves.
#[derive(Component, Clone, Debug, PartialEq)]
pub enum PickPreview {
    /// Plain text.
    Text(String),
    /// A conversation, such as a saved session's line.
    Conversation {
        /// The conversation.
        conversation: Conversation,
        /// The entry to show first; the end with `None`.
        at: Option<MessageId>,
    },
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
        if !app.is_plugin_added::<CurrentAgentPlugin>() {
            app.add_plugins(CurrentAgentPlugin);
        }
        app.add_observer(pick);
    }
}

/// Runs the chosen items' commands, in list order, for the picker's agent,
/// else the one shown, and triggers [`Picked`] before despawning the
/// picker; answers a question, a closed one refused.
fn pick(
    pick: On<Pick>,
    pickers: Query<&Picker>,
    mut asks: Query<(&Ask, &mut Decision), (Without<Outcome>, Without<Settled>)>,
    shown: Res<ShownAgent>,
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
        Some(chosen) => {
            let ids = chosen
                .positions()
                .into_iter()
                .map(|index| ask.choices.get(index));
            let ids: Option<Vec<String>> = ids.map(|c| c.map(|c| c.id.clone())).collect();
            let typed = chosen.text.is_some();
            ids.filter(|ids| ask.multiple || ids.len() == 1 || (ids.is_empty() && typed))
                .map(|choices| {
                    let answer = Answer {
                        choices,
                        text: chosen.text,
                    };
                    answer.outcome()
                })
        }
        None => Some(Outcome::refused("The user closed the question.")),
    };
    if let Some(answer) = answer {
        decision.decide(pick.by.clone(), answer);
    }
}

#[cfg(test)]
mod tests;
