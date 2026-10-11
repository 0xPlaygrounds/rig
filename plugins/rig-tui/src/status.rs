//! The status line: what the view shows under the transcript of the agent
//! it shows. Each agent, and its running turn, has its [`StatusItems`], and
//! the app its [`AppStatus`]; a plugin sets its items there in
//! [`StatusSystems`]. A plugin that also builds without the terminal view
//! does it behind a default `tui` cargo feature. The plugin guide has an
//! example and the default plugins' places.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;

use rig_ecs::agent::{Agent, TurnOf};

/// One piece of the status line: what it says, how, and its place.
#[derive(Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Clone, Debug, Default, PartialEq)]
pub struct StatusItem {
    /// What it says.
    pub text: String,
    /// How it is drawn.
    pub tone: Tone,
    /// Its side of the line.
    pub side: Side,
    /// Its place on its side, lowest first.
    pub order: u8,
    /// How long it stays when the line is too narrow: the items with the
    /// lowest go first, and one at `u8::MAX` never.
    pub keep: u8,
}

impl StatusItem {
    /// An empty item on `side` at `order`, which stays by `keep`.
    pub const fn at(side: Side, order: u8, keep: u8) -> Self {
        Self {
            text: String::new(),
            tone: Tone::Plain,
            side,
            order,
            keep,
        }
    }

    /// This item saying `text` in `tone`.
    pub fn says(&self, text: impl Into<String>, tone: Tone) -> Self {
        Self {
            text: text.into(),
            tone,
            ..*self
        }
    }
}

/// Status line items, one at each place (side and order). On an agent and
/// on its running turn, which every agent and turn has, they are shown
/// while the agent is; a view redraws when they change.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Clone, Debug, Default, PartialEq)]
pub struct StatusItems(Vec<StatusItem>);

impl StatusItems {
    /// Shows `item` at its place, in place of the one there; an empty
    /// item leaves the place empty.
    pub fn show(&mut self, item: StatusItem) {
        self.0
            .retain(|old| (old.side, old.order) != (item.side, item.order));
        if !item.text.is_empty() {
            self.0.push(item);
        }
    }

    /// The items, in no set order.
    pub fn iter(&self) -> impl Iterator<Item = &StatusItem> {
        self.0.iter()
    }
}

/// The app's status line items, shown with every agent: a plugin shows
/// one with `status.0.show(item)`.
#[derive(Resource, Reflect, Clone, Debug, Default)]
#[reflect(Resource, Clone, Debug, Default)]
pub struct AppStatus(pub StatusItems);

/// The side of the status line a [`StatusItem`] is on.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Side {
    /// After the items before it, from the left edge.
    #[default]
    Left,
    /// The meter, at the right edge.
    Right,
}

/// How a view draws a [`StatusItem`]'s text.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tone {
    /// As the text around it.
    #[default]
    Plain,
    /// Bold, such as the model.
    Bold,
    /// Dimmed, such as the meter.
    Dim,
    /// Green, such as `idle`.
    Green,
    /// Yellow, a warning or work in progress.
    Yellow,
    /// Red, an error or a context nearly full.
    Red,
    /// Cyan, the app's own state, such as a reload.
    Cyan,
    /// Magenta, other agents.
    Magenta,
}

/// The systems in `PostUpdate` that bring the status line items up to
/// date; views draw after them.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StatusSystems;

/// Gives every agent and turn its [`StatusItems`], and the app its
/// [`AppStatus`].
pub(crate) fn add(app: &mut App) {
    app.register_required_components::<Agent, StatusItems>()
        .register_required_components::<TurnOf, StatusItems>()
        .init_resource::<AppStatus>();
}
