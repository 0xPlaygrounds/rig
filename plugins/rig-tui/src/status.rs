//! The status line: what the view shows under the transcript of the agent
//! it shows. Each agent, and its running turn, has its [`StatusItems`], and
//! the app its [`AppStatus`]; a plugin shows its items there by name in
//! [`StatusSystems::Show`], and replaces or hides another plugin's in
//! [`StatusSystems::Edit`]. A plugin that also builds without the terminal
//! view does it behind a default `tui` cargo feature. The plugin guide has
//! an example and the default plugins' names and places.
//!
//! ```no_run
//! use rig_harness::prelude::*;
//! use rig_tui::{Side, StatusItem, StatusItems, StatusSystems, Tone};
//!
//! /// A gauge in place of rig-telemetry's `context` item. rig-telemetry
//! /// shows its item again whenever its meter changes, so this replaces
//! /// it every frame.
//! fn gauge(mut agents: Query<&mut StatusItems, With<Agent>>) {
//!     for mut items in &mut agents {
//!         items.show(StatusItem::new("context", Side::Right, 40, 5).says("▮▮▯▯", Tone::Dim));
//!     }
//! }
//!
//! fn build(app: &mut App) {
//!     app.add_systems(PostUpdate, gauge.in_set(StatusSystems::Edit));
//! }
//! ```

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;

use rig_ecs::agent::{Agent, TurnOf};
use serde::{Deserialize, Serialize};

/// One piece of the status line: its name, what it says, how, and its
/// place.
#[derive(Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Clone, Debug, Default, PartialEq)]
pub struct StatusItem {
    /// What it is, such as `context`: one item of a name shows in a
    /// [`StatusItems`], and another plugin replaces or hides it by name.
    pub name: &'static str,
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
    /// How narrow it may be cut, ending in `…`, when the line is too
    /// narrow, before it goes (by `keep`); 0 for an item never cut.
    pub min_width: u16,
}

impl StatusItem {
    /// An empty item named `name` on `side` at `order`, which stays by
    /// `keep`.
    pub const fn new(name: &'static str, side: Side, order: u8, keep: u8) -> Self {
        Self {
            name,
            text: String::new(),
            tone: Tone::Plain,
            side,
            order,
            keep,
            min_width: 0,
        }
    }

    /// This item, cut down to no fewer than `min_width` columns when the
    /// line is too narrow, before it goes.
    pub const fn shrinks_to(mut self, min_width: u16) -> Self {
        self.min_width = min_width;
        self
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

/// Status line items, one of each name. On an agent and on its running
/// turn, which every agent and turn has, they are shown while the agent
/// is, by place (side and order), items at one place by name; a view
/// redraws when they change.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Clone, Debug, Default, PartialEq)]
pub struct StatusItems(Vec<StatusItem>);

impl StatusItems {
    /// Shows `item` in place of the one of its name; an empty item hides
    /// it.
    pub fn show(&mut self, item: StatusItem) {
        self.hide(item.name);
        if !item.text.is_empty() {
            self.0.push(item);
        }
    }

    /// Hides the item named `name`. Its owner shows it again the next
    /// time it shows it, so a plugin that hides another's item hides it
    /// as often, in [`StatusSystems::Edit`].
    pub fn hide(&mut self, name: &str) {
        self.0.retain(|old| old.name != name);
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

/// The side of the status line a [`StatusItem`] is on, read from settings
/// as `left` or `right`.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
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
/// date, in order; views draw after them.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StatusSystems {
    /// Each plugin shows its own items.
    Show,
    /// A plugin replaces or hides another plugin's items, by name.
    Edit,
}

/// Gives every agent and turn its [`StatusItems`], and the app its
/// [`AppStatus`].
pub(crate) fn add(app: &mut App) {
    app.register_required_components::<Agent, StatusItems>()
        .register_required_components::<TurnOf, StatusItems>()
        .init_resource::<AppStatus>()
        .configure_sets(
            PostUpdate,
            (StatusSystems::Show, StatusSystems::Edit).chain(),
        );
}

#[cfg(test)]
mod tests;
