//! The status line's items: facts an agent's plugins show about it, its
//! running turn and the app, such as its model or its context, which a
//! view draws under its transcript. Each agent, and its running turn, has
//! its [`StatusItems`], and the app its [`AppStatus`]; a plugin shows its
//! items there by name in [`StatusSystems::Show`], and replaces or hides
//! another plugin's in [`StatusSystems::Edit`], as plain text in a
//! [`Tone`] or in runs of their own tones ([`StatusItem::styled`]).
//!
//! They are the agent's: a worker publishes each change as the session's
//! `status` event ([`StatusEvent`]), and a view's mirror puts it on the
//! mirrored agent, its turn or its app, where the terminal view draws it.
//!
//! ```no_run
//! use rig_harness::prelude::*;
//! use rig_harness::status::{Side, StatusItem, StatusItems, StatusSystems, Tone};
//!
//! /// A gauge in place of `rig_basics::telemetry`'s `context` item. It
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

use std::borrow::Cow;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig_ecs::agent::{AgentId, TurnOf};
use rig_ecs::prelude::Agent;
use serde::{Deserialize, Serialize};

use crate::worker::{Publish, WorkerOutbox};

/// One piece of the status line: its name, what it says, how, and its
/// place.
#[derive(Reflect, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[reflect(Clone, Debug, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct StatusItem {
    /// What it is, such as `context`: one item of a name shows in a
    /// [`StatusItems`], and another plugin replaces or hides it by name.
    #[serde(rename = "id")]
    pub name: Cow<'static, str>,
    /// What it says.
    pub text: String,
    /// How it is drawn.
    #[serde(default)]
    pub tone: Tone,
    /// Its side of the line.
    #[serde(default)]
    pub side: Side,
    /// Its place on its side, lowest first.
    #[serde(default)]
    pub order: u8,
    /// How long it stays when the line is too narrow: the items with the
    /// lowest go first, and one at `u8::MAX` never.
    #[serde(default)]
    pub keep: u8,
    /// How narrow it may be cut, ending in `…`, when the line is too
    /// narrow, before it goes (by `keep`); 0 for an item never cut.
    #[serde(default)]
    pub min_width: u16,
    /// Its text in runs of their own tones, drawn in place of `text` in
    /// its `tone`; see [`StatusItem::styled`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub runs: Vec<(String, Tone)>,
}

impl StatusItem {
    /// An empty item named `name` on `side` at `order`, which stays by
    /// `keep`.
    pub const fn new(name: &'static str, side: Side, order: u8, keep: u8) -> Self {
        Self {
            name: Cow::Borrowed(name),
            text: String::new(),
            tone: Tone::Plain,
            side,
            order,
            keep,
            min_width: 0,
            runs: Vec::new(),
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
            runs: Vec::new(),
            ..self.clone()
        }
    }

    /// This item saying `runs`, each in its own tone, such as a gauge
    /// coloured by how full it is; its `text` is the runs' text.
    ///
    /// ```
    /// use rig_harness::status::{Side, StatusItem, Tone};
    ///
    /// let gauge = StatusItem::new("context", Side::Right, 40, 5);
    /// let gauge = gauge.styled([("▮▮", Tone::Red), ("▯▯", Tone::Dim)]);
    /// assert_eq!(gauge.text, "▮▮▯▯");
    /// ```
    pub fn styled(&self, runs: impl IntoIterator<Item = (impl Into<String>, Tone)>) -> Self {
        let runs: Vec<(String, Tone)> = runs
            .into_iter()
            .map(|(text, tone)| (text.into(), tone))
            .collect();
        Self {
            text: runs.iter().map(|(text, _)| text.as_str()).collect(),
            tone: Tone::Plain,
            runs,
            ..self.clone()
        }
    }
}

/// Status line items of an agent or of its running turn, one per name
/// (every agent and turn has this component). The line shows those of the
/// shown agent, its turn and the [`AppStatus`], ordered by side, then
/// `order`, then name. A change redraws it.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[reflect(Component, Clone, Debug, Default, PartialEq)]
#[serde(transparent)]
pub struct StatusItems(Vec<StatusItem>);

impl StatusItems {
    /// Shows `item` in place of the one of its name; an empty item hides
    /// it.
    pub fn show(&mut self, item: StatusItem) {
        self.hide(&item.name);
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
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Resource, Clone, Debug, Default)]
pub struct AppStatus(pub StatusItems);

/// The side of the status line a [`StatusItem`] is on, read from settings
/// as `left` or `right`.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    /// Left-aligned, after the items of lower `order`.
    #[default]
    Left,
    /// Right-aligned, after the left side.
    Right,
}

/// How a view draws a [`StatusItem`]'s text.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tone {
    /// As the text around it.
    #[default]
    Plain,
    /// Bold, such as the model.
    Bold,
    /// Dimmed, such as the right side's items.
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

/// The session's `status` event: the items of an agent (`agentId`), of
/// its running turn (`turn`), or of the app (neither), as they stand now.
/// Its `items` hold `{id, text}` at least, as every client reads them.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StatusEvent {
    /// The agent whose items these are; `None` for the app's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// Whether they are the agent's running turn's.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub turn: bool,
    /// The items.
    pub items: StatusItems,
}

impl StatusEvent {
    /// The event's kind.
    pub const KIND: &'static str = "status";
}

/// Gives every agent and turn its [`StatusItems`] and the app its
/// [`AppStatus`]; in a worker, publishes each change as a
/// [`StatusEvent`].
pub struct StatusPlugin;

impl Plugin for StatusPlugin {
    fn build(&self, app: &mut App) {
        add(app);
        app.add_systems(Last, publish.run_if(resource_exists::<WorkerOutbox>));
    }
}

/// The items' components and sets, in an agent and in a view alike.
pub fn add(app: &mut App) {
    if app.world().contains_resource::<AppStatus>() {
        return;
    }
    app.register_required_components::<Agent, StatusItems>()
        .register_required_components::<TurnOf, StatusItems>()
        .init_resource::<AppStatus>()
        .configure_sets(
            PostUpdate,
            (StatusSystems::Show, StatusSystems::Edit).chain(),
        );
}

/// Publishes the items that changed this frame.
fn publish(
    app: Res<AppStatus>,
    agents: Query<(&AgentId, &StatusItems), (With<Agent>, Changed<StatusItems>)>,
    turns: Query<(&TurnOf, &StatusItems), Changed<StatusItems>>,
    ids: Query<&AgentId>,
    mut out: Publish,
) {
    let mut send = |event: StatusEvent| {
        let data = serde_json::to_value(&event).unwrap_or_default();
        out.raw(StatusEvent::KIND, data, None);
    };
    if app.is_changed() {
        send(StatusEvent {
            agent_id: None,
            turn: false,
            items: app.0.clone(),
        });
    }
    for (id, items) in &agents {
        send(StatusEvent {
            agent_id: Some(id.0.clone()),
            turn: false,
            items: items.clone(),
        });
    }
    for (TurnOf(agent), items) in &turns {
        if let Ok(id) = ids.get(*agent) {
            send(StatusEvent {
                agent_id: Some(id.0.clone()),
                turn: true,
                items: items.clone(),
            });
        }
    }
}
