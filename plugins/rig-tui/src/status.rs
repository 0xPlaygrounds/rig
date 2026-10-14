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

use ratatui::layout::{Constraint, Layout};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::Widget;
use rig_ecs::agent::{ActiveTurn, Agent, TurnOf};
use rig_harness::primary::Shown;
use serde::{Deserialize, Serialize};
use unicode_width::UnicodeWidthChar;

use super::panel::{PanelCanvas, Placement, TuiPanel};

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

/// A status line: `agent`'s items (the agent shown's, with `None`),
/// those of its running turn and the app's, by place, fit to its panel's
/// width; with no agent, `no agent` where its model would be. It draws in
/// one row of its [`TuiPanel`]. The default screen's shows the agent
/// shown, under the transcript.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[require(TuiPanel = TuiPanel::new(Placement::Bottom(Constraint::Length(1))))]
pub struct StatusLine {
    /// The agent whose items it shows; `None` for the one shown.
    pub agent: Option<Entity>,
}

/// Draws each status line with room into its canvas.
pub(crate) fn draw_status_lines(
    mut lines: Query<(&StatusLine, &mut PanelCanvas)>,
    (items, active): (Query<&StatusItems>, Query<&ActiveTurn>),
    (shown, app): (Res<Shown>, Option<Res<AppStatus>>),
) {
    for (line, mut canvas) in &mut lines {
        let area = canvas.area();
        if area.is_empty() {
            continue;
        }
        let agent = line.agent.or(shown.0);
        let turn = agent.and_then(|agent| active.get(agent).ok());
        let own = agent.and_then(|agent| items.get(agent).ok());
        let turn = turn.and_then(|turn| items.get(turn.turn()).ok());
        let app = app.as_deref().map(|app| &app.0);
        let (mut left, mut right) = status_pieces(agent.is_some(), [app, own, turn]);
        fit(&mut left, &mut right, usize::from(area.width));
        let (line, usage) = (join(left, LEFT_GAP), join(right, RIGHT_GAP));
        let usage_width = u16::try_from(usage.width()).unwrap_or(u16::MAX);
        let [status, meter] =
            Layout::horizontal([Constraint::Min(0), Constraint::Length(usage_width)]).areas(area);
        let buffer = canvas.buffer_mut();
        line.render(status, buffer);
        usage.render(meter, buffer);
    }
}

/// A piece of the status line, from a [`StatusItem`]. When the line does
/// not fit, [`fit`] cuts or drops the pieces with the lowest `keep` first.
struct Piece {
    keep: u8,
    /// How narrow it may be cut; 0 for never.
    min: usize,
    span: Span<'static>,
}

impl Piece {
    fn new(keep: u8, span: Span<'static>) -> Self {
        Self { keep, min: 0, span }
    }

    /// The same piece, cut down to no fewer than `min` columns before it
    /// goes.
    fn shrinking_to(mut self, min: usize) -> Self {
        self.min = min;
        self
    }

    fn width(&self) -> usize {
        self.span.width()
    }

    /// Whether it can still be cut.
    fn can_shrink(&self) -> bool {
        self.min > 0 && self.width() > self.min
    }

    /// Cuts it by `over` columns, to no fewer than its `min`, ending in `…`.
    fn shrink(&mut self, over: usize) {
        let target = self.width().saturating_sub(over).max(self.min);
        let mut text = String::new();
        let mut width = 0;
        for character in self.span.content.chars() {
            let next = width + character.width().unwrap_or(0);
            if next + 1 > target {
                break;
            }
            text.push(character);
            width = next;
        }
        text.push('…');
        self.span.content = text.into();
    }
}

/// The gap between the pieces on the left, and between the left and the
/// meter.
const LEFT_GAP: &str = "  ";
/// The gap between the meter's pieces.
const RIGHT_GAP: &str = " ";

/// The width of `pieces` joined by `gap`.
fn joined_width(pieces: &[Piece], gap: &str) -> usize {
    let gaps = pieces.len().saturating_sub(1) * gap.len();
    pieces.iter().map(Piece::width).sum::<usize>() + gaps
}

/// Cuts, then drops, the pieces of `left` and of the meter on the `right`
/// with the lowest `keep` until both fit `width` with a gap between them:
/// one that may shrink is cut to its `min` before it goes. What keeps
/// `u8::MAX` stays, cut where it may; past that the line is cut at the
/// edge.
fn fit(left: &mut Vec<Piece>, right: &mut Vec<Piece>, width: usize) {
    loop {
        let separator = if right.is_empty() { 0 } else { LEFT_GAP.len() };
        let used = joined_width(left, LEFT_GAP) + separator + joined_width(right, RIGHT_GAP);
        let Some(over) = used.checked_sub(width).filter(|over| *over > 0) else {
            return;
        };
        let lowest = |pieces: &[Piece]| {
            pieces
                .iter()
                .enumerate()
                .filter(|(_, piece)| piece.keep < u8::MAX || piece.can_shrink())
                .min_by_key(|(_, piece)| piece.keep)
                .map(|(index, piece)| (piece.keep, index))
        };
        let (side, index) = match (lowest(right.as_slice()), lowest(left.as_slice())) {
            (Some((on_right, index)), on_left)
                if on_left.is_none_or(|(on_left, _)| on_right <= on_left) =>
            {
                (&mut *right, index)
            }
            (_, Some((_, index))) => (&mut *left, index),
            (_, None) => return,
        };
        match side.get_mut(index) {
            Some(piece) if piece.can_shrink() => piece.shrink(over),
            _ => {
                side.remove(index);
            }
        }
    }
}

/// `pieces` as one line, `gap` between them.
fn join(pieces: Vec<Piece>, gap: &'static str) -> Line<'static> {
    let mut spans = Vec::with_capacity(pieces.len() * 2);
    for (index, piece) in pieces.into_iter().enumerate() {
        if index > 0 {
            spans.push(Span::from(gap));
        }
        spans.push(piece.span);
    }
    Line::from(spans)
}

/// The status line's pieces, left and right: the app's items, the shown
/// agent's and its turn's, by `order`, then name, with `no agent` where an
/// agent's model would be when none is shown.
fn status_pieces(shown: bool, items: [Option<&StatusItems>; 3]) -> (Vec<Piece>, Vec<Piece>) {
    let no_agent = StatusItem::new("agent", Side::Left, 30, u8::MAX).says("no agent", Tone::Dim);
    let mut items: Vec<&StatusItem> = items
        .into_iter()
        .flatten()
        .flat_map(StatusItems::iter)
        .collect();
    if !shown {
        items.push(&no_agent);
    }
    items.sort_by_key(|item| (item.order, item.name));
    let piece = |item: &StatusItem| {
        let style = match item.tone {
            Tone::Plain => Style::new(),
            Tone::Bold => Style::new().bold(),
            Tone::Dim => Style::new().dim(),
            Tone::Green => Style::new().green(),
            Tone::Yellow => Style::new().yellow(),
            Tone::Red => Style::new().red(),
            Tone::Cyan => Style::new().cyan(),
            Tone::Magenta => Style::new().magenta(),
        };
        Piece::new(item.keep, Span::styled(item.text.clone(), style))
            .shrinking_to(usize::from(item.min_width))
    };
    let (left, right): (Vec<_>, Vec<_>) =
        items.into_iter().partition(|item| item.side == Side::Left);
    (
        left.into_iter().map(piece).collect(),
        right.into_iter().map(piece).collect(),
    )
}

#[cfg(test)]
mod tests;
