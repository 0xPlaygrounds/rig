//! The status line: what the view shows under the transcript of the agent
//! it shows. Each agent, and its running turn, has its [`StatusItems`], and
//! the app its [`AppStatus`]; a plugin shows its items there by name in
//! [`StatusSystems::Show`], and replaces or hides another plugin's in
//! [`StatusSystems::Edit`], as plain text in a [`Tone`] or in runs of their
//! own styles ([`StatusItem::styled`]). A plugin that also builds without
//! the terminal view does it behind a default `tui` cargo feature. The
//! plugin guide has an example and the default plugins' names and places.
//! A [`StatusLine`] draws one in a panel, for the shown agent or another,
//! and [`line()`] fits any items to a width.
//!
//! ```no_run
//! use rig_harness::prelude::*;
//! use rig_tui::{Side, StatusItem, StatusItems, StatusSystems, Tone};
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

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use bevy_reflect::prelude::*;

use ratatui::layout::{Constraint, Layout};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::Widget;
use rig_ecs::agent::{ActiveTurn, Agent, TurnOf};
use rig_harness::main_agent::ShownAgent;
use serde::{Deserialize, Serialize};
use unicode_width::UnicodeWidthChar;

use super::editor::Editor;
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
    /// Its text in runs of their own styles, drawn in place of `text` in
    /// its `tone`; see [`StatusItem::styled`].
    #[reflect(ignore)]
    pub styled: Option<Line<'static>>,
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
            styled: None,
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
            styled: None,
            ..self.clone()
        }
    }

    /// This item saying `line`, each run in its own style, such as a gauge
    /// coloured by how full it is; its `text` is the line's text.
    ///
    /// ```
    /// use rig_tui::ratatui::style::Stylize;
    /// use rig_tui::ratatui::text::{Line, Span};
    /// use rig_tui::{Side, StatusItem};
    ///
    /// let gauge = StatusItem::new("context", Side::Right, 40, 5);
    /// let gauge = gauge.styled(Line::from(vec![Span::from("▮▮").red(), Span::from("▯▯").dim()]));
    /// assert_eq!(gauge.text, "▮▮▯▯");
    /// ```
    pub fn styled(&self, line: Line<'static>) -> Self {
        Self {
            text: line.to_string(),
            tone: Tone::Plain,
            styled: Some(line),
            ..self.clone()
        }
    }
}

/// Status line items of an agent or of its running turn, one per name
/// (every agent and turn has this component). The line shows those of the
/// shown agent, its turn and the [`AppStatus`], ordered by side, then
/// `order`, then name. A change redraws it.
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
    /// Left-aligned, after the items of lower `order`.
    #[default]
    Left,
    /// Right-aligned, after the left side.
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

/// A status line: `agent`'s items (the shown agent's, with `None`),
/// those of its running turn and the app's, by place, fit to its panel's
/// width; with no agent, `no agent` where its model would be. It draws in
/// one row of its [`TuiPanel`]. On an [`Editor`] it is
/// drawn by the editor instead, in its bottom row while nothing is typed;
/// the default screen's is so, for the shown agent.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[require(TuiPanel = TuiPanel::new(Placement::Bottom(Constraint::Length(1))))]
pub struct StatusLine {
    /// The agent whose items it shows; `None` for the one shown.
    pub agent: Option<Entity>,
}

/// What a [`StatusLine`] shows: the agents' and turns' items, the app's,
/// and which agent is shown.
#[derive(SystemParam)]
pub(crate) struct StatusSources<'w, 's> {
    items: Query<'w, 's, &'static StatusItems>,
    active: Query<'w, 's, &'static ActiveTurn>,
    shown: Res<'w, ShownAgent>,
    app: Option<Res<'w, AppStatus>>,
}

impl StatusSources<'_, '_> {
    /// `status`'s line fit to `width`: its left and its right.
    pub(crate) fn line(&self, status: &StatusLine, width: u16) -> (Line<'static>, Line<'static>) {
        let agent = status.agent.or(self.shown.0);
        let turn = agent.and_then(|agent| self.active.get(agent).ok());
        let own = agent.and_then(|agent| self.items.get(agent).ok());
        let turn = turn.and_then(|turn| self.items.get(turn.turn()).ok());
        let app = self.app.as_deref().map(|app| &app.0);
        let no_agent =
            StatusItem::new("agent", Side::Left, 30, u8::MAX).says("no agent", Tone::Dim);
        let no_agent = agent.is_none().then_some(&no_agent);
        let items = [app, own, turn]
            .into_iter()
            .flatten()
            .flat_map(StatusItems::iter);
        self::line(items.chain(no_agent), width)
    }
}

/// Draws each status line with room into its canvas, but those an editor
/// draws.
pub(crate) fn draw_status_lines(
    mut lines: Query<(&StatusLine, &mut PanelCanvas), Without<Editor>>,
    sources: StatusSources,
) {
    for (line, mut canvas) in &mut lines {
        let area = canvas.area();
        if area.is_empty() {
            continue;
        }
        let (left, right) = sources.line(line, area.width);
        let right_width = u16::try_from(right.width()).unwrap_or(u16::MAX);
        let [left_area, right_area] =
            Layout::horizontal([Constraint::Min(0), Constraint::Length(right_width)]).areas(area);
        let buffer = canvas.buffer_mut();
        left.render(left_area, buffer);
        right.render(right_area, buffer);
    }
}

/// A piece of the status line, from a [`StatusItem`]. When the line does
/// not fit, [`fit`] cuts or drops the pieces with the lowest `keep` first.
struct Piece {
    keep: u8,
    /// How narrow it may be cut; 0 for never.
    min: usize,
    line: Line<'static>,
}

impl Piece {
    fn new(keep: u8, line: impl Into<Line<'static>>) -> Self {
        let line = line.into();
        Self { keep, min: 0, line }
    }

    /// The same piece, cut down to no fewer than `min` columns before it
    /// goes.
    fn shrinking_to(mut self, min: usize) -> Self {
        self.min = min;
        self
    }

    fn width(&self) -> usize {
        self.line.width()
    }

    /// Whether it can still be cut.
    fn can_shrink(&self) -> bool {
        self.min > 0 && self.width() > self.min
    }

    /// Cuts it by `over` columns, to no fewer than its `min`, ending in `…`
    /// in the style of the run it cuts.
    fn shrink(&mut self, over: usize) {
        let target = self.width().saturating_sub(over).max(self.min);
        let mut spans: Vec<Span<'static>> = Vec::new();
        let mut width = 0;
        let mut cut = Style::new();
        'runs: for span in &self.line.spans {
            let mut text = String::new();
            cut = span.style;
            for character in span.content.chars() {
                let next = width + character.width().unwrap_or(0);
                if next + 1 > target {
                    spans.push(Span::styled(text, span.style));
                    break 'runs;
                }
                text.push(character);
                width = next;
            }
            spans.push(Span::styled(text, span.style));
        }
        spans.push(Span::styled("…", cut));
        self.line.spans = spans;
    }
}

/// The gap between the pieces on the left, and between the two sides.
const LEFT_GAP: &str = "  ";
/// The gap between the pieces on the right.
const RIGHT_GAP: &str = " ";

/// The width of `pieces` joined by `gap`.
fn joined_width(pieces: &[Piece], gap: &str) -> usize {
    let gaps = pieces.len().saturating_sub(1) * gap.len();
    pieces.iter().map(Piece::width).sum::<usize>() + gaps
}

/// Cuts, then drops, the pieces on the `left` and on the `right`
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
        spans.extend(piece.line.spans);
    }
    Line::from(spans)
}

/// A status line of `items` fit to `width` columns: the items on the left,
/// and those on the right; each by place (`order`, then name),
/// cut or dropped by `keep` when they do not fit. A plugin drawing a status
/// line of its own, such as in a panel, draws these two.
pub fn line<'a>(
    items: impl IntoIterator<Item = &'a StatusItem>,
    width: u16,
) -> (Line<'static>, Line<'static>) {
    let (mut left, mut right) = pieces(items);
    fit(&mut left, &mut right, usize::from(width));
    (join(left, LEFT_GAP), join(right, RIGHT_GAP))
}

/// The status line's pieces of `items`, left and right, by `order`, then
/// name.
fn pieces<'a>(items: impl IntoIterator<Item = &'a StatusItem>) -> (Vec<Piece>, Vec<Piece>) {
    let mut items: Vec<&StatusItem> = items.into_iter().collect();
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
        let line = match &item.styled {
            Some(line) => line.clone(),
            None => Line::from(Span::styled(item.text.clone(), style)),
        };
        Piece::new(item.keep, line).shrinking_to(usize::from(item.min_width))
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
