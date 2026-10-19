//! What a plugin draws in the terminal view: a [`TuiPanel`] entity takes a
//! side of the transcript or a box over the screen, and an ordinary system
//! of the plugin draws it into the entity's [`PanelCanvas`] in
//! [`TuiSystems::Draw`], with whatever queries and resources it needs. The
//! transcript is laid out in what the panels leave.
//!
//! A frame is drawn only when something drawn changed. The view watches the
//! agents and the panels' [`TuiPanel`] components; a panel whose own state
//! changed asks with a [`RequestRedraw`] message.
//!
//! ```no_run
//! use rig_harness::prelude::*;
//! use rig_tui::ratatui::layout::Constraint;
//! use rig_tui::ratatui::widgets::Paragraph;
//! use rig_tui::{PanelCanvas, Placement, Theme, TuiPanel, TuiSystems};
//!
//! #[derive(Component)]
//! struct Working;
//!
//! fn spawn(mut commands: Commands) {
//!     commands.spawn((Working, TuiPanel::new(Placement::Right(Constraint::Length(30)))));
//! }
//!
//! fn draw(
//!     mut panels: Query<&mut PanelCanvas, With<Working>>,
//!     turns: Query<&TurnOf>,
//!     theme: Res<Theme>,
//! ) {
//!     let busy = turns.iter().count();
//!     for mut canvas in &mut panels {
//!         canvas.render(Paragraph::new(format!("{busy} working")).block(theme.panel("turns")));
//!     }
//! }
//!
//! App::new().add_systems(Startup, spawn).add_systems(PostUpdate, draw.in_set(TuiSystems::Draw));
//! ```

use std::cmp::Reverse;

use bevy_ecs::change_detection::Tick;
use bevy_ecs::prelude::*;
use bevy_ecs::query::SpawnDetails;
use bevy_reflect::prelude::*;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::widgets::Widget;

/// The terminal view's steps in `PostUpdate`, in order, after the
/// [`StatusSystems`](crate::StatusSystems).
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TuiSystems {
    /// Before the frame is laid out: a plugin updates what it shows and
    /// its [`TuiPanel`]s here, such as a placement that depends on the
    /// [`TerminalSize`].
    Prepare,
    /// The view decides whether to draw a frame and lays it out: first
    /// every panel by its placement (the input, the status line, the
    /// plugins' panels), then each [`TranscriptPane`](crate::TranscriptPane)
    /// in what its panel got, its rows laid out and its scroll placed
    /// ([`TranscriptView::top`](crate::TranscriptView::top),
    /// [`DrawnTranscript`](crate::DrawnTranscript)).
    Layout,
    /// After the frame is laid out and every pane's scroll placed: a plugin
    /// fits its [`TuiPanel`]s to what the panes show here, such as a header
    /// shown while a pane is scrolled up. When a panel was spawned or
    /// changed here, the view lays the frame out again, panels then panes,
    /// before [`Draw`](Self::Draw), so the frame that placed the scroll
    /// draws the panel fitted to it.
    Fit,
    /// Plugins draw their panels into their [`PanelCanvas`]. Runs only for
    /// a frame that is drawn.
    Draw,
    /// The view composites the panels' canvases (the transcript, status
    /// line and input are panels too) and draws its overlays over them:
    /// the picker or question, the completion list and the caret.
    Render,
}

/// Asks the terminal view to draw a frame, for state of a plugin it does
/// not watch, such as an animation's step.
#[derive(Message, Clone, Copy, Debug, Default)]
pub struct RequestRedraw;

/// Asks the terminal view to write every cell of the area again after its
/// next frame, changed or not, and to draw that frame: for a plugin that
/// drew over cells itself (an image in the cells), so they show the view's
/// text again once it is gone. The area is clipped to the screen; the
/// cells go out before that frame's
/// [`TerminalWrite`](crate::TerminalWrite)s.
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub struct RedrawArea(pub Rect);

/// The size of the terminal, from its start and each resize.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TerminalSize(pub Rect);

/// A panel in the terminal view: the default screen's transcript, input
/// and status line are panels too, spawned first. Panels are laid out one
/// after another, in the order spawned, each taking its side of what is left,
/// after the rows under the input ([`Placement::BelowInput`]); a panel
/// that would leave less than 20 columns or 3 rows gets no area this frame.
/// Despawn the entity, or remove the component, to remove the panel.
#[derive(Component, Reflect, Clone, Debug, PartialEq, Eq)]
#[require(PanelCanvas)]
#[reflect(opaque, Component, Clone, Debug, PartialEq)]
pub struct TuiPanel {
    /// Where it goes.
    pub placement: Placement,
}

impl TuiPanel {
    /// A panel at `placement`.
    pub fn new(placement: Placement) -> Self {
        Self { placement }
    }
}

/// Where a [`TuiPanel`] goes, with the size it asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// Rows above what is left: above the transcript.
    Top(Constraint),
    /// Rows under what is left: under the transcript, above the status
    /// line, which the default screen spawns first.
    Bottom(Constraint),
    /// Columns left of the transcript.
    Left(Constraint),
    /// Columns right of the transcript.
    Right(Constraint),
    /// Rows at the bottom of the screen, under the input box, such as a row
    /// of suggestions; several stack in the order spawned.
    BelowInput(Constraint),
    /// A box centred over the whole screen, drawn over everything but the
    /// view's own overlays (the picker and the completion list).
    Over {
        /// Its width.
        width: Constraint,
        /// Its height.
        height: Constraint,
    },
    /// What the other panels leave, such as the default screen's
    /// transcript; several such panels share it.
    Rest,
    /// Column `part` (from 0) of `of` equal columns of another panel, or
    /// of the picker shown, such as a preview beside a picker's list or
    /// two transcripts compared side by side; drawn over it.
    Within {
        /// The panel, or the picker's entity.
        panel: Entity,
        /// Which column.
        part: u16,
        /// How many columns.
        of: u16,
    },
}

/// Where a [`TuiPanel`] is drawn this frame and what was drawn there. The
/// view sets the area and clears the buffer before [`TuiSystems::Draw`].
#[derive(Component, Clone, Debug, Default)]
pub struct PanelCanvas {
    buffer: Buffer,
}

impl PanelCanvas {
    /// The panel's area on the screen; empty when it has no room.
    pub fn area(&self) -> Rect {
        self.buffer.area
    }

    /// Draws `widget` over the whole area.
    pub fn render(&mut self, widget: impl Widget) {
        let area = self.buffer.area;
        widget.render(area, &mut self.buffer);
    }

    /// The buffer, for drawing parts of the area; its area is [`Self::area`].
    pub fn buffer_mut(&mut self) -> &mut Buffer {
        &mut self.buffer
    }

    /// Starts the frame's drawing at `area`.
    pub(super) fn reset(&mut self, area: Rect) {
        if self.buffer.area == area {
            self.buffer.reset();
        } else {
            self.buffer = Buffer::empty(area);
        }
    }

    /// Copies what was drawn onto `target`, within its area.
    pub(super) fn copy_to(&self, target: &mut Buffer) {
        for position in self.buffer.area.intersection(target.area).positions() {
            if let (Some(cell), Some(slot)) =
                (self.buffer.cell(position), target.cell_mut(position))
            {
                slot.clone_from(cell);
            }
        }
    }
}

/// The spawn ticks of the panels, for [`spawn_order`].
pub(super) type PanelSpawns<'w, 's> = Query<'w, 's, SpawnDetails, With<TuiPanel>>;

/// Where `panel` comes in the order panels are laid out and drawn: the
/// order they were spawned, the default screen's first. Bevy hands a
/// despawned entity's index out again, so the spawn tick orders them, and
/// the index only those spawned at one tick (`Entity`'s own order is not
/// the index's).
pub(super) fn spawn_order(now: Tick, spawns: &PanelSpawns, panel: Entity) -> (Reverse<u32>, u32) {
    let spawned = spawns.get(panel).map(SpawnDetails::spawn_tick);
    let age = spawned.map_or(0, |spawned| now.get().wrapping_sub(spawned.get()));
    (Reverse(age), panel.index_u32())
}

/// The smallest transcript a panel may leave.
pub(super) const MIN_TRANSCRIPT: (u16, u16) = (20, 3);

/// The rows a panel below the input asks for on a screen `height` rows
/// high.
pub(super) fn rows_of(size: Constraint, height: u16) -> u16 {
    let [rows, _] = Layout::vertical([size, Constraint::Fill(1)]).areas(Rect::new(0, 0, 1, height));
    rows.height
}

/// The rows each of the panels below the input that asked for `asked`
/// rows gets, in order, out of `room`: a panel that does not fit whole is
/// skipped.
pub(super) fn below_rows(asked: impl IntoIterator<Item = u16>, mut room: u16) -> Vec<u16> {
    asked
        .into_iter()
        .map(|rows| match room.checked_sub(rows) {
            Some(left) => {
                room = left;
                rows
            }
            None => 0,
        })
        .collect()
}

/// A panel to lay out: its entity, where it goes, how many rows it fits its
/// content in when it does (an editor, a strip) in place of its
/// placement's size, and its canvas.
pub(super) type ToLayOut<'a> = (Entity, &'a TuiPanel, Option<u16>, Mut<'a, PanelCanvas>);

/// Lays the panels out on `screen`, in the order given, in these steps:
/// 1. those below the input, stacked at the bottom while what they leave
///    keeps the rows the others fit their content in and [`MIN_TRANSCRIPT`];
/// 2. each side panel takes its side of what is left (one that fits its
///    content as many rows as that takes);
/// 3. an `Over` box goes over the screen;
/// 4. those that take the `Rest` share what is left;
/// 5. those `Within` another take their column of it, or of a box of
///    `boxes` (the picker shown).
pub(super) fn lay_out<'a>(
    panels: impl Iterator<Item = ToLayOut<'a>>,
    screen: Rect,
    boxes: &[(Entity, Rect)],
) {
    let panels: Vec<ToLayOut<'a>> = panels.collect();
    let fitted: u16 = panels
        .iter()
        .filter(|(_, panel, ..)| !matches!(panel.placement, Placement::BelowInput(_)))
        .filter_map(|(_, _, fits, _)| *fits)
        .sum();
    let asked = panels
        .iter()
        .filter_map(|(_, panel, fits, _)| match panel.placement {
            Placement::BelowInput(size) => Some(fits.unwrap_or(rows_of(size, screen.height))),
            _ => None,
        });
    let room = (screen.height.saturating_sub(fitted)).saturating_sub(MIN_TRANSCRIPT.1);
    let below_rows = below_rows(asked, room);
    let total: u16 = below_rows.iter().sum();
    let [mut main, mut below] =
        Layout::vertical([Constraint::Min(0), Constraint::Length(total)]).areas(screen);
    let mut below_rows = below_rows.into_iter();
    let (mut rest, mut within) = (Vec::new(), Vec::new());
    let mut areas: Vec<(Entity, Rect)> = boxes.to_vec();
    for (entity, panel, fits, mut canvas) in panels {
        let fit = |size| fits.map_or(size, Constraint::Length);
        let area = match panel.placement {
            Placement::Rest => {
                rest.push((entity, canvas));
                continue;
            }
            Placement::Within { panel, part, of } => {
                within.push((panel, (part, of), canvas));
                continue;
            }
            Placement::Over { width, height } => screen.centered(width, height),
            Placement::BelowInput(_) => {
                let rows = below_rows.next().unwrap_or(0);
                let [panel, left] =
                    Layout::vertical([Constraint::Length(rows), Constraint::Min(0)]).areas(below);
                below = left;
                if rows == 0 { Rect::default() } else { panel }
            }
            Placement::Top(size) => {
                let [panel, rest] = Layout::vertical([fit(size), Constraint::Min(0)]).areas(main);
                take(&mut main, panel, rest)
            }
            Placement::Bottom(size) => {
                let [rest, panel] = Layout::vertical([Constraint::Min(0), fit(size)]).areas(main);
                take(&mut main, panel, rest)
            }
            Placement::Left(size) => {
                let [panel, rest] = Layout::horizontal([size, Constraint::Min(0)]).areas(main);
                take(&mut main, panel, rest)
            }
            Placement::Right(size) => {
                let [rest, panel] = Layout::horizontal([Constraint::Min(0), size]).areas(main);
                take(&mut main, panel, rest)
            }
        };
        // Drawing changes only the buffer: not a change to redraw for.
        canvas.bypass_change_detection().reset(area);
        areas.push((entity, area));
    }
    for (entity, mut canvas) in rest {
        canvas.bypass_change_detection().reset(main);
        areas.push((entity, main));
    }
    for (host, (part, of), mut canvas) in within {
        let host = areas.iter().find(|(entity, _)| *entity == host);
        let host = host.map_or(Rect::default(), |(_, area)| *area);
        let columns = Layout::horizontal(vec![Constraint::Fill(1); usize::from(of.max(1))]);
        let column = columns.split(host).get(usize::from(part)).copied();
        canvas
            .bypass_change_detection()
            .reset(column.unwrap_or_default());
    }
}

/// `panel` when `rest` is still big enough for the transcript, which it
/// then becomes; otherwise nothing.
fn take(main: &mut Rect, panel: Rect, rest: Rect) -> Rect {
    if rest.width < MIN_TRANSCRIPT.0 || rest.height < MIN_TRANSCRIPT.1 {
        return Rect::default();
    }
    *main = rest;
    panel
}
