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
//! use rig_harness::tui::ratatui::layout::Constraint;
//! use rig_harness::tui::ratatui::widgets::{Block, Paragraph};
//! use rig_harness::tui::{PanelCanvas, Placement, TuiPanel, TuiSystems};
//!
//! #[derive(Component)]
//! struct Working;
//!
//! fn spawn(mut commands: Commands) {
//!     commands.spawn((Working, TuiPanel::new(Placement::Right(Constraint::Length(30)))));
//! }
//!
//! fn draw(mut panels: Query<&mut PanelCanvas, With<Working>>, agents: Query<&Activity>) {
//!     let busy = agents.iter().filter(|activity| activity.is_busy()).count();
//!     for mut canvas in &mut panels {
//!         canvas.render(Paragraph::new(format!("{busy} working")).block(Block::bordered()));
//!     }
//! }
//!
//! App::new().add_systems(Startup, spawn).add_systems(PostUpdate, draw.in_set(TuiSystems::Draw));
//! ```

use bevy_ecs::prelude::*;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::widgets::Widget;

/// The terminal view's steps in `PostUpdate`, in order, after the agents'
/// [`ActivitySystems`](crate::plugins::activity::ActivitySystems).
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TuiSystems {
    /// Before the frame is laid out: a plugin updates what it shows and
    /// its [`TuiPanel`]s here, such as a placement that depends on the
    /// [`TuiScreen`].
    Prepare,
    /// The view decides whether to draw a frame, and lays out the input,
    /// the status line, the panels and the transcript.
    Layout,
    /// Plugins draw their panels into their [`PanelCanvas`]. Runs only for
    /// a frame that is drawn.
    Draw,
    /// The view draws the frame: the transcript, the status line, the
    /// input, the panels, and its own overlays over everything.
    Render,
}

/// Asks the terminal view to draw a frame, for state of a plugin it does
/// not watch, such as an animation's step.
#[derive(Message, Clone, Copy, Debug, Default)]
pub struct RequestRedraw;

/// The size of the terminal, from its start and each resize.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TuiScreen(pub Rect);

/// Marks the agent the terminal view shows and sends what is typed to.
#[derive(Component, Clone, Copy, Debug, Default)]
#[component(storage = "SparseSet")]
pub struct Focused;

/// A panel of a plugin in the terminal view. Panels are laid out one after
/// another, in entity order, each taking its side of what is left above the
/// status line; a panel that would leave the transcript less than 20
/// columns or 3 rows gets no area this frame. Despawn the entity, or
/// remove the component, to remove the panel.
#[derive(Component, Clone, Debug, PartialEq, Eq)]
#[require(PanelCanvas)]
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
    /// Rows above the transcript.
    Top(Constraint),
    /// Rows under the transcript, above the status line.
    Bottom(Constraint),
    /// Columns left of the transcript.
    Left(Constraint),
    /// Columns right of the transcript.
    Right(Constraint),
    /// A box centred over the whole screen, drawn over everything but the
    /// view's own overlays (pickers, completion, a failed rebuild).
    Over {
        /// Its width.
        width: Constraint,
        /// Its height.
        height: Constraint,
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

/// The smallest transcript a panel may leave.
const MIN_MAIN: (u16, u16) = (20, 3);

/// Lays the panels out in `main`, the area above the status line, and
/// `screen`, the whole screen; returns what is left for the transcript.
pub(super) fn lay_out<'a>(
    panels: impl Iterator<Item = (&'a TuiPanel, Mut<'a, PanelCanvas>)>,
    mut main: Rect,
    screen: Rect,
) -> Rect {
    for (panel, mut canvas) in panels {
        let area = match panel.placement {
            Placement::Over { width, height } => screen.centered(width, height),
            Placement::Top(size) => {
                let [panel, rest] = Layout::vertical([size, Constraint::Min(0)]).areas(main);
                take(&mut main, panel, rest)
            }
            Placement::Bottom(size) => {
                let [rest, panel] = Layout::vertical([Constraint::Min(0), size]).areas(main);
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
    }
    main
}

/// `panel` when `rest` is still big enough for the transcript, which it
/// then becomes; otherwise nothing.
fn take(main: &mut Rect, panel: Rect, rest: Rect) -> Rect {
    if rest.width < MIN_MAIN.0 || rest.height < MIN_MAIN.1 {
        return Rect::default();
    }
    *main = rest;
    panel
}
