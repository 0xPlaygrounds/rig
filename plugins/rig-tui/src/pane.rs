//! Transcript panes: a [`TranscriptPane`] draws a conversation in its
//! panel with the transcript's own layout, looks and state, live for an
//! agent (the reply streaming in, running calls, the inbox, notices) or
//! as a value (a saved session). Each pane has its own
//! [`TranscriptView`] (what is selected and lit, where it is
//! scrolled), its own scroll and, once a plugin adds one, its own
//! [`DrawnTranscript`]. Render systems select a pane through
//! [`crate::RenderBatch::pane`]. The default screen's transcript is one,
//! marked [`MainPane`].
//!
//! ```no_run
//! use rig_harness::prelude::*;
//! use rig_tui::ratatui::layout::Constraint;
//! use rig_tui::{PaneKeys, Placement, TranscriptPane, TranscriptSource, TuiPanel};
//!
//! /// Shows a subagent beside the main transcript, live.
//! fn beside(subagent: Entity, mut commands: Commands) {
//!     commands.spawn((
//!         TranscriptPane::new(TranscriptSource::Agent(subagent)).titled("helper"),
//!         TuiPanel::new(Placement::Right(Constraint::Percentage(40))),
//!         PaneKeys,
//!     ));
//! }
//! ```

use bevy_ecs::lifecycle::HookContext;
use bevy_ecs::prelude::*;
use bevy_ecs::world::DeferredWorld;
use crossterm::event::{KeyCode, KeyModifiers};
use ratatui::layout::Rect;
use ratatui::style::Stylize;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, Widget};
use rig_core::conversation::Conversation;
use rig_ecs::agent::Conversation as AgentConversation;
use rig_harness::main_agent::ShownAgent;

use super::keys::{InputAction, KeyBinding, KeyPress, KeyScope};
use super::panel::{PanelCanvas, Placement, TuiPanel};
use super::render::{Showing, decorate, lay_out_rendered};
use super::renderers::{RenderKey, RenderRequest, Renderers};
use super::theme::Theme;
use super::transcript::{Below, Part, Replaced, Scroll, Transcript};
use super::view::{DrawnTranscript, TranscriptView};

/// A transcript in a panel: `source`'s conversation, drawn as the default
/// screen draws its own, under a title row when it has a `title`. It
/// takes what the other panels leave unless its [`TuiPanel`] says
/// otherwise. Its rows are laid out and its scroll placed in
/// [`TuiSystems::Layout`](crate::TuiSystems::Layout), so its
/// [`TranscriptView::top`] and [`DrawnTranscript`] are current before
/// [`TuiSystems::Draw`](crate::TuiSystems::Draw). With [`PaneKeys`] and the
/// [`KeyFocus`](crate::KeyFocus), the keys scroll and select in
/// it.
#[derive(Component, Clone, Debug)]
#[require(TuiPanel = TuiPanel::new(Placement::Rest), TranscriptView, PaneState)]
pub struct TranscriptPane {
    /// What it draws.
    pub source: TranscriptSource,
    /// The title on its first row, on the
    /// [`panel_surface`](crate::Theme::panel_surface); `None` draws no
    /// title row.
    pub title: Option<String>,
}

impl TranscriptPane {
    /// A pane drawing `source`, with no title row.
    pub fn new(source: TranscriptSource) -> Self {
        Self {
            source,
            title: None,
        }
    }

    /// The same pane, under a title row reading `title`.
    pub fn titled(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }
}

/// What a [`TranscriptPane`] draws.
#[derive(Clone, Debug)]
pub enum TranscriptSource {
    /// The shown agent ([`ShownAgent`]), whichever it is; with none, the
    /// notices about every agent.
    Shown,
    /// This agent, live: the reply streaming in, its running calls, its
    /// inbox and its notices.
    Agent(Entity),
    /// A conversation no agent of this app holds, such as a saved session
    /// read with `rig_memory::read_conversation`: its messages and asides,
    /// with no live parts or notices.
    Value(Conversation),
}

/// Marks the transcript pane the keys scroll and select in while
/// no other pane has the [`KeyFocus`](crate::KeyFocus): the default
/// screen's. An app that builds its own screen marks its own.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct MainPane;

/// The main transcript pane's view: what is selected and lit,
/// and where it is. A system with it is skipped when there is no such
/// pane, as in an app without the default screen.
///
/// ```no_run
/// use rig_harness::prelude::*;
/// use rig_tui::{MainTranscript, TranscriptItem};
///
/// fn select_first(mut view: MainTranscript, conversations: Query<&Conversation>) {
///     let first = conversations.iter().find_map(|conversation| conversation.id(0));
///     view.selected = first.map(TranscriptItem::Message);
/// }
/// ```
pub type MainTranscript<'w, 's> = Single<'w, 's, &'static mut TranscriptView, With<MainPane>>;

/// Binds the transcript's keys in a pane's own scope
/// ([`KeyScope::Panel`]), on entities it owns: Up and Down scroll a row,
/// PageUp and PageDown a page, and Alt+Up and Alt+Down select. They work
/// while the pane has the [`KeyFocus`](crate::KeyFocus); Esc gives the keys
/// back.
#[derive(Component, Clone, Copy, Debug, Default)]
#[component(on_add = bind_pane_keys)]
pub struct PaneKeys;

/// A pane's rows and scroll, laid out in `Layout` and drawn in `Draw`.
#[derive(Component, Default)]
pub(crate) struct PaneState {
    /// Its messages' rows, kept between frames.
    pub(crate) transcript: Transcript,
    pub(crate) revision: u64,
    pub(crate) width: u16,
    pub(crate) theme: Option<Theme>,
    pub(crate) source: Option<Entity>,
    /// The outputs of its items outside any message.
    pub(crate) rendered: std::collections::HashMap<RenderKey, RenderRequest>,
    /// Where it is scrolled.
    pub(crate) scroll: Scroll,
    /// The parts laid out last, whose items, top to bottom, the selection
    /// moves through.
    pub(crate) parts: Vec<Part>,
    /// The rows to draw this frame, and what is below them.
    pub(crate) rows: Vec<Line<'static>>,
    pub(crate) below: Below,
    /// Where its rows go, inside its border.
    pub(crate) area: Rect,
}

impl PaneState {
    /// Rows a page key moves: the pane's height but one.
    pub(crate) fn page(&self) -> usize {
        usize::from(self.area.height).saturating_sub(1).max(1)
    }
}

/// Binds [`PaneKeys`]' keys in the pane's scope.
fn bind_pane_keys(mut world: DeferredWorld, context: HookContext) {
    let pane = context.entity;
    let alt = KeyModifiers::ALT;
    let keys = [
        (
            KeyPress::key(KeyCode::Up),
            InputAction::ScrollUp,
            "Scroll up",
        ),
        (
            KeyPress::key(KeyCode::Down),
            InputAction::ScrollDown,
            "Scroll down",
        ),
        (
            KeyPress::key(KeyCode::PageUp),
            InputAction::PageUp,
            "Page up",
        ),
        (
            KeyPress::key(KeyCode::PageDown),
            InputAction::PageDown,
            "Page down",
        ),
        (
            KeyPress::new(KeyCode::Up, alt),
            InputAction::SelectUp,
            "Select the item above",
        ),
        (
            KeyPress::new(KeyCode::Down, alt),
            InputAction::SelectDown,
            "Select the item below",
        ),
    ];
    let mut commands = world.commands();
    for (press, action, label) in keys {
        let binding = KeyBinding::new(press, label).in_scope(KeyScope::Panel(pane));
        let name = Name::new(format!("key:{}", binding.keys));
        commands.spawn((name, binding, action, ChildOf(pane)));
    }
}

/// Lays out every pane with room: its rows at its width, its scroll
/// placed at its height, its [`TranscriptView::top`] and width noted (a
/// change to either changes the view), and its [`DrawnTranscript`] set.
/// In [`TuiSystems::Layout`](crate::TuiSystems::Layout), after the panels
/// are laid out, and again with them after
/// [`TuiSystems::Fit`](crate::TuiSystems::Fit) when a panel changed there.
pub(crate) fn lay_out_panes(
    world: &mut World,
    changes: &mut bevy_ecs::system::SystemState<(
        Query<Entity, Or<(Changed<AgentConversation>, Changed<Replaced>)>>,
        Query<Entity, Changed<TranscriptPane>>,
    )>,
) {
    let Ok((agents, panes)) = changes.get(world) else {
        return;
    };
    let changed_agents: std::collections::HashSet<Entity> = agents.iter().collect();
    let changed_panes: std::collections::HashSet<Entity> = panes.iter().collect();
    let entities: Vec<Entity> = world
        .query_filtered::<Entity, With<TranscriptPane>>()
        .iter(world)
        .collect();
    let revision = world
        .get_resource::<crate::RenderRevision>()
        .copied()
        .unwrap_or_default()
        .0;
    let shown_agent = world.get_resource::<ShownAgent>().and_then(|shown| shown.0);
    let Some(theme) = world.get_resource::<Theme>().copied() else {
        return;
    };
    for entity in entities {
        let Some(pane) = world.get::<TranscriptPane>(entity).cloned() else {
            continue;
        };
        let Some(canvas) = world.get::<PanelCanvas>(entity) else {
            continue;
        };
        let area = inner(&pane, canvas.area());
        if area.is_empty() {
            continue;
        }
        let Some(mut view) = world.get::<TranscriptView>(entity).cloned() else {
            continue;
        };
        let Some(mut stored) = world.get_mut::<PaneState>(entity) else {
            continue;
        };
        let mut state = std::mem::take(stored.bypass_change_detection());
        let agent = match &pane.source {
            TranscriptSource::Shown => shown_agent,
            TranscriptSource::Agent(agent) => Some(*agent),
            TranscriptSource::Value(_) => None,
        };
        if state.revision != revision
            || state.width != area.width
            || state.theme != Some(theme)
            || state.source != agent
            || changed_panes.contains(&entity)
        {
            state.transcript.clear();
            state.revision = revision;
            state.width = area.width;
            state.theme = Some(theme);
            state.source = agent;
            state.rendered.clear();
        }
        let showing = match (&pane.source, agent) {
            (TranscriptSource::Value(conversation), _) => {
                Showing::Value(conversation, entity, changed_panes.contains(&entity))
            }
            (_, Some(agent)) => Showing::Agent(agent, changed_agents.contains(&agent)),
            (_, None) => Showing::Nothing,
        };
        let cached = std::mem::take(&mut state.rendered);
        let mut renderers = Renderers::new((Some(entity), agent, area.width), cached);
        let parts = lay_out_rendered(
            world,
            (&mut state.transcript, &mut renderers),
            (&view, &showing, area.width),
        )
        .unwrap_or_default();
        state.rendered = renderers.cached.into_inner();
        let visible = state.transcript.visible(
            &parts,
            (area.width, usize::from(area.height)),
            &mut state.scroll,
            &view.highlights,
        );
        // Copying every row as text costs the whole transcript, so only a
        // pane whose rows a plugin reads gets them.
        let drawn = world.get::<DrawnTranscript>(entity).is_some().then(|| {
            let on_screen = visible.first..visible.first + visible.rows.len();
            state
                .transcript
                .laid_out(agent, &parts, &view.highlights, on_screen)
                .at(area)
        });
        let selected = view
            .selected
            .as_ref()
            .and_then(|item| state.transcript.rows_of(&parts, item));
        state.rows = decorate(visible.rows, visible.first, selected, &theme);
        state.below = visible.below;
        state.parts = parts;
        state.area = area;
        view.set_top(visible.top);
        view.set_width(area.width);
        if let Some(mut stored) = world.get_mut::<PaneState>(entity) {
            *stored.bypass_change_detection() = state;
        }
        if let Some(mut old) = world.get_mut::<TranscriptView>(entity) {
            if old.top() != view.top() {
                old.set_top(view.top().cloned());
            }
            if old.width() != Some(area.width) {
                old.set_width(area.width);
            }
        }
        if let Some((mut old, drawn)) = world.get_mut::<DrawnTranscript>(entity).zip(drawn) {
            old.set_if_neq(drawn);
        }
    }
}

/// Where `pane`'s rows go in its panel's `area`: under its title row,
/// when it has one.
fn inner(pane: &TranscriptPane, area: Rect) -> Rect {
    match pane.title {
        Some(_) => Block::new().title("").inner(area),
        None => area,
    }
}

/// Draws every pane laid out this frame into its canvas: its
/// title row, its rows, and a marker of what is below a pane scrolled up. In
/// [`TuiSystems::Draw`](crate::TuiSystems::Draw).
pub(crate) fn draw_panes(
    mut panes: Query<(&TranscriptPane, &mut PaneState, &mut PanelCanvas)>,
    theme: Res<Theme>,
) {
    for (pane, mut state, mut canvas) in &mut panes {
        let state = state.bypass_change_detection();
        let area = canvas.area();
        if area.is_empty() {
            continue;
        }
        let buffer = canvas.buffer_mut();
        if let Some((title, row)) = pane.title.as_ref().zip(area.rows().next()) {
            Line::from(format!(" {title}"))
                .bold()
                .bg(theme.panel_surface)
                .render(row, buffer);
        }
        // Each row's own background fills its width, blank rows too.
        for (row, area) in std::mem::take(&mut state.rows)
            .iter()
            .zip(state.area.rows())
        {
            buffer.set_style(area, row.style);
            row.render(area, buffer);
        }
        draw_below(buffer, state.below, state.area, &theme);
    }
}

/// A marker at the bottom right of a pane scrolled up: what is below and
/// how to get there.
fn draw_below(buffer: &mut ratatui::buffer::Buffer, below: Below, area: Rect, theme: &Theme) {
    let text = match below {
        Below::Nothing => return,
        Below::More => " ↓ more below (PgDn) ",
        Below::New => " ↓ new output (PgDn) ",
    };
    let width = u16::try_from(Span::from(text).width()).unwrap_or(u16::MAX);
    if area.height == 0 || area.width < width {
        return;
    }
    let marker = Rect::new(area.right() - width, area.bottom() - 1, width, 1);
    Clear.render(marker, buffer);
    Line::from(text)
        .bold()
        .style(theme.selection())
        .render(marker, buffer);
}

#[cfg(test)]
mod tests;
