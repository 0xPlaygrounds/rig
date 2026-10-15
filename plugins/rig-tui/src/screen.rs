//! The default screen, built from the view's public widgets only: a
//! [`TranscriptPane`] over the shown agent, which takes what the other
//! panels leave, a [`StatusLine`] under it, the [`InlineStrip`] of a
//! question asked beside the work, and the input, an [`Editor`]. An app
//! that builds its own screen adds [`TuiViewPlugin`](crate::TuiViewPlugin)
//! without this one and spawns its own, marking the transcript and the
//! editor the keys go to [`MainPane`] and [`MainEditor`].

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use ratatui::layout::Constraint;

use crate::{
    Editor, InlineStrip, MainEditor, MainPane, PaneKeys, Placement, StatusLine, TranscriptPane,
    TranscriptSource, TuiPanel,
};

/// Spawns the default screen's widgets, as ordinary entities, first, so
/// the panels other plugins add go above the status line or beside the
/// transcript. The [`TuiPlugin`](crate::TuiPlugin) adds it.
#[derive(Default)]
pub struct DefaultScreenPlugin;

impl Plugin for DefaultScreenPlugin {
    fn build(&self, app: &mut App) {
        let bottom = |rows| TuiPanel::new(Placement::Bottom(Constraint::Length(rows)));
        let world = app.world_mut();
        // Laid out in this order from the bottom: the input, the strip
        // above it, the status line above that.
        world.spawn((
            Name::new("tui:input"),
            Editor::default(),
            MainEditor,
            bottom(3),
        ));
        world.spawn((Name::new("tui:strip"), InlineStrip));
        world.spawn((Name::new("tui:status"), StatusLine::default()));
        world.spawn((
            Name::new("tui:transcript"),
            TranscriptPane::new(TranscriptSource::Shown),
            MainPane,
            PaneKeys,
        ));
    }
}
