//! The summary in the terminal view: a "Summary" entry that shows all of
//! it. It takes the place of `rig_basics::compaction`'s look for its summaries.

use rig_basics::compaction::summary_author;
use rig_harness::prelude::*;
use rig_tui::ratatui::style::Style;
use rig_tui::ratatui::text::Line;
use rig_tui::{AppToolRenderersExt, EntryRenderer, EntryView, text_rows};

/// Draws summaries as a "Summary" entry, in place of `rig_basics::compaction`'s look.
pub(crate) fn add(app: &mut App) {
    let AuthorKind::Plugin(origin) = summary_author().kind else {
        return;
    };
    let world = app.world_mut();
    let mut renderers = world.query::<(Entity, &EntryRenderer)>();
    let theirs: Vec<Entity> = renderers
        .iter(world)
        .filter(|(_, renderer)| renderer.plugin == origin)
        .map(|(entity, _)| entity)
        .collect();
    for renderer in theirs {
        world.entity_mut(renderer).insert(Disabled);
    }
    app.add_entry_renderer(&origin, summary);
}

/// "Summary", then the summary.
fn summary(entry: &EntryView<'_>) -> Vec<Line<'static>> {
    let style = Style::new().magenta();
    let mut lines = vec![Line::styled("Summary", style.bold())];
    lines.extend(text_rows(entry.text, style.dim()));
    lines
}
