//! The summary in the terminal view: a "Summary" entry that shows its
//! headings, or all of it once expanded. It takes the place of
//! rig-compaction's look for its summaries.

use rig_compaction::summary_author;
use rig_harness::prelude::*;
use rig_tui::ratatui::style::Style;
use rig_tui::ratatui::text::Line;
use rig_tui::{AppToolRenderersExt, EntryRenderer, EntryView, excerpt};

/// Draws summaries as a "Summary" entry, in place of rig-compaction's look.
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

/// "Summary", then its headings, or all of it when expanded.
fn summary(entry: &EntryView<'_>) -> Vec<Line<'static>> {
    let style = Style::new().magenta();
    let mark = if entry.expanded { "▾" } else { "▸" };
    let mut lines = vec![Line::styled(format!("{mark} Summary"), style.bold())];
    let text = if entry.expanded {
        entry.text.to_owned()
    } else {
        let headings = entry.text.lines().filter(|line| line.starts_with('#'));
        headings.collect::<Vec<_>>().join("\n")
    };
    lines.extend(excerpt(&text, usize::MAX, style.dim()));
    lines
}
