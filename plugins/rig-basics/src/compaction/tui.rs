//! The summary in the terminal view: a block that says it stands for the
//! earlier messages, then all of it. A plugin
//! that draws summaries its own way disables this renderer
//! (`entry:compaction`) and adds its own for [`summary_author`]'s plugin.

use rig_ecs::prelude::*;
use rig_tui::ratatui::style::Style;
use rig_tui::ratatui::text::Line;
use rig_tui::{AppToolRenderersExt, EntryView, text_rows};

use crate::compaction::summary_author;

/// Draws the summary's look in the terminal view.
pub(crate) fn add(app: &mut App) {
    if let AuthorKind::Plugin(plugin) = summary_author().kind {
        app.add_entry_renderer(&plugin, summary_lines);
    }
}

/// A summary's lines: a banner, then the summary.
fn summary_lines(summary: &EntryView<'_>) -> Vec<Line<'static>> {
    let style = Style::new().magenta();
    let mut lines = vec![Line::styled(
        "── the earlier messages were compacted into this summary ──",
        style.bold(),
    )];
    lines.extend(text_rows(summary.text, style.dim()));
    lines
}
