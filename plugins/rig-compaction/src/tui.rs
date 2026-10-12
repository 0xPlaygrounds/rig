//! The summary in the terminal view: a block that says it stands for the
//! earlier messages, its first lines, or all of it once expanded. A plugin
//! that draws summaries its own way disables this renderer
//! (`entry:compaction`) and adds its own for [`summary_origin`]'s plugin.

use rig_ecs::prelude::*;
use rig_tui::ratatui::style::Style;
use rig_tui::ratatui::text::Line;
use rig_tui::{AppToolRenderersExt, EntryView, excerpt};

use crate::summary_origin;

/// Lines of a summary shown collapsed.
const SUMMARY_LINES: usize = 12;

/// Draws the summary's look in the terminal view.
pub(crate) fn add(app: &mut App) {
    if let OriginKind::Plugin(origin) = summary_origin().kind {
        app.add_entry_renderer(&origin, summary_lines);
    }
}

/// A summary's lines: a banner, then its first lines, or all of it.
fn summary_lines(summary: &EntryView<'_>) -> Vec<Line<'static>> {
    let style = Style::new().magenta();
    let mut lines = vec![Line::styled(
        "── the earlier messages were compacted into this summary ──",
        style.bold(),
    )];
    let limit = if summary.expanded {
        usize::MAX
    } else {
        SUMMARY_LINES
    };
    lines.extend(excerpt(summary.text, limit, style.dim()));
    lines
}
