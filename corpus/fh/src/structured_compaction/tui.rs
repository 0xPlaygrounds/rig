//! The summary in the terminal view: a "Summary" entry that shows all of
//! it. It takes the place of `rig_basics::compaction`'s look for its summaries.

use rig_basics::compaction::{SummaryLooks, summary_author};
use rig_harness::prelude::*;
use rig_tui::ratatui::style::Style;
use rig_tui::ratatui::text::Line;
use rig_tui::{EntryView, RenderBatch, RenderSystems, TranscriptRender, text_rows};

/// Draws summaries as a "Summary" entry, after `rig_basics::compaction`'s
/// look, so its lines are the ones drawn.
pub(crate) fn add(app: &mut App) {
    app.add_systems(
        TranscriptRender,
        draw.after(SummaryLooks).in_set(RenderSystems::Render),
    );
}

/// Draws the summaries.
fn draw(mut batch: ResMut<RenderBatch>) {
    let AuthorKind::Plugin(origin) = summary_author().kind else {
        return;
    };
    for (entry, output) in batch.entries_mut(&origin) {
        output.lines = summary(&entry.view()).into_iter().map(Into::into).collect();
    }
}

/// "Summary", then the summary.
fn summary(entry: &EntryView<'_>) -> Vec<Line<'static>> {
    let style = Style::new().magenta();
    let mut lines = vec![Line::styled("Summary", style.bold())];
    lines.extend(text_rows(entry.text, style.dim()));
    lines
}
