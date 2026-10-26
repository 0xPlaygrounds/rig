//! The summary in the terminal view: a block that says it stands for the
//! earlier messages, then all of it. A plugin that draws summaries its own
//! way draws [`summary_author`]'s entries in a system after
//! [`SummaryLooks`].

use rig_ecs::prelude::*;
use rig_tui::ratatui::style::Style;
use rig_tui::ratatui::text::Line;
use rig_tui::{EntryView, RenderBatch, RenderSystems, TranscriptRender, text_rows};

use crate::compaction::summary_author;

/// The system that draws summaries in the terminal view, in
/// [`RenderSystems::Render`].
#[derive(SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SummaryLooks;

/// Draws the summary's look in the terminal view.
pub(crate) fn add(app: &mut App) {
    app.add_systems(
        TranscriptRender,
        draw.in_set(SummaryLooks).in_set(RenderSystems::Render),
    );
}

/// Draws the summaries.
fn draw(mut batch: ResMut<RenderBatch>) {
    let AuthorKind::Plugin(plugin) = summary_author().kind else {
        return;
    };
    for (summary, output) in batch.entries_mut(&plugin) {
        output.lines = summary_lines(&summary.view());
    }
}

/// A summary's lines: a banner, then the summary.
fn summary_lines(summary: &EntryView<'_>) -> Vec<Line<'static>> {
    let style = Style::new().magenta();
    let mut lines = vec![Line::styled(
        "The earlier messages were compacted into this summary",
        style.bold(),
    )];
    lines.extend(text_rows(summary.text, style.dim()));
    lines
}
