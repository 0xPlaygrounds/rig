//! Compaction checkpoints in the terminal view, drawn from their saved
//! summary state when available.

use rig_ecs::prelude::*;
use rig_memory::{Compacted, SummaryState};
use rig_tui::ratatui::style::Style;
use rig_tui::ratatui::text::Line;
use rig_tui::{EntryView, RenderBatch, RenderSystems, Theme, TranscriptRender, text_rows};

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

fn draw(mut batch: ResMut<RenderBatch>, agents: Query<&Conversation>, theme: Res<Theme>) {
    let AuthorKind::Plugin(plugin) = summary_author().kind else {
        return;
    };
    let conversation = batch.agent.and_then(|agent| agents.get(agent).ok());
    let states: Vec<_> = conversation
        .into_iter()
        .flat_map(|conversation| Compacted::of(conversation))
        .map(|(_, compacted)| compacted)
        .collect();
    let width = usize::from(batch.width);
    for (summary, output) in batch.entries_mut(&plugin) {
        let state = states
            .iter()
            .find(|state| Some(state.summary) == summary.id);
        output.lines = summary_lines(&summary.view(), state.map(|it| &it.state), &theme, width);
    }
}

/// A summary's banner, markdown and tracked paths, or its plain text without a record.
fn summary_lines(
    entry: &EntryView<'_>,
    state: Option<&SummaryState>,
    theme: &Theme,
    width: usize,
) -> Vec<Line<'static>> {
    let style = Style::new().magenta();
    let mut lines = vec![Line::styled(
        "The earlier messages were compacted into this summary",
        style.bold(),
    )];
    let Some(state) = state else {
        lines.extend(text_rows(entry.text, style.dim()));
        return lines;
    };
    lines.extend(rig_tui::markdown::render(
        &state.summary,
        theme,
        Some(width),
    ));
    for (index, set) in state.tracked.iter().enumerate() {
        let later = state.tracked.get(index + 1..).unwrap_or_default();
        let values: Vec<_> = set
            .values
            .iter()
            .filter(|value| !later.iter().any(|set| set.values.contains(*value)))
            .map(String::as_str)
            .collect();
        if values.is_empty() {
            continue;
        }
        let label = match set.name.as_str() {
            "read-files" => "Read",
            "modified-files" => "Changed",
            name => name,
        };
        lines.push(Line::styled(
            format!("{label}: {}", values.join(", ")),
            style.dim(),
        ));
    }
    lines
}

#[cfg(test)]
mod tests;
