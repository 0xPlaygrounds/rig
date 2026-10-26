//! The summary in the terminal view: a block that says it stands for the
//! earlier messages, then the summary drawn from its saved state when
//! available, below the messages it replaced ([`Replaced`]). A plugin that
//! draws summaries its own way draws [`summary_author`]'s entries in a
//! system after [`SummaryLooks`].

use rig_core::conversation::{Entry, Spliced};
use rig_ecs::fork::AgentForked;
use rig_ecs::prelude::*;
use rig_ecs::session_log::SessionStore;
use rig_memory::{Compacted, SummaryState, summarized_in_log};
use rig_tui::ratatui::style::Style;
use rig_tui::ratatui::text::Line;
use rig_tui::{
    EntryView, RenderBatch, RenderSystems, Replaced, Theme, TranscriptRender, text_rows,
};

use crate::compaction::summary_author;

/// The system that draws summaries in the terminal view, in
/// [`RenderSystems::Render`].
#[derive(SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SummaryLooks;

/// Draws the summary's look in the terminal view, and keeps what each
/// summary replaced on screen.
pub(crate) fn add(app: &mut App) {
    app.add_systems(
        TranscriptRender,
        draw.in_set(SummaryLooks).in_set(RenderSystems::Render),
    )
    .add_systems(PostStartup, replaced_from_logs)
    .add_observer(replaced_in_fork);
}

/// Keeps the entries `spliced`, a summary just put in `agent`'s
/// conversation, took out of it, from `taken`, the entries before it.
pub(crate) fn keep_replaced(
    commands: &mut Commands,
    agent: Entity,
    spliced: &Spliced,
    taken: Vec<Entry>,
) {
    let Some(&summary) = spliced.changed.first() else {
        return;
    };
    let out = |entry: &Entry| entry.id.is_some_and(|id| spliced.removed.contains(&id));
    let taken: Vec<Entry> = taken.into_iter().filter(out).collect();
    commands
        .entity(agent)
        .entry::<Replaced>()
        .or_default()
        .and_modify(move |mut replaced| {
            replaced.0.entry(summary).or_insert(taken);
        });
}

/// What each restored agent's summaries replaced, read from its log.
fn replaced_from_logs(
    store: Option<Res<SessionStore>>,
    agents: Query<(Entity, &AgentId), With<Conversation>>,
    mut commands: Commands,
) {
    let Some(store) = store else {
        return;
    };
    for (agent, id) in &agents {
        match summarized_in_log(&*store.0, &id.0) {
            Ok(replaced) if replaced.is_empty() => {}
            Ok(replaced) => {
                commands.entity(agent).try_insert(Replaced(replaced));
            }
            Err(error) => {
                bevy_log::warn!(
                    "what the compactions of {} replaced is not shown: {error}",
                    id.0
                );
            }
        }
    }
}

/// A fork draws what its source's summaries replaced: its log is a copy
/// of the source's.
fn replaced_in_fork(forked: On<AgentForked>, replaced: Query<&Replaced>, mut commands: Commands) {
    if let Ok(replaced) = replaced.get(forked.from) {
        commands.entity(forked.fork).try_insert(replaced.clone());
    }
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
