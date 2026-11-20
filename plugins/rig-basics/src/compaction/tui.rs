//! The summary in the terminal view: a block that says it stands for the
//! earlier messages, then the summary and its file lists as its entry holds
//! them, below the messages it replaced ([`Replaced`]) since the conversation
//! was last replaced whole (a reset of its log). A plugin that
//! draws summaries its own way draws [`summary_author`]'s entries in a
//! system after [`SummaryLooks`].

use std::collections::BTreeMap;

use rig_core::conversation::{Change, Entry, MessageId, Spliced};
use rig_ecs::fork::AgentForked;
use rig_ecs::prelude::*;
use rig_ecs::session_log::{ConversationChanged, SessionStore};
use rig_memory::{SummaryState, summarized_in_log, summarized_prefix, summary_in};
use rig_tui::ratatui::style::Style;
use rig_tui::ratatui::text::Line;
use rig_tui::{LookLine, RenderBatch, RenderSystems, Replaced, Theme, TranscriptRender};

use crate::compaction::summary_author;

/// The system that draws summaries in the terminal view, in
/// [`RenderSystems::Render`]. A plugin that draws them its own way runs
/// after it.
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
    .add_systems(Update, forget_at_resets.after(AgentLoop))
    .add_observer(replaced_in_fork);
}

/// Drops what an agent's summaries replaced before its conversation was replaced whole, as a
/// restart from its log, which reads nothing across the reset, does. A summary saved after the
/// reset keeps what it replaced. Changing the archive lays the transcript out again.
fn forget_at_resets(
    mut changed: MessageReader<ConversationChanged>,
    mut agents: Query<&mut Replaced>,
) {
    let mut since = BTreeMap::<Entity, Vec<MessageId>>::new();
    for ConversationChanged { agent, change } in changed.read() {
        if matches!(change.change, Change::Reset { .. }) {
            since.insert(*agent, Vec::new());
        } else if let (Some(summaries), Some(summary)) =
            (since.get_mut(agent), summary_in(&change.change))
        {
            summaries.push(summary);
        }
    }
    for (agent, summaries) in since {
        if let Ok(mut replaced) = agents.get_mut(agent) {
            replaced.0.retain(|summary, _| summaries.contains(summary));
        }
    }
}

/// Keeps `taken`, the entries `spliced`, a summary just put in `agent`'s
/// conversation, took out of it, on screen.
pub(crate) fn keep(commands: &mut Commands, agent: Entity, spliced: &Spliced, taken: Vec<Entry>) {
    let Some(&summary) = spliced.changed.first() else {
        return;
    };
    let mut agent = commands.entity(agent);
    agent
        .entry::<Replaced>()
        .or_default()
        .and_modify(move |mut replaced| {
            replaced.0.insert(summary, taken);
        });
}

/// What each restored agent's summaries replaced, read from its log; an
/// agent with no summary has nothing to read.
fn replaced_from_logs(
    store: Option<Res<SessionStore>>,
    agents: Query<(Entity, &AgentId, &Conversation)>,
    mut commands: Commands,
) {
    let Some(store) = store else {
        return;
    };
    for (agent, id, conversation) in &agents {
        if summarized_prefix(conversation) > 0 {
            read_replaced(&store, (agent, id), &mut commands);
        }
    }
}

/// What a fork's summaries replaced, read from its own log, a copy of its
/// source's up to where it forked: a summary its source made later is no
/// part of it.
fn replaced_in_fork(
    forked: On<AgentForked>,
    store: Option<Res<SessionStore>>,
    ids: Query<&AgentId>,
    mut commands: Commands,
) {
    if let (Some(store), Ok(id)) = (store, ids.get(forked.fork)) {
        read_replaced(&store, (forked.fork, id), &mut commands);
    }
}

/// Puts what the summaries in `agent`'s log, `id`, replaced on it.
fn read_replaced(store: &SessionStore, (agent, id): (Entity, &AgentId), commands: &mut Commands) {
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

/// Draws the summaries, each from its own entry's text.
fn draw(mut batch: ResMut<RenderBatch>, theme: Res<Theme>) {
    let AuthorKind::Plugin(plugin) = summary_author().kind else {
        return;
    };
    let width = usize::from(batch.width);
    for (summary, output) in batch.entries_mut(&plugin) {
        let state = SummaryState::read(&summary.text);
        output.lines = summary_lines(&state, &theme, width);
    }
}

/// A summary's banner, markdown and file lists.
fn summary_lines(state: &SummaryState, theme: &Theme, width: usize) -> Vec<LookLine> {
    let style = Style::new().magenta();
    let mut lines = vec![LookLine::from(Line::styled(
        "The earlier messages were compacted into this summary",
        style.bold(),
    ))];
    lines.extend(rig_tui::markdown::render(
        &state.summary,
        theme,
        Some(width),
    ));
    for (label, files) in [
        ("Read", &state.read_files),
        ("Changed", &state.modified_files),
    ] {
        if !files.is_empty() {
            let text = format!("{label}: {}", files.join(", "));
            lines.push(Line::styled(text, style.dim()).into());
        }
    }
    lines
}

#[cfg(test)]
mod tests;
