//! The summary in the terminal view: a block that says it stands for the
//! earlier messages, then the summary drawn from its saved state when
//! available, below the messages it replaced ([`Replaced`]). A plugin that
//! draws summaries its own way draws [`summary_author`]'s entries in a
//! system after [`SummaryLooks`].

use std::collections::BTreeMap;

use rig_core::conversation::{Entry, MessageId, Spliced};
use rig_ecs::fork::AgentForked;
use rig_ecs::prelude::*;
use rig_ecs::session_log::SessionStore;
use rig_memory::{Compacted, Summarized, SummaryState, summarized_in_log};
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

/// The records of the summaries a later compaction took out of the
/// agent's conversation, by summary id, so the summaries drawn above it
/// ([`Replaced`]) are drawn from their state too. Display only.
#[derive(Component, Clone, Debug, Default)]
struct ReplacedRecords(BTreeMap<MessageId, Compacted>);

/// The entries before a summary's place, and their compaction records,
/// read just before the summary goes in.
pub(crate) struct Before(Vec<Entry>, Vec<Compacted>);

impl Before {
    /// The first `end` entries of `conversation` and its records.
    pub(crate) fn of(conversation: &rig_core::conversation::Conversation, end: usize) -> Self {
        let entries = (0..end).filter_map(|at| conversation.entry(at)).collect();
        let records = Compacted::of(conversation).map(|(_, record)| record);
        Self(entries, records.collect())
    }

    /// Keeps the entries and records `spliced`, a summary just put in
    /// `agent`'s conversation, took out of it.
    pub(crate) fn keep(self, commands: &mut Commands, agent: Entity, spliced: &Spliced) {
        let Some(&summary) = spliced.changed.first() else {
            return;
        };
        // The entries before the summary are the ones it took out.
        let Self(taken, records) = self;
        let out = |id: &MessageId| spliced.removed.contains(id);
        let records = records.into_iter().filter(|record| out(&record.summary));
        let records: Vec<_> = records.map(|record| (record.summary, record)).collect();
        let mut agent = commands.entity(agent);
        agent
            .entry::<Replaced>()
            .or_default()
            .and_modify(move |mut replaced| {
                replaced.0.entry(summary).or_insert(taken);
            });
        agent
            .entry::<ReplacedRecords>()
            .or_default()
            .and_modify(move |mut kept| {
                for (summary, record) in records {
                    kept.0.entry(summary).or_insert(record);
                }
            });
    }
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
        read_replaced(&store, (agent, id), &mut commands);
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
        Ok(Summarized { replaced, .. }) if replaced.is_empty() => {}
        Ok(Summarized { replaced, records }) => {
            let kept = (Replaced(replaced), ReplacedRecords(records));
            commands.entity(agent).try_insert(kept);
        }
        Err(error) => {
            bevy_log::warn!(
                "what the compactions of {} replaced is not shown: {error}",
                id.0
            );
        }
    }
}

/// Draws the summaries, each from its record: the conversation's, or the
/// one kept when a later compaction took it out.
fn draw(
    mut batch: ResMut<RenderBatch>,
    agents: Query<(&Conversation, Option<&ReplacedRecords>)>,
    theme: Res<Theme>,
) {
    let AuthorKind::Plugin(plugin) = summary_author().kind else {
        return;
    };
    let agent = batch.agent.and_then(|agent| agents.get(agent).ok());
    let (conversation, kept) = agent.unzip();
    let live = conversation.into_iter().flat_map(|it| Compacted::of(it));
    let kept = kept.flatten().into_iter().flat_map(|kept| kept.0.values());
    let states: Vec<Compacted> = live
        .map(|(_, record)| record)
        .chain(kept.cloned())
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
    // A path a later set holds (changed) is not listed again in an earlier
    // one (read), as the model-facing message does.
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
