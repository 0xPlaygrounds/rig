//! SP-10: edit churn monitor.
//!
//! Counts, for each file an agent's tools change, the changing calls, the
//! lines added and removed, and the lines reverted: lines the same agent
//! added earlier in the session and now removes again, matched by their
//! text. What the tools publish as `FilesChanged` is all it reads, so a
//! change the user or another program makes is never counted; a line the
//! agent added that someone else removed since (the file as the next call
//! found it no longer holds it) is forgotten, so the agent's later edits
//! never count it as reverted.
//!
//! When one file gets more than `limit` changing calls within an agent's
//! last `turns` turns, that agent's status line says `churn: <file> ×N`
//! and a notice says so once per episode; with `nudge = true`, the first
//! model request after an episode starts carries one short note naming the
//! file. `/churn` shows the session's table, subagents under their own
//! names. The counts are saved with the session, so a resume restores them.
//!
//! ```toml
//! [churn]
//! limit = 6      # changing calls of one file, more than this is churn
//! turns = 3      # over the agent's last this many turns
//! nudge = false  # tell the model once per episode
//! ```

use std::collections::BTreeMap;
use std::path::Path;

use rig_harness::prelude::*;
use rig_harness::rig_core::message::Message;
use rig_harness::rig_core::tool::{FileChange, FilesChanged, HunkLine};
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests;

/// The plugin's name: its notes' author.
const NAME: &str = "churn-monitor";
/// The key its counts are saved under.
const KEY: &str = "churn-monitor.churn";
/// What an agent not spawned by another is called in `/churn`.
const MAIN: &str = "main";

/// Tracks per-file edit churn of each agent's tools; see the crate docs.
#[derive(Default)]
pub struct ChurnMonitorPlugin;

/// The plugin's stage system: in `PrepareSystems::Edit`, the nudge goes
/// into the request it is due on.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChurnSystems;

impl Plugin for ChurnMonitorPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<ChurnSettings>("churn")
            .init_resource::<ProjectRoot>()
            .register_type::<Churn>()
            .init_resource::<Churn>();
        let _ = app.saved_as::<Churn>(KEY);
        app.configure_sets(AgentPass, ChurnSystems.in_set(PrepareSystems::Edit))
            .add_systems(AgentPass, nudge.in_set(ChurnSystems))
            .add_command("churn", "Per-file edit churn of this session", show_table)
            .add_observer(count_changes)
            .add_observer(turn_started);
        #[cfg(feature = "tui")]
        status::add(app);
    }
}

/// The `[churn]` settings.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct ChurnSettings {
    /// More changing calls of one file than this within the window is
    /// churn.
    pub limit: u32,
    /// The window: an agent's last this many turns, the running one
    /// included.
    pub turns: u64,
    /// Whether the model gets a note when an episode starts.
    pub nudge: bool,
}

impl Default for ChurnSettings {
    fn default() -> Self {
        Self {
            limit: 6,
            turns: 3,
            nudge: false,
        }
    }
}

/// The session's churn, by agent id; saved with the session.
#[derive(Resource, Reflect, Serialize, Deserialize, Clone, Debug, Default)]
#[reflect(opaque, Resource, Saved, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Churn {
    /// Each agent's churn, by its [`AgentId`].
    pub agents: BTreeMap<String, AgentChurn>,
}

/// One agent's churn.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct AgentChurn {
    /// What `/churn` calls it: `main`, or a subagent's title.
    pub name: String,
    /// Its counts, by file as shown (relative to the project when inside).
    pub files: BTreeMap<String, FileChurn>,
    /// The index of its newest turn.
    turn: u64,
    /// Its changing calls of each file, by turn index, for the turns in
    /// the window.
    recent: BTreeMap<u64, BTreeMap<String, u32>>,
    /// Whether an episode is under way.
    churning: bool,
    /// The file a note is due for in its next model request.
    nudge: Option<String>,
}

/// One agent's counts for one file.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
#[serde(default)]
pub struct FileChurn {
    /// Calls that changed it.
    pub calls: u32,
    /// Lines added.
    pub added: u32,
    /// Lines removed.
    pub removed: u32,
    /// Lines removed that the agent added earlier.
    pub reverted: u32,
    /// Lines the agent added that are still there, by text, with how many.
    live: BTreeMap<String, u32>,
}

impl FileChurn {
    /// Adds one call's change of the file.
    fn record(&mut self, change: &FileChange) {
        self.calls += 1;
        // Lines someone else removed since are no longer the agent's.
        match &change.before_text {
            Some(before) => {
                let there = multiset(before.lines());
                self.live.retain(|line, count| {
                    *count = (*count).min(there.get(line.as_str()).copied().unwrap_or(0));
                    *count > 0
                });
            }
            None if change.before.is_none() => self.live.clear(),
            None => {}
        }
        if change.hunks.is_empty() {
            self.added += count(change.added);
            self.removed += count(change.removed);
            return;
        }
        for line in change.hunks.iter().flat_map(|hunk| &hunk.lines) {
            match line {
                HunkLine::Added(text) => {
                    self.added += 1;
                    *self.live.entry(bare(text).to_owned()).or_default() += 1;
                }
                HunkLine::Removed(text) => {
                    self.removed += 1;
                    if let Some(live) = self.live.get_mut(bare(text))
                        && *live > 0
                    {
                        *live -= 1;
                        self.reverted += 1;
                    }
                }
                HunkLine::Same(_) => {}
            }
        }
        self.live.retain(|_, count| *count > 0);
    }
}

impl AgentChurn {
    /// The file with the most changing calls in the window, and how many.
    pub fn worst(&self, settings: &ChurnSettings) -> Option<(&str, u32)> {
        let from = self.turn.saturating_sub(settings.turns.saturating_sub(1));
        let mut sums: BTreeMap<&str, u32> = BTreeMap::new();
        for (_, files) in self.recent.range(from..) {
            for (file, calls) in files {
                *sums.entry(file.as_str()).or_default() += calls;
            }
        }
        sums.into_iter().max_by_key(|(_, calls)| *calls)
    }

    /// Drops turns before the window, and starts or ends an episode;
    /// `Some(file)` when one starts.
    fn settle(&mut self, settings: &ChurnSettings) -> Option<String> {
        let from = self.turn.saturating_sub(settings.turns.saturating_sub(1));
        self.recent.retain(|turn, _| *turn >= from);
        let over = self
            .worst(settings)
            .filter(|(_, calls)| *calls > settings.limit)
            .map(|(file, _)| file.to_owned());
        match (over, self.churning) {
            (Some(file), false) => {
                self.churning = true;
                self.nudge = settings.nudge.then(|| file.clone());
                Some(file)
            }
            (None, true) => {
                self.churning = false;
                self.nudge = None;
                None
            }
            _ => None,
        }
    }
}

/// `text` without its line ending.
fn bare(text: &str) -> &str {
    text.trim_end_matches(['\n', '\r'])
}

/// The lines of `lines` with how often each occurs.
fn multiset<'a>(lines: impl Iterator<Item = &'a str>) -> BTreeMap<&'a str, u32> {
    let mut counts = BTreeMap::new();
    for line in lines {
        *counts.entry(bare(line)).or_default() += 1;
    }
    counts
}

fn count(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// `path` relative to the project root or the agent's directory when
/// inside one of them.
fn shown(path: &str, roots: &[&Path]) -> String {
    let path = Path::new(path);
    let inside = roots.iter().find_map(|root| path.strip_prefix(root).ok());
    inside.unwrap_or(path).to_string_lossy().replace('\\', "/")
}

/// Counts each settled tool call's published file changes for its agent.
/// A call restored after a restart and answered from its record carries
/// no published facts, so nothing is counted twice.
fn count_changes(
    settled: On<Add<Settled>>,
    calls: Query<
        (&Published, &ForAgent, Option<&CallOf>),
        (With<ToolRequest>, Without<Interrupted>),
    >,
    turns: Query<&Turn>,
    agents: Query<(&AgentId, &Name, Has<SpawnedBy>, Option<&WorkDir>)>,
    (root, settings): (Res<ProjectRoot>, Res<ChurnSettings>),
    mut churn: ResMut<Churn>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((published, &ForAgent(agent), call_of)) = calls.get(settled.entity) else {
        return;
    };
    let Ok(Some(FilesChanged(changes))) = published.0.get::<FilesChanged>() else {
        return;
    };
    let Ok((id, name, spawned, work_dir)) = agents.get(agent) else {
        return;
    };
    if changes.is_empty() {
        return;
    }
    let state = churn.agents.entry(id.0.clone()).or_default();
    state.name = if spawned {
        name.as_str().to_owned()
    } else {
        MAIN.to_owned()
    };
    let turn = call_of
        .and_then(|&CallOf(turn)| turns.get(turn).ok())
        .map_or(state.turn, |turn| turn.index);
    state.turn = state.turn.max(turn);
    let roots: Vec<&Path> = work_dir
        .map(|dir| dir.0.as_path())
        .into_iter()
        .chain([root.0.as_path()])
        .collect();
    for change in &changes {
        let file = shown(&change.path, &roots);
        state.files.entry(file.clone()).or_default().record(change);
        *state
            .recent
            .entry(turn)
            .or_default()
            .entry(file)
            .or_default() += 1;
    }
    if let Some(file) = state.settle(&settings) {
        let calls = state.worst(&settings).map_or(0, |(_, calls)| calls);
        let text = format!(
            "churn: {file} changed by {calls} calls in the last {} turns",
            settings.turns
        );
        notices.write(Notice::warn(agent, text));
    }
}

/// A new turn moves its agent's window on, which may end an episode.
fn turn_started(
    added: On<Add<TurnOf>>,
    turns: Query<(&Turn, &TurnOf)>,
    agents: Query<&AgentId>,
    settings: Res<ChurnSettings>,
    mut churn: ResMut<Churn>,
) {
    let Ok((turn, &TurnOf(agent))) = turns.get(added.entity) else {
        return;
    };
    let Ok(id) = agents.get(agent) else {
        return;
    };
    // Only agents that changed a file have a window to move.
    let Some(state) = churn.bypass_change_detection().agents.get_mut(&id.0) else {
        return;
    };
    if turn.index > state.turn {
        state.turn = turn.index;
        state.settle(&settings);
        churn.set_changed();
    }
}

/// The note due on an agent's first turn request after an episode
/// started, added to what is sent, not to the conversation.
fn nudge(
    mut calls: Query<(&mut ModelRequest, &ForAgent), (Added<Admitted>, With<TurnCall>)>,
    agents: Query<&AgentId>,
    settings: Res<ChurnSettings>,
    mut churn: ResMut<Churn>,
) {
    for (mut call, &ForAgent(agent)) in &mut calls {
        let Ok(id) = agents.get(agent) else {
            continue;
        };
        // Changed, and so saved, only when a note is taken.
        let agents = &mut churn.bypass_change_detection().agents;
        let Some(state) = agents.get_mut(&id.0) else {
            continue;
        };
        let Some(file) = state.nudge.take() else {
            continue;
        };
        let calls = state.worst(&settings).map_or(0, |(_, calls)| calls);
        churn.set_changed();
        let text = format!(
            "[churn monitor] You have changed {file} in {calls} calls over your last {} turns, \
             adding and removing the same lines. Step back and rethink the approach before \
             editing it again.",
            settings.turns
        );
        if let Some(history) = &mut call.history {
            history.push(Entry::new(Message::user(text)).by(Author::plugin(NAME)));
        }
    }
}

/// `/churn`: the session's table, most changing calls first.
fn show_table(In(args): In<CommandArgs>, churn: Res<Churn>, mut notices: MessageWriter<Notice>) {
    notices.write(Notice::info(args.agent, table(&churn)));
}

/// The session's churn as a table: file, agent, calls, lines added,
/// removed and reverted, sorted by calls.
pub fn table(churn: &Churn) -> String {
    let mut rows: Vec<(&str, &str, &FileChurn)> = churn
        .agents
        .values()
        .flat_map(|agent| {
            let files = agent.files.iter();
            files.map(|(file, counts)| (file.as_str(), agent.name.as_str(), counts))
        })
        .collect();
    if rows.is_empty() {
        return "No file was changed by the agents' tools in this session.".to_owned();
    }
    rows.sort_by(|a, b| {
        b.2.calls
            .cmp(&a.2.calls)
            .then(a.0.cmp(b.0))
            .then(a.1.cmp(b.1))
    });
    let cells: Vec<[String; 6]> = rows
        .iter()
        .map(|(file, agent, counts)| {
            [
                (*file).to_owned(),
                (*agent).to_owned(),
                counts.calls.to_string(),
                format!("+{}", counts.added),
                format!("−{}", counts.removed),
                counts.reverted.to_string(),
            ]
        })
        .collect();
    let header = ["file", "agent", "calls", "+lines", "−lines", "reverted"].map(str::to_owned);
    let mut widths = [0; 6];
    for row in std::iter::once(&header).chain(&cells) {
        for (width, cell) in widths.iter_mut().zip(row) {
            *width = (*width).max(cell.chars().count());
        }
    }
    let line = |row: &[String; 6]| {
        let padded = row.iter().zip(widths).map(|(cell, width)| {
            let pad = width.saturating_sub(cell.chars().count());
            format!("{cell}{}", " ".repeat(pad))
        });
        padded.collect::<Vec<_>>().join("  ").trim_end().to_owned()
    };
    std::iter::once(&header)
        .chain(&cells)
        .map(line)
        .collect::<Vec<_>>()
        .join("\n")
}

/// `churn: <file> ×N` in an agent's status line while it churns.
#[cfg(feature = "tui")]
mod status {
    use rig_harness::prelude::*;
    use rig_tui::{Side, StatusItem, StatusItems, StatusSystems, Tone};

    use super::{Churn, ChurnSettings};

    /// After the agents at work; kept long when the line is narrow.
    const ITEM: StatusItem = StatusItem::new("churn", Side::Left, 75, 17);

    pub(super) fn add(app: &mut App) {
        app.add_systems(
            PostUpdate,
            show.in_set(StatusSystems::Show)
                .run_if(resource_changed::<Churn>),
        );
    }

    fn show(
        mut agents: Query<(&AgentId, &mut StatusItems), With<Agent>>,
        churn: Res<Churn>,
        settings: Res<ChurnSettings>,
    ) {
        for (id, mut items) in &mut agents {
            let worst = churn
                .agents
                .get(&id.0)
                .and_then(|state| state.worst(&settings));
            let item = match worst {
                Some((file, calls)) if calls > settings.limit => {
                    ITEM.says(format!("churn: {file} ×{calls}"), Tone::Yellow)
                }
                _ => ITEM.says("", Tone::Dim),
            };
            items.show(item);
        }
    }
}
