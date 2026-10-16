//! SP-18: a "worklist" plugin in two versions of one saved shape, and what
//! each does with a session the other saved.
//!
//! Both keep each agent's checklist as a `Follows` component under the key
//! [`KEY`]. Version 1 ([`v1::WorklistV1Plugin`]) saves a list of
//! strings, a done one led by `[x] `; version 2 ([`v2::WorklistV2Plugin`])
//! saves items with a status and the turn they were added in, at version 2
//! of the key with one upgrade step that reads version 1. The kernel reads
//! an old value through the step without writing it back, so the first
//! change is what writes the new shape. A value no step reads (a newer one,
//! or one of no known shape) is set aside by the kernel and never written
//! over; the plugin reads that from `Shelved` and shows the agent's list
//! read-only (newer) or empty with one warning (unreadable).

use rig_harness::prelude::*;
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::widgets::{Block, Paragraph};
use rig_tui::{PanelCanvas, Placement, RequestRedraw, TuiPanel, TuiSystems};

pub mod v1;
pub mod v2;

/// The key both versions keep an agent's checklist under.
pub const KEY: &str = "worklist.items";

/// What leads a done item in version 1's saved strings.
pub const DONE: &str = "[x] ";

/// On an agent whose checklist a newer worklist saved: shown empty and
/// read-only, and never written.
#[derive(Component, Debug)]
#[component(storage = "SparseSet")]
pub struct Newer;

/// On an agent whose saved checklist has no shape this version reads: it
/// starts empty, and nothing is written until `/work reset`.
#[derive(Component, Debug)]
#[component(storage = "SparseSet")]
pub struct Unreadable;

/// The panel showing the shown agent's checklist.
#[derive(Component, Debug)]
pub struct WorklistPanel;

/// A version's saved checklist, as its panel shows it.
trait Shows: Component {
    /// One line per item.
    fn lines(&self) -> Vec<String>;
}

/// What both versions add: the panel, its redraws and the marks from what
/// the kernel set aside.
fn build_common<T: Shows>(app: &mut App) {
    app.world_mut().spawn((
        Name::new("worklist: panel"),
        WorklistPanel,
        TuiPanel::new(Placement::Right(Constraint::Length(34))),
    ));
    app.add_systems(Update, (mark_set_aside, redraw::<T>))
        .add_systems(PostUpdate, draw::<T>.in_set(TuiSystems::Draw));
}

/// Marks each agent whose checklist the kernel set aside, once, with a
/// notice: [`Newer`] when its version is newer than this build reads,
/// else [`Unreadable`] with a warning.
fn mark_set_aside(
    set_aside: Res<Shelved>,
    (keys, upgrades): (Res<SavedKeys>, Res<SavedUpgrades>),
    agents: Query<(Entity, &AgentId, Has<Newer>, Has<Unreadable>)>,
    mut seen: Local<usize>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let fresh = set_aside.0.get(*seen..).unwrap_or_default();
    *seen = set_aside.0.len();
    let reads = keys.path(KEY).map_or(1, |path| upgrades.0.version(path));
    for value in fresh.iter().filter(|value| value.key == KEY) {
        let found = agents
            .iter()
            .find(|(_, id, ..)| value.agent.as_ref() == Some(*id));
        let Some((agent, _, newer, unreadable)) = found else {
            continue;
        };
        if newer || unreadable {
            continue;
        }
        if value.saved.v > reads {
            commands.entity(agent).insert(Newer);
            let text = "worklist: saved by a newer worklist; shown read-only, left as saved";
            notices.write(Notice::warn(agent, text));
        } else {
            commands.entity(agent).insert(Unreadable);
            let text = format!(
                "worklist: the saved list is unreadable ({}); starting empty, it is left as saved until /work reset",
                value.why
            );
            notices.write(Notice::warn(agent, text));
        }
    }
}

/// Asks for a frame when a checklist or its marks changed.
fn redraw<T: Shows>(
    changed: Query<(), Or<(Changed<T>, Added<Newer>, Added<Unreadable>)>>,
    mut removed: RemovedComponents<T>,
    mut redraws: MessageWriter<RequestRedraw>,
) {
    if !changed.is_empty() || removed.read().next().is_some() {
        redraws.write(RequestRedraw);
    }
}

/// Draws the shown agent's checklist (the focused one, else the main one).
fn draw<T: Shows>(
    mut panels: Query<&mut PanelCanvas, With<WorklistPanel>>,
    agents: Query<(Option<&T>, Has<Newer>, Has<Unreadable>, Has<SpawnedBy>), With<Agent>>,
    focused: Option<Res<ShownAgent>>,
) {
    let shown = focused.and_then(|shown| agents.get(shown.0?).ok());
    let shown = shown.or_else(|| agents.iter().find(|(.., spawned)| !spawned));
    let Some((list, newer, unreadable, ..)) = shown else {
        return;
    };
    let title = match (newer, unreadable) {
        (true, _) => "worklist (read-only)",
        (_, true) => "worklist (unreadable)",
        _ => "worklist",
    };
    let lines = match list {
        Some(list) if !newer && !unreadable => list.lines(),
        _ => Vec::new(),
    };
    for mut canvas in &mut panels {
        canvas.render(Paragraph::new(lines.join("\n")).block(Block::bordered().title(title)));
    }
}

/// The verb of a `/work` line and what follows it.
fn verb(args: &str) -> (&str, &str) {
    let args = args.trim();
    args.split_once(' ')
        .map_or((args, ""), |(verb, rest)| (verb, rest.trim()))
}

/// The index of the item a 1-based `n` names in a list of `len`.
fn index(n: &str, len: usize) -> Option<usize> {
    let n: usize = n.parse().ok()?;
    let at = n.checked_sub(1)?;
    (at < len).then_some(at)
}

/// Why `/work <verb>` is refused for an agent with these marks, if it is.
fn refusal(verb: &str, newer: bool, unreadable: bool) -> Option<&'static str> {
    if newer {
        Some("worklist: saved by a newer worklist; it is read-only here")
    } else if unreadable && verb != "reset" {
        Some("worklist: the saved list is unreadable; /work reset starts a new one")
    } else {
        None
    }
}

#[cfg(test)]
mod tests;
