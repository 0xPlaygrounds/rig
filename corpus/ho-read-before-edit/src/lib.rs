//! Refuse edits to files the agent has not read (spec `H01-read-before-edit.md`).
//!
//! Each agent keeps, in its saved [`ReadFiles`], the content hash of every
//! file it last read with a successful `read`, or changed with a
//! successful `edit` or `write`: the hashes those tools published
//! ([`FilesRead`], [`FileChange::after`]) of the bytes they read or wrote.
//! A Gate system judges each `edit`, and each `write` over an existing
//! file, when it is about to run, by the file its footprint names: a file
//! the agent never read, or one whose content changed on disk since, is
//! refused, and the model reads why. A conversation compacted into a
//! summary forgets what its agent read.

use std::collections::HashMap;
use std::path::Path;

use rig_harness::prelude::*;
use rig_harness::rig_core::tool::{FileChange, FilesChanged, FilesRead, content_hash};
use rig_harness::rig_memory::summary_in;
use rig_tools::{Edit, Read, Write};

/// The content hash of each file the agent last read or wrote itself, by
/// resolved path: what its model has seen of the file. Saved with the
/// session.
#[derive(Component, Reflect, Default, Debug)]
#[reflect(Component, Saved)]
pub struct ReadFiles(pub HashMap<String, String>);

/// The gate that refuses edits of files not read, for others to order
/// against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReadBeforeEditSystems;

/// Refuses `edit` and `write` on an existing file the agent has not read
/// as it is on disk now.
#[derive(Default)]
pub struct ReadBeforeEditPlugin;

impl Plugin for ReadBeforeEditPlugin {
    fn build(&self, app: &mut App) {
        app.register_required_components::<Agent, ReadFiles>()
            .configure_sets(AgentPass, ReadBeforeEditSystems.in_set(Stage::Gate))
            .add_systems(AgentPass, refuse_unread.in_set(ReadBeforeEditSystems))
            .add_observer(remember)
            .add_systems(Update, forget.after(AgentLoop));
    }
}

/// Refuses each `edit`, and each `write` over an existing file, whose
/// file the agent has not read, or has not read as it is now.
fn refuse_unread(
    mut calls: Query<
        (&ToolRequest, &Footprint, &ForAgent, &mut Decision),
        (ReadyToStart, ToolCalls),
    >,
    agents: Query<&ReadFiles>,
) {
    for (request, footprint, &ForAgent(agent), mut decision) in &mut calls {
        let name = request.call.function.name.as_str();
        let Footprint(Touches::Writes(Scope::Paths(paths))) = footprint else {
            continue;
        };
        if !matches!(name, Edit::NAME | Write::NAME) {
            continue;
        }
        let read = agents.get(agent).ok();
        let why = paths.iter().find_map(|path| unread(read, path));
        if let Some(why) = why {
            decision.decide(
                Decider::plugin("ho-read-before-edit"),
                Outcome::refused(why),
            );
        }
    }
}

/// Why the agent that read `read` may not change the file at `path`, if it
/// may not: it exists, and the agent has not read it as it is now. A new
/// file, or one that is not a file, is the tool's own business.
fn unread(read: Option<&ReadFiles>, path: &Path) -> Option<String> {
    if !path.is_file() {
        return None;
    }
    let shown = path.display();
    let Some(seen) = read.and_then(|read| read.0.get(&*path.to_string_lossy())) else {
        return Some(format!(
            "Read {shown} first: it has not been read in this conversation."
        ));
    };
    let now = rig_tools::fs::read_text(&path.to_string_lossy())
        .ok()
        .map(|text| content_hash(text.as_bytes()).0);
    (now.as_ref() != Some(seen))
        .then(|| format!("{shown} changed since you read it; read it again."))
}

/// Remembers the hash of each file a successful `read`, `edit` or `write`
/// of the agent read or left, as the call settles.
fn remember(
    settled: On<Add<Settled>>,
    calls: Query<(&ToolRequest, &Outcome, &Published, &ForAgent), ToolCalls>,
    mut agents: Query<&mut ReadFiles>,
) {
    let Ok((request, outcome, published, &ForAgent(agent))) = calls.get(settled.entity) else {
        return;
    };
    let Ok(mut known) = agents.get_mut(agent) else {
        return;
    };
    if !outcome.tool_result().is_success() {
        return;
    }
    let files: Vec<(String, Option<_>)> = match request.call.function.name.as_str() {
        Read::NAME => {
            let read = published.0.get::<FilesRead>().ok().flatten();
            let read = read.unwrap_or_default().0.into_iter();
            read.map(|file| (file.path, Some(file.hash))).collect()
        }
        Edit::NAME | Write::NAME => {
            let changed = published.0.get::<FilesChanged>().ok().flatten();
            let changed = changed.unwrap_or_default().0.into_iter();
            changed
                .map(|FileChange { path, after, .. }| (path, after.and_then(|it| it.hash)))
                .collect()
        }
        _ => return,
    };
    for (path, hash) in files {
        match hash {
            Some(hash) => known.0.insert(path, hash.0),
            None => known.0.remove(&path),
        };
    }
}

/// Forgets what an agent read once its conversation is compacted into a
/// summary: its model no longer holds the files' content.
fn forget(mut committed: MessageReader<ConversationChanged>, mut agents: Query<&mut ReadFiles>) {
    for ConversationChanged { agent, change } in committed.read() {
        if summary_in(&change.change).is_some()
            && let Ok(mut read) = agents.get_mut(*agent)
        {
            read.0.clear();
        }
    }
}

#[cfg(test)]
mod tests;
