//! Refuse edits to files the agent has not read (spec `H01-read-before-edit.md`).
//!
//! Each agent keeps, in its saved [`ReadFiles`], the content hash of every
//! file it last read with a successful `read`, or changed with a
//! successful `edit` or `write`, by canonical path, taken from the files
//! the call published ([`FilesRead`], [`FilesChanged`]) as it settles. A
//! Gate system judges each `edit`, and each `write` over an existing file,
//! when it is about to run: a file the agent never read, or one whose
//! content changed on disk since, is refused, and the model reads why. A
//! conversation compacted into a summary forgets what its agent read.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rig_harness::prelude::*;
use rig_harness::rig_core::tool::{FilesChanged, FilesRead, Tool};
use rig_tools::{Edit, Read, Write};
use sha2::{Digest, Sha256};

/// The content hash of each file the agent last read or wrote itself, by
/// canonical path: what its model has seen of the file. Saved with the
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

/// `path` with symlinks and `..` resolved, when it names an existing file.
fn canonical(path: &Path) -> Option<PathBuf> {
    let path = std::fs::canonicalize(path).ok()?;
    path.is_file().then_some(path)
}

/// The hash of the file at `path` as it is on disk now.
fn hash(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    Some(format!("{:x}", Sha256::digest(bytes)))
}

/// Refuses each `edit`, and each `write` over an existing file, whose
/// file the agent has not read, or has not read as it is now.
fn refuse_unread(
    calls: Query<(Entity, &ToolRequest, &ForAgent), Undecided>,
    agents: Query<&ReadFiles>,
    mut commands: Commands,
) {
    for (call, request, &ForAgent(agent)) in &calls {
        let function = &request.call.function;
        if !matches!(function.name.as_str(), Edit::NAME | Write::NAME) {
            continue;
        }
        let Some(named) = function
            .arguments
            .get("path")
            .and_then(|path| path.as_str())
        else {
            continue;
        };
        // A new file, or what the tool refuses itself.
        let Some(path) = canonical(Path::new(named)) else {
            continue;
        };
        let seen = agents
            .get(agent)
            .ok()
            .and_then(|read| read.0.get(&*path.to_string_lossy()));
        let why = match seen {
            None => format!("Read {named} first: it has not been read in this conversation."),
            Some(seen) if hash(&path).as_ref() != Some(seen) => {
                format!("{named} changed since you read it; read it again.")
            }
            Some(_) => continue,
        };
        commands.entity(call).insert_if_new(Outcome::refused(why));
    }
}

/// Remembers the hash of each file a successful `read`, `edit` or `write`
/// of the agent's read or wrote, as the call settles.
fn remember(
    settled: On<Add<Settled>>,
    calls: Query<(&ToolRequest, &Outcome, &Published, &ForAgent)>,
    mut agents: Query<&mut ReadFiles>,
) {
    let Ok((request, outcome, published, &ForAgent(agent))) = calls.get(settled.entity) else {
        return;
    };
    if !outcome.tool_result().is_success() {
        return;
    }
    let paths: Vec<String> = match request.call.function.name.as_str() {
        Read::NAME => published
            .0
            .get::<FilesRead>()
            .ok()
            .flatten()
            .unwrap_or_default()
            .paths()
            .map(str::to_owned)
            .collect(),
        Edit::NAME | Write::NAME => {
            let changed = published.0.get::<FilesChanged>().ok().flatten();
            changed
                .unwrap_or_default()
                .paths()
                .map(str::to_owned)
                .collect()
        }
        _ => return,
    };
    let Ok(mut read) = agents.get_mut(agent) else {
        return;
    };
    for path in paths {
        if let Some(path) = canonical(Path::new(&path))
            && let Some(hash) = hash(&path)
        {
            read.0.insert(path.to_string_lossy().into_owned(), hash);
        }
    }
}

/// Forgets what an agent read once its conversation is compacted into a
/// summary: its model no longer holds the files' content.
fn forget(
    mut committed: MessageReader<Committed>,
    mut agents: Query<(&Conversation, &mut ReadFiles)>,
) {
    for change in committed.read() {
        if let Committed::Splice {
            agent,
            removed,
            changed,
        } = change
            && let Ok((conversation, mut read)) = agents.get_mut(*agent)
            && compacted(conversation, removed, changed)
        {
            read.0.clear();
        }
    }
}

/// Whether a splice that took out messages put a compaction's summary in
/// one of the `changed` messages of `conversation`.
fn compacted(conversation: &Conversation, removed: &[MessageId], changed: &[MessageId]) -> bool {
    let summary = Origin::plugin("compaction");
    let changed = changed.iter().filter_map(|id| conversation.index_of(*id));
    let mut entries = changed.filter_map(|at| conversation.entry(at));
    !removed.is_empty()
        && entries.any(|entry| entry.origins.iter().any(|(_, origin)| *origin == summary))
}

#[cfg(test)]
mod tests;
