//! Nested project context (spec `29-nested-context.md`): the instruction
//! files (`AGENTS.md`, or `CLAUDE.md`) of the project's subdirectories
//! reach an agent when it first works there, not up front.
//!
//! When a tool call names a `path` or a `cwd` under the project root, each
//! directory between the root and that place is checked for an instruction
//! file. The root is left out: rig's `AgentsMdPlugin` puts the root's file
//! in the system prompt, so adding it here would give it twice. A Review
//! system adds each one this agent was not given yet to the call's result
//! before it is committed, and again, as a change, one whose text changed
//! since. What each agent was given is saved with the session and
//! forgotten when its conversation is condensed, so a compacted agent is
//! told again on its next access. The Review system is in the public
//! [`NestedContextSystems`] set, so another reviewer that replaces tool
//! outcomes orders against it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rig_harness::prelude::*;
use rig_harness::rig_core::effect::Answered;
use rig_harness::rig_core::message::ToolResultContent;
use rig_harness::rig_core::tool::ToolOutput;
use rig_harness::rig_memory::summary_in;
use rig_tools::fs::read_prefix;

#[cfg(test)]
mod tests;

/// The most bytes of one instruction file given.
const MAX_FILE_BYTES: usize = 32 * 1024;

/// Adds nested instruction files to the results of the calls working
/// under them.
#[derive(Default)]
pub struct NestedContextPlugin;

impl Plugin for NestedContextPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .register_required_components::<Agent, NestedLoaded>()
            .add_systems(Update, forget_compacted.after(AgentLoop))
            .configure_sets(AgentPass, NestedContextSystems.in_set(Stage::Review))
            .add_systems(AgentPass, add_rules.in_set(NestedContextSystems));
    }
}

/// The Review system that appends nested rules to tool results, for
/// another reviewer of tool outcomes to order against: `.after` it to see
/// the rules in the result, `.before` it to have them appended to yours.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NestedContextSystems;

/// The nested instruction files this agent was given, by their path from
/// the project root, with the text it was given.
#[derive(Component, Reflect, Default, Debug)]
#[reflect(Component, Saved)]
pub struct NestedLoaded(pub BTreeMap<String, String>);

/// On a call whose result was checked: once is enough.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Checked;

/// Adds to each tool result, before it is committed, the instruction
/// files of the place the call worked in that its agent does not hold
/// yet, or that changed since it was given them.
fn add_rules(
    mut calls: Query<
        (Entity, &ToolRequest, &ForAgent, &mut Outcome),
        (ReadyToSettle, ToolCalls, Without<Checked>),
    >,
    mut agents: Query<&mut NestedLoaded>,
    root: Res<ProjectRoot>,
    mut commands: Commands,
) {
    for (call, request, &ForAgent(agent), mut outcome) in &mut calls {
        commands.entity(call).insert(Checked);
        let Ok(Answered::ToolResult { result }) = &outcome.0 else {
            continue;
        };
        let (Some(place), Ok(mut loaded)) = (
            place(&request.call.function.arguments),
            agents.get_mut(agent),
        ) else {
            continue;
        };
        let mut notes = Vec::new();
        for (key, dir, text) in nested_files(&root.0, &place) {
            let note = match loaded.0.get(&key) {
                None => format!("Rules for `{dir}/` (from its {}):\n{text}", file_name(&key)),
                Some(given) if *given != text => format!("{key} changed:\n{text}"),
                Some(_) => continue,
            };
            notes.push(note);
            loaded.0.insert(key, text);
        }
        if notes.is_empty() {
            continue;
        }
        let mut content = result.output().as_content().to_vec();
        content.push(ToolResultContent::text(notes.join("\n\n")));
        if let Ok(output) = ToolOutput::content(content) {
            *outcome = Outcome::from(result.clone().with_output(output));
        }
    }
}

/// The place a call works in: its `cwd`, else its `path`.
fn place(arguments: &serde_json::Map<String, serde_json::Value>) -> Option<PathBuf> {
    let named = |key| arguments.get(key).and_then(serde_json::Value::as_str);
    named("cwd").or_else(|| named("path")).map(PathBuf::from)
}

/// The instruction files of the directories from below `root` down to
/// `place` (a file's directory, or a directory), most general first: each
/// with its path from `root`, its directory's path from `root`, and its
/// text. None when `place` is not under `root`.
fn nested_files(root: &Path, place: &Path) -> Vec<(String, String, String)> {
    let (Some(root), Some(dir)) = (existing(root), existing(&root.join(place))) else {
        return Vec::new();
    };
    let Ok(below) = dir.strip_prefix(&root) else {
        return Vec::new();
    };
    let mut files = Vec::new();
    let mut at = root.clone();
    for part in below {
        at.push(part);
        let names = ["AGENTS.md", "CLAUDE.md"].map(|name| at.join(name));
        let Some(file) = names.into_iter().find(|path| path.is_file()) else {
            continue;
        };
        let Ok(contents) = read_prefix(&file, MAX_FILE_BYTES) else {
            continue;
        };
        let text = contents.text.trim().to_owned();
        let (Ok(key), Ok(dir)) = (file.strip_prefix(&root), at.strip_prefix(&root)) else {
            continue;
        };
        if !text.is_empty() {
            files.push((key.display().to_string(), dir.display().to_string(), text));
        }
    }
    files
}

/// The nearest directory of `path` that exists, `path` itself when it is
/// one, without symbolic links.
fn existing(path: &Path) -> Option<PathBuf> {
    path.ancestors()
        .filter(|dir| dir.is_dir())
        .find_map(|dir| dir.canonicalize().ok())
}

/// The file name of the instruction file at `key`.
fn file_name(key: &str) -> &str {
    Path::new(key)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(key)
}

/// A conversation compacted into a summary drops the results it
/// summarized: the agent is given each file again on its next access.
fn forget_compacted(
    mut committed: MessageReader<ConversationChanged>,
    mut agents: Query<&mut NestedLoaded>,
) {
    for ConversationChanged { agent, change } in committed.read() {
        if summary_in(&change.change).is_some()
            && let Ok(mut loaded) = agents.get_mut(*agent)
            && !loaded.0.is_empty()
        {
            loaded.0.clear();
        }
    }
}
