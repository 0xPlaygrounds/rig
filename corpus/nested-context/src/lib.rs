//! Nested project context (spec `29-nested-context.md`): the instruction
//! files (`AGENTS.md`, or `CLAUDE.md`) of the project's subdirectories
//! reach an agent when it first works there, not up front.
//!
//! When a tool call names a `path` or a `cwd` under the project root, each
//! directory between the root (exclusive: its file is in the system prompt
//! already) and that place is checked for an instruction file. A Review
//! system adds each one this agent was not given yet to the call's result
//! before it is committed, and again, as a change, one whose text changed
//! since. What each agent was given is saved with the session and
//! forgotten when its conversation is condensed, so a compacted agent is
//! told again on its next access. `/context` lists the files given.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rig_basics::ContextEntry;
use rig_harness::prelude::*;
use rig_harness::rig_core::effect::Outcome as Answered;
use rig_harness::rig_core::message::ToolResultContent;
use rig_harness::rig_core::tool::ToolOutput;
use rig_tools::context::{MAX_FILE_BYTES, instruction_file};
use rig_tools::fs::read_prefix;

#[cfg(test)]
mod tests;

/// Adds nested instruction files to the results of the calls working
/// under them.
#[derive(Default)]
pub struct NestedContextPlugin;

impl Plugin for NestedContextPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .register_required_components::<Agent, NestedLoaded>()
            .add_observer(forget_on_condense)
            .add_systems(AgentPass, add_rules.in_set(Stage::Review))
            .add_systems(Update, list_loaded);
        app.world_mut()
            .spawn((NestedEntry, ContextEntry::new("nested rules", "")));
    }
}

/// The nested instruction files this agent was given, by their path from
/// the project root, with the text it was given.
#[derive(Component, Reflect, Default, Debug)]
#[reflect(Component, Saved)]
pub struct NestedLoaded(pub BTreeMap<String, String>);

/// On a call whose result was checked: once is enough.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Checked;

/// The plugin's line in `/context`.
#[derive(Component)]
struct NestedEntry;

/// Adds to each tool result, before it is committed, the instruction
/// files of the place the call worked in that its agent does not hold
/// yet, or that changed since it was given them.
fn add_rules(
    calls: Query<(Entity, &ToolRequest, &ForAgent, &Outcome), (Unreviewed, Without<Checked>)>,
    mut agents: Query<&mut NestedLoaded>,
    root: Res<ProjectRoot>,
    mut commands: Commands,
) {
    for (call, request, &ForAgent(agent), outcome) in &calls {
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
            let result = result.clone().with_output(output);
            commands.entity(call).insert(Outcome::from(result));
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
        let Some(file) = instruction_file(&at) else {
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

/// A condensed conversation drops the results it summarized: the agent is
/// given each file again on its next access. A restore inserts the
/// summary before the saved components, so what was given then comes back.
fn forget_on_condense(insert: On<Insert<Condensed>>, mut agents: Query<&mut NestedLoaded>) {
    if let Ok(mut loaded) = agents.get_mut(insert.entity)
        && !loaded.0.is_empty()
    {
        loaded.0.clear();
    }
}

/// Keeps the `/context` line current: the nested files each agent holds.
fn list_loaded(
    changed: Query<(), Changed<NestedLoaded>>,
    mut gone: RemovedComponents<NestedLoaded>,
    agents: Query<(&NestedLoaded, Option<&Name>)>,
    mut entry: Single<&mut ContextEntry, With<NestedEntry>>,
) {
    if changed.is_empty() && gone.read().count() == 0 {
        return;
    }
    let lines: Vec<String> = agents
        .iter()
        .filter(|(loaded, _)| !loaded.0.is_empty())
        .map(|(loaded, name)| {
            let files: Vec<&str> = loaded.0.keys().map(String::as_str).collect();
            let name = name.map_or("agent", Name::as_str);
            format!("{name}: {}", files.join(", "))
        })
        .collect();
    entry.text = lines.join("; ");
}
