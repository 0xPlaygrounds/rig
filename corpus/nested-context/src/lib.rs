//! Nested project context (spec `29-nested-context.md`): the instruction
//! files (`AGENTS.md`, or `CLAUDE.md`) of the project's subdirectories
//! reach an agent when it first works there, not up front.
//!
//! When a tool call names a `path` or a `cwd` under the project root, each
//! directory between the root (exclusive: its file is in the system prompt
//! already) and that place is checked for an instruction file. One this
//! agent was not given yet goes to it as a note, which the kernel sends
//! with the next model call, in the message holding the call's result; one
//! it was given whose text changed since goes again, as a change. What
//! each agent was given is saved with the session, and forgotten when its
//! conversation is condensed, so a compacted agent is told again on its
//! next access. `/context` also lists the nested files each agent holds.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rig_harness::prelude::*;
use rig_tools::context::{MAX_FILE_BYTES, instruction_file};
use rig_tools::fs::read_prefix;

#[cfg(test)]
mod tests;

/// The name the notes are sent under.
const PLUGIN: &str = "nested-context";

/// Sends nested instruction files to the agent working under them.
#[derive(Default)]
pub struct NestedContextPlugin;

impl Plugin for NestedContextPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .register_required_components::<Agent, NestedLoaded>()
            .add_observer(on_tool_call)
            .add_observer(forget_on_condense)
            .add_observer(list_on_context);
    }
}

/// The project root: the directory whose instruction file, and those above
/// it, the system prompt holds already. The working directory by default,
/// as the basics plugin reads it; relative tool paths resolve against it.
#[derive(Resource, Clone, Debug)]
pub struct ProjectRoot(pub PathBuf);

impl Default for ProjectRoot {
    fn default() -> Self {
        Self(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    }
}

/// The nested instruction files this agent was given, by their path from
/// the project root, with the text it was given.
#[derive(Component, Reflect, Default, Debug)]
#[reflect(Component, Saved)]
pub struct NestedLoaded(pub BTreeMap<String, String>);

/// Checks the place each tool call works in, as it starts, and sends the
/// agent the instruction files there it does not hold yet or that changed.
fn on_tool_call(
    call: On<Add<CallOf>>,
    calls: Query<(&ToolCallRun, &CallOf)>,
    turns: Query<&TurnOf>,
    mut agents: Query<&mut NestedLoaded>,
    root: Res<ProjectRoot>,
    mut commands: Commands,
) {
    let Ok((run, CallOf(turn))) = calls.get(call.entity) else {
        return;
    };
    let Ok(&TurnOf(agent)) = turns.get(*turn) else {
        return;
    };
    let Ok(mut loaded) = agents.get_mut(agent) else {
        return;
    };
    let Some(place) = place(&run.call.function.arguments) else {
        return;
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
    if !notes.is_empty() {
        let note = Deliver::new(agent, notes.join("\n\n"), DeliveryMode::Note);
        commands.trigger(note.with_origin(Origin::plugin(PLUGIN)));
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

/// A condensed conversation drops the notes it summarized: the agent is
/// given each file again on its next access. A restore's condensed start
/// (`upto` 0) summarizes nothing new.
fn forget_on_condense(
    insert: On<Insert<Condensed>>,
    condensed: Query<&Condensed>,
    mut agents: Query<&mut NestedLoaded>,
) {
    let summarized = condensed.get(insert.entity).is_ok_and(|c| c.upto > 0);
    if let (true, Ok(mut loaded)) = (summarized, agents.get_mut(insert.entity))
        && !loaded.0.is_empty()
    {
        loaded.0.clear();
    }
}

/// `/context`, the basics plugin's command, also lists the nested files
/// the agent holds.
fn list_on_context(
    run: On<RunCommand>,
    agents: Query<&NestedLoaded>,
    mut notices: MessageWriter<Notice>,
) {
    if run.line.split_whitespace().next() != Some("context") {
        return;
    }
    let Ok(loaded) = agents.get(run.entity) else {
        return;
    };
    let names: Vec<&str> = loaded.0.keys().map(String::as_str).collect();
    let listed = if names.is_empty() {
        "none".to_owned()
    } else {
        names.join(", ")
    };
    notices.write(Notice::info(
        run.entity,
        format!("Nested instruction files given: {listed}."),
    ));
}
