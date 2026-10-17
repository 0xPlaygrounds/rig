//! SN-11: leftover debug code detector.
//!
//! When an agent's turn would end having changed files, the lines the turn
//! added (each file compared with its content before the turn) are matched
//! against patterns by file extension: `dbg!(`, `console.log(`,
//! `breakpoint()` and the like, and `TODO`, `FIXME` or `XXX` in a comment.
//! If any match, the agent is sent one message listing each as
//! `path:line: text`, asking it to remove each one or say why it stays, and
//! its turn carries on. At most one nudge per turn the user started; what
//! the agent kept then is the file's content before the next turn, so it is
//! never raised again.
//!
//! The content before the turn is read in `Gate`, as each call that writes
//! named files (its `Footprint`) is about to run, the first time the turn
//! touches each file. A call that may write anywhere, such as `shell`,
//! reports what it changed (its checkpoint's `FilesChanged`): a file it is
//! the first to change was missing before (created) or is read, as the
//! call settles, from the turn's checkpoint taken before its first such
//! call (`TurnCheckpoint`), which keeps content in the session's scratch.
//! A subagent's calls count for the agent at the root of its spawns, whose
//! turn ends with them. `/leftovers off|on` switches it for the session,
//! saved with it.
//!
//! The project's `.rig/leftovers.toml` adds patterns:
//!
//! ```toml
//! # Words that mark a leftover in a comment, in every file.
//! comments = ["HACK"]
//!
//! # Text that marks a leftover, by file extension; "*" for every file.
//! [patterns]
//! go = ["fmt.Println(\"DEBUG"]
//! "*" = ["DO NOT COMMIT"]
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_basics::checkpoint::{CheckpointPlugin, TurnCheckpoint};
use rig_harness::prelude::*;
use rig_harness::rig_core::tool::{ChangeKind, FilesChanged};
use rig_harness::rig_tools::workspace::Workspace;
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// The command, and who the agent is told the nudge is from.
const NAME: &str = "leftovers";
/// The project's own patterns, from its root.
pub const CONFIG: &str = ".rig/leftovers.toml";

/// Nudges an agent once per turn about debug code the turn added.
#[derive(Default)]
pub struct LeftoverDetectorPlugin;

impl Plugin for LeftoverDetectorPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<CheckpointPlugin>() {
            app.add_plugins(CheckpointPlugin);
        }
        app.init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .init_resource::<LeftoverSwitch>()
            .register_required_components::<Agent, TurnFiles>()
            .add_systems(
                AgentPass,
                (
                    snapshot.in_set(Stage::Gate),
                    check_turn.in_set(ReviewSystems::Act),
                ),
            )
            .add_command(
                NAME,
                "Nudge the agent about debug code its turn added: /leftovers off|on",
                toggle,
            )
            .add_observer(snapshot_read)
            .add_observer(checkpointed)
            .add_observer(scanned)
            .add_observer(reset);
    }
}

/// Whether the checks are off for this session, saved with it.
#[derive(Resource, Reflect, Clone, Debug, Default)]
#[reflect(Resource, Saved)]
pub struct LeftoverSwitch {
    /// No snapshot is taken and no turn is checked.
    pub off: bool,
}

/// On each agent: the content, before its running turn, of each file the
/// turn (its subagents' too) is about to change or changed; `None` for a
/// file that was not there or not text.
#[derive(Component, Default, Debug)]
pub struct TurnFiles {
    before: BTreeMap<PathBuf, Option<String>>,
    reading: BTreeSet<PathBuf>,
    nudged: bool,
}

impl TurnFiles {
    /// The files the running turn is about to change or changed, each
    /// with its content before the turn.
    pub fn before(&self) -> &BTreeMap<PathBuf, Option<String>> {
        &self.before
    }
}

/// A line a turn added that looks like leftover debug code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Leftover {
    /// The file, as the workspace names it.
    pub path: PathBuf,
    /// Its line number, from 1.
    pub line: usize,
    /// The line, trimmed.
    pub text: String,
}

/// What marks a leftover: text by file extension (`*` for every file),
/// and words in a comment, in every file.
#[derive(Clone, Debug)]
pub struct Patterns {
    /// Text that marks a leftover anywhere in a line, by extension.
    pub by_extension: BTreeMap<String, Vec<String>>,
    /// Words that mark a leftover in a comment.
    pub comments: Vec<String>,
}

impl Default for Patterns {
    /// The built-in patterns.
    fn default() -> Self {
        let rust = ["dbg!(", "println!(\"DEBUG", "todo!()", "unimplemented!()"];
        let js = [
            "console.log(",
            "debugger;",
            "it.only(",
            "describe.only(",
            "test.only(",
        ];
        let python = ["breakpoint()", "pdb.set_trace()"];
        let owned = |patterns: &[&str]| patterns.iter().map(|&p| p.to_owned()).collect();
        let mut by_extension = BTreeMap::new();
        by_extension.insert("rs".to_owned(), owned(&rust));
        for extension in ["js", "jsx", "mjs", "cjs", "ts", "tsx", "mts", "cts"] {
            by_extension.insert(extension.to_owned(), owned(&js));
        }
        by_extension.insert("py".to_owned(), owned(&python));
        Self {
            by_extension,
            comments: owned(&["TODO", "FIXME", "XXX"]),
        }
    }
}

/// `.rig/leftovers.toml`.
#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct ConfigFile {
    comments: Vec<String>,
    patterns: BTreeMap<String, Vec<String>>,
}

impl Patterns {
    /// These patterns and those of `text`, a `leftovers.toml`.
    ///
    /// # Errors
    ///
    /// When `text` does not read as one.
    pub fn extended(mut self, text: &str) -> Result<Self, String> {
        let file: ConfigFile = toml::from_str(text).map_err(|error| error.to_string())?;
        self.comments.extend(file.comments);
        for (extension, patterns) in file.patterns {
            self.by_extension
                .entry(extension)
                .or_default()
                .extend(patterns);
        }
        Ok(self)
    }

    /// Whether `line` of the file `path` looks like a leftover.
    pub fn matches(&self, path: &Path, line: &str) -> bool {
        let extension = path.extension().and_then(OsStr::to_str).unwrap_or_default();
        let mut texts = [extension, "*"]
            .into_iter()
            .filter_map(|key| self.by_extension.get(key))
            .flatten();
        if texts.any(|text| line.contains(text.as_str())) {
            return true;
        }
        comment(line)
            .is_some_and(|comment| self.comments.iter().any(|word| has_word(comment, word)))
    }
}

/// The comment in `line`: what follows its first comment marker, or the
/// line itself when it continues a block comment (`* …`).
fn comment(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    if trimmed.starts_with('*') {
        return Some(trimmed);
    }
    ["//", "/*", "#", "--", "<!--"]
        .into_iter()
        .filter_map(|marker| line.find(marker).map(|at| (at, marker.len())))
        .min()
        .and_then(|(at, len)| line.get(at + len..))
}

/// Whether `word` is in `text` as a word of its own.
fn has_word(text: &str, word: &str) -> bool {
    let part = |c: char| c.is_alphanumeric() || c == '_';
    text.match_indices(word).any(|(at, _)| {
        let before = text.get(..at).and_then(|head| head.chars().next_back());
        let after = text
            .get(at + word.len()..)
            .and_then(|tail| tail.chars().next());
        !before.is_some_and(part) && !after.is_some_and(part)
    })
}

/// The lines of `after` that are not in `before`, each with its number
/// from 1.
pub fn added_lines<'a>(before: &str, after: &'a str) -> Vec<(usize, &'a str)> {
    let lines: Vec<&str> = after.lines().collect();
    let diff = similar::TextDiff::from_lines(before, after);
    diff.ops()
        .iter()
        .filter(|op| op.tag() != similar::DiffTag::Equal)
        .flat_map(|op| op.new_range())
        .filter_map(|at| lines.get(at).map(|&line| (at + 1, line)))
        .collect()
}

/// What a check of a turn's files found, and why the project's patterns
/// could not be read, if they could not.
struct Scan {
    found: Vec<Leftover>,
    problem: Option<String>,
}

/// The text of `path` in `workspace`; `None` for a missing file, or one
/// that is not text.
fn read(workspace: &dyn Workspace, path: &Path) -> Option<String> {
    workspace.read_text(path).ok()
}

/// Checks each of `before`'s files as it is now against its content
/// before, with the built-in patterns and the project's.
fn scan(
    workspace: &dyn Workspace,
    project: &Path,
    before: &BTreeMap<PathBuf, Option<String>>,
) -> Scan {
    let config = project.join(CONFIG);
    let (patterns, problem) = match read(workspace, &config) {
        None => (Patterns::default(), None),
        Some(text) => match Patterns::default().extended(&text) {
            Ok(patterns) => (patterns, None),
            Err(why) => (
                Patterns::default(),
                Some(format!("{}: {why}", config.display())),
            ),
        },
    };
    let mut found = Vec::new();
    for (path, was) in before {
        let Some(now) = read(workspace, path) else {
            continue;
        };
        let added = added_lines(was.as_deref().unwrap_or_default(), &now);
        found.extend(
            added
                .into_iter()
                .filter(|(_, line)| patterns.matches(path, line))
                .map(|(line, text)| Leftover {
                    path: path.clone(),
                    line,
                    text: text.trim().to_owned(),
                }),
        );
    }
    Scan { found, problem }
}

/// The agent at the root of `agent`'s spawns, whose turn a subagent's
/// changes count for.
fn root_of(mut agent: Entity, parents: &Query<&SpawnedBy>) -> Entity {
    while let Ok(&SpawnedBy(parent)) = parents.get(agent) {
        agent = parent;
    }
    agent
}

/// The pool a remote workspace's reads wait on.
fn pool() -> &'static TaskPool {
    IoTaskPool::get_or_init(TaskPool::default)
}

/// A file's content before the turn, read off the main thread.
struct Reading {
    agent: Entity,
    path: PathBuf,
    text: Option<String>,
}

/// Keeps, before each call that writes named files runs, the content of
/// each the turn has not touched yet: at once in this machine's workspace,
/// else off the main thread under a hold on the call.
fn snapshot(
    calls: Query<(Entity, &Footprint, &ForAgent, &Decision), (ReadyToStart, ToolCalls)>,
    parents: Query<&SpawnedBy>,
    mut agents: Query<&mut TurnFiles>,
    (switch, workspace, wake): (Res<LeftoverSwitch>, Res<ProjectWorkspace>, Res<Wake>),
    mut commands: Commands,
) {
    if switch.off {
        return;
    }
    for (call, footprint, &ForAgent(agent), decision) in &calls {
        let Touches::Writes(Scope::Paths(paths)) = &footprint.0 else {
            continue;
        };
        let root = root_of(agent, &parents);
        let Ok(mut files) = agents.get_mut(root) else {
            continue;
        };
        if decision.is_decided() {
            continue;
        }
        for path in paths {
            if files.before.contains_key(path) || files.reading.contains(path) {
                continue;
            }
            if workspace.0.host().is_none() {
                let text = read(&*workspace.0, path);
                files.before.insert(path.clone(), text);
                continue;
            }
            files.reading.insert(path.clone());
            let (workspace, path) = (workspace.0.clone(), path.clone());
            let label = Name::new(format!("leftovers: reading {}", path.display()));
            let work = async move {
                let (from, at) = (workspace.clone(), path.clone());
                let text = blocking(move || Ok(read(&*from, &at))).await.ok().flatten();
                Reading {
                    agent: root,
                    path,
                    text,
                }
            };
            commands.spawn((label, HoldOf(call), Running::spawn(pool(), &wake, work)));
        }
    }
}

/// Keeps a file's content read off the main thread, and lets its call go.
fn snapshot_read(
    read: On<Insert<Done<Reading>>>,
    snapshots: Query<&Done<Reading>>,
    mut agents: Query<&mut TurnFiles>,
    mut commands: Commands,
) {
    let Ok(Done(snapshot)) = snapshots.get(read.entity) else {
        return;
    };
    if let Ok(mut files) = agents.get_mut(snapshot.agent) {
        files.reading.remove(&snapshot.path);
        let text = snapshot.text.clone();
        files.before.entry(snapshot.path.clone()).or_insert(text);
    }
    commands.entity(read.entity).despawn();
}

/// Keeps the content before the turn of each file a call that may write
/// anywhere is the first of the turn to change: none for a file it
/// created, else the file as the turn's checkpoint from before its first
/// such call kept it, a read of the session's scratch on this machine.
fn checkpointed(
    settled: On<Add<Settled>>,
    calls: Query<(&Footprint, &Published, &ForAgent, &CallOf), With<ToolRequest>>,
    turns: Query<&TurnCheckpoint>,
    parents: Query<&SpawnedBy>,
    mut agents: Query<&mut TurnFiles>,
    switch: Res<LeftoverSwitch>,
) {
    let Ok((footprint, published, &ForAgent(agent), &CallOf(turn))) = calls.get(settled.entity)
    else {
        return;
    };
    if switch.off || footprint.0 != Touches::Writes(Scope::Anything) {
        return;
    }
    let Ok(FilesChanged(changes)) = published
        .0
        .get::<FilesChanged>()
        .map(Option::unwrap_or_default)
    else {
        return;
    };
    let Ok(mut files) = agents.get_mut(root_of(agent, &parents)) else {
        return;
    };
    let first = turns.get(turn).ok().and_then(TurnCheckpoint::first);
    for change in changes {
        let path = PathBuf::from(change.path);
        if files.before.contains_key(&path) {
            continue;
        }
        let was = match (change.kind, &first) {
            (ChangeKind::Created, _) => None,
            // Not kept: larger than a checkpoint keeps, or no scratch.
            (ChangeKind::Modified, Some(first)) => match first.content(&path) {
                Some(Ok(bytes)) => Some(String::from_utf8_lossy(&bytes).into_owned()),
                _ => continue,
            },
            _ => continue,
        };
        files.before.insert(path, was);
    }
}

/// A check of a turn's files done off the main thread.
struct Scanned {
    agent: Entity,
    scan: Scan,
}

/// Checks the files an agent's turn changed as the turn would end with an
/// answer, once per turn that finds something: at once in this machine's
/// workspace, else off the main thread under a hold on the turn.
fn check_turn(
    proposed: Query<(Entity, &TurnOf, &TurnOutcome), Added<TurnOutcome>>,
    spawned: Query<(), With<SpawnedBy>>,
    mut agents: Query<&mut TurnFiles>,
    (switch, workspace, root, wake): (
        Res<LeftoverSwitch>,
        Res<ProjectWorkspace>,
        Res<ProjectRoot>,
        Res<Wake>,
    ),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if switch.off {
        return;
    }
    for (turn, &TurnOf(agent), outcome) in &proposed {
        if !matches!(outcome, TurnOutcome::Answered(_)) || spawned.contains(agent) {
            continue;
        }
        let Ok(mut files) = agents.get_mut(agent) else {
            continue;
        };
        if files.nudged || files.before.is_empty() {
            continue;
        }
        let before = files.before.clone();
        if workspace.0.host().is_none() {
            let scan = scan(&*workspace.0, &root.0, &before);
            files.nudged = nudge(&mut commands, &mut notices, agent, &root.0, &scan);
            continue;
        }
        let (workspace, project) = (workspace.0.clone(), root.0.clone());
        let work = async move {
            let scan = blocking(move || Ok(scan(&*workspace, &project, &before))).await;
            let scan = scan.unwrap_or_else(|error| Scan {
                found: Vec::new(),
                problem: Some(error.to_string()),
            });
            Scanned { agent, scan }
        };
        let label = Name::new("leftovers: checking the added lines");
        commands.spawn((label, HoldOf(turn), Running::spawn(pool(), &wake, work)));
    }
}

/// Nudges the agent about what a check off the main thread found, and lets
/// its turn go.
fn scanned(
    done: On<Insert<Done<Scanned>>>,
    scans: Query<&Done<Scanned>>,
    mut agents: Query<&mut TurnFiles>,
    root: Res<ProjectRoot>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(Done(Scanned { agent, scan })) = scans.get(done.entity) else {
        return;
    };
    let nudged = nudge(&mut commands, &mut notices, *agent, &root.0, scan);
    if let Ok(mut files) = agents.get_mut(*agent) {
        files.nudged |= nudged;
    }
    commands.entity(done.entity).despawn();
}

/// Sends `agent` the leftovers `scan` found, as its turn's next step, and
/// warns of patterns that could not be read; whether it sent any.
fn nudge(
    commands: &mut Commands,
    notices: &mut MessageWriter<Notice>,
    agent: Entity,
    project: &Path,
    scan: &Scan,
) -> bool {
    if let Some(problem) = &scan.problem {
        notices.write(Notice::warn(agent, problem.clone()));
    }
    if scan.found.is_empty() {
        return false;
    }
    let mut text = "These lines your turn added look like leftover debug code:\n".to_owned();
    for leftover in &scan.found {
        let path = leftover
            .path
            .strip_prefix(project)
            .unwrap_or(&leftover.path);
        let line = format!(
            "- {}:{}: {}\n",
            path.display(),
            leftover.line,
            leftover.text
        );
        text.push_str(&line);
    }
    text.push_str("Remove each one, or say why it should stay.");
    let author = Author::plugin(NAME).titled("leftover debug code");
    commands.trigger(Deliver::new(agent, text, DeliveryMode::Queue).with_author(author));
    true
}

/// A turn that ended leaves nothing to compare the next one with.
fn reset(
    settled: On<Add<Settled>>,
    turns: Query<&ForAgent, With<Turn>>,
    mut agents: Query<&mut TurnFiles>,
) {
    if let Ok(&ForAgent(agent)) = turns.get(settled.entity)
        && let Ok(mut files) = agents.get_mut(agent)
    {
        *files = TurnFiles::default();
    }
}

/// `/leftovers off|on`: switches the checks for this session; without an
/// argument, says whether they are on.
fn toggle(
    In(args): In<CommandArgs>,
    mut switch: ResMut<LeftoverSwitch>,
    mut notices: MessageWriter<Notice>,
) {
    match args.args.trim() {
        "off" => switch.off = true,
        "on" => switch.off = false,
        "" => {}
        _ => {
            notices.write(Notice::error(args.agent, "/leftovers takes off or on"));
            return;
        }
    }
    let state = if switch.off { "off" } else { "on" };
    notices.write(Notice::info(
        args.agent,
        format!("Leftover checks are {state}."),
    ));
}
