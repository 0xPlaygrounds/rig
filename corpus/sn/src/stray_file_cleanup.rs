//! Stray file cleanup: the files an agent's tools created while it worked on
//! a prompt, offered for deletion when its turn ends.
//!
//! - A call's new files are the ones it published as created
//!   ([`FilesChanged`], [`ChangeKind::Created`]): a file tool names what it
//!   wrote, and the harness's checkpoint of a call that may write anywhere
//!   (such as `shell`) names the files that appeared while it ran, each
//!   call from a snapshot taken just before it, so a file made between two
//!   calls is in neither. Ignored files count only under a `watch_ignored`
//!   path, which the plugin adds to every checkpoint ([`CheckpointAlso`]).
//!   A subagent's files count for the agent at the top of its tree.
//! - When a top-level agent's turn would end, the end is held while the
//!   files still there are measured. Where someone can answer, a
//!   multiple [`StrayPicker`] "Files created this turn" offers them,
//!   scratch-looking names ticked; then a [`StrayActions`] picker asks what
//!   to do with the ticked ones: **Delete selected** moves them into
//!   `.rig/trash/<timestamp>/`, **Add selected to .gitignore** ignores
//!   them, **Keep all** (or closing either picker) leaves them. A run nobody
//!   answers, such as `--print`, deletes nothing and prints the list after
//!   the answer.
//! - The user's next message to the agent carries a note naming the files
//!   deleted.
//! - `/restore-strays` lists the recent batches; `/restore-strays <batch>`
//!   puts one back.
//!
//! Settings, in `settings.toml`:
//!
//! ```toml
//! [stray]
//! scratch = ["tmp*", "scratch*", "*.log", "*.bak", "*.orig"]
//! watch_ignored = ["out.log", "dist"]
//! ```

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bevy_ecs::system::SystemParam;
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_basics::checkpoint::{CheckpointAlso, CheckpointPlugin};
use rig_basics::print::PrintSection;
use rig_harness::prelude::*;
use rig_harness::rig_core::tool::{ChangeKind, FilesChanged};
use rig_tools::workspace::{Stat, Workspace};
use serde::{Deserialize, Serialize};

/// The command that lists and restores deleted batches.
pub const RESTORE: &str = "restore-strays";

/// Where deleted files go, under the project root.
pub const TRASH: &str = ".rig/trash";

/// The trash's list of batches, under the project root.
const INDEX: &str = ".rig/trash/index.json";

/// How many batches `/restore-strays` lists.
const RECENT: usize = 10;

/// The picker's title.
pub const TITLE: &str = "Files created this turn";

/// The title of `/restore-strays`' picker.
pub const RESTORE_TITLE: &str = "Restore deleted strays";

/// The `[stray]` table of `settings.toml`.
#[derive(Resource, Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StraySettings {
    /// File-name globs selected at first in the picker.
    pub scratch: Vec<String>,
    /// Paths, relative to the project, under which ignored files still
    /// count as created.
    pub watch_ignored: Vec<PathBuf>,
}

impl Default for StraySettings {
    fn default() -> Self {
        Self {
            scratch: ["tmp*", "scratch*", "*.log", "*.bak", "*.orig"]
                .map(str::to_owned)
                .into(),
            watch_ignored: Vec::new(),
        }
    }
}

/// On every agent: the files its work (and its subagents') created this
/// turn, and the ones the user deleted that it has not been told of.
#[derive(Component, Default, Debug)]
pub struct Strays {
    /// The new files, by absolute path.
    pub created: BTreeSet<PathBuf>,
    /// The deleted files, as the user saw them, for the agent's next
    /// message.
    pub deleted: Vec<String>,
}

/// A file created this turn, as offered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stray {
    /// Its path under the project root (absolute outside it).
    pub path: String,
    /// Its absolute path.
    pub absolute: PathBuf,
    /// Its size in bytes.
    pub size: u64,
}

/// On the multiple picker "Files created this turn": the files, in the
/// picker's order.
#[derive(Component, Clone, Debug)]
pub struct StrayPicker {
    /// The agent whose turn made them.
    pub agent: Entity,
    /// The files.
    pub files: Vec<Stray>,
}

/// On the picker that asks what to do with the files ticked.
#[derive(Component, Clone, Debug)]
pub struct StrayActions {
    /// The agent whose turn made them.
    pub agent: Entity,
    /// The files ticked.
    pub files: Vec<Stray>,
}

/// The stray plugin's stage systems: the Prepare system that adds the
/// note of deleted files to an agent's next message, in
/// [`PrepareSystems::Edit`].
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StraySystems;

/// The plugin.
#[derive(Default)]
pub struct StrayFilesPlugin;

impl Plugin for StrayFilesPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<CheckpointPlugin>() {
            app.add_plugins(CheckpointPlugin);
        }
        app.add_setting::<StraySettings>("stray");
        let watched = app.world().get_resource::<StraySettings>();
        let watched = watched
            .map(|it| it.watch_ignored.clone())
            .unwrap_or_default();
        app.world_mut()
            .get_resource_or_init::<CheckpointAlso>()
            .0
            .extend(watched);
        app.init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .register_required_components::<Agent, Strays>()
            .configure_sets(AgentPass, StraySystems.in_set(PrepareSystems::Edit))
            .add_systems(
                AgentPass,
                (
                    end_of_turn.in_set(Stage::Review),
                    tell_deleted.in_set(StraySystems),
                ),
            )
            .add_command(
                RESTORE,
                "Restore stray files deleted after a turn: /restore-strays [batch]",
                restore,
            )
            .add_observer(files_written)
            .add_observer(measured)
            .add_observer(ticked)
            .add_observer(act)
            .add_observer(reported)
            .add_observer(batches_listed);
        app.world_mut()
            .spawn((StraySection, PrintSection::new("strays", "")));
    }
}

/// The `--print` section listing the files.
#[derive(Component)]
struct StraySection;

/// A hold on a turn's end while its files are measured.
#[derive(Component)]
struct TurnEnd {
    agent: Entity,
    files: Vec<PathBuf>,
}

/// The files of a turn that still exist.
#[derive(Default)]
struct Existing(Vec<Stray>);

/// File work for an agent: deleting, ignoring or restoring.
#[derive(Component)]
struct Job {
    agent: Entity,
}

/// What a [`Job`] did.
#[derive(Default)]
struct Report {
    lines: Vec<String>,
    deleted: Vec<String>,
    restored: Vec<String>,
}

/// `/restore-strays`' read of the trash, for an agent.
#[derive(Component)]
struct BatchList {
    agent: Entity,
}

/// The trash's batches, oldest first.
#[derive(Default)]
struct Batches(Vec<Batch>);

/// The files one deletion moved to the trash, by path under the project.
#[derive(Serialize, Deserialize, Clone, Default)]
struct Batch {
    id: String,
    files: Vec<String>,
}

/// What off-thread file work needs.
#[derive(SystemParam)]
struct Work<'w> {
    workspace: Res<'w, ProjectWorkspace>,
    root: Res<'w, ProjectRoot>,
    wake: Res<'w, Wake>,
}

impl Work<'_> {
    /// Runs `work` with the workspace and the project root, off the main
    /// thread.
    fn run<T: Default + Send + Sync + 'static>(
        &self,
        work: impl FnOnce(&dyn Workspace, &Path) -> T + Send + 'static,
    ) -> Running {
        let (workspace, root) = (Arc::clone(&self.workspace.0), self.root.0.clone());
        let pool = IoTaskPool::get_or_init(TaskPool::default);
        let blocking = blocking(move || Ok(work(&*workspace, &root)));
        Running::spawn(pool, &self.wake, async move {
            blocking.await.unwrap_or_default()
        })
    }
}

/// The agent at the top of `agent`'s tree.
fn top(mut agent: Entity, parents: &Query<&SpawnedBy>) -> Entity {
    while let Ok(SpawnedBy(parent)) = parents.get(agent) {
        agent = *parent;
    }
    agent
}

/// Keeps the files a tool call published as created.
fn files_written(
    settled: On<Add<Settled>>,
    calls: Query<(&Published, &ForAgent), With<ToolRequest>>,
    parents: Query<&SpawnedBy>,
    mut agents: Query<&mut Strays>,
) {
    let Ok((published, &ForAgent(agent))) = calls.get(settled.entity) else {
        return;
    };
    let Ok(Some(FilesChanged(changes))) = published.0.get::<FilesChanged>() else {
        return;
    };
    if let Ok(mut strays) = agents.get_mut(top(agent, &parents)) {
        let created = changes
            .into_iter()
            .filter(|change| change.kind == ChangeKind::Created);
        strays
            .created
            .extend(created.map(|change| PathBuf::from(change.path)));
    }
}

/// Holds a top-level agent's turn as it would end, while the files it
/// created are measured.
fn end_of_turn(
    turns: Query<(Entity, &TurnOf), Added<TurnOutcome>>,
    agents: Query<&Strays, Without<SpawnedBy>>,
    work: Work,
    mut commands: Commands,
) {
    for (turn, &TurnOf(agent)) in &turns {
        let Ok(strays) = agents.get(agent) else {
            continue;
        };
        if strays.created.is_empty() {
            continue;
        }
        let files: Vec<PathBuf> = strays.created.iter().cloned().collect();
        let measure = files.clone();
        let running = work.run(move |workspace, root| Existing(existing(workspace, root, measure)));
        let name = Name::new("listing the files created this turn");
        commands.spawn((name, HoldOf(turn), TurnEnd { agent, files }, running));
    }
}

/// Offers the files of a turn that still exist, or prints them where
/// nobody answers, and lets the turn end.
fn measured(
    done: On<Insert<Done<Existing>>>,
    ends: Query<(&TurnEnd, &Done<Existing>)>,
    mut agents: Query<&mut Strays>,
    mut sections: Query<&mut PrintSection, With<StraySection>>,
    answerers: Answerers,
    settings: Res<StraySettings>,
    mut commands: Commands,
) {
    let Ok((end, Done(Existing(files)))) = ends.get(done.entity) else {
        return;
    };
    let agent = end.agent;
    if let Ok(mut strays) = agents.get_mut(agent) {
        strays.created.retain(|file| !end.files.contains(file));
    }
    commands.entity(done.entity).despawn();
    if files.is_empty() {
        return;
    }
    if answerers.can_answer(Some(agent)) {
        let mut picker = Picker::new(TITLE)
            .for_agent(agent)
            .body("Tick the files to act on.")
            .multiple();
        for file in files {
            picker.items.push(PickItem {
                label: file.path.clone(),
                command: String::new(),
                detail: Some(size(file.size)),
                more: None,
                checked: looks_scratch(&settings.scratch, &file.path),
            });
        }
        let files = files.clone();
        commands.spawn((picker, StrayPicker { agent, files }));
        return;
    }
    let lines: Vec<String> = files
        .iter()
        .map(|file| format!("  {} ({})", file.path, size(file.size)))
        .collect();
    let text = format!("{TITLE}:\n{}", lines.join("\n"));
    for mut section in &mut sections {
        section.text.clone_from(&text);
    }
}

/// Asks what to do with the files ticked.
fn ticked(picked: On<Picked>, offers: Query<&StrayPicker>, mut commands: Commands) {
    let Ok(offer) = offers.get(picked.entity) else {
        return;
    };
    // Closed: keep all.
    let Some(Chosen {
        items: positions, ..
    }) = &picked.chosen
    else {
        return;
    };
    let files: Vec<Stray> = (positions.iter())
        .filter_map(|at| offer.files.get(*at).cloned())
        .collect();
    if files.is_empty() {
        return;
    }
    let names: Vec<&str> = files.iter().map(|file| file.path.as_str()).collect();
    let picker = Picker::new(TITLE)
        .for_agent(offer.agent)
        .body(format!("Selected: {}", names.join(", ")))
        .item("Delete selected", "")
        .item("Add selected to .gitignore", "")
        .item("Keep all", "");
    let agent = offer.agent;
    commands.spawn((picker, StrayActions { agent, files }));
}

/// Deletes or ignores the files ticked, or keeps them.
fn act(picked: On<Picked>, actions: Query<&StrayActions>, work: Work, mut commands: Commands) {
    let Ok(StrayActions { agent, files }) = actions.get(picked.entity) else {
        return;
    };
    let (job, files) = (Job { agent: *agent }, files.clone());
    match picked
        .chosen
        .as_ref()
        .and_then(|chosen| chosen.items.first())
    {
        Some(0) => {
            let running = work.run(move |workspace, root| trash(workspace, root, &files));
            commands.spawn((job, running));
        }
        Some(1) => {
            let running = work.run(move |workspace, root| ignore(workspace, root, &files));
            commands.spawn((job, running));
        }
        // Keep all, or closed.
        _ => {}
    }
}

/// Says what a job did, and keeps what it deleted for the agent's next
/// message.
fn reported(
    done: On<Insert<Done<Report>>>,
    jobs: Query<(&Job, &Done<Report>)>,
    mut agents: Query<&mut Strays>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok((&Job { agent }, Done(report))) = jobs.get(done.entity) else {
        return;
    };
    for line in &report.lines {
        notices.write(Notice::info(agent, line.clone()));
    }
    if let Ok(mut strays) = agents.get_mut(agent) {
        strays.deleted.extend(report.deleted.iter().cloned());
        strays
            .deleted
            .retain(|file| !report.restored.contains(file));
    }
    commands.entity(done.entity).despawn();
}

/// Adds a note naming the files the user deleted to the next message the
/// user writes to the agent.
fn tell_deleted(
    mut messages: Query<(&mut Incoming, &ForAgent), (Added<Admitted>, IncomingMessages)>,
    mut agents: Query<&mut Strays>,
) {
    for (mut message, &ForAgent(agent)) in &mut messages {
        if message.author.kind != AuthorKind::User || message.command().is_some() {
            continue;
        }
        let Ok(mut strays) = agents.get_mut(agent) else {
            continue;
        };
        if strays.deleted.is_empty() {
            continue;
        }
        let names = std::mem::take(&mut strays.deleted).join(", ");
        let note = format!(
            "The user deleted these files you created, moving them to {TRASH}: {names}. \
             They are no longer in the project; do not refer to them."
        );
        let attachment = Attachment {
            label: "deleted stray files".to_owned(),
            content: UserContent::text(note),
        };
        message.parts.insert(0, MessagePart::Attachment(attachment));
    }
}

/// `/restore-strays [batch]`: lists the recent batches, or restores one.
fn restore(In(args): In<CommandArgs>, work: Work, mut commands: Commands) {
    let id = args.args.trim().to_owned();
    if id.is_empty() {
        let read = work.run(|workspace, root| Batches(read_index(workspace, root)));
        commands.spawn((BatchList { agent: args.agent }, read));
    } else {
        let job = Job { agent: args.agent };
        commands.spawn((
            job,
            work.run(move |workspace, root| put_back(workspace, root, &id)),
        ));
    }
}

/// Offers the recent batches to restore, newest first, or lists them where
/// nobody answers.
fn batches_listed(
    done: On<Insert<Done<Batches>>>,
    lists: Query<(&BatchList, &Done<Batches>)>,
    answerers: Answerers,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok((&BatchList { agent }, Done(Batches(batches)))) = lists.get(done.entity) else {
        return;
    };
    commands.entity(done.entity).despawn();
    let recent: Vec<&Batch> = batches.iter().rev().take(RECENT).collect();
    if recent.is_empty() {
        notices.write(Notice::info(agent, "No deleted stray files to restore."));
        return;
    }
    if !answerers.can_answer(Some(agent)) {
        for batch in recent {
            let line = format!("/{RESTORE} {}: {}", batch.id, batch.files.join(", "));
            notices.write(Notice::info(agent, line));
        }
        return;
    }
    let mut picker = Picker::new(RESTORE_TITLE).for_agent(agent);
    for batch in recent {
        picker.items.push(PickItem {
            label: batch.files.join(", "),
            command: format!("{RESTORE} {}", batch.id),
            detail: Some(format!("batch {}", batch.id)),
            more: None,
            checked: false,
        });
    }
    commands.spawn(picker);
}

/// `path` under the project `root`, as the user knows it.
fn under(workspace: &dyn Workspace, root: &Path, path: &Path) -> Option<PathBuf> {
    path.strip_prefix(workspace.canonical(root))
        .ok()
        .map(Path::to_path_buf)
}

/// Each of `files` that is still a regular file, with its size.
fn existing(workspace: &dyn Workspace, root: &Path, files: Vec<PathBuf>) -> Vec<Stray> {
    let shown = |path: &Path| {
        under(workspace, root, path)
            .unwrap_or_else(|| path.to_path_buf())
            .display()
            .to_string()
    };
    files
        .into_iter()
        .filter_map(|absolute| match workspace.stat(&absolute) {
            Ok(Stat::File { size, .. }) => Some(Stray {
                path: shown(&absolute),
                absolute,
                size,
            }),
            _ => None,
        })
        .collect()
}

/// Moves `files` into a new batch of the trash, keeping their paths.
fn trash(workspace: &dyn Workspace, root: &Path, files: &[Stray]) -> Report {
    let id = now_ms().to_string();
    let batch = root.join(TRASH).join(&id);
    let mut report = Report::default();
    // The trash is never one of the project's untracked files.
    workspace
        .write(&root.join(TRASH).join(".gitignore"), b"*\n")
        .ok();
    for file in files {
        let Some(path) = under(workspace, root, &file.absolute) else {
            report
                .lines
                .push(format!("{} is outside the project; kept.", file.path));
            continue;
        };
        match workspace.rename(&file.absolute, &batch.join(&path)) {
            Ok(()) => report.deleted.push(path.display().to_string()),
            Err(error) => report.lines.push(format!(
                "Could not move {} to the trash: {error}",
                file.path
            )),
        }
    }
    if report.deleted.is_empty() {
        return report;
    }
    let mut batches = read_index(workspace, root);
    batches.push(Batch {
        id: id.clone(),
        files: report.deleted.clone(),
    });
    if let Err(error) = write_index(workspace, root, &batches) {
        report
            .lines
            .push(format!("Could not list batch {id}: {error}"));
    }
    report.lines.push(format!(
        "Moved {} to {TRASH}/{id}; /{RESTORE} {id} brings them back.",
        report.deleted.join(", ")
    ));
    report
}

/// Adds `files` to the project's `.gitignore`.
fn ignore(workspace: &dyn Workspace, root: &Path, files: &[Stray]) -> Report {
    let path = root.join(".gitignore");
    let mut text = workspace.read_text(&path).unwrap_or_default();
    let mut added = Vec::new();
    for file in files {
        let Some(relative) = under(workspace, root, &file.absolute) else {
            continue;
        };
        let line = format!("/{}", relative.display());
        if !text.lines().any(|existing| existing == line) {
            if !text.is_empty() && !text.ends_with('\n') {
                text.push('\n');
            }
            text.push_str(&line);
            text.push('\n');
            added.push(file.path.clone());
        }
    }
    let line = match workspace.write(&path, text.as_bytes()) {
        Ok(()) => format!("Added {} to .gitignore.", added.join(", ")),
        Err(error) => format!("Could not write .gitignore: {error}"),
    };
    Report {
        lines: vec![line],
        ..Report::default()
    }
}

/// Moves the files of batch `id` back where they were, unless something
/// is there now.
fn put_back(workspace: &dyn Workspace, root: &Path, id: &str) -> Report {
    let mut report = Report::default();
    let mut batches = read_index(workspace, root);
    let Some(batch) = batches.iter_mut().find(|batch| batch.id == id) else {
        report.lines.push(format!("No deleted batch {id}."));
        return report;
    };
    let from = root.join(TRASH).join(id);
    let mut kept = Vec::new();
    for file in std::mem::take(&mut batch.files) {
        let to = root.join(&file);
        if !matches!(workspace.stat(&to), Ok(Stat::Missing)) {
            report
                .lines
                .push(format!("{file} exists again; its copy stays in the trash."));
            kept.push(file);
            continue;
        }
        match workspace.rename(&from.join(&file), &to) {
            Ok(()) => report.restored.push(file),
            Err(error) => {
                report
                    .lines
                    .push(format!("Could not restore {file}: {error}"));
                kept.push(file);
            }
        }
    }
    batch.files = kept;
    batches.retain(|batch| !batch.files.is_empty());
    if let Err(error) = write_index(workspace, root, &batches) {
        report
            .lines
            .push(format!("Could not update the trash: {error}"));
    }
    if !report.restored.is_empty() {
        report
            .lines
            .push(format!("Restored {}.", report.restored.join(", ")));
    }
    report
}

/// The trash's batches, oldest first.
fn read_index(workspace: &dyn Workspace, root: &Path) -> Vec<Batch> {
    let text = workspace.read_text(&root.join(INDEX)).unwrap_or_default();
    serde_json::from_str(&text).unwrap_or_default()
}

/// Writes the trash's batches.
fn write_index(workspace: &dyn Workspace, root: &Path, batches: &[Batch]) -> std::io::Result<()> {
    let text = serde_json::to_string_pretty(batches).map_err(std::io::Error::other)?;
    workspace.write(&root.join(INDEX), text.as_bytes())
}

/// Whether the file's name matches one of the `scratch` globs.
fn looks_scratch(scratch: &[String], path: &str) -> bool {
    let name = Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    scratch
        .iter()
        .any(|glob| glob::Pattern::new(glob).is_ok_and(|pattern| pattern.matches(&name)))
}

/// `bytes` for people.
fn size(bytes: u64) -> String {
    match bytes {
        0..1024 => format!("{bytes} B"),
        1024..1_048_576 => format!("{} KB", bytes / 1024),
        _ => format!("{} MB", bytes / 1_048_576),
    }
}

#[cfg(test)]
mod tests;
