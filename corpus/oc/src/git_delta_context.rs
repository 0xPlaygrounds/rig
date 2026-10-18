//! OC-03: what changed in the git repository since the agent's last turn,
//! told to the model with the user's next message.
//!
//! As the app starts, and as each turn of an agent ends, the plugin takes a
//! snapshot of the agent's directory (its [`WorkDir`], else the
//! [`ProjectRoot`]): rig-tools' checkpoint [`snapshot`], a git tree written
//! with a private index, beside the branch and `HEAD`. When a turn the
//! user's own message started sends its first request, a new snapshot is
//! compared with the last one: a branch switch or a moved `HEAD` and the
//! files changed since, with their line counts, become a note. Changes the
//! agent's own `edit`, `write` and `shell` calls made are in the snapshot
//! its turn ended with, so they are never reported.
//!
//! The note goes into every request of that turn, as a message before the
//! user's, in the request's `history` view, which is never saved; the
//! conversation keeps it as a transcript [`Note`] (an aside, which no model
//! reads), so a resumed session and an export show it, a restarted turn
//! sends it again, and later turns never repeat it. Nothing is added when
//! nothing changed, or outside a git work tree on this machine.
//! `/gitdelta off|on` turns it off and on for the session.

use std::path::{Path, PathBuf};

use bevy_ecs::system::SystemParam;
use bevy_log::warn;
use rig_basics::checkpoint::CheckpointAlso;
use rig_harness::prelude::*;
use rig_harness::rig_core::message::Message;
use rig_harness::rig_core::tool::{ChangeKind, FileChange};
use rig_harness::rig_tools::checkpoint::{Snapshot, Taking, snapshot};

#[cfg(test)]
mod tests;

/// The plugin's name, as the author of what it adds.
pub const NAME: &str = "git-delta";

/// The most commit subjects a note lists.
const MAX_COMMITS: usize = 5;

/// The most files a note lists.
const MAX_FILES: usize = 20;

/// The title of the transcript note that keeps what the model was told.
pub const NOTE_TITLE: &str = "Repository changes since the last turn";

/// Adds what changed in the repository since an agent's last turn to the
/// request of the user's next turn; `/gitdelta off|on`.
#[derive(Default)]
pub struct GitDeltaContextPlugin;

/// The plugin's stage system, which edits each turn's request in
/// `PrepareSystems::Edit`, for others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GitDeltaSystems;

/// Whether `/gitdelta off` turned the plugin off for the session; saved
/// with it.
#[derive(Resource, Reflect, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[reflect(Resource, Saved, Default)]
pub struct GitDeltaOff(pub bool);

/// The repository as last seen: a snapshot of a directory's files, its
/// branch and `HEAD`. Only for this run: a snapshot does not outlive it.
#[derive(Component, Clone, Debug)]
pub struct SeenRepository {
    /// The directory the snapshot is of.
    pub root: PathBuf,
    /// Its files.
    pub files: Snapshot,
    /// The branch, or `HEAD` when detached.
    pub branch: String,
    /// The commit `HEAD` names; `None` before the first commit.
    pub head: Option<String>,
}

/// The repository of the [`ProjectRoot`] as the app started, the
/// baseline of an agent's first turn.
#[derive(Resource, Default)]
struct AtStart(Option<SeenRepository>);

/// On a turn whose request was checked: its first request was, so later
/// ones only repeat the note it saved.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Checked;

impl Plugin for GitDeltaContextPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Processes>()
            .init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .init_resource::<GitDeltaOff>()
            .init_resource::<AtStart>()
            .configure_sets(AgentPass, GitDeltaSystems.in_set(PrepareSystems::Edit))
            .add_systems(AgentPass, annotate.in_set(GitDeltaSystems))
            .add_systems(Startup, at_start)
            .add_observer(remember)
            .add_command(
                "gitdelta",
                "Tell the model what changed in the repository between turns: on or off",
                toggle,
            );
    }
}

/// What a snapshot of the repository needs.
#[derive(SystemParam)]
struct Repository<'w, 's> {
    processes: Res<'w, Processes>,
    workspace: Res<'w, ProjectWorkspace>,
    root: Res<'w, ProjectRoot>,
    scratch: Option<Res<'w, Scratch>>,
    also: Option<Res<'w, CheckpointAlso>>,
    dirs: Query<'w, 's, &'static WorkDir>,
}

impl Repository<'_, '_> {
    /// The directory `agent` works in.
    fn root_of(&self, agent: Option<Entity>) -> PathBuf {
        agent
            .and_then(|agent| self.dirs.get(agent).ok())
            .map_or_else(|| self.root.0.clone(), |dir| dir.0.clone())
    }

    /// The repository at `root` as it is now, from `earlier` (so only
    /// what changed is hashed again); `None` outside a git work tree on
    /// this machine. Taken inline: the workspace is this machine's, and
    /// git keeps the cost to what changed.
    fn see(&self, root: &Path, earlier: Option<&SeenRepository>) -> Option<SeenRepository> {
        if self.workspace.0.host().is_some() {
            return None;
        }
        git(
            &self.processes,
            root,
            &["rev-parse", "--is-inside-work-tree"],
        )?;
        let head = git(
            &self.processes,
            root,
            &["rev-parse", "-q", "--verify", "HEAD"],
        );
        let branch = git(
            &self.processes,
            root,
            &["symbolic-ref", "-q", "--short", "HEAD"],
        );
        // A directory of its own, so no checkpoint of the harness's shares
        // the private index.
        let keep = self.scratch.as_ref().and_then(|scratch| scratch.0.clone());
        let taking = Taking {
            keep: keep.map(|keep| keep.join(NAME)),
            also: self
                .also
                .as_ref()
                .map(|also| also.0.clone())
                .unwrap_or_default(),
            launcher: self.processes.launcher(),
        };
        let earlier = earlier.filter(|earlier| earlier.root == root);
        let files = snapshot(
            &*self.workspace.0,
            root,
            &taking,
            earlier.map(|it| &it.files),
        );
        match files {
            Ok(files) => Some(SeenRepository {
                root: root.to_path_buf(),
                files,
                branch: branch.unwrap_or_else(|| "HEAD".to_owned()),
                head,
            }),
            Err(error) => {
                warn!("{NAME}: no snapshot of {}: {error}", root.display());
                None
            }
        }
    }
}

/// The project's repository as the app starts.
fn at_start(repository: Repository, mut start: ResMut<AtStart>) {
    let root = repository.root.0.clone();
    start.0 = repository.see(&root, None);
}

/// Each turn that ends leaves its agent's repository as the baseline of
/// its next turn, with what the turn's own calls changed in it.
fn remember(
    settled: On<Add<Settled>>,
    turns: Query<&ForAgent, With<Turn>>,
    seen: Query<&SeenRepository>,
    repository: Repository,
    off: Res<GitDeltaOff>,
    mut commands: Commands,
) {
    let Ok(&ForAgent(agent)) = turns.get(settled.entity) else {
        return;
    };
    if off.0 {
        return;
    }
    let root = repository.root_of(Some(agent));
    if let Some(now) = repository.see(&root, seen.get(agent).ok()) {
        commands.entity(agent).try_insert(now);
    }
}

/// Puts the note of what changed since the agent's last turn into each
/// request of a turn the user's message started: worked out, and kept in
/// the conversation, as its first request is built.
fn annotate(
    mut calls: Query<(&mut ModelRequest, &CallOf, &ForAgent), (Added<Admitted>, With<TurnCall>)>,
    turns: Query<(&Turn, Has<Checked>, Has<Restored>)>,
    mut agents: Query<(&mut Conversation, Option<&SeenRepository>)>,
    (repository, start, off): (Repository, Res<AtStart>, Res<GitDeltaOff>),
    mut commands: Commands,
) {
    for (mut call, &CallOf(turn), &ForAgent(agent)) in &mut calls {
        let Ok((turn_data, checked, restored)) = turns.get(turn) else {
            continue;
        };
        let Ok((mut conversation, seen)) = agents.get_mut(agent) else {
            continue;
        };
        let first = turn_data.start;
        let users = conversation
            .request(first)
            .is_some_and(|request| request.is_users_own());
        if !checked {
            commands.entity(turn).try_insert(Checked);
            if users && !restored && !off.0 {
                let root = repository.root_of(Some(agent));
                let was = seen.or(start.0.as_ref().filter(|it| it.root == root));
                let now = repository.see(&root, was);
                if let (Some(was), Some(now)) = (was, &now)
                    && let Some(text) = describe(&repository.processes, was, now)
                {
                    let at = conversation.start_of(first);
                    let after = at.checked_sub(1).and_then(|at| conversation.id(at));
                    let note = Note::new(NOTE_TITLE, text);
                    conversation.add_aside(note.aside(Author::plugin(NAME), after));
                }
                if let Some(now) = now {
                    commands.entity(agent).try_insert(now);
                }
            }
        }
        let Some(text) = saved_note(&conversation, first) else {
            continue;
        };
        if let Some(history) = &mut call.history {
            let at = history.start_of(first);
            let by = Author::plugin(NAME);
            let entry = Entry::new(Message::user(text)).by(by.clone());
            history.splice(at..at, [entry], by);
        }
    }
}

/// The text of the note the plugin kept for the turn that begins at
/// `first`: the newest just before it.
fn saved_note(conversation: &Conversation, first: MessageId) -> Option<String> {
    let at = conversation.start_of(first);
    let after = at.checked_sub(1).and_then(|at| conversation.id(at));
    let author = Author::plugin(NAME);
    let mut notes = conversation.asides().filter_map(|(_, aside)| {
        let mine = aside.author == author && aside.after == after;
        mine.then(|| Note::of(aside)).flatten()
    });
    notes.next_back().map(|note| note.text)
}

/// What changed from `was` to `now`, for the model; `None` when nothing
/// did.
fn describe(processes: &Processes, was: &SeenRepository, now: &SeenRepository) -> Option<String> {
    let mut lines = Vec::new();
    if was.branch != now.branch || was.head != now.head {
        let range = match (&was.head, &now.head) {
            (Some(was), Some(now)) => format!("{was}..{now}"),
            (None, Some(now)) => now.clone(),
            (_, None) => String::new(),
        };
        let max = format!("-n{MAX_COMMITS}");
        let subjects = if range.is_empty() {
            None
        } else {
            git(processes, &now.root, &["log", "--format=%s", &max, &range])
        };
        let subjects: Vec<String> = subjects
            .iter()
            .flat_map(|it| it.lines())
            .map(str::to_owned)
            .collect();
        let commits = match subjects.is_empty() {
            true => "none".to_owned(),
            false => subjects.join("; "),
        };
        lines.push(match was.branch == now.branch {
            true => format!("Branch `{}` moved; new commits: {commits}", now.branch),
            false => format!(
                "Branch is now `{}` (was `{}`); new commits: {commits}",
                now.branch, was.branch
            ),
        });
    }
    let changes = match now.files.changes_since(&was.files) {
        Ok(changes) => changes,
        Err(error) => {
            warn!("{NAME}: cannot compare snapshots: {error}");
            Vec::new()
        }
    };
    if !changes.is_empty() {
        lines.push("Files changed outside your tool calls:".to_owned());
        lines.extend(
            changes
                .iter()
                .take(MAX_FILES)
                .map(|change| file_line(&now.root, change)),
        );
        if let Some(more) = changes.len().checked_sub(MAX_FILES).filter(|n| *n > 0) {
            lines.push(format!("- … and {more} more"));
        }
    }
    if lines.is_empty() {
        return None;
    }
    lines.insert(
        0,
        "The repository changed since your last turn (not by you):".to_owned(),
    );
    Some(lines.join("\n"))
}

/// One changed file, relative to `root`, with its line counts.
fn file_line(root: &Path, change: &FileChange) -> String {
    let path = Path::new(&change.path);
    let shown = path.strip_prefix(root).unwrap_or(path).display();
    let kind = match change.kind {
        ChangeKind::Created => " (new)",
        ChangeKind::Modified => "",
        ChangeKind::Deleted => " (deleted)",
    };
    format!("- {shown} +{} −{}{kind}", change.added, change.removed)
}

/// `/gitdelta off|on`: turning it on takes the repository as it is now as
/// the agent's baseline, so what changed while it was off is not told.
fn toggle(
    In(args): In<CommandArgs>,
    mut off: ResMut<GitDeltaOff>,
    seen: Query<&SeenRepository>,
    repository: Repository,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let now_off = match args.args.as_str() {
        "off" => true,
        "on" => false,
        "" => {
            let state = if off.0 { "off" } else { "on" };
            notices.write(Notice::info(args.agent, format!("/gitdelta is {state}.")));
            return;
        }
        _ => {
            notices.write(Notice::error(args.agent, "/gitdelta takes on or off"));
            return;
        }
    };
    if off.0 && !now_off {
        let root = repository.root_of(Some(args.agent));
        if let Some(now) = repository.see(&root, seen.get(args.agent).ok()) {
            commands.entity(args.agent).try_insert(now);
        }
    }
    off.0 = now_off;
    let state = if now_off { "off" } else { "on" };
    notices.write(Notice::info(args.agent, format!("/gitdelta is {state}.")));
}

/// What git prints in `dir` for `args`, trimmed; `None` when it fails.
fn git(processes: &Processes, dir: &Path, args: &[&str]) -> Option<String> {
    let output = processes
        .launcher()
        .command("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}
