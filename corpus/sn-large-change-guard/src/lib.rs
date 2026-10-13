//! SN-02: the large-change guard. Every call that writes named paths is
//! previewed first ([`Preview`]): the harness runs its tool on an overlay
//! of the workspace and puts the files it would change on the call. The
//! guard then reads those files as they are now, off the main thread, for
//! the share of each file replaced, the size it would have, and for a
//! delete whether git tracks it. Under every threshold the call runs at
//! once. Over one, the user is asked, naming the agent (or subagent) and
//! the damage, with the diff: allow, allow this file for the session
//! (saved with it), deny, or deny with a reason, which the model reads in
//! the call's refusal. A run nobody can answer follows `on_headless`.
//! Thresholds are read from `RIG_HOME/large-change.toml`, then the
//! project's `.rig/large-change.toml`, key by key.

use std::ffi::OsString;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::prelude::*;
use rig_harness::rig_core::tool::{ChangeKind, FileChange};
use rig_tools::MAX_FILE_BYTES;
use rig_tools::fs::file_change;
use rig_tools::process::{Launcher, Spawn};
use rig_tools::workspace::{Stat, Workspace};
use serde::Deserialize;
use serde::de::DeserializeOwned;

/// The config file's name, under `RIG_HOME` and under the project's `.rig`.
pub const CONFIG: &str = "large-change.toml";

/// The refusal a run nobody can answer gives under `on_headless = "deny"`.
pub const HEADLESS: &str = "large change needs approval; not available non-interactively";

/// Who decides in the effect log.
const NAME: &str = "large-change-guard";

/// What a run nobody can answer does with a large change.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OnHeadless {
    /// Refuses the call with [`HEADLESS`].
    #[default]
    Deny,
    /// Lets it run.
    Allow,
}

/// `large-change.toml`: when a change is large enough to ask about.
#[derive(Resource, Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct LargeChangeConfig {
    /// Asks when more lines than this are removed.
    pub removed_lines: usize,
    /// Asks when more than this percentage of an existing file's lines is
    /// removed.
    pub replaced_percent: usize,
    /// Asks when the file would be larger than this many KB.
    pub result_kb: u64,
    /// Asks before deleting a file git tracks.
    pub delete_tracked: bool,
    /// What a run nobody can answer does.
    pub on_headless: OnHeadless,
}

impl Default for LargeChangeConfig {
    /// 80 lines, 60 %, 256 KB, tracked deletes, refused headless.
    fn default() -> Self {
        Self {
            removed_lines: 80,
            replaced_percent: 60,
            result_kb: 256,
            delete_tracked: true,
            on_headless: OnHeadless::Deny,
        }
    }
}

/// The files the user allowed every large change to for the rest of the
/// session, by resolved path. Saved with the session, so `/resume` and
/// `/reload` keep them.
#[derive(Resource, Reflect, Clone, Debug, Default)]
#[reflect(Resource, Saved)]
pub struct AllowedFiles {
    /// The resolved paths.
    pub files: Vec<String>,
}

impl AllowedFiles {
    fn has(&self, path: &str) -> bool {
        self.files.iter().any(|file| file == path)
    }
}

/// The guard's Gate systems, which preview the calls that write named
/// paths and hold them while their files are measured, for others to
/// order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LargeChangeSystems;

/// Asks before a large file change.
#[derive(Default)]
pub struct LargeChangeGuardPlugin;

impl Plugin for LargeChangeGuardPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<AllowedFiles>()
            .register_type::<AllowedFiles>()
            .init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .init_resource::<Processes>()
            .configure_sets(AgentPass, LargeChangeSystems.in_set(Stage::Gate))
            .add_systems(
                AgentPass,
                (preview_writes, measure_previewed).in_set(LargeChangeSystems),
            )
            .add_observer(measured)
            .add_observer(answered);
    }

    /// Reads the config once every plugin set the project root and its
    /// workspace, unless the app has one already, such as a test's.
    fn finish(&self, app: &mut App) {
        let world = app.world();
        if world.contains_resource::<LargeChangeConfig>() {
            return;
        }
        let home = RigHome::of(world).root().join(CONFIG);
        let project = world.resource::<ProjectRoot>().0.join(".rig").join(CONFIG);
        let workspace = world.resource::<ProjectWorkspace>().0.clone();
        let texts = [
            std::fs::read(&home).ok().map(|bytes| (home, bytes)),
            bytes_of(&*workspace, &project).map(|bytes| (project, bytes)),
        ];
        let (config, problems) = read_config::<LargeChangeConfig>(texts.into_iter().flatten());
        app.insert_resource(config);
        for problem in problems {
            app.add_systems(Startup, move |mut notices: MessageWriter<Notice>| {
                notices.write(Notice::warn(None, problem.clone()));
            });
        }
    }
}

/// `T` from the files' tables, a later file's keys over an earlier one's,
/// and what did not read.
fn read_config<T: DeserializeOwned + Default>(
    files: impl IntoIterator<Item = (PathBuf, Vec<u8>)>,
) -> (T, Vec<String>) {
    let mut table = toml::Table::new();
    let mut problems = Vec::new();
    for (path, bytes) in files {
        match String::from_utf8_lossy(&bytes).parse::<toml::Table>() {
            Ok(more) => table.extend(more),
            Err(failure) => problems.push(format!("{}: {failure}", path.display())),
        }
    }
    match toml::Value::Table(table).try_into() {
        Ok(config) => (config, problems),
        Err(failure) => {
            problems.push(format!("{CONFIG}: {failure}; using the default thresholds"));
            (T::default(), problems)
        }
    }
}

/// What a call would do to one file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Change {
    /// The file, as the tool resolved it.
    pub path: String,
    /// Lines removed.
    pub removed: usize,
    /// Lines added.
    pub added: usize,
    /// The file's lines now; `None` when there is no file yet.
    pub old_lines: Option<usize>,
    /// The size it would have, in bytes.
    pub size: u64,
    /// Whether it would be deleted.
    pub deleted: bool,
    /// Whether git tracks it (looked up for a delete only).
    pub tracked: bool,
    /// The change as a unified diff.
    pub diff: String,
}

impl Change {
    /// Whether it crosses one of `config`'s thresholds.
    pub fn is_large(&self, config: &LargeChangeConfig) -> bool {
        let replaced = self
            .old_lines
            .is_some_and(|old| old > 0 && self.removed * 100 > config.replaced_percent * old);
        (self.deleted && self.tracked && config.delete_tracked)
            || self.removed > config.removed_lines
            || replaced
            || self.size > config.result_kb * 1024
    }

    /// The damage in a line, such as
    /// `src/app.rs: −190 / +10 lines (95% of file replaced), 1 KB`.
    pub fn line(&self, root: &Path) -> String {
        let path = Path::new(&self.path);
        let path = path.strip_prefix(root).unwrap_or(path).display();
        let mut line = match self.old_lines {
            None => format!("{path}: new file, +{} lines", self.added),
            Some(old) => {
                let percent = (self.removed * 100).checked_div(old).unwrap_or(0);
                format!(
                    "{path}: −{} / +{} lines ({percent}% of file replaced)",
                    self.removed, self.added
                )
            }
        };
        if self.deleted {
            line.push_str(if self.tracked {
                ", deletes a tracked file"
            } else {
                ", deletes the file"
            });
        } else {
            let _ = write!(line, ", {} KB", self.size.div_ceil(1024));
        }
        line
    }
}

/// The bytes of the file `path` in `workspace`; `None` when there is no
/// file.
fn bytes_of(workspace: &dyn Workspace, path: &Path) -> Option<Vec<u8>> {
    match workspace.stat(path).ok()? {
        Stat::File { .. } => Some(workspace.read(path, MAX_FILE_BYTES).ok()?.bytes),
        Stat::Dir | Stat::Other | Stat::Missing => None,
    }
}

/// Whether git tracks `path`, asked in its directory.
fn tracked(workspace: &dyn Workspace, launcher: &Launcher, path: &Path) -> bool {
    let args: Vec<OsString> = vec!["ls-files".into(), "--error-unmatch".into(), path.into()];
    workspace
        .spawn(
            launcher,
            path.parent(),
            "git".as_ref(),
            &args,
            Spawn::default(),
        )
        .and_then(|process| process.output(Some(Duration::from_secs(10))))
        .is_ok_and(|output| output.status.success())
}

/// The bytes `diff`'s hunks add and remove, each line with its newline.
/// The preview gives the lines a change adds, not the size it leaves, so
/// the size is the file's now, grown by this.
fn bytes_changed(diff: &str) -> (u64, u64) {
    let hunks = diff.lines().skip_while(|line| !line.starts_with("@@"));
    hunks.fold((0, 0), |(added, removed), line| {
        let len = line.len() as u64;
        match line.as_bytes().first() {
            Some(b'+') => (added + len, removed),
            Some(b'-') => (added, removed + len),
            _ => (added, removed),
        }
    })
}

/// What `change` does to its file as it is now in `workspace`.
fn measure(workspace: &dyn Workspace, launcher: &Launcher, change: FileChange) -> Change {
    let path = PathBuf::from(&change.path);
    let before = bytes_of(workspace, &path);
    let text = before.as_deref().map(String::from_utf8_lossy);
    let deleted = change.kind == ChangeKind::Deleted;
    // A deletion's preview carries no diff: the lines it loses are the file.
    let diff = match change.diff {
        Some(diff) => diff,
        None if deleted => file_change(&change.path, text.as_deref(), "")
            .diff
            .unwrap_or_default(),
        None => String::new(),
    };
    let size = if deleted {
        0
    } else {
        let (added, removed) = bytes_changed(&diff);
        let now = before.as_ref().map_or(0, |bytes| bytes.len() as u64);
        (now + added).saturating_sub(removed)
    };
    Change {
        removed: change.removed,
        added: change.added,
        old_lines: text.map(|text| text.lines().count()),
        size,
        deleted,
        tracked: deleted && tracked(workspace, launcher, &path),
        diff,
        path: change.path,
    }
}

/// The changes the user may be asked about: all but those to files
/// allowed for the session, and a deletion whose content another change
/// of the call keeps, as a move's source.
fn at_stake(changes: &[FileChange], allowed: &AllowedFiles) -> Vec<FileChange> {
    let moved = |change: &FileChange| {
        change.kind == ChangeKind::Deleted
            && change.before.is_some()
            && changes.iter().any(|other| other.after == change.before)
    };
    changes
        .iter()
        .filter(|change| !allowed.has(&change.path) && !moved(change))
        .cloned()
        .collect()
}

/// On a previewed call measured already.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Checked;

/// The hold of a call while its files are measured.
#[derive(Component)]
struct Measuring {
    call: Entity,
    agent: Entity,
}

/// What the measuring found, on the hold.
#[derive(Component)]
struct Measured(Vec<Change>);

impl From<Vec<Change>> for Measured {
    fn from(changes: Vec<Change>) -> Self {
        Self(changes)
    }
}

/// On the guard's question: the call it holds, its large files and the
/// damage.
#[derive(Component, Clone)]
struct GuardAsk {
    call: Entity,
    paths: Vec<String>,
    line: String,
}

/// Previews each call about to run that writes named paths, unless every
/// one is allowed for the session.
fn preview_writes(
    calls: Query<(Entity, &Footprint, &Decision), (Undecided, ToolCalls, Without<Preview>)>,
    allowed: Res<AllowedFiles>,
    mut commands: Commands,
) {
    for (call, Footprint(touches), decision) in &calls {
        let Touches::Writes(Scope::Paths(paths)) = touches else {
            continue;
        };
        let all_allowed = paths
            .iter()
            .all(|path| allowed.has(&path.to_string_lossy()));
        if !decision.is_decided() && !all_allowed {
            commands.entity(call).insert(Preview);
        }
    }
}

/// Holds each previewed call while the files it would change are measured
/// as they are now. A call that could not be previewed, such as one the
/// tool would refuse, runs as it would have.
fn measure_previewed(
    calls: Query<
        (Entity, &Previewed, &ForAgent, &Decision),
        (Undecided, ToolCalls, Without<Checked>),
    >,
    allowed: Res<AllowedFiles>,
    dirs: Query<&WorkDir>,
    root: Res<ProjectRoot>,
    (workspace, processes): (Res<ProjectWorkspace>, Res<Processes>),
    wake: Res<Wake>,
    mut commands: Commands,
) {
    for (call, Previewed(previewed), &ForAgent(agent), decision) in &calls {
        commands.entity(call).insert(Checked);
        let Ok(changes) = previewed else {
            continue;
        };
        let changes = at_stake(changes, &allowed);
        if decision.is_decided() || changes.is_empty() {
            continue;
        }
        let dir = dirs.get(agent).map_or(&root.0, |dir| &dir.0);
        let launcher = processes.launcher().in_dir(dir);
        let workspace: Arc<dyn Workspace> = workspace.0.clone();
        let work = async move {
            let measured = move || {
                let measure = |change| measure(&*workspace, &launcher, change);
                Ok(changes.into_iter().map(measure).collect::<Vec<_>>())
            };
            blocking(measured).await.unwrap_or_default()
        };
        let pool = IoTaskPool::get_or_init(TaskPool::default);
        commands.spawn((
            Name::new("measuring a file change"),
            Measuring { call, agent },
            HoldOf(call),
            Running::spawn_into::<Measured, _>(pool, &wake, work),
        ));
    }
}

/// Lets a measured call run when its changes are small or their files
/// allowed; otherwise asks, or follows `on_headless` when nobody can
/// answer.
#[allow(clippy::too_many_arguments)]
fn measured(
    added: On<Add<Measured>>,
    holds: Query<(&Measuring, &Measured)>,
    agents: Query<(Option<&Name>, Has<SpawnedBy>)>,
    allowed: Res<AllowedFiles>,
    config: Res<LargeChangeConfig>,
    root: Res<ProjectRoot>,
    answerers: Answerers,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let hold = added.entity;
    let Ok((&Measuring { call, agent }, Measured(changes))) = holds.get(hold) else {
        return;
    };
    // The hold goes whatever happens next: a question holds the call instead.
    commands.entity(hold).despawn();
    let large: Vec<&Change> = changes
        .iter()
        .filter(|change| change.is_large(&config) && !allowed.has(&change.path))
        .collect();
    if large.is_empty() {
        return;
    }
    let lines: Vec<String> = large.iter().map(|change| change.line(&root.0)).collect();
    let line = lines.join("; ");
    if !answerers.can_answer(Some(agent)) {
        if config.on_headless == OnHeadless::Deny
            && let Ok(mut decision) = calls.get_mut(call)
        {
            decision.decide(
                Decider::plugin(NAME),
                Outcome::refused(format!("{HEADLESS}: {line}")),
            );
        }
        return;
    }
    let who = match agents.get(agent) {
        Ok((Some(name), true)) => format!("subagent \"{name}\""),
        Ok((Some(name), false)) => name.to_string(),
        _ => "the agent".to_owned(),
    };
    let diffs: Vec<&str> = large.iter().map(|change| change.diff.as_str()).collect();
    let ask = Ask::new(format!("{who}: {line}"))
        .body(format!(
            "{}\n\nType a reason to deny with it.",
            diffs.join("\n")
        ))
        .choice("allow", "Allow")
        .choice("allow_file", "Allow this file for the session")
        .choice("deny", "Deny")
        .choice("deny_reason", "Deny with reason")
        .free_text();
    let paths = large.iter().map(|change| change.path.clone()).collect();
    commands.spawn((
        ask,
        ForAgent(agent),
        HoldOf(call),
        GuardAsk { call, paths, line },
    ));
}

/// Acts on the user's answer: lets the call run, allows its files for the
/// session, refuses it with the damage and the user's reason, or asks for
/// the reason.
fn answered(
    added: On<Add<Outcome>>,
    asks: Query<(&Outcome, &GuardAsk, &ForAgent)>,
    mut allowed: ResMut<AllowedFiles>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let ask = added.entity;
    let Ok((outcome, guard, &ForAgent(agent))) = asks.get(ask) else {
        return;
    };
    let refusal = match Answer::of(outcome) {
        Some(answer) if answer.is("allow") => None,
        Some(answer) if answer.is("allow_file") => {
            allowed.files.extend(guard.paths.iter().cloned());
            None
        }
        Some(answer) if answer.is("deny_reason") => {
            let why = Ask::new(format!("Why deny {}?", guard.line)).initial("");
            commands.spawn((why, ForAgent(agent), HoldOf(guard.call), guard.clone()));
            commands.entity(ask).remove::<HoldOf>();
            return;
        }
        Some(Answer::Text(reason)) if !reason.trim().is_empty() => Some(format!(
            "The user denied this large change: {}. Their reason: {}",
            guard.line,
            reason.trim()
        )),
        _ => Some(format!(
            "The user denied this large change: {}.",
            guard.line
        )),
    };
    if let (Some(why), Ok(mut decision)) = (refusal, calls.get_mut(guard.call)) {
        decision.decide(Decider::plugin(NAME), Outcome::refused(why));
    }
    commands.entity(ask).remove::<HoldOf>();
}

#[cfg(test)]
mod tests;
