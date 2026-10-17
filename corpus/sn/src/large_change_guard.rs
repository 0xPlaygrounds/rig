//! SN-02: the large-change guard. Every call that writes named paths is
//! previewed first ([`Preview`]): the harness runs its tool on an overlay
//! of the workspace and puts the files it would change on the call, each
//! with its diff and its size and lines before and after. Under every
//! threshold the call runs at once. Over one, the user is asked, naming
//! the agent (or subagent) and the damage, with the diff: allow, allow
//! this file for the session (saved with it), deny, or deny with a reason,
//! which the model reads in the call's refusal. A run nobody can answer
//! follows `on_headless`. A deletion is held while git is asked whether
//! it tracks the file. The thresholds are `[large-change]` in
//! `settings.toml`, which a project's `.rig/settings.toml` may override.

use std::ffi::OsString;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_basics::preview::{Preview, PreviewPlugin, Previewed};
use rig_harness::prelude::*;
use rig_harness::rig_core::tool::{ChangeKind, FileChange, FileState};
use rig_tools::process::{Launcher, Spawn};
use rig_tools::workspace::Workspace;
use serde::Deserialize;

/// The settings table, in `settings.toml` and a project's
/// `.rig/settings.toml`.
pub const SECTION: &str = "large-change";

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

/// `[large-change]`: when a change is large enough to ask about.
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
/// paths and judge them once previewed, for others to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LargeChangeSystems;

/// Asks before a large file change.
#[derive(Default)]
pub struct LargeChangeGuardPlugin;

impl Plugin for LargeChangeGuardPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<PreviewPlugin>() {
            app.add_plugins(PreviewPlugin);
        }
        app.add_project_setting::<LargeChangeConfig>(SECTION)
            .init_resource::<AllowedFiles>()
            .register_type::<AllowedFiles>()
            .init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .init_resource::<Processes>()
            .configure_sets(AgentPass, LargeChangeSystems.in_set(Stage::Gate))
            .add_systems(
                AgentPass,
                (preview_writes, judge_previewed).in_set(LargeChangeSystems),
            )
            .add_observer(measured)
            .add_observer(answered);
    }
}

/// What a call would do to one file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Change {
    /// The change as the preview found it.
    pub file: FileChange,
    /// Whether git tracks the file (looked up for a deletion only).
    pub tracked: bool,
}

impl Change {
    fn deleted(&self) -> bool {
        self.file.kind == ChangeKind::Deleted
    }

    /// The file's lines before the change, when it existed and was read.
    fn old_lines(&self) -> Option<usize> {
        self.file.before.as_ref().and_then(|before| before.lines)
    }

    /// Whether it crosses one of `config`'s thresholds.
    pub fn is_large(&self, config: &LargeChangeConfig) -> bool {
        let removed = self.file.removed;
        let replaced = self
            .old_lines()
            .is_some_and(|old| old > 0 && removed * 100 > config.replaced_percent * old);
        let size = self.file.after.as_ref().map_or(0, |after| after.bytes);
        (self.deleted() && self.tracked && config.delete_tracked)
            || removed > config.removed_lines
            || replaced
            || size > config.result_kb * 1024
    }

    /// The damage in a line, such as
    /// `src/app.rs: −190 / +10 lines (95% of file replaced), 1 KB`.
    pub fn line(&self, root: &Path) -> String {
        let FileChange {
            path,
            kind,
            added,
            removed,
            after,
            ..
        } = &self.file;
        let path = Path::new(path);
        let path = path.strip_prefix(root).unwrap_or(path).display();
        let mut line = match (kind, self.old_lines()) {
            (ChangeKind::Created, _) => format!("{path}: new file, +{added} lines"),
            (_, Some(old)) => {
                let percent = (removed * 100).checked_div(old).unwrap_or(0);
                format!("{path}: −{removed} / +{added} lines ({percent}% of file replaced)")
            }
            (_, None) => format!("{path}: −{removed} / +{added} lines"),
        };
        match after {
            Some(after) => {
                let _ = write!(line, ", {} KB", after.bytes.div_ceil(1024));
            }
            None if self.tracked => line.push_str(", deletes a tracked file"),
            None => line.push_str(", deletes the file"),
        }
        line
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

/// The changes the user may be asked about: all but those to files
/// allowed for the session, and a deletion whose content another change
/// of the call keeps, as a move's source.
fn at_stake(changes: &[FileChange], allowed: &AllowedFiles) -> Vec<FileChange> {
    let hash = |state: &Option<FileState>| state.as_ref().and_then(|it| it.hash.clone());
    let moved = |change: &FileChange| {
        change.kind == ChangeKind::Deleted
            && hash(&change.before).is_some()
            && changes
                .iter()
                .any(|other| hash(&other.after) == hash(&change.before))
    };
    changes
        .iter()
        .filter(|change| !allowed.has(&change.path) && !moved(change))
        .cloned()
        .collect()
}

/// On a previewed call judged already.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Checked;

/// A previewed call's changes on their way to being judged: at once, or
/// on a hold of the call once git said which deleted files it tracks.
#[derive(Component)]
struct Measuring {
    call: Entity,
    agent: Entity,
}

/// The changes to judge, on the [`Measuring`] entity.
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
    calls: Query<(Entity, &Footprint, &Decision), (ReadyToStart, ToolCalls, Without<Preview>)>,
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

/// Judges each previewed call by its changes. One that deletes a file is
/// held while git is asked whether it tracks it. A call that could not be
/// previewed, such as one the tool would refuse, runs as it would have.
fn judge_previewed(
    calls: Query<
        (Entity, &Previewed, &ForAgent, &Decision),
        (ReadyToStart, ToolCalls, Without<Checked>),
    >,
    allowed: Res<AllowedFiles>,
    config: Res<LargeChangeConfig>,
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
        let measuring = Measuring { call, agent };
        let deletes = changes.iter().any(|it| it.kind == ChangeKind::Deleted);
        if !(deletes && config.delete_tracked) {
            let untracked = |file| Change {
                file,
                tracked: false,
            };
            let changes = changes.into_iter().map(untracked).collect::<Vec<_>>();
            commands.spawn((measuring, Measured(changes)));
            continue;
        }
        let dir = dirs.get(agent).map_or(&root.0, |dir| &dir.0);
        let launcher = processes.launcher().in_dir(dir);
        let workspace: Arc<dyn Workspace> = workspace.0.clone();
        let paths: Vec<_> = changes
            .iter()
            .map(|it| (it.kind == ChangeKind::Deleted).then(|| it.path.clone()))
            .collect();
        let work = async move {
            let look = move || {
                let tracked = |path: Option<String>| {
                    path.is_some_and(|path| tracked(&*workspace, &launcher, path.as_ref()))
                };
                Ok(paths.into_iter().map(tracked).collect::<Vec<_>>())
            };
            // Unanswered, a deleted file counts as tracked.
            let tracked = blocking(look).await.unwrap_or_default();
            let tracked = tracked.into_iter().chain(std::iter::repeat(true));
            let changes = changes.into_iter().zip(tracked);
            changes
                .map(|(file, tracked)| Change {
                    tracked: tracked && file.kind == ChangeKind::Deleted,
                    file,
                })
                .collect::<Vec<_>>()
        };
        let pool = IoTaskPool::get_or_init(TaskPool::default);
        commands.spawn((
            Name::new("asking git about a deletion"),
            measuring,
            HoldOf(call),
            Running::spawn_into::<Measured, _>(pool, &wake, work),
        ));
    }
}

/// Lets a call run when its changes are small or their files allowed;
/// otherwise asks, or follows `on_headless` when nobody can answer.
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
    let measuring = added.entity;
    let Ok((&Measuring { call, agent }, Measured(changes))) = holds.get(measuring) else {
        return;
    };
    // It goes whatever happens next: a question holds the call instead.
    commands.entity(measuring).despawn();
    let large: Vec<&Change> = changes
        .iter()
        .filter(|change| change.is_large(&config) && !allowed.has(&change.file.path))
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
    let diffs: Vec<String> = large
        .iter()
        .map(|change| change.file.unified())
        .filter(|diff| !diff.is_empty())
        .collect();
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
    let paths = large
        .iter()
        .map(|change| change.file.path.clone())
        .collect();
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
        Some(Answer {
            text: Some(reason), ..
        }) if !reason.trim().is_empty() => Some(format!(
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
