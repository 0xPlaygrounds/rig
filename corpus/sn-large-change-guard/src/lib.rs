//! SN-02: the large-change guard. Before a `write`, `edit`, `delete` or
//! overwriting `move` runs, a Gate system holds it while the file it would
//! change is measured off the main thread: lines removed and added, the
//! share of the existing file replaced, the size it would have, and for a
//! delete whether git tracks the file. Under every threshold the hold goes
//! at once. Over one, the user is asked, naming the agent (or subagent)
//! and the damage, with a diff preview: allow, allow this file for the
//! session (saved with it), deny, or deny with a reason, which the model
//! reads in the call's refusal. A run nobody can answer follows
//! `on_headless`. Thresholds are read from `RIG_HOME/large-change.toml`,
//! then the project's `.rig/large-change.toml`, key by key.

use std::ffi::OsString;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::prelude::*;
use rig_tools::edit::apply;
use rig_tools::fs::file_change;
use rig_tools::process::{Launcher, Spawn};
use rig_tools::workspace::{Stat, Workspace};
use rig_tools::{
    Delete, DeleteArgs, Edit, EditArgs, MAX_FILE_BYTES, Move, MoveArgs, Write, WriteArgs,
};
use serde::Deserialize;
use serde::de::DeserializeOwned;

/// The config file's name, under `RIG_HOME` and under the project's `.rig`.
pub const CONFIG: &str = "large-change.toml";

/// The refusal a run nobody can answer gives under `on_headless = "deny"`.
pub const HEADLESS: &str = "large change needs approval; not available non-interactively";

/// How many diff lines the question shows.
const PREVIEW_LINES: usize = 60;

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

/// The guard's Gate system, which holds the file-changing calls it
/// measures, for others to order against.
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
            .add_systems(AgentPass, hold_to_measure.in_set(LargeChangeSystems))
            .add_observer(measured)
            .add_observer(answered);
    }

    /// Reads the config once every plugin set the project root, unless the
    /// app has one already, such as a test's.
    fn finish(&self, app: &mut App) {
        if app.world().contains_resource::<LargeChangeConfig>() {
            return;
        }
        let root = app.world().resource::<ProjectRoot>().0.clone();
        let files = [
            RigHome::of(app.world()).root().join(CONFIG),
            root.join(".rig").join(CONFIG),
        ];
        let (config, problems) = read_config::<LargeChangeConfig>(&files);
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
fn read_config<T: DeserializeOwned + Default>(files: &[PathBuf]) -> (T, Vec<String>) {
    let mut table = toml::Table::new();
    let mut problems = Vec::new();
    for path in files {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        match text.parse::<toml::Table>() {
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

/// What a call would do to a file, as the tool would do it.
enum Plan {
    /// The file gets this text.
    Write(PathBuf, String),
    /// The file gets these edits.
    Edit(PathBuf, EditArgs),
    /// The file is deleted.
    Delete(PathBuf),
    /// The file `to` is replaced by `from`.
    Overwrite { from: PathBuf, to: PathBuf },
}

impl Plan {
    /// The plan of a call of one of rig-tools' file tools, on the paths its
    /// footprint resolved; `None` for any other call.
    fn of(call: &ToolCall, Footprint(touches): &Footprint) -> Option<Self> {
        let Touches::Writes(Scope::Paths(paths)) = touches else {
            return None;
        };
        let args = serde_json::Value::Object(call.function.arguments.clone());
        let first = paths.first()?.clone();
        Some(match call.function.name.as_str() {
            Write::NAME => Self::Write(first, parse::<WriteArgs>(args)?.content),
            Edit::NAME => Self::Edit(first, parse(args)?),
            Delete::NAME => parse::<DeleteArgs>(args).map(|_| Self::Delete(first))?,
            Move::NAME if parse::<MoveArgs>(args)?.overwrite => Self::Overwrite {
                from: first,
                to: paths.get(1)?.clone(),
            },
            _ => return None,
        })
    }

    /// The file whose content is at stake.
    fn path(&self) -> &Path {
        match self {
            Self::Write(path, _) | Self::Edit(path, _) | Self::Delete(path) => path,
            Self::Overwrite { to, .. } => to,
        }
    }
}

fn parse<T: DeserializeOwned>(args: serde_json::Value) -> Option<T> {
    serde_json::from_value(args).ok()
}

/// What a call would do to its file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Change {
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
    /// `src/app.rs: −190 / +10 lines (95% of file replaced)`.
    pub fn line(&self, path: &str) -> String {
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

    /// The diff, cut to [`PREVIEW_LINES`].
    fn preview(&self) -> String {
        let lines: Vec<&str> = self.diff.lines().collect();
        let mut text = lines
            .iter()
            .take(PREVIEW_LINES)
            .copied()
            .collect::<Vec<_>>()
            .join("\n");
        if let Some(more) = lines.len().checked_sub(PREVIEW_LINES).filter(|n| *n > 0) {
            let _ = write!(text, "\n… {more} more lines");
        }
        text
    }
}

/// The text of `path` in `workspace`, lossily, as the file tools diff it;
/// `None` when there is no file.
fn text_of(workspace: &dyn Workspace, path: &Path) -> Option<String> {
    match workspace.stat(path).ok()? {
        Stat::File { .. } => {
            let contents = workspace.read(path, MAX_FILE_BYTES).ok()?;
            Some(String::from_utf8_lossy(&contents.bytes).into_owned())
        }
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

/// What `plan` would do; `None` when the tool would refuse it anyway or it
/// loses nothing, such as a move onto no file.
fn measure(workspace: &dyn Workspace, launcher: &Launcher, plan: &Plan) -> Option<Change> {
    let path = plan.path();
    let before = text_of(workspace, path);
    let shown = path.to_string_lossy();
    let (after, deleted) = match plan {
        Plan::Write(_, content) => (content.clone(), false),
        Plan::Edit(_, args) => (
            apply(&shown, before.as_deref()?, &args.edits).ok()?.text,
            false,
        ),
        Plan::Delete(_) => (String::new(), true),
        Plan::Overwrite { from, .. } => {
            before.as_ref()?;
            (text_of(workspace, from)?, false)
        }
    };
    let change = file_change(&shown, before.as_deref(), &after);
    Some(Change {
        removed: change.removed,
        added: change.added,
        old_lines: before.as_deref().map(|text| text.lines().count()),
        size: after.len() as u64,
        deleted,
        tracked: deleted && tracked(workspace, launcher, path),
        diff: change.diff.unwrap_or_default(),
    })
}

/// On a call measured already, or allowed.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Checked;

/// The hold of a call while its file is measured.
#[derive(Component)]
struct Measuring {
    call: Entity,
    agent: Entity,
    path: PathBuf,
}

/// What the measuring found, on the hold.
#[derive(Component)]
struct Measured(Option<Change>);

impl From<Option<Change>> for Measured {
    fn from(change: Option<Change>) -> Self {
        Self(change)
    }
}

/// On the guard's question: the call it holds, its file and the damage.
#[derive(Component)]
struct GuardAsk {
    call: Entity,
    path: PathBuf,
    line: String,
}

fn pool() -> &'static TaskPool {
    IoTaskPool::get_or_init(TaskPool::default)
}

/// Holds each file-changing call about to run while its file is measured,
/// unless its file is allowed for the session.
#[allow(clippy::too_many_arguments)]
fn hold_to_measure(
    calls: Query<
        (Entity, &ToolRequest, &ForAgent, &Footprint, &Decision),
        (Undecided, ToolCalls, Without<Checked>),
    >,
    allowed: Res<AllowedFiles>,
    dirs: Query<&WorkDir>,
    root: Res<ProjectRoot>,
    workspace: Res<ProjectWorkspace>,
    processes: Res<Processes>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    for (call, request, &ForAgent(agent), footprint, decision) in &calls {
        if decision.is_decided() {
            continue;
        }
        commands.entity(call).insert(Checked);
        let Some(plan) = Plan::of(&request.call, footprint) else {
            continue;
        };
        let path = plan.path().to_owned();
        if is_allowed(&allowed, &path) {
            continue;
        }
        let dir = dirs.get(agent).map_or(&root.0, |dir| &dir.0);
        let launcher = processes.launcher().in_dir(dir);
        let workspace: Arc<dyn Workspace> = workspace.0.clone();
        let work = async move {
            blocking(move || Ok(measure(&*workspace, &launcher, &plan)))
                .await
                .ok()
                .flatten()
        };
        let name = format!("measuring the change to {}", path.display());
        commands.spawn((
            Name::new(name),
            Measuring { call, agent, path },
            HoldOf(call),
            Running::spawn_into::<Measured, _>(pool(), &wake, work),
        ));
    }
}

fn is_allowed(allowed: &AllowedFiles, path: &Path) -> bool {
    let path = path.to_string_lossy();
    allowed.files.iter().any(|file| *file == path)
}

/// `path` as the user reads it: from the project's root when under it.
fn shown(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

/// Lets a measured call run when its change is small or its file allowed;
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
    let hold = added.entity;
    let Ok((measuring, Measured(change))) = holds.get(hold) else {
        return;
    };
    // The hold goes whatever happens next: a question holds the call instead.
    commands.entity(hold).despawn();
    let Some(change) = change.as_ref().filter(|change| change.is_large(&config)) else {
        return;
    };
    if is_allowed(&allowed, &measuring.path) {
        return;
    }
    let line = change.line(&shown(&measuring.path, &root.0));
    let (call, agent) = (measuring.call, measuring.agent);
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
    let ask = Ask::new(format!("{who}: {line}"))
        .body(format!(
            "{}\n\nType a reason to deny with it.",
            change.preview()
        ))
        .choice("allow", "Allow")
        .choice("allow_file", "Allow this file for the session")
        .choice("deny", "Deny")
        .choice("deny_reason", "Deny with reason")
        .free_text();
    let path = measuring.path.clone();
    commands.spawn((
        ask,
        ForAgent(agent),
        HoldOf(call),
        GuardAsk { call, path, line },
    ));
}

/// Acts on the user's answer: lets the call run, allows its file for the
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
            allowed
                .files
                .push(guard.path.to_string_lossy().into_owned());
            None
        }
        Some(answer) if answer.is("deny_reason") => {
            let why = Ask::new(format!("Why deny {}?", guard.line)).initial("");
            let again = GuardAsk {
                call: guard.call,
                path: guard.path.clone(),
                line: guard.line.clone(),
            };
            commands.spawn((why, ForAgent(agent), HoldOf(guard.call), again));
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
