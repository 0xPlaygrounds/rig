//! Plugin corpus: 32. Run checks after the agent edits (spec
//! `32-check-after-turn.md`).
//!
//! The project's checks, `<project>/.rig/checks.toml`, run when an agent's
//! turn ends having edited files one of them watches:
//!
//! ```toml
//! max_rounds = 3          # automatic fix rounds per user message
//!
//! [[check]]
//! name = "clippy"
//! command = "cargo clippy --all-targets -- -D warnings"
//! when = "**/*.rs"        # a glob over the edited paths, from the project
//! ```
//!
//! The files a turn edited are what its tool calls published they changed
//! (`FilesChanged`). Each matching check is a `check` effect of the agent,
//! one at a time, whose command runs as `Processes` say, so a sandbox
//! plugin's launcher wraps it too. All pass: a "checks passed" notice. One
//! fails (its `ExitStatus`): the agent is sent its name and the last lines
//! of its output and starts a turn to fix it, at most `max_rounds` times per
//! user message. A user message, or stopping the agent, cancels the checks.

use std::collections::{BTreeSet, VecDeque};
use std::path::{Path, PathBuf};

use bevy_tasks::{IoTaskPool, TaskPool};
use glob::Pattern;
use rig_harness::prelude::*;
use rig_harness::rig_core::effect::{Answered, EffectKind};
use rig_harness::rig_core::error::ErrorReport;
use rig_harness::rig_core::tool::{ExitStatus, FilesChanged};
use rig_tools::{MAX_BYTES, OutputLimit, Shell, ShellArgs};
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests;

/// The effect kind of a check, and who the agent is told sent a failure.
const CHECK: &str = "check";
/// Fix rounds per user message unless the file says.
const DEFAULT_MAX_ROUNDS: u32 = 3;
/// Lines of a failing check's output the agent is sent.
const TAIL_LINES: usize = 200;
/// How long a check may run: the shell's longest timeout.
const TIMEOUT_SECS: u64 = 600;

/// Runs the project's checks after a turn that edited files they watch.
#[derive(Default)]
pub struct CheckAfterTurnPlugin;

impl Plugin for CheckAfterTurnPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .init_resource::<Processes>()
            .register_required_components::<Agent, CheckState>()
            .add_observer(note_edits)
            .add_observer(check_after_turn)
            .add_observer(run_check)
            .add_observer(check_done)
            .add_observer(follow_successor::<CheckRun>)
            .add_observer(cancel_on_user)
            .add_observer(cancel_on_interrupt);
        #[cfg(feature = "tui")]
        app.add_systems(
            PostUpdate,
            status::show.in_set(rig_tui::StatusSystems::Show),
        );
    }
}

/// The project's checks.
#[derive(Clone, Debug)]
pub struct Checks {
    /// Automatic fix rounds per user message.
    pub max_rounds: u32,
    /// The checks, in the order they run.
    pub checks: Vec<Check>,
}

/// One check.
#[derive(Clone, Debug)]
pub struct Check {
    /// Its name, as the agent and the user are told.
    pub name: String,
    /// The shell command; it passes when it exits with 0.
    pub command: String,
    /// The edited paths, relative to the project, that make it run.
    pub when: Pattern,
}

/// `checks.toml`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChecksFile {
    max_rounds: Option<u32>,
    #[serde(default)]
    check: Vec<CheckEntry>,
}

/// A `[[check]]` of `checks.toml`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckEntry {
    name: String,
    command: String,
    when: String,
}

impl Checks {
    /// The checks of `text`, a `checks.toml`.
    pub fn parse(text: &str) -> Result<Self, String> {
        let file: ChecksFile = toml::from_str(text).map_err(|error| error.to_string())?;
        let checks = file
            .check
            .into_iter()
            .map(|entry| match Pattern::new(&entry.when) {
                Ok(when) => Ok(Check {
                    name: entry.name,
                    command: entry.command,
                    when,
                }),
                Err(error) => Err(format!("check `{}`: `when`: {error}", entry.name)),
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            max_rounds: file.max_rounds.unwrap_or(DEFAULT_MAX_ROUNDS),
            checks,
        })
    }

    /// The checks of `root/.rig/checks.toml`, read afresh; none when the
    /// file is missing.
    fn load(root: &Path) -> Result<Option<Self>, String> {
        let path = root.join(".rig").join("checks.toml");
        match std::fs::read_to_string(&path) {
            Ok(text) => Self::parse(&text)
                .map(Some)
                .map_err(|why| format!("{}: {why}", path.display())),
            Err(_) => Ok(None),
        }
    }

    /// The checks that watch any of `edited`, in order.
    fn matching(&self, edited: &BTreeSet<PathBuf>) -> VecDeque<Check> {
        self.checks
            .iter()
            .filter(|check| edited.iter().any(|path| check.when.matches_path(path)))
            .cloned()
            .collect()
    }
}

/// An agent's checks: what its turns edited since the last checks, the
/// fix rounds since the user's last message, and the check running.
#[derive(Component, Default, Debug)]
pub struct CheckState {
    edited: BTreeSet<PathBuf>,
    rounds: u32,
    running: Option<String>,
}

impl CheckState {
    /// The check running, if any.
    pub fn running(&self) -> Option<&str> {
        self.running.as_deref()
    }
}

/// On a `check` effect: the check, the checks after it, and the fix rounds
/// allowed.
#[derive(Component)]
struct CheckRun {
    check: Check,
    left: VecDeque<Check>,
    max_rounds: u32,
}

/// How a check's command ended: the `check` effect's outcome.
#[derive(Serialize, Deserialize)]
struct Ran {
    status: ExitStatus,
    output: String,
}

/// Notes the files a settled tool call published it changed.
fn note_edits(
    settled: On<Add<Settled>>,
    calls: Query<(&Published, &ForAgent)>,
    root: Res<ProjectRoot>,
    mut states: Query<&mut CheckState>,
) {
    let Ok((published, &ForAgent(agent))) = calls.get(settled.entity) else {
        return;
    };
    let Ok(Some(changed)) = published.0.get::<FilesChanged>() else {
        return;
    };
    if let Ok(mut state) = states.get_mut(agent) {
        let paths = changed.paths().filter_map(|path| relative(&root.0, path));
        state.edited.extend(paths);
    }
}

/// Starts the checks that watch what an answered turn edited.
fn check_after_turn(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    root: Res<ProjectRoot>,
    mut states: Query<&mut CheckState>,
    runs: Query<(Entity, &ForAgent), With<CheckRun>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((TurnOutcome::Answered(_), &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    let Ok(mut state) = states.get_mut(agent) else {
        return;
    };
    if state.edited.is_empty() {
        return;
    }
    let edited = std::mem::take(&mut state.edited);
    let checks = match Checks::load(&root.0) {
        Ok(Some(checks)) => checks,
        Ok(None) => return,
        Err(why) => {
            notices.write(Notice::error(agent, format!("checks: {why}")));
            return;
        }
    };
    let mut left = checks.matching(&edited);
    let Some(check) = left.pop_front() else {
        return;
    };
    stop(agent, &runs, &mut commands);
    let run = CheckRun {
        check,
        left,
        max_rounds: checks.max_rounds,
    };
    spawn_check(&mut commands, &mut state, agent, run);
}

/// Spawns `run` as a `check` effect of `agent`.
fn spawn_check(commands: &mut Commands, state: &mut CheckState, agent: Entity, run: CheckRun) {
    let Check { name, command, .. } = &run.check;
    state.running = Some(name.clone());
    let kind = EffectKind::Custom {
        kind: CHECK.into(),
        payload: serde_json::json!({ "name": name, "command": command }),
    };
    let label = Name::new(format!("check {name}"));
    let request = EffectRequest {
        kind,
        handler: None,
    };
    commands.spawn((request, ForAgent(agent), label, run));
}

/// Runs a check's command once its effect started, as `Processes` say, in
/// the project unless they name a directory.
fn run_check(
    started: On<Add<Started>>,
    runs: Query<&CheckRun>,
    processes: Res<Processes>,
    root: Res<ProjectRoot>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Ok(run) = runs.get(started.entity) else {
        return;
    };
    let mut launcher = processes.launcher();
    if launcher.cwd.is_none() {
        launcher = launcher.in_dir(&root.0);
    }
    let shell = Shell {
        launcher,
        spill: None,
        confine: false,
        cut: Some(OutputLimit {
            lines: TAIL_LINES,
            bytes: MAX_BYTES,
        }),
        ..Shell::default()
    };
    let args = ShellArgs::new(run.check.command.clone()).timeout_secs(TIMEOUT_SECS);
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    let work = async move {
        match shell.run(args).await {
            Ok(ran) => {
                let ran = Ran {
                    status: ran.status,
                    output: ran.output,
                };
                Outcome::custom(serde_json::to_value(ran).unwrap_or_default())
            }
            Err(error) => Outcome(Err(ErrorReport::from(error))),
        }
    };
    commands
        .entity(started.entity)
        .insert(Running::answering(pool, &wake, work));
}

/// A check settled: the next one runs, or the agent hears how they went.
fn check_done(
    settled: On<Add<Settled>>,
    runs: Query<(&CheckRun, &Outcome, &ForAgent)>,
    mut states: Query<&mut CheckState>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((run, outcome, &ForAgent(agent))) = runs.get(settled.entity) else {
        return;
    };
    let Ok(mut state) = states.get_mut(agent) else {
        return;
    };
    state.running = None;
    let Check { name, command, .. } = &run.check;
    let ran = match &outcome.0 {
        Ok(Answered::Custom { payload }) => serde_json::from_value::<Ran>(payload.clone()).ok(),
        _ => None,
    };
    let Some(ran) = ran else {
        let why = match &outcome.0 {
            Err(report) => report.message.clone(),
            Ok(_) => "it was answered without running".to_owned(),
        };
        notices.write(Notice::error(
            agent,
            format!("check {name} could not run: {why}"),
        ));
        return;
    };
    if ran.status.success() {
        let mut left = run.left.clone();
        match left.pop_front() {
            Some(check) => {
                let next = CheckRun {
                    check,
                    left,
                    max_rounds: run.max_rounds,
                };
                spawn_check(&mut commands, &mut state, agent, next);
            }
            None => {
                notices.write(Notice::info(agent, "checks passed"));
            }
        }
        return;
    }
    if state.rounds >= run.max_rounds {
        let line = format!("checks still failing after {} rounds", run.max_rounds);
        notices.write(Notice::error(agent, line));
        return;
    }
    state.rounds += 1;
    let message = format!(
        "The check `{name}` (`{command}`) failed after your changes. The last \
         {TAIL_LINES} lines of its output:\n\n{}\n\nFix the cause; the checks run \
         again when your turn ends.",
        ran.output.trim_end()
    );
    let deliver = Deliver::new(agent, message, DeliveryMode::Queue);
    commands.trigger(deliver.with_author(Author::plugin(CHECK).titled(name.clone())));
}

/// A message the user sends starts over: the checks running stop, and the
/// fix rounds count from none.
fn cancel_on_user(
    started: On<Add<Started>>,
    messages: Query<(&Incoming, &ForAgent)>,
    runs: Query<(Entity, &ForAgent), With<CheckRun>>,
    mut states: Query<&mut CheckState>,
    mut commands: Commands,
) {
    let Ok((message, &ForAgent(agent))) = messages.get(started.entity) else {
        return;
    };
    if message.author.kind != AuthorKind::User || message.command().is_some() {
        return;
    }
    stop(agent, &runs, &mut commands);
    if let Ok(mut state) = states.get_mut(agent) {
        state.rounds = 0;
        state.running = None;
    }
}

/// A stop that reaches the agent stops its checks.
fn cancel_on_interrupt(
    interrupt: On<StopReached>,
    runs: Query<(Entity, &ForAgent), With<CheckRun>>,
    mut states: Query<&mut CheckState>,
    mut commands: Commands,
) {
    stop(interrupt.event_target(), &runs, &mut commands);
    if let Ok(mut state) = states.get_mut(interrupt.event_target()) {
        state.running = None;
    }
}

/// Despawns `agent`'s checks, which kills their commands; each is recorded
/// as cancelled.
fn stop(agent: Entity, runs: &Query<(Entity, &ForAgent), With<CheckRun>>, commands: &mut Commands) {
    for (run, &ForAgent(of)) in runs {
        if of == agent {
            commands.entity(run).despawn();
        }
    }
}

/// `path`, as a tool published it, relative to the project `root`; `None`
/// outside it. rig-tools' file tools publish absolute paths; a relative
/// one is taken as the call named it, from the project.
fn relative(root: &Path, path: &str) -> Option<PathBuf> {
    let joined = root.join(path);
    let inside = joined.strip_prefix(root).ok()?;
    (!inside.as_os_str().is_empty()).then(|| inside.to_path_buf())
}

/// "checking: `<name>`" in the status line while a check runs.
#[cfg(feature = "tui")]
mod status {
    use rig_harness::prelude::*;
    use rig_tui::{Side, StatusItem, StatusItems, Tone};

    use super::CheckState;

    /// After the agent's status; gone before the reasoning setting.
    const CHECKING: StatusItem = StatusItem::new("check-after-turn", Side::Left, 56, 10);

    pub(super) fn show(mut agents: Query<(&CheckState, &mut StatusItems), Changed<CheckState>>) {
        for (state, mut items) in &mut agents {
            let text = state
                .running()
                .map(|name| format!("checking: {name}"))
                .unwrap_or_default();
            items.show(CHECKING.says(text, Tone::Dim));
        }
    }
}
