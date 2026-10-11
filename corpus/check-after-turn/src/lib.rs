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
//! The files a turn edited are the paths of its successful `write` and
//! `edit` calls. Each matching check runs, one at a time, as a call of the
//! agent's registered `shell` tool outside any turn, so a plugin that
//! replaces `shell`, such as a sandbox, runs the checks too. All pass: a
//! "checks passed" notice. One fails: the agent is sent its name and the
//! last lines of its output and starts a turn to fix it, at most
//! `max_rounds` times per user message. A user message, or stopping the
//! agent, cancels the checks running.

use std::collections::{BTreeSet, VecDeque};
use std::path::{Component as PathPart, Path, PathBuf};

use bevy_log::warn;
use glob::Pattern;
use rig_ecs::turn::ToolStarter;
use rig_harness::prelude::*;
use rig_harness::rig_core::message::{ToolFunction, ToolName};
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// Who the agent is told sent a failure.
const ORIGIN: &str = "checks";
/// Fix rounds per user message unless the file says.
const DEFAULT_MAX_ROUNDS: u32 = 3;
/// Lines of a failing check's output the agent is sent.
const TAIL_LINES: usize = 200;
/// The tools whose successful calls edited the file in their `path`.
const EDITING_TOOLS: [&str; 2] = ["write", "edit"];
/// How long a check may run: the `shell` tool's longest timeout.
const TIMEOUT_SECS: u64 = 600;
/// The tool each check runs as.
const SHELL: &str = "shell";

/// Runs the project's checks after a turn that edited files they watch.
#[derive(Default)]
pub struct CheckAfterTurnPlugin;

impl Plugin for CheckAfterTurnPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<Checks>() {
            match std::env::current_dir() {
                Ok(root) => app.insert_resource(Checks::load(root)),
                Err(error) => {
                    warn!("checks: no working directory: {error}");
                    return;
                }
            };
        }
        app.register_required_components::<Agent, CheckState>()
            .add_observer(record_edit)
            .add_observer(check_after_turn)
            .add_observer(check_done)
            .add_observer(cancel_on_user)
            .add_observer(cancel_on_interrupt);
        #[cfg(feature = "tui")]
        app.add_systems(PostUpdate, status::show.in_set(rig_tui::StatusSystems));
    }
}

/// The project's checks.
#[derive(Resource, Clone, Debug)]
pub struct Checks {
    /// The project directory: edited paths are matched relative to it. The
    /// checks run where the `shell` tool runs, the working directory, which
    /// it is when loaded from there.
    pub root: PathBuf,
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
    /// The edited paths that make it run.
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
    /// The checks of `text`, a `checks.toml`, for the project `root`.
    pub fn parse(root: PathBuf, text: &str) -> Result<Self, String> {
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
            root,
            max_rounds: file.max_rounds.unwrap_or(DEFAULT_MAX_ROUNDS),
            checks,
        })
    }

    /// The checks of `root/.rig/checks.toml`; none when it is missing or
    /// wrong, which is logged.
    fn load(root: PathBuf) -> Self {
        let path = root.join(".rig").join("checks.toml");
        let parsed = match std::fs::read_to_string(&path) {
            Ok(text) => Self::parse(root.clone(), &text),
            Err(_) => Err(String::new()),
        };
        parsed.unwrap_or_else(|why| {
            if !why.is_empty() {
                warn!("checks: {}: {why}", path.display());
            }
            Self {
                root,
                max_rounds: DEFAULT_MAX_ROUNDS,
                checks: Vec::new(),
            }
        })
    }

    /// `path`, as a tool call named it, relative to the project; `None`
    /// outside it.
    fn relative(&self, path: &str) -> Option<PathBuf> {
        let path = Path::new(path);
        let path = match path.strip_prefix(&self.root) {
            Ok(inside) => inside,
            Err(_) if path.is_absolute() => return None,
            Err(_) => path,
        };
        let path: PathBuf = path
            .components()
            .filter(|part| !matches!(part, PathPart::CurDir))
            .collect();
        (!path.as_os_str().is_empty()).then_some(path)
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

/// On a check's `shell` call: the agent it checks for, the check, and the
/// checks after it.
#[derive(Component)]
struct CheckRun {
    agent: Entity,
    check: Check,
    left: VecDeque<Check>,
}

/// Notes the file a turn's successful `write` or `edit` call changed.
fn record_edit(
    output: On<Insert<ToolOutput>>,
    calls: Query<(&ToolOutput, &ToolCallRun, &CallOf)>,
    turns: Query<&TurnOf>,
    checks: Res<Checks>,
    mut states: Query<&mut CheckState>,
) {
    let Ok((ToolOutput(result), run, &CallOf(turn))) = calls.get(output.entity) else {
        return;
    };
    let function = &run.call.function;
    if !EDITING_TOOLS.contains(&function.name.as_str()) || !result.is_success() {
        return;
    }
    let path = function
        .arguments
        .get("path")
        .and_then(|path| path.as_str());
    let Some(path) = path.and_then(|path| checks.relative(path)) else {
        return;
    };
    if let Ok(&TurnOf(agent)) = turns.get(turn)
        && let Ok(mut state) = states.get_mut(agent)
    {
        state.edited.insert(path);
    }
}

/// Starts the checks that watch what an answered turn edited.
fn check_after_turn(
    ended: On<TurnEnded>,
    checks: Res<Checks>,
    mut states: Query<&mut CheckState>,
    runs: Query<(Entity, &CheckRun)>,
    starter: ToolStarter,
    mut commands: Commands,
) {
    let agent = ended.entity;
    if !matches!(ended.outcome, TurnOutcome::Answered(_)) {
        return;
    }
    let Ok(mut state) = states.get_mut(agent) else {
        return;
    };
    if state.edited.is_empty() {
        return;
    }
    let mut left = checks.matching(&std::mem::take(&mut state.edited));
    let Some(check) = left.pop_front() else {
        return;
    };
    stop_runs(agent, &runs, &mut commands);
    start_check(&starter, &mut commands, &mut state, (agent, check, left));
}

/// Runs `check` for `agent` as a call of the `shell` tool, with the checks
/// `left` after it.
fn start_check(
    starter: &ToolStarter,
    commands: &mut Commands,
    state: &mut CheckState,
    (agent, check, left): (Entity, Check, VecDeque<Check>),
) {
    let Ok(shell) = ToolName::new(SHELL) else {
        return;
    };
    let arguments = serde_json::json!({ "command": check.command, "timeout_secs": TIMEOUT_SECS });
    let call = ToolCall::from_wire(
        format!("check-{}", check.name),
        ToolFunction::new(shell, arguments),
    );
    let run = starter.run(call, None);
    state.running = Some(check.name.clone());
    let name = Name::new(format!("check {}", check.name));
    let entity = commands
        .spawn((name, CheckRun { agent, check, left }, run.clone()))
        .id();
    starter.start(commands, entity, agent, &run);
}

/// A check finished: the next one runs, or the agent hears how they went.
fn check_done(
    output: On<Insert<ToolOutput>>,
    runs: Query<(&ToolOutput, &CheckRun)>,
    checks: Res<Checks>,
    mut states: Query<&mut CheckState>,
    starter: ToolStarter,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((ToolOutput(result), run)) = runs.get(output.entity) else {
        return;
    };
    commands.entity(output.entity).despawn();
    let agent = run.agent;
    let Ok(mut state) = states.get_mut(agent) else {
        return;
    };
    state.running = None;
    let text = result.output().render();
    if !result.is_success() {
        let why = tail(&text, 5);
        let line = format!("check {} could not run: {why}", run.check.name);
        notices.write(Notice::error(agent, line));
        return;
    }
    if passed(&text) {
        let mut left = run.left.clone();
        match left.pop_front() {
            Some(next) => start_check(&starter, &mut commands, &mut state, (agent, next, left)),
            None => {
                notices.write(Notice::info(agent, "checks passed"));
            }
        }
        return;
    }
    if state.rounds >= checks.max_rounds {
        let line = format!("checks still failing after {} rounds", checks.max_rounds);
        notices.write(Notice::error(agent, line));
        return;
    }
    state.rounds += 1;
    let Check { name, command, .. } = &run.check;
    let output = tail(&text, TAIL_LINES);
    let message = format!(
        "The check `{name}` (`{command}`) failed after your changes. The last \
         {TAIL_LINES} lines of its output:\n\n{output}\n\nFix the cause; the checks \
         run again when your turn ends."
    );
    let deliver = Deliver::new(agent, message, DeliveryMode::Queue);
    commands.trigger(deliver.with_origin(Origin::plugin(ORIGIN).titled(name.clone())));
}

/// A message the user sends starts over: the checks running stop, and the
/// fix rounds count from none.
fn cancel_on_user(
    deliver: On<Deliver>,
    runs: Query<(Entity, &CheckRun)>,
    mut states: Query<&mut CheckState>,
    mut commands: Commands,
) {
    let user = deliver.origin.kind == OriginKind::User;
    if !user || deliver.command().is_some() || deliver.text.trim().is_empty() {
        return;
    }
    stop_runs(deliver.entity, &runs, &mut commands);
    if let Ok(mut state) = states.get_mut(deliver.entity) {
        state.rounds = 0;
        state.running = None;
    }
}

/// Stopping the agent stops its checks.
fn cancel_on_interrupt(
    interrupt: On<Interrupt>,
    runs: Query<(Entity, &CheckRun)>,
    mut states: Query<&mut CheckState>,
    mut commands: Commands,
) {
    stop_runs(interrupt.entity, &runs, &mut commands);
    if let Ok(mut state) = states.get_mut(interrupt.entity) {
        state.running = None;
    }
}

/// Despawns `agent`'s check calls, which cancels their commands.
fn stop_runs(agent: Entity, runs: &Query<(Entity, &CheckRun)>, commands: &mut Commands) {
    for (entity, run) in runs {
        if run.agent == agent {
            commands.entity(entity).despawn();
        }
    }
}

/// Whether the `shell` tool's `output` ends in a zero exit code: the tool
/// says how the command ended only in its text.
fn passed(output: &str) -> bool {
    output.trim_end().ends_with("[exit code 0]")
}

/// The last `lines` lines of `text`.
fn tail(text: &str, lines: usize) -> String {
    let all: Vec<&str> = text.trim_end().lines().collect();
    let skip = all.len().saturating_sub(lines);
    all.into_iter().skip(skip).collect::<Vec<_>>().join("\n")
}

/// "checking: <name>" in the status line while a check runs.
#[cfg(feature = "tui")]
mod status {
    use rig_harness::prelude::*;
    use rig_tui::{Side, StatusItem, StatusItems, Tone};

    use super::CheckState;

    /// After the agent's status; gone before the reasoning setting.
    const CHECKING: StatusItem = StatusItem::at(Side::Left, 56, 10);

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
