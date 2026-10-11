//! Plugin corpus (held-out): H15: Debounced background check after edits
//! (spec `H15-background-verify.md`).
//!
//! A tool call by any agent that changed a file matching `[verify] files`
//! (default `*.rs`) schedules the project's check (`[verify] command`,
//! default `cargo check --message-format short`) after a quiet period
//! (`[verify] quiet_ms`, default 1500): a later edit restarts the wait and
//! kills a check that runs. The check is a `verify` effect of the app, one
//! at a time, whose command runs as `Processes` say.
//!
//! A failure is a [`DeliveryMode::Note`] to each agent that edited since the
//! last pass, with the first error lines: it starts no turn, and goes with
//! the agent's next model call. While the agent is idle the note is held,
//! and let go as a message that starts its next turn is about to start, so
//! it goes in just before it. Only the newest result counts: a result
//! withdraws every note not sent yet, and a pass after a failure an agent
//! saw tells it the check passes again. `/verify on|off|now`.

use std::collections::HashSet;
use std::path::Path;

use bevy_tasks::{IoTaskPool, TaskPool};
use glob::Pattern;
use rig_harness::prelude::*;
use rig_harness::rig_core::effect::{EffectKind, Outcome as Answered};
use rig_harness::rig_core::error::ErrorReport;
use rig_harness::rig_core::tool::{ExitStatus, FilesChanged};
use rig_tools::{Shell, ShellArgs};
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests;

/// The check's effect kind, and the sender of its notes.
const VERIFY: &str = "verify";
/// Error lines a failure note carries.
const ERROR_LINES: usize = 20;
/// How long a check may run.
const TIMEOUT_SECS: u64 = 600;

/// Checks the project in the background after edits.
#[derive(Default)]
pub struct BackgroundVerifyPlugin;

impl Plugin for BackgroundVerifyPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<VerifySettings>("verify")
            .init_resource::<ProjectRoot>()
            .init_resource::<Processes>()
            .init_resource::<Verify>()
            .add_command(
                "verify",
                "Background check after edits: /verify on|off|now",
                command,
            )
            .add_systems(
                AgentPass,
                release_notes.in_set(Stage::Gate).after(CommandSystems),
            )
            .add_observer(on_edit)
            .add_observer(due)
            .add_observer(run_check)
            .add_observer(checked)
            .add_observer(note_sent);
        #[cfg(feature = "tui")]
        status::add(app);
    }
}

/// The `[verify]` table of the user's settings.
#[derive(Resource, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VerifySettings {
    /// A glob over the edited paths, relative to the project.
    pub files: String,
    /// The shell command; the check passes when it exits with 0.
    pub command: String,
    /// The quiet period after the last edit, in milliseconds.
    pub quiet_ms: u64,
}

impl Default for VerifySettings {
    fn default() -> Self {
        Self {
            files: "*.rs".to_owned(),
            command: "cargo check --message-format short".to_owned(),
            quiet_ms: 1500,
        }
    }
}

/// Where the project's check is.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum CheckStatus {
    /// No check ran yet, or checks are off.
    #[default]
    None,
    /// A check waits for quiet, or runs.
    Pending,
    /// The last check passed.
    Passed,
    /// The last check failed with this many error lines.
    Failed(usize),
}

/// The plugin's state, for the whole project.
#[derive(Resource, Debug)]
pub struct Verify {
    /// Whether edits schedule checks.
    pub enabled: bool,
    /// Where the check is.
    pub status: CheckStatus,
    /// Bumped by every edit, so only the newest wait starts a check.
    scheduled: u64,
    /// The agents that edited since the last pass.
    edited: HashSet<Entity>,
    /// The agents that were sent a failure, and not that it passes again.
    saw_failure: HashSet<Entity>,
}

impl Default for Verify {
    fn default() -> Self {
        Self {
            enabled: true,
            status: CheckStatus::None,
            scheduled: 0,
            edited: HashSet::new(),
            saw_failure: HashSet::new(),
        }
    }
}

/// The quiet period numbered `0` ended.
#[derive(Event)]
struct CheckDue(u64);

/// On the `verify` effect: the command it runs.
#[derive(Component)]
struct CheckRun(String);

/// On a note to an agent: whether it reports a failure.
#[derive(Component)]
struct CheckNote {
    failure: bool,
}

/// On the hold that keeps a note while its agent is idle.
#[derive(Component)]
struct IdleHold;

/// How the check's command ended: the `verify` effect's outcome.
#[derive(Serialize, Deserialize)]
struct Ran {
    status: ExitStatus,
    output: String,
}

/// A settled tool call that changed a watched file schedules a check.
fn on_edit(
    settled: On<Add<Settled>>,
    calls: Query<(&Published, &Outcome, &ForAgent), With<ToolRequest>>,
    settings: Res<VerifySettings>,
    root: Res<ProjectRoot>,
    mut verify: ResMut<Verify>,
    runs: Query<Entity, With<CheckRun>>,
    mut commands: Commands,
) {
    let Ok((published, outcome, &ForAgent(agent))) = calls.get(settled.entity) else {
        return;
    };
    let Ok(Some(changed)) = published.0.get::<FilesChanged>() else {
        return;
    };
    let Ok(files) = Pattern::new(&settings.files) else {
        return;
    };
    let watched = changed.paths().any(|path| {
        let path = Path::new(path);
        files.matches_path(path.strip_prefix(&root.0).unwrap_or(path))
    });
    if !verify.enabled || !watched || !outcome.tool_result().is_success() {
        return;
    }
    verify.edited.insert(agent);
    schedule(&mut verify, &runs, &mut commands, settings.quiet_ms);
}

/// Kills the running check and starts the wait for quiet again.
fn schedule(
    verify: &mut Verify,
    runs: &Query<Entity, With<CheckRun>>,
    commands: &mut Commands,
    quiet_ms: u64,
) {
    for run in runs {
        commands.entity(run).despawn();
    }
    verify.scheduled += 1;
    verify.status = CheckStatus::Pending;
    commands
        .delayed()
        .duration(Duration::from_millis(quiet_ms))
        .trigger(CheckDue(verify.scheduled));
}

/// Starts the check once the newest wait ended.
fn due(
    due: On<CheckDue>,
    verify: Res<Verify>,
    settings: Res<VerifySettings>,
    mut commands: Commands,
) {
    if due.0 != verify.scheduled || !verify.enabled {
        return;
    }
    let kind = EffectKind::Custom {
        kind: VERIFY.into(),
        payload: serde_json::json!({ "command": settings.command }),
    };
    let request = EffectRequest {
        kind,
        handler: None,
    };
    commands.spawn((
        request,
        CheckRun(settings.command.clone()),
        Name::new(VERIFY),
    ));
}

/// Runs the check's command once its effect started.
fn run_check(
    started: On<Add<Started>>,
    runs: Query<&CheckRun>,
    processes: Res<Processes>,
    root: Res<ProjectRoot>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let Ok(CheckRun(command)) = runs.get(started.entity) else {
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
        cut: None,
    };
    let args = ShellArgs::new(command.clone()).timeout_secs(TIMEOUT_SECS);
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    let work = async move {
        match shell.run(args).await {
            Ok(ran) => Outcome::custom(
                serde_json::to_value(Ran {
                    status: ran.status,
                    output: ran.output,
                })
                .unwrap_or_default(),
            ),
            Err(error) => Outcome(Err(ErrorReport::from(error))),
        }
    };
    commands
        .entity(started.entity)
        .insert(Running::answering(pool, &wake, work));
}

/// The check ended: every note not sent yet is withdrawn, and the agents
/// hear the new result.
fn checked(
    settled: On<Add<Settled>>,
    runs: Query<(&CheckRun, &Outcome)>,
    unsent: Query<Entity, (With<CheckNote>, Without<Outcome>)>,
    agents: Query<Has<ActiveTurn>, With<Agent>>,
    mut verify: ResMut<Verify>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((CheckRun(command), outcome)) = runs.get(settled.entity) else {
        return;
    };
    let ran = match &outcome.0 {
        Ok(Answered::Custom { payload }) => serde_json::from_value::<Ran>(payload.clone()).ok(),
        _ => None,
    };
    let Some(ran) = ran else {
        verify.status = CheckStatus::None;
        notices.write(Notice::error(
            None,
            format!("verify: `{command}` could not run"),
        ));
        return;
    };
    for note in &unsent {
        commands.entity(note).despawn();
    }
    let mut send = |agent: Entity, text: String, failure: bool| {
        let Ok(busy) = agents.get(agent) else {
            return;
        };
        let message = Incoming::new(text, DeliveryMode::Note).with_origin(Origin::plugin(VERIFY));
        let note = commands
            .spawn((message, ForAgent(agent), CheckNote { failure }))
            .id();
        if !busy {
            commands.spawn((IdleHold, HoldOf(note)));
        }
    };
    if ran.status.success() {
        verify.status = CheckStatus::Passed;
        verify.edited.clear();
        for &agent in &verify.saw_failure {
            send(agent, format!("`{command}` passes again."), false);
        }
        return;
    }
    let errors: Vec<&str> = error_lines(&ran.output).collect();
    verify.status = CheckStatus::Failed(errors.len());
    let shown: Vec<&str> = match errors.is_empty() {
        true => ran.output.lines().take(ERROR_LINES).collect(),
        false => errors.into_iter().take(ERROR_LINES).collect(),
    };
    let text = format!(
        "`{command}` fails after your edits. Its first errors:\n\n{}",
        shown.join("\n")
    );
    for &agent in &verify.edited {
        send(agent, text.clone(), true);
    }
}

/// The lines of a check's output that report an error, without the
/// summary that the crate does not compile.
fn error_lines(output: &str) -> impl Iterator<Item = &str> {
    output.lines().filter(|line| {
        (line.contains("error:") || line.contains("error["))
            && !line.starts_with("error: could not compile")
            && !line.starts_with("error: aborting")
    })
}

/// Lets go of each held note once its agent is busy, or a message that
/// starts its next turn is about to start: the older note goes in first.
fn release_notes(
    holds: Query<(Entity, &HoldOf), With<IdleHold>>,
    notes: Query<&ForAgent, With<CheckNote>>,
    agents: Query<Has<ActiveTurn>>,
    starting: Query<(&Incoming, &ForAgent), Undecided>,
    mut commands: Commands,
) {
    for (hold, &HoldOf(note)) in &holds {
        let Ok(&ForAgent(agent)) = notes.get(note) else {
            continue;
        };
        let busy = agents.get(agent).unwrap_or_default();
        let turn_starts = starting.iter().any(|(message, &ForAgent(to))| {
            to == agent && message.mode != DeliveryMode::Note && message.command().is_none()
        });
        if busy || turn_starts {
            commands.entity(hold).despawn();
        }
    }
}

/// A note that went into its agent's conversation: the agent saw the
/// failure, or that it passes again.
fn note_sent(
    settled: On<Add<Settled>>,
    notes: Query<(&CheckNote, &ForAgent)>,
    mut verify: ResMut<Verify>,
) {
    let Ok((note, &ForAgent(agent))) = notes.get(settled.entity) else {
        return;
    };
    match note.failure {
        true => verify.saw_failure.insert(agent),
        false => verify.saw_failure.remove(&agent),
    };
}

/// `/verify on|off|now`.
fn command(
    In(args): In<CommandArgs>,
    mut verify: ResMut<Verify>,
    runs: Query<Entity, With<CheckRun>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    match args.args.as_str() {
        "on" => {
            verify.enabled = true;
            notices.write(Notice::info(args.agent, "Checks after edits are on."));
        }
        "off" => {
            verify.enabled = false;
            verify.scheduled += 1;
            verify.status = CheckStatus::None;
            for run in &runs {
                commands.entity(run).despawn();
            }
            notices.write(Notice::info(args.agent, "Checks after edits are off."));
        }
        "now" => {
            verify.enabled = true;
            schedule(&mut verify, &runs, &mut commands, 0);
        }
        _ => {
            notices.write(Notice::error(args.agent, "/verify on|off|now"));
        }
    }
}

/// `check ✓`, `check ✗ 3 errors` or `check …` in the status line.
#[cfg(feature = "tui")]
mod status {
    use rig_harness::prelude::*;
    use rig_tui::{AppStatus, Side, StatusItem, StatusItems, StatusSystems, Tone};

    use super::{CheckStatus, Verify};

    /// After the agents at work; gone early when the line is narrow.
    const CHECK: StatusItem = StatusItem::at(Side::Left, 75, 9);

    pub(super) fn add(app: &mut App) {
        app.init_resource::<AppStatus>().add_systems(
            PostUpdate,
            show.in_set(StatusSystems)
                .run_if(resource_changed::<Verify>),
        );
    }

    fn show(verify: Res<Verify>, mut status: ResMut<AppStatus>) {
        let item = match verify.status {
            CheckStatus::None => CHECK.says("", Tone::Dim),
            CheckStatus::Pending => CHECK.says("check …", Tone::Yellow),
            CheckStatus::Passed => CHECK.says("check ✓", Tone::Green),
            CheckStatus::Failed(0) => CHECK.says("check ✗", Tone::Red),
            CheckStatus::Failed(errors) => {
                CHECK.says(format!("check ✗ {errors} errors"), Tone::Red)
            }
        };
        let items: &mut StatusItems = &mut status.0;
        items.show(item);
    }
}
