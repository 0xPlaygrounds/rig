//! Plugin corpus (held-out): H15: Debounced background check after edits
//! (spec `H15-background-verify.md`).
//!
//! A tool call by any agent that changed a file matching `[verify] files`
//! (default `*.rs`) schedules the project's check (`[verify] command`,
//! default `cargo check --message-format short`) after a quiet period
//! (`[verify] quiet_ms`, default 1500), a [`WakeAt`] that a later edit puts
//! off; the edit also kills a check that runs. The check is a `verify`
//! effect of the app, one at a time, whose command runs as `Processes` say.
//!
//! A failure goes to each agent that edited since the last pass, with the
//! first error lines, and starts no turn. A busy agent gets it as a
//! [`DeliveryMode::Note`], which goes with its next model call; an idle
//! one with its next message from the user, as an attachment added in
//! Prepare. Only the newest result counts: a result withdraws every note
//! not sent yet, and a pass after a failure an agent saw tells it the
//! check passes again. `/verify on|off|now`.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use bevy_tasks::{IoTaskPool, TaskPool};
use glob::Pattern;
use rig_coding_tools::AttachSystems;
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

/// The plugin's stage system: the Prepare edit that adds a note to an idle
/// agent's next message from the user, after the files it names.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VerifySystems;

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
            .configure_sets(
                AgentPass,
                VerifySystems.in_set(Stage::Prepare).after(AttachSystems),
            )
            .add_systems(AgentPass, attach_notes.in_set(VerifySystems))
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
    /// The agents that edited since the last pass.
    edited: HashSet<Entity>,
    /// The agents that were sent a failure, and not that it passes again.
    saw_failure: HashSet<Entity>,
    /// The note each idle agent gets with its next message from the user.
    waiting: HashMap<Entity, CheckNote>,
}

impl Default for Verify {
    fn default() -> Self {
        Self {
            enabled: true,
            status: CheckStatus::None,
            edited: HashSet::new(),
            saw_failure: HashSet::new(),
            waiting: HashMap::new(),
        }
    }
}

/// The wait for quiet: a [`WakeAt`] that each edit puts off.
#[derive(Component)]
struct Quiet;

/// On the `verify` effect: the command it runs.
#[derive(Component)]
struct CheckRun(String);

/// A note about the check, and on a busy agent's note message.
#[derive(Component, Clone, Debug)]
struct CheckNote {
    text: String,
    failure: bool,
}

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
    pending: Query<Entity, Or<(With<Quiet>, With<CheckRun>)>>,
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
    schedule(&mut verify, &pending, &mut commands, settings.quiet_ms);
}

/// Calls off the wait and the check, then waits for quiet again.
fn schedule(
    verify: &mut Verify,
    pending: &Query<Entity, Or<(With<Quiet>, With<CheckRun>)>>,
    commands: &mut Commands,
    quiet_ms: u64,
) {
    for entity in pending {
        commands.entity(entity).despawn();
    }
    verify.status = CheckStatus::Pending;
    let wait = WakeAt::after(Duration::from_millis(quiet_ms));
    commands.spawn((Quiet, wait, Name::new("verify quiet")));
}

/// Starts the check once it was quiet long enough.
fn due(
    due: On<Add<Due>>,
    quiet: Query<(), With<Quiet>>,
    settings: Res<VerifySettings>,
    mut commands: Commands,
) {
    if !quiet.contains(due.entity) {
        return;
    }
    commands.entity(due.entity).despawn();
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

/// Runs the check's command once its effect started; despawning the
/// effect kills it.
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
    mut unsent: Query<(Entity, &mut Decision), (With<CheckNote>, Without<Outcome>)>,
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
    verify.waiting.clear();
    for (note, mut decision) in &mut unsent {
        commands.entity(note).remove::<CheckNote>();
        decision.decide(Outcome::custom("withdrawn"));
    }
    let (note, to) = match ran.status.success() {
        true => {
            verify.status = CheckStatus::Passed;
            verify.edited.clear();
            let text = format!("`{command}` passes again.");
            (
                CheckNote {
                    text,
                    failure: false,
                },
                &verify.saw_failure,
            )
        }
        false => {
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
            (
                CheckNote {
                    text,
                    failure: true,
                },
                &verify.edited,
            )
        }
    };
    let mut idle = Vec::new();
    for &agent in to {
        match agents.get(agent) {
            Ok(true) => {
                let message = Incoming::new(note.text.clone(), DeliveryMode::Note)
                    .with_origin(Origin::plugin(VERIFY));
                commands.spawn((message, ForAgent(agent), note.clone()));
            }
            Ok(false) => idle.push(agent),
            Err(_) => {}
        }
    }
    for agent in idle {
        verify.waiting.insert(agent, note.clone());
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

/// Adds an idle agent's waiting note to its next message from the user.
fn attach_notes(
    mut typed: Query<(&mut Incoming, &ForAgent), (Added<Admitted>, IncomingMessages)>,
    mut verify: ResMut<Verify>,
) {
    for (mut typed, &ForAgent(agent)) in &mut typed {
        if typed.origin.kind != OriginKind::User || typed.command().is_some() {
            continue;
        }
        let Some(note) = verify.waiting.remove(&agent) else {
            continue;
        };
        saw(&mut verify, agent, note.failure);
        typed.attachments.push(Attachment {
            label: VERIFY.to_owned(),
            content: UserContent::text(note.text),
        });
    }
}

/// A busy agent's note went into its conversation.
fn note_sent(
    settled: On<Add<Settled>>,
    notes: Query<(&CheckNote, &ForAgent)>,
    mut verify: ResMut<Verify>,
) {
    if let Ok((note, &ForAgent(agent))) = notes.get(settled.entity) {
        saw(&mut verify, agent, note.failure);
    }
}

/// `agent` was told that the check fails, or that it passes again.
fn saw(verify: &mut Verify, agent: Entity, failure: bool) {
    match failure {
        true => verify.saw_failure.insert(agent),
        false => verify.saw_failure.remove(&agent),
    };
}

/// `/verify on|off|now`.
fn command(
    In(args): In<CommandArgs>,
    mut verify: ResMut<Verify>,
    pending: Query<Entity, Or<(With<Quiet>, With<CheckRun>)>>,
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
            verify.status = CheckStatus::None;
            for entity in &pending {
                commands.entity(entity).despawn();
            }
            notices.write(Notice::info(args.agent, "Checks after edits are off."));
        }
        "now" => {
            verify.enabled = true;
            schedule(&mut verify, &pending, &mut commands, 0);
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
