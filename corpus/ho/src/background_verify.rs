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
//! [`DeliveryMode::Inform`], which goes with its next model call; an idle
//! one with its next message from the user, as an attachment added in
//! Prepare. A note whose agent's turn would end before another call
//! waits for the user's message instead, so that it stays withdrawable.
//! Only the newest result counts: a result withdraws every note not sent
//! yet, and a pass after a failure an agent saw tells it the check passes
//! again. `/verify on|off|now`.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use bevy_tasks::{IoTaskPool, TaskPool};
use glob::Pattern;
use rig_harness::prelude::*;
use rig_harness::rig_core::effect::{Answered, EffectKind};
use rig_harness::rig_core::error::ErrorReport;
use rig_harness::rig_core::tool::{ExitStatus, FilesChanged};
use rig_harness::rig_tools::{Shell, ShellArgs};
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

/// The plugin's stage systems: in Review, a busy agent's notes wait for
/// the user when its turn would end; in Prepare, the waiting note goes
/// with the user's next message, after the files it names; in Gate, a
/// note to an agent gone idle waits for the user too.
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
            .add_systems(
                AgentPass,
                (
                    before_turn_end.in_set(Stage::Review),
                    attach_notes.in_set(PrepareSystems::Edit),
                    to_idle_agents.in_set(Stage::Gate).after(AskSystems),
                )
                    .in_set(VerifySystems),
            )
            .add_observer(on_edit)
            .add_observer(due)
            .add_observer(run_check)
            .add_observer(checked)
            .add_observer(release)
            .add_observer(note_settled);
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
    /// How many checks ended: the newest result's number.
    round: u64,
}

impl Default for Verify {
    fn default() -> Self {
        Self {
            enabled: true,
            status: CheckStatus::None,
            edited: HashSet::new(),
            saw_failure: HashSet::new(),
            waiting: HashMap::new(),
            round: 0,
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
    /// The result it tells of ([`Verify::round`]).
    round: u64,
}

/// On a message from the user: the note attached to it.
#[derive(Component)]
struct Carries(CheckNote);

/// On a note taken back before its agent's turn ends: the hold that keeps
/// the turn from ending (and so from committing the note) until the
/// note is answered.
#[derive(Component)]
struct Holding(Entity);

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
        ..Shell::default()
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
    verify.round += 1;
    for (note, mut decision) in &mut unsent {
        commands.entity(note).remove::<CheckNote>();
        decision.decide(
            Decider::plugin("ho-background-verify"),
            Outcome::custom("withdrawn"),
        );
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
                    round: verify.round,
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
                    round: verify.round,
                },
                &verify.edited,
            )
        }
    };
    let mut idle = Vec::new();
    for &agent in to {
        match agents.get(agent) {
            Ok(true) => {
                let message = Incoming::new(note.text.clone(), DeliveryMode::Inform)
                    .with_author(Author::plugin(VERIFY));
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

/// A busy agent's turn would end (its end is proposed, and Review is the
/// last stage before the kernel ends it): its notes not sent yet would be
/// committed to its conversation, beyond a later result's reach, so each
/// is taken back and waits for the user's next message, and the turn is
/// held until the kernel applied that.
fn before_turn_end(
    ending: Query<(Entity, &TurnOf), With<TurnOutcome>>,
    mut notes: Query<
        (Entity, &CheckNote, &ForAgent, &mut Decision),
        (With<Started>, Without<Outcome>, IncomingMessages),
    >,
    mut verify: ResMut<Verify>,
    mut commands: Commands,
) {
    let ending: HashMap<Entity, Entity> = ending.iter().map(|(turn, of)| (of.0, turn)).collect();
    for (entity, note, &ForAgent(agent), mut decision) in &mut notes {
        let Some(&turn) = ending.get(&agent) else {
            continue;
        };
        if wait_for_user(
            &mut verify,
            &mut commands,
            entity,
            agent,
            note,
            &mut decision,
        ) {
            let hold = commands
                .spawn((HoldOf(turn), Name::new("verify note")))
                .id();
            commands.entity(entity).insert(Holding(hold));
        }
    }
}

/// A note answered: the turn it held may end (the kernel despawned the
/// hold already if the turn carried on).
fn release(answered: On<Add<Outcome>>, holding: Query<&Holding>, mut commands: Commands) {
    if let Ok(&Holding(hold)) = holding.get(answered.entity) {
        commands.entity(hold).try_despawn();
    }
}

/// A note about to go to an agent whose turn ended meanwhile would be
/// committed at once: it waits for the user's next message instead.
fn to_idle_agents(
    mut notes: Query<
        (Entity, &CheckNote, &ForAgent, &mut Decision),
        (ReadyToStart, IncomingMessages),
    >,
    busy: Query<Has<ActiveTurn>>,
    mut verify: ResMut<Verify>,
    mut commands: Commands,
) {
    for (entity, note, &ForAgent(agent), mut decision) in &mut notes {
        if !busy.get(agent).unwrap_or(true) {
            wait_for_user(
                &mut verify,
                &mut commands,
                entity,
                agent,
                note,
                &mut decision,
            );
        }
    }
}

/// Takes the note message `entity` back; its note waits for the user's
/// next message to `agent`. Whether it was not taken already.
fn wait_for_user(
    verify: &mut Verify,
    commands: &mut Commands,
    entity: Entity,
    agent: Entity,
    note: &CheckNote,
    decision: &mut Decision,
) -> bool {
    if !decision.decide(
        Decider::plugin("ho-background-verify"),
        Outcome::custom("waits for the user"),
    ) {
        return false;
    }
    commands.entity(entity).remove::<CheckNote>();
    verify.waiting.insert(agent, note.clone());
    true
}

/// Adds an idle agent's waiting note to its next message from the user.
fn attach_notes(
    mut typed: Query<(Entity, &mut Incoming, &ForAgent), (Added<Admitted>, IncomingMessages)>,
    mut verify: ResMut<Verify>,
    mut commands: Commands,
) {
    for (entity, mut typed, &ForAgent(agent)) in &mut typed {
        let note = typed.mode != DeliveryMode::Inform && typed.command().is_none();
        if typed.author.kind != AuthorKind::User || !note {
            continue;
        }
        let Some(note) = verify.waiting.remove(&agent) else {
            continue;
        };
        let attachment = Attachment {
            label: VERIFY.to_owned(),
            content: UserContent::text(note.text.clone()),
        };
        typed.parts.insert(0, MessagePart::Attachment(attachment));
        commands.entity(entity).insert(Carries(note));
    }
}

/// A note went into an agent's conversation, or the user's message that
/// carries one did not (it was taken over, or handed back at a turn's
/// end), and the note waits again unless a newer result came.
fn note_settled(
    settled: On<Add<Settled>>,
    notes: Query<(
        Option<&CheckNote>,
        Option<&Carries>,
        Has<DeliveredIn>,
        &ForAgent,
    )>,
    mut verify: ResMut<Verify>,
) {
    let Ok((note, carries, delivered, &ForAgent(agent))) = notes.get(settled.entity) else {
        return;
    };
    match (note, carries) {
        (Some(note), _) => saw(&mut verify, agent, note.failure),
        (None, Some(Carries(note))) if delivered => saw(&mut verify, agent, note.failure),
        (None, Some(Carries(note))) if note.round == verify.round => {
            verify.waiting.entry(agent).or_insert_with(|| note.clone());
        }
        _ => {}
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
    const CHECK: StatusItem = StatusItem::new("ho-background-verify", Side::Left, 75, 9);

    pub(super) fn add(app: &mut App) {
        app.init_resource::<AppStatus>().add_systems(
            PostUpdate,
            show.in_set(StatusSystems::Show)
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
