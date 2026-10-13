//! The agent's side of the `rig` launcher protocol
//! ([`rig::harness_protocol`]): how the agent was started ([`Invoked`]),
//! the launcher's path for rebuilds, the launcher's startup notice, the
//! failed build it started after, and the ready file, in the launcher's
//! run directory ([`LauncherRun`]), that tells it this build started.
//!
//! The fronts plugins offer and the options they take
//! ([`cli`]) are checked against how rig was started once
//! every plugin is built: an option no plugin takes, an unknown command,
//! or an interactive terminal no front takes, is refused with exit code 2
//! once the launcher knows the build started, so it is not rolled back.
//! The secrets the launcher passed on are [`Secrets`].
//!
//! A failed build, the launcher's before this start or `/reload`'s, goes
//! into the primary agent's conversation from [`BUILD_ORIGIN`]
//! ([`note_build_failure`]): the model reads why, and where the whole
//! output and the files the build is made from are.

use std::ffi::OsString;
use std::fmt::Write as _;
use std::io::IsTerminal;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::error;
use rig::harness_protocol::{Home, Invocation, RunDir, env};

use super::cli;
use super::exit::{Exit, ExitReason, reason_from_exit};
use super::session::leave_session;
use crate::describe::{Describing, describing};
use crate::secrets::Secrets;
use crate::settings::Settings;
use rig_core::conversation::Author;
use rig_ecs::agent::Notice;

use crate::primary::{PrimaryQuery, primary};
use rig_ecs::inbox::{Deliver, DeliveryMode};
use rig_ecs::journal::SessionRestored;

/// The plugin name in the [`Author`] of a failed build's note.
pub const BUILD_ORIGIN: &str = "build";

/// How this process was started, for the plugins that read their part of
/// it: the agent's arguments (`--print` for `rig-print`, `--model` for
/// `rig-models`, the plugins' own options with [`Invoked::flag`] and
/// commands with [`Invoked::command`]) and whether stdin is a terminal. [`LauncherPlugin`]
/// inserts it from the process unless the app inserted one before.
#[derive(Resource, Clone, Debug, Default)]
pub struct Invoked {
    /// The agent's arguments.
    pub args: Invocation,
    /// Whether stdin is a terminal.
    pub terminal: bool,
}

impl Invoked {
    /// How this process was started.
    pub fn from_env() -> Self {
        let args = Invocation::from_env().unwrap_or_else(|failure| {
            // The launcher checked them; the binary run alone says why it
            // runs as usual.
            eprintln!("rig: {failure}; starting as usual");
            Invocation::default()
        });
        Self {
            args,
            terminal: std::io::stdin().is_terminal(),
        }
    }

    /// Whether someone sits at the terminal: no `--print` and stdin is a
    /// terminal. The terminal view runs then, `--print` otherwise.
    pub fn interactive(&self) -> bool {
        self.args.print.is_none() && self.terminal
    }
}

/// The run directory of the launcher that started this agent, from
/// [`env::RUN`]. [`LauncherPlugin`] inserts it unless the app inserted one
/// before.
#[derive(Resource, Clone, Debug)]
pub struct LauncherRun(pub RunDir);

/// Inserts [`Invoked`], [`LauncherRun`] and [`Secrets`], shows the
/// launcher's startup notice, hands a failed build to the primary agent,
/// and writes the ready file.
pub struct LauncherPlugin;

impl Plugin for LauncherPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<Invoked>() {
            app.insert_resource(Invoked::from_env());
        }
        if !app.world().contains_resource::<LauncherRun>()
            && let Some(run) = RunDir::from_env()
        {
            app.insert_resource(LauncherRun(run));
        }
        if !app.world().contains_resource::<Secrets>() {
            app.insert_resource(Secrets::from_env());
        }
        if describing().is_some() {
            app.insert_resource(Describing);
        }
        app.add_systems(Startup, launcher_notice)
            .add_systems(
                Update,
                deliver_build_failure
                    .run_if(resource_exists::<SessionRestored>)
                    .run_if(resource_exists::<StartBuildFailure>),
            )
            .add_systems(Last, signal_ready.before(reason_from_exit));
    }

    /// Refuses a run the build cannot take: see [`cli`].
    fn cleanup(&self, app: &mut App) {
        if let Some(refusal) = cli::refusal(app) {
            refuse(app, refusal);
        }
    }
}

/// Marks a run refused, until it exits.
#[derive(Resource)]
struct Refused;

/// Refuses this run, once the app is built: says `why` on stderr now,
/// before any front takes the terminal, and in the launcher's
/// [`RunDir::refusal`], keeps the run out of its session's directory
/// (`leave_session`), so the launcher names the run's own log, and exits
/// with code 2 at the end of the first frame. Only the first refusal is
/// said.
pub(crate) fn refuse(app: &mut App, why: String) {
    if app.world().contains_resource::<Refused>() {
        return;
    }
    leave_session(app.world_mut());
    let stop = if why.ends_with('.') { "" } else { "." };
    let said = format!("rig: {why}{stop}\n");
    eprint!("{said}");
    if let Some(run) = app.world().get_resource::<LauncherRun>()
        && let Err(failure) = std::fs::write(run.0.refusal(), said)
    {
        error!("could not tell the launcher why the run is refused: {failure}");
    }
    app.insert_resource(Refused).add_systems(
        Last,
        exit_refused
            .after(signal_ready)
            .run_if(resource_exists::<Refused>),
    );
}

/// Exits a refused run once the launcher knows that the build started: it
/// is not rolled back.
fn exit_refused(mut commands: Commands) {
    commands.remove_resource::<Refused>();
    commands.trigger(Exit(ExitReason::Failed(2)));
}

/// The failed build the launcher started after, until the primary agent
/// has it.
#[derive(Resource)]
struct StartBuildFailure(String);

/// The note on a failed build of `what`, for the model: `summary` (the
/// reason and the first errors), where the whole output is, and the files
/// the agent is built from.
pub fn build_failure_note(what: &str, summary: &str) -> String {
    let home = Home::from_env();
    let project = home.project();
    let mut note = format!(
        "Building the agent ({what}) failed; the previous build keeps running.\n\n{}\n\n\
         Whole build output: {}\n\
         Plugin list: {}\n\
         Generated project: {} and {}",
        summary.trim(),
        home.build_log().display(),
        home.config().display(),
        project.join("Cargo.toml").display(),
        project.join("src/main.rs").display(),
    );
    if let Some(source) = std::env::var_os("RIG_SOURCE").filter(|source| !source.is_empty()) {
        let _ = write!(
            note,
            "\nrig-harness source (RIG_SOURCE): {}",
            std::path::Path::new(&source).display()
        );
    }
    note
}

/// Puts `note`, on a failed build, in `agent`'s conversation from the
/// plugin [`BUILD_ORIGIN`], as a [`DeliveryMode::Note`]: an idle agent gets it at
/// once without a turn starting (logged as halted, so a restore does not
/// answer it and the user's next message joins it); a busy agent's model
/// reads it with the turn's next call.
pub fn note_build_failure(commands: &mut Commands, agent: Entity, note: String) {
    let note = Deliver::new(agent, note, DeliveryMode::Note);
    commands.trigger(note.with_author(Author::plugin(BUILD_ORIGIN)));
}

/// The launcher that started this agent, if one did: it sets both
/// [`env::LAUNCHER`] and [`env::SESSION`]. A nested agent, run by this
/// agent's shell, inherits only the launcher's path, which the model uses.
pub fn executable() -> Option<OsString> {
    std::env::var_os(env::SESSION).filter(|session| !session.is_empty())?;
    std::env::var_os(env::LAUNCHER).filter(|launcher| !launcher.is_empty())
}

/// Shows the launcher's notice, such as a rollback, and what is wrong with
/// the user's settings, at startup.
fn launcher_notice(
    settings: Option<Res<Settings>>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    if let Ok(notice) = std::env::var(env::NOTICE)
        && !notice.is_empty()
    {
        notices.write(Notice::info(None, notice));
    }
    for problem in settings.iter().flat_map(|settings| settings.problems()) {
        notices.write(Notice::error(None, problem.clone()));
    }
    if let Ok(failure) = std::env::var(env::BUILD_FAILURE)
        && !failure.trim().is_empty()
    {
        commands.insert_resource(StartBuildFailure(failure));
    }
}

/// Puts the launcher's failed build in the primary agent's conversation,
/// once the session is restored.
fn deliver_build_failure(
    failure: Res<StartBuildFailure>,
    agents: PrimaryQuery,
    mut commands: Commands,
) {
    let Some(agent) = primary(&agents) else {
        return;
    };
    note_build_failure(
        &mut commands,
        agent,
        build_failure_note("before this start", &failure.0),
    );
    commands.remove_resource::<StartBuildFailure>();
}

/// Tells the launcher, if one started this agent, that this build started:
/// every plugin built, the session restored and the first frame drawn
/// without a plain `AppExit`. An exit asked for with
/// [`Exit`](super::exit::Exit), which gives its [`ExitReason`] at once, is
/// deliberate, such as a finished replay or a plugin's refusal to start,
/// so it counts as started.
fn signal_ready(
    mut signalled: Local<bool>,
    exits: MessageReader<AppExit>,
    reason: Option<Res<ExitReason>>,
    run: Option<Res<LauncherRun>>,
) {
    if *signalled || (!exits.is_empty() && reason.is_none()) {
        return;
    }
    *signalled = true;
    let Some(run) = run else {
        return;
    };
    if let Err(failure) = std::fs::write(run.0.ready(), b"") {
        error!("could not write the ready file: {failure}");
    }
}

#[cfg(test)]
mod tests;
