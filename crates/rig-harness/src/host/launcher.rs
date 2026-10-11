//! The agent's side of the `rig` launcher protocol
//! ([`rig::harness_protocol`]): how the agent was started ([`Invoked`]),
//! the launcher's path for rebuilds, the launcher's startup notice, the
//! failed build it started after, and the ready file that tells it this
//! build started.
//!
//! The fronts plugins offer and the options they take
//! ([`cli`]) are checked against how rig was started once
//! every plugin is built: an option no plugin takes, or an interactive
//! terminal no front takes, is refused with exit code 2 once the launcher
//! knows the build started, so it is not rolled back.
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
use rig::harness_protocol::{Home, Invocation, env};

use super::cli::{self, Cli};
use super::session::SessionPaths;
use crate::services::RigHome;
use crate::settings::Settings;
use rig_ecs::agent::Notice;

use crate::primary::{PrimaryQuery, primary};
use rig_ecs::inbox::{Deliver, DeliveryMode, Origin};
use rig_ecs::journal::SessionRestored;

/// The plugin name in the [`Origin`] of a failed build's note.
pub const BUILD_ORIGIN: &str = "build";

/// How this process was started, for the plugins that read their part of
/// it: the agent's arguments (`--print` for `rig-print`, `--model` for
/// `rig-models`, the plugins' own options with [`Invoked::flag`]) and
/// whether stdin is a terminal. [`LauncherPlugin`]
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

/// Inserts [`Invoked`], shows the launcher's startup notice, hands a
/// failed build to the primary agent, and writes the ready file.
pub struct LauncherPlugin;

impl Plugin for LauncherPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<Invoked>() {
            app.insert_resource(Invoked::from_env());
        }
        app.add_systems(Startup, launcher_notice)
            .add_systems(
                Update,
                deliver_build_failure
                    .run_if(resource_exists::<SessionRestored>)
                    .run_if(resource_exists::<StartBuildFailure>),
            )
            .add_systems(Last, signal_ready);
    }

    /// Refuses a run the build cannot take: see [`cli`].
    fn cleanup(&self, app: &mut App) {
        if let Some(refusal) = cli::refusal(app) {
            app.insert_resource(Refused(refusal))
                .add_systems(Last, exit_refused.after(signal_ready));
        }
    }
}

/// Why this run cannot go on.
#[derive(Resource)]
struct Refused(String);

/// Says why the run cannot go on, and exits once the launcher knows that
/// the build started: it is not rolled back.
fn exit_refused(refused: Res<Refused>, mut exit: MessageWriter<AppExit>) {
    eprintln!("rig: {}.", refused.0);
    exit.write(AppExit::from_code(2));
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
    commands.trigger(note.with_origin(Origin::plugin(BUILD_ORIGIN)));
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
/// without an exit request. It also leaves the plugins' options for the
/// launcher's `rig help`.
fn signal_ready(
    mut signalled: Local<bool>,
    exits: MessageReader<AppExit>,
    paths: Option<Res<SessionPaths>>,
    cli: Option<Res<Cli>>,
    home: Option<Res<RigHome>>,
) {
    if *signalled || !exits.is_empty() {
        return;
    }
    *signalled = true;
    let Some(paths) = paths.filter(|_| executable().is_some()) else {
        return;
    };
    if let Err(failure) = std::fs::write(paths.ready(), b"") {
        error!("could not write the ready file: {failure}");
    }
    let flags = home
        .map_or_else(Home::from_env, |home| home.0.clone())
        .flags();
    let usage = cli.map(|cli| cli.usage()).unwrap_or_default();
    if std::fs::read_to_string(&flags).ok().as_deref() != Some(usage.as_str()) {
        std::fs::write(&flags, usage).ok();
    }
}

#[cfg(test)]
mod tests;
