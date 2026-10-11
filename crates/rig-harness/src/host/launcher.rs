//! The agent's side of the `rig` launcher protocol
//! ([`rig::harness_protocol`]): how the agent was started ([`Invoked`]),
//! the launcher's path for rebuilds, the launcher's startup notice, the
//! failed build it started after, and the ready file that tells it this
//! build started.
//!
//! How rig was started picks its front: the terminal view (`rig-tui`) runs
//! on an interactive terminal, `--print` (`rig-print`) with `-p` or with
//! stdin piped in. A build without `rig-tui` started on an interactive
//! terminal says so and exits.
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

use super::session::SessionPaths;
use crate::PluginSource;
use rig_ecs::agent::{Notice, PrimaryQuery, primary};
use rig_ecs::inbox::{Deliver, DeliveryMode, Origin};
use rig_ecs::journal::SessionRestored;

/// The plugin name in the [`Origin`] of a failed build's note.
pub const BUILD_ORIGIN: &str = "build";

/// The terminal view's package, the front of an interactive run.
const TERMINAL_VIEW: &str = "rig-tui";

/// How this process was started, for the plugins that read their part of
/// it: the agent's arguments (`--print` for `rig-print`, `--model` for
/// `rig-models`) and whether stdin is a terminal. [`LauncherPlugin`]
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

    /// Ends an interactive run of a build without the terminal view.
    fn cleanup(&self, app: &mut App) {
        let world = app.world_mut();
        let interactive = world
            .get_resource::<Invoked>()
            .is_some_and(Invoked::interactive);
        let mut plugins = world.query::<&PluginSource>();
        if interactive
            && !plugins
                .iter(world)
                .any(|plugin| plugin.krate == TERMINAL_VIEW)
        {
            app.add_systems(Last, exit_without_front.after(signal_ready));
        }
    }
}

/// Says that nothing shows an interactive run, and exits once the launcher
/// knows that the build started: it is not rolled back.
fn exit_without_front(mut exit: MessageWriter<AppExit>) {
    eprintln!(
        "rig: this build has no front for an interactive terminal: enable {TERMINAL_VIEW} \
         (`rig plugin add rig_tui::TuiPlugin --crate {TERMINAL_VIEW}`), or run with -p or \
         with stdin piped in."
    );
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

/// Shows the launcher's notice, such as a rollback, at startup.
fn launcher_notice(mut notices: MessageWriter<Notice>, mut commands: Commands) {
    if let Ok(notice) = std::env::var(env::NOTICE)
        && !notice.is_empty()
    {
        notices.write(Notice::info(None, notice));
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
/// without an exit request.
fn signal_ready(
    mut signalled: Local<bool>,
    exits: MessageReader<AppExit>,
    paths: Option<Res<SessionPaths>>,
) {
    if *signalled || !exits.is_empty() {
        return;
    }
    *signalled = true;
    if let Some(paths) = paths.filter(|_| executable().is_some())
        && let Err(failure) = std::fs::write(paths.ready(), b"")
    {
        error!("could not write the ready file: {failure}");
    }
}
