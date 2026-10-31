//! The agent's side of the `rig` launcher protocol
//! ([`rig_harness_protocol`]): how the agent was started ([`Invoked`]),
//! the launcher's path for rebuilds, the launcher's startup notice, the
//! failed build it started after, and the ready file, in the launcher's
//! run directory ([`LauncherRun`]), that tells it this build started. The
//! agent exits once that launcher is gone, killed alone or not.
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
//! ([`inform_build_failure`]): the model reads why, and where the whole
//! build output is.

use std::io::{ErrorKind, IsTerminal};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::{error, warn};
use rig_harness_protocol::{Home, Invocation, REFUSED_EXIT_CODE, RunDir, env};

use super::cli;
use super::exit::{Exit, ExitReason, reason_from_exit};
use super::session::leave_session;
use crate::describe::{Describing, describing};
use crate::secrets::Secrets;
use crate::settings::Settings;
use rig_core::conversation::Author;
use rig_ecs::agent::Notice;
use rig_ecs::tasks::Wake;

use crate::main_agent::{MainAgentQuery, main_agent};
use rig_ecs::inbox::{Deliver, DeliveryMode};
use rig_ecs::session_log::SessionRestored;

/// The plugin name in the [`Author`] of a failed build's note.
pub const BUILD_ORIGIN: &str = "build";

/// How this process was started, for the plugins that read their part of
/// it: the agent's arguments (`--print` for `rig_basics::print`, `--model` for
/// `rig_basics::models`, the plugins' own options with [`Invoked::flag`] and
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
        !self.args.is_headless() && self.terminal
    }
}

/// The run directory of the launcher that started this agent, from
/// [`env::RUN`]. [`LauncherPlugin`] inserts it unless the app inserted one
/// before.
#[derive(Resource, Clone, Debug)]
pub struct LauncherRun(pub RunDir);

/// Inserts [`Invoked`], [`LauncherRun`] and [`Secrets`], lists the
/// process groups the agent starts for the launcher to kill should the
/// agent die without stopping them, shows the launcher's startup notice,
/// hands a failed build to the primary agent, and writes the ready file.
pub struct LauncherPlugin;

impl Plugin for LauncherPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<Invoked>() {
            app.insert_resource(Invoked::from_env());
        }
        if !app.world().contains_resource::<LauncherRun>()
            && let Some(run) = RunDir::from_env()
        {
            rig_tools::process::record_groups(run.groups());
            app.insert_resource(LauncherRun(run));
        }
        watch_launcher(app);
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

    /// Refuses a run the build cannot take: see [`cli`]. Else tells the
    /// launcher when the run's front takes the terminal's screen
    /// ([`RunDir::screen`]).
    fn cleanup(&self, app: &mut App) {
        if let Some(refusal) = cli::refusal(app) {
            refuse(app, refusal);
        }
        let world = app.world();
        let terminal = !world.contains_resource::<Refused>()
            && world
                .get_resource::<Invoked>()
                .is_some_and(|invoked| invoked.terminal)
            && world
                .get_resource::<cli::Cli>()
                .is_some_and(cli::Cli::terminal);
        if let (true, Some(run)) = (terminal, world.get_resource::<LauncherRun>())
            && let Err(failure) = std::fs::write(run.0.screen(), b"")
        {
            warn!("the launcher will not put the terminal back after the agent: {failure}");
        }
    }
}

/// Exits the app, as a signal to it does, once the launcher in
/// [`LauncherRun`] is gone. A launcher signalled alone, as `kill` and
/// `timeout` do, leaves its agent behind, which would otherwise run on by
/// itself, even editing files under `--watch`. A thread, started as the
/// plugins are built so it also sees a launcher gone before the first
/// frame, waits for the run's lock, which the launcher holds while it
/// lives and the system frees when it dies however it died; the std-only
/// launcher needs no signal handling.
pub(crate) fn watch_launcher(app: &mut App) {
    let Some(run) = app.world().get_resource::<LauncherRun>() else {
        return;
    };
    let gone = LauncherGone::default();
    let (lock, watched) = (run.0.lock(), gone.clone());
    let watching = std::thread::Builder::new()
        .name("rig-launcher-watch".to_owned())
        .spawn(move || {
            if launcher_left(&lock) {
                watched.gone.store(true, Ordering::Release);
                if let Some(wake) = watched.wake.get() {
                    wake.wake();
                }
            }
        });
    match watching {
        Ok(_) => {
            app.insert_resource(gone)
                .add_systems(PostStartup, wake_when_launcher_gone)
                .add_systems(First, exit_when_launcher_gone);
        }
        Err(failure) => warn!("the agent will not exit with its launcher: {failure}"),
    }
}

/// Set by the watching thread once the launcher is gone, which then wakes
/// the loop once the loop's wake is in place.
#[derive(Resource, Clone, Default)]
pub(crate) struct LauncherGone {
    gone: Arc<AtomicBool>,
    wake: Arc<OnceLock<Wake>>,
}

impl LauncherGone {
    /// Whether the launcher is gone.
    pub(crate) fn is_gone(&self) -> bool {
        self.gone.load(Ordering::Acquire)
    }
}

/// Gives the watching thread the loop's wake, and wakes the loop for a
/// launcher already gone.
fn wake_when_launcher_gone(gone: Res<LauncherGone>, wake: Option<Res<Wake>>) {
    if let Some(wake) = wake {
        gone.wake.get_or_init(|| wake.clone());
        if gone.is_gone() {
            wake.wake();
        }
    }
}

/// Waits until the launcher that holds `lock` is gone: whether it is, or
/// `false` when that cannot be told.
fn launcher_left(lock: &Path) -> bool {
    let file = match rig_harness_protocol::lock_file(lock, false) {
        Ok(file) => file,
        Err(failure) if failure.kind() == ErrorKind::NotFound => return true,
        Err(failure) => {
            warn!("the agent will not exit with its launcher: {failure}");
            return false;
        }
    };
    file.lock()
        .inspect_err(|failure| warn!("the agent will not exit with its launcher: {failure}"))
        .is_ok()
}

/// Asks the app to exit once, as for a signal, when the launcher is gone.
fn exit_when_launcher_gone(
    gone: Res<LauncherGone>,
    mut asked: Local<bool>,
    mut commands: Commands,
) {
    if !*asked && gone.is_gone() {
        *asked = true;
        commands.trigger(Exit(ExitReason::Signal));
    }
}

/// Marks a run refused ([`AppCliExt::refuse_run`](super::cli::AppCliExt::refuse_run))
/// from `cleanup` until it exits at the end of its first frame: a front
/// leaves the terminal alone meanwhile.
#[derive(Resource, Debug)]
pub struct Refused;

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
    let stop = if why.ends_with(['.', '?', '!']) {
        ""
    } else {
        "."
    };
    let said = format!("rig: {why}{stop}\n");
    eprint!("{said}");
    if let Some(run) = app.world().get_resource::<LauncherRun>() {
        if let Err(failure) = std::fs::write(run.0.refusal(), said) {
            error!("could not tell the launcher why the run is refused: {failure}");
        }
        // No front takes the screen.
        std::fs::remove_file(run.0.screen()).ok();
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
    commands.trigger(Exit(ExitReason::Failed(REFUSED_EXIT_CODE)));
}

/// The failed build the launcher started after, until the primary agent
/// has it.
#[derive(Resource)]
struct StartBuildFailure(String);

/// The message on a failed build of `what`, for the model: `summary` (the
/// reason and the first errors) and where the whole output is. It does not
/// point at the plugin list, the generated project or rig's source, which
/// only the user changes.
pub fn build_failure_message(what: &str, summary: &str) -> String {
    format!(
        "Building the agent ({what}) failed; the previous build keeps running.\n\n{}\n\n\
         Whole build output: {}",
        summary.trim(),
        Home::from_env().build_log().display(),
    )
}

/// Puts `text`, on a failed build, in `agent`'s conversation from the
/// plugin [`BUILD_ORIGIN`], as a [`DeliveryMode::Inform`]: an idle agent gets it at
/// once without a turn starting (logged as halted, so a restore does not
/// answer it and the user's next message joins it); a busy agent's model
/// reads it with the turn's next call.
pub fn inform_build_failure(commands: &mut Commands, agent: Entity, text: String) {
    let inform = Deliver::new(agent, text, DeliveryMode::Inform);
    commands.trigger(inform.with_author(Author::plugin(BUILD_ORIGIN)));
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
    agents: MainAgentQuery,
    mut commands: Commands,
) {
    let Some(agent) = main_agent(&agents) else {
        return;
    };
    inform_build_failure(
        &mut commands,
        agent,
        build_failure_message("before this start", &failure.0),
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
