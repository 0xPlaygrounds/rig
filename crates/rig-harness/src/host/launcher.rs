//! The agent's side of the `rig` launcher protocol
//! ([`rig::harness_protocol`]): the launcher's path for rebuilds, the system
//! prompt section on extending the agent with plugins, the launcher's
//! startup notice, the failed build it started after, and the ready file
//! that tells it this build started.
//!
//! A failed build, the launcher's before this start or `/reload`'s, goes
//! into the primary agent's conversation from [`BUILD_ORIGIN`]
//! (`note_build_failure`): the model reads why, and where the whole
//! output and the files the build is made from are.

use std::ffi::OsString;
use std::fmt::Write as _;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::error;
use rig::harness_protocol::{Home, env};

use super::session::SessionPaths;
use rig_ecs::agent::{Notice, PrimaryQuery, primary};
use rig_ecs::inbox::{Deliver, DeliveryMode, Origin, OriginKind};
use rig_ecs::journal::SessionLog;

/// The plugin name in the [`Origin`] of a failed build's note.
pub const BUILD_ORIGIN: &str = "build";

/// Tells the model it is the rig harness and how it extends itself
/// (`extending`), shows the launcher's startup notice, hands a failed build
/// to the primary agent, and writes the ready file.
pub struct LauncherPlugin;

impl Plugin for LauncherPlugin {
    fn build(&self, app: &mut App) {
        // Only an agent the launcher started can rebuild itself with
        // plugins, so only it is told how.
        if let Some(launcher) = executable() {
            app.world_mut().spawn((
                Name::new("prompt:rig_harness"),
                super::extending::section(std::path::Path::new(&launcher)),
            ));
        }
        app.add_systems(Startup, launcher_notice)
            .add_systems(Update, deliver_build_failure)
            .add_systems(Last, signal_ready);
    }
}

/// The failed build the launcher started after, until the primary agent
/// has it.
#[derive(Resource)]
struct StartBuildFailure(String);

/// The [`Origin`] of a failed build's note.
pub fn build_origin() -> Origin {
    Origin {
        kind: OriginKind::Plugin(BUILD_ORIGIN.to_owned()),
        from: None,
        request: None,
    }
}

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

/// Puts `note`, on a failed build, in `agent`'s conversation from
/// [`build_origin`], as a [`DeliveryMode::Note`]: an idle agent gets it at
/// once without a turn starting (logged as halted, so a restore does not
/// answer it and the user's next message joins it); a busy agent's model
/// reads it with the turn's next call.
pub(crate) fn note_build_failure(commands: &mut Commands, agent: Entity, note: String) {
    commands.trigger(Deliver {
        entity: agent,
        text: note,
        origin: build_origin(),
        mode: DeliveryMode::Note,
        attachments: Vec::new(),
    });
}

/// The launcher that started this agent, if one did.
pub(crate) fn executable() -> Option<OsString> {
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
/// once the session is restored and logged.
fn deliver_build_failure(
    failure: Option<Res<StartBuildFailure>>,
    agents: PrimaryQuery,
    log: Option<Res<SessionLog>>,
    mut commands: Commands,
) {
    let Some(failure) = failure else {
        return;
    };
    let Some(agent) = primary(&agents).filter(|_| log.is_some_and(|log| log.is_live())) else {
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
