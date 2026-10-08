//! The agent's side of the `rig` launcher protocol
//! ([`rig::harness_protocol`]): the launcher's path for rebuilds, its startup
//! notice, and the ready file that tells it this build started.

use std::ffi::OsString;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::error;
use rig::harness_protocol::env;

use crate::core::agent::Notice;
use crate::core::journal::SessionPaths;

/// Shows the launcher's startup notice and writes the ready file.
pub struct LauncherPlugin;

impl Plugin for LauncherPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, launcher_notice)
            .add_systems(Last, signal_ready);
    }
}

/// The launcher that started this agent, if one did.
pub(crate) fn executable() -> Option<OsString> {
    std::env::var_os(env::LAUNCHER).filter(|launcher| !launcher.is_empty())
}

/// Shows the launcher's notice, such as a rollback, at startup.
fn launcher_notice(mut notices: MessageWriter<Notice>) {
    if let Ok(notice) = std::env::var(env::NOTICE)
        && !notice.is_empty()
    {
        notices.write(Notice::info(None, notice));
    }
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
