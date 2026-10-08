//! The agent's side of the `rig` launcher protocol: the reload exit code,
//! the launcher's path for rebuilds, its startup notice, and the ready file
//! that tells it this build started.

use std::ffi::OsString;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::error;

use crate::core::agent::Notice;

/// The exit code that asks the launcher to restart on the staged build. The
/// launcher (`src/launcher/mod.rs` in the `rig` crate) repeats it.
pub const RELOAD_EXIT_CODE: u8 = 75;

/// Shows the launcher's startup notice and writes the ready file.
pub struct LauncherPlugin;

impl Plugin for LauncherPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, launcher_notice)
            .add_systems(Last, signal_ready);
    }
}

/// The launcher that started this agent, from `RIG_LAUNCHER`.
pub fn executable() -> Option<OsString> {
    std::env::var_os("RIG_LAUNCHER")
}

/// Shows the launcher's notice, such as a rollback, at startup.
fn launcher_notice(mut notices: MessageWriter<Notice>) {
    if let Ok(notice) = std::env::var("RIG_NOTICE")
        && !notice.is_empty()
    {
        notices.write(Notice::info(None, notice));
    }
}

/// Tells the launcher this build started: every plugin built, the session
/// restored and the first frame drawn without an exit request.
fn signal_ready(mut signalled: Local<bool>, exits: MessageReader<AppExit>) {
    if *signalled || !exits.is_empty() {
        return;
    }
    *signalled = true;
    if let Some(path) = std::env::var_os("RIG_READY_FILE")
        && let Err(failure) = std::fs::write(&path, b"")
    {
        error!("could not write the ready file: {failure}");
    }
}
