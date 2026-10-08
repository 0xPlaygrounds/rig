//! The handshake with the `rig` launcher, kept out of the agent core: the
//! reload exit code, `RIG_READY_FILE` written after the first full frame
//! so the launcher keeps this binary, and `RIG_NOTICE` shown on start.

use std::path::Path;

use bevy::prelude::*;

use crate::core::{Notice, write_atomic};

/// The exit code that asks the `rig` launcher to start the newest build.
pub const RELOAD_EXIT_CODE: u8 = 75;

/// Talks to the `rig` launcher. Added by [`crate::run`].
pub(crate) struct LauncherPlugin;

impl Plugin for LauncherPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, launcher_notice)
            .add_systems(Last, mark_ready.run_if(run_once));
    }
}

/// After the first full frame, tells the launcher this binary works by
/// creating `RIG_READY_FILE`.
fn mark_ready() {
    if let Some(ready) = std::env::var_os("RIG_READY_FILE").filter(|value| !value.is_empty())
        && let Err(error) = write_atomic(Path::new(&ready), b"ready")
    {
        error!("cannot write the ready file: {error}");
    }
}

/// Shows the launcher's `RIG_NOTICE`, such as a rollback, to the user.
fn launcher_notice(mut notices: MessageWriter<Notice>) {
    if let Ok(text) = std::env::var("RIG_NOTICE")
        && !text.is_empty()
    {
        notices.write(Notice::error(None, text));
    }
}
