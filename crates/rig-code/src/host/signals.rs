//! A clean exit on SIGINT, SIGTERM and SIGHUP: a signal exits like
//! `/quit`, so running calls and a `/reload` build are stopped with their
//! process groups and the session is saved.
//!
//! Bevy's `TerminalCtrlCHandlerPlugin` force-exits on a second signal
//! (`references/bevy/crates/bevy_app/src/terminal_ctrl_c_handler.rs:55-59`).
//! Closing a terminal sends two SIGHUPs within a millisecond, one from the
//! tty hangup and one from the shell, so that handler skipped the save and
//! orphaned every child. Here a repeat signal forces the exit only once the
//! clean exit had [`FORCE_AFTER`] to finish. Installing this handler first
//! also makes Bevy's plugin, if `DefaultPlugins` adds it, skip its own
//! (`:93-96`).

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::warn;

/// The exit code of a signalled exit, as a shell reports Ctrl+C.
const SIGNAL_EXIT_CODE: u8 = 130;

/// How long a clean exit may take before a repeat signal forces it.
const FORCE_AFTER: Duration = Duration::from_secs(2);

/// When the first signal arrived.
static SIGNALLED: OnceLock<Instant> = OnceLock::new();

/// Exits the app cleanly on SIGINT, SIGTERM and SIGHUP.
pub struct ExitOnSignalPlugin;

impl Plugin for ExitOnSignalPlugin {
    fn build(&self, app: &mut App) {
        if let Err(failure) = ctrlc::try_set_handler(on_signal) {
            warn!("signals will not exit cleanly: {failure}");
        }
        app.add_systems(First, exit_on_signal);
    }
}

/// Runs on `ctrlc`'s thread for every signal.
fn on_signal() {
    let first = *SIGNALLED.get_or_init(Instant::now);
    if first.elapsed() >= FORCE_AFTER {
        std::process::exit(SIGNAL_EXIT_CODE.into());
    }
}

/// Asks the app to exit once a signal arrived.
fn exit_on_signal(mut exit: MessageWriter<AppExit>) {
    if SIGNALLED.get().is_some() {
        exit.write(AppExit::from_code(SIGNAL_EXIT_CODE));
    }
}
