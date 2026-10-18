//! A clean exit on SIGINT, SIGTERM and SIGHUP: a signal exits like
//! `/quit`, so running calls and a `/reload` build are stopped with their
//! process groups and the session logs get their last records.
//!
//! Bevy's `TerminalCtrlCHandlerPlugin` force-exits on a second signal
//! (`references/bevy/crates/bevy_app/src/terminal_ctrl_c_handler.rs:55-59`).
//! Closing a terminal sends two SIGHUPs within a millisecond, one from the
//! tty hangup and one from the shell, so that handler skipped the last
//! log writes and orphaned every child. Here the signals of the first two
//! seconds ask for one [`Exit`], which a plugin may hold, and a signal
//! after them forces the exit. Installing this handler first also makes
//! Bevy's plugin, if `DefaultPlugins` adds it, skip its own (`:93-96`).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::warn;

use rig_ecs::tasks::Wake;

use super::exit::{Exit, ExitReason, SIGNAL_EXIT_CODE};

/// How long a clean exit may take before a repeat signal forces it; the
/// signals before then ask for one exit.
const FORCE_AFTER: Duration = Duration::from_secs(2);

/// The signals of this process.
static SIGNALS: Signals = Signals::new();

/// When the first signal arrived, and whether the app was asked to exit
/// for it.
struct Signals {
    first: OnceLock<Instant>,
    asked: AtomicBool,
}

impl Signals {
    const fn new() -> Self {
        Self {
            first: OnceLock::new(),
            asked: AtomicBool::new(false),
        }
    }

    /// Notes a signal; whether it forces the exit, coming after the clean
    /// exit had its time.
    fn signal(&self) -> bool {
        self.first.get_or_init(Instant::now).elapsed() >= FORCE_AFTER
    }

    /// Whether a signal came that the app was not asked to exit for yet;
    /// once per process.
    fn take(&self) -> bool {
        self.first.get().is_some() && !self.asked.swap(true, Ordering::AcqRel)
    }
}

/// Wakes the loop for a signal; set at startup, once the loop's own wake
/// is in place.
static WAKE: OnceLock<Wake> = OnceLock::new();

/// Exits the app cleanly on SIGINT, SIGTERM and SIGHUP.
pub struct ExitOnSignalPlugin;

impl Plugin for ExitOnSignalPlugin {
    fn build(&self, app: &mut App) {
        static INSTALLED: OnceLock<()> = OnceLock::new();
        INSTALLED.get_or_init(install);
        app.add_systems(PostStartup, |wake: Option<Res<Wake>>| {
            if let Some(wake) = wake {
                WAKE.get_or_init(|| wake.clone());
            }
        })
        .add_systems(First, exit_on_signal);
    }
}

/// Installs the handler once per process. On Unix, `signal-hook` replaces
/// whatever disposition the process inherited: a program started with
/// SIGINT and SIGTERM ignored, as `nohup` and a shell's `&` leave them,
/// still exits cleanly on them (M-8), where `ctrlc` refuses to install.
#[cfg(unix)]
fn install() {
    use signal_hook::consts::{SIGHUP, SIGINT, SIGTERM};
    match signal_hook::iterator::Signals::new([SIGINT, SIGTERM, SIGHUP]) {
        Ok(mut signals) => {
            let spawned = std::thread::Builder::new()
                .name("signals".into())
                .spawn(move || {
                    for _ in signals.forever() {
                        on_signal();
                    }
                });
            if let Err(failure) = spawned {
                warn!("signals will not exit cleanly: {failure}");
            }
        }
        Err(failure) => warn!("signals will not exit cleanly: {failure}"),
    }
}

#[cfg(not(unix))]
fn install() {
    if let Err(failure) = ctrlc::try_set_handler(on_signal) {
        warn!("signals will not exit cleanly: {failure}");
    }
}

/// Runs on the signal thread for every signal.
fn on_signal() {
    if SIGNALS.signal() {
        std::process::exit(SIGNAL_EXIT_CODE.into());
    }
    if let Some(wake) = WAKE.get() {
        wake.wake();
    }
}

/// Asks the app to exit once for the signals that arrived.
fn exit_on_signal(mut commands: Commands) {
    if SIGNALS.take() {
        commands.trigger(Exit(ExitReason::Signal));
    }
}

#[cfg(test)]
mod tests;
