//! A clean exit on SIGINT, SIGTERM and SIGHUP, and, for a front whose
//! output is the run's ([`ExitWhenUnread`]), once nobody reads stdout:
//! each exits like `/quit`, so running calls and a `/reload` build are
//! stopped with their process groups and the session logs get their last
//! records. A reader gone, such as `head` after its first line, counts as a
//! signal, as SIGPIPE would.
//!
//! SIGTERM is always heard. SIGINT and SIGHUP are not when the process
//! started with them ignored, as a shell starts a script's `&` job and
//! `nohup` its command: whoever started rig chose that a Ctrl+C at the
//! terminal, or the terminal closing, leaves it running.
//!
//! Closing a terminal sends two SIGHUPs within a millisecond, one from the
//! tty hangup and one from the shell. The signals of the first two seconds
//! ask for one [`Exit`], which a plugin may hold, and a signal after them
//! forces the exit.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::warn;

use rig_ecs::tasks::Wake;

use super::exit::{Exit, ExitReason, SIGNAL_EXIT_CODE};
use super::launcher::LauncherGone;

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

/// Exits the app cleanly on SIGINT, SIGTERM and SIGHUP, and once nobody
/// reads stdout.
pub struct ExitOnSignalPlugin;

impl Plugin for ExitOnSignalPlugin {
    fn build(&self, app: &mut App) {
        if let Err(failure) = listen(on_signal) {
            warn!("signals will not exit cleanly: {failure}");
        }
        app.add_systems(PostStartup, |wake: Option<Res<Wake>>| {
            if let Some(wake) = wake {
                WAKE.get_or_init(|| wake.clone());
            }
        })
        .add_systems(First, exit_on_signal);
        #[cfg(unix)]
        app.add_systems(
            First,
            exit_when_unread.run_if(resource_exists::<ExitWhenUnread>),
        );
    }
}

/// While it exists, the app exits, as for a signal, once nobody reads
/// stdout any more ([`stdout_unread`]): a front whose output is the run's,
/// such as `-p`, inserts it.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct ExitWhenUnread;

/// Starts the thread that calls `heard` for each SIGTERM, and each SIGINT
/// and SIGHUP the process did not start with ignored; the signals it hears.
#[cfg(unix)]
fn listen(heard: fn()) -> std::io::Result<Vec<i32>> {
    use signal_hook::consts::{SIGHUP, SIGINT, SIGTERM};

    let signals: Vec<i32> = [SIGINT, SIGTERM, SIGHUP]
        .into_iter()
        .filter(|&signal| signal == SIGTERM || !ignored(signal))
        .collect();
    let mut incoming = signal_hook::iterator::Signals::new(&signals)?;
    std::thread::Builder::new()
        .name("rig-signals".to_owned())
        .spawn(move || incoming.forever().for_each(|_| heard()))?;
    Ok(signals)
}

/// Starts the handler that calls `heard` for each Ctrl+C.
#[cfg(not(unix))]
fn listen(heard: fn()) -> Result<(), ctrlc::Error> {
    ctrlc::try_set_handler(heard)
}

/// Whether `signal` is ignored.
#[cfg(unix)]
fn ignored(signal: i32) -> bool {
    // SAFETY: an all-zero `sigaction` is a valid value to read into, and a
    // null new action makes `sigaction` only read the current one.
    unsafe {
        let mut current: libc::sigaction = std::mem::zeroed();
        libc::sigaction(signal, std::ptr::null(), &mut current) == 0
            && current.sa_sigaction == libc::SIG_IGN
    }
}

/// What tells whether the run is interrupted already: a signal arrived,
/// or the launcher that started it is gone. A front that waits before the
/// first frame, as `-p` does for what is piped in, stops waiting once it
/// is, and holds the work: the first frame exits.
pub fn interruption(world: &World) -> impl Fn() -> bool + Send + 'static {
    let launcher = world.get_resource::<LauncherGone>().cloned();
    move || SIGNALS.first.get().is_some() || launcher.as_ref().is_some_and(LauncherGone::is_gone)
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

/// Asks the app to exit once, as for a signal, when nobody reads stdout
/// any more. The loop runs a frame at least every second, so a run whose
/// reader left stops within about that.
#[cfg(unix)]
fn exit_when_unread(mut asked: Local<bool>, mut commands: Commands) {
    if !*asked && stdout_unread() {
        *asked = true;
        commands.trigger(Exit(ExitReason::Signal));
    }
}

/// Whether nobody reads stdout any more: see [`ExitWhenUnread`]. Never
/// elsewhere than unix.
pub fn stdout_unread() -> bool {
    #[cfg(unix)]
    return reader_gone(libc::STDOUT_FILENO);
    #[cfg(not(unix))]
    false
}

/// Whether nobody reads `fd` any more: a pipe whose reader closed (POLLERR
/// on Linux, POLLHUP on macOS), or a terminal or socket that hung up. A
/// file, or a descriptor that is not open, has no reader to lose. macOS
/// reports a hangup only to a poll for writing, so it asks for that, and
/// reads only the error bits.
#[cfg(unix)]
fn reader_gone(fd: i32) -> bool {
    let mut poll = libc::pollfd {
        fd,
        events: libc::POLLOUT,
        revents: 0,
    };
    // SAFETY: one valid `pollfd` and no wait.
    let ready = unsafe { libc::poll(&mut poll, 1, 0) };
    ready > 0 && poll.revents & (libc::POLLERR | libc::POLLHUP) != 0
}

#[cfg(test)]
mod tests;
