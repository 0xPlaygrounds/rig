//! While the agent runs, a signal that would end the launcher (SIGTERM,
//! SIGINT, SIGHUP; Ctrl+C elsewhere than unix) is passed on to the agent,
//! and the launcher waits for it, at most [`STOP_WAIT`], before it ends:
//! the agent's last lines come before the shell's prompt, and a script's
//! `wait` returns once the agent is gone. On unix the launcher then ends as
//! the signal would have ended it ([`end_as_heard`]). A signal the launcher
//! was started ignoring, as under `nohup`, stays ignored.

use std::process::Child;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// How long the agent has to end once told to: its own clean exit takes
/// at most a few seconds, and an ACP router's at most about eleven, while
/// its sessions answer and end.
pub(super) const STOP_WAIT: Duration = Duration::from_secs(15);

/// How long the launcher's own closing output may take after a signal.
const CLOSE_WAIT: Duration = Duration::from_secs(1);

/// Runs `close`, the launcher's closing output after a signal, for at most
/// [`CLOSE_WAIT`]: a terminal or pipe whose reader stopped would otherwise
/// hold it.
pub(super) fn bounded(close: impl FnOnce() + Send + 'static) {
    let (done, closed) = std::sync::mpsc::channel();
    let spawned = std::thread::Builder::new().spawn(move || {
        close();
        done.send(()).ok();
    });
    if spawned.is_ok() {
        closed.recv_timeout(CLOSE_WAIT).ok();
    }
}

/// The first signal heard while the agent ran; 0 while none was.
static HEARD: AtomicI32 = AtomicI32::new(0);

/// How many ending signals were heard while the agent ran.
static TIMES: AtomicUsize = AtomicUsize::new(0);

/// The signal heard while the agent ran, if one was.
pub(super) fn heard() -> Option<i32> {
    Some(HEARD.load(Ordering::SeqCst)).filter(|signal| *signal != 0)
}

/// Passes each heard signal on to the agent it watches, as a terminal
/// would (a repeat ends an agent's clean exit at once), and kills the agent
/// if it has not ended [`STOP_WAIT`] after the first.
pub(super) struct Watch {
    told: Option<Instant>,
    /// How many signals it passed on.
    passed: usize,
    killed: bool,
    #[cfg(unix)]
    replaced: Vec<(libc::c_int, libc::sigaction)>,
}

impl Watch {
    /// Hears the ending signals until dropped; if one was heard by then,
    /// they stay heard, so the launcher's cleanup is not cut short.
    pub(super) fn start() -> Self {
        #[cfg(not(any(unix, target_family = "wasm")))]
        console::listen();
        Self {
            told: None,
            passed: 0,
            killed: false,
            #[cfg(unix)]
            replaced: unix::listen(),
        }
    }

    /// Called as the launcher waits for `child`: tells it of each signal
    /// heard, and kills it once it had [`STOP_WAIT`] to end.
    pub(super) fn pass_on(&mut self, child: &mut Child) {
        let Some(signal) = heard() else {
            return;
        };
        let times = TIMES.load(Ordering::SeqCst).max(1);
        if self.passed < times {
            self.passed = times;
            #[cfg(unix)]
            unix::tell(child, signal);
            #[cfg(not(unix))]
            let _ = signal;
        }
        let told = *self.told.get_or_insert_with(Instant::now);
        if !self.killed && told.elapsed() >= STOP_WAIT {
            self.killed = true;
            child.kill().ok();
        }
    }
}

impl Drop for Watch {
    fn drop(&mut self) {
        #[cfg(unix)]
        if heard().is_none() {
            unix::restore(&self.replaced);
        }
        #[cfg(not(any(unix, target_family = "wasm")))]
        console::WAITING.store(false, Ordering::SeqCst);
    }
}

/// Ends the launcher as the signal heard while the agent ran would have,
/// after its cleanup; returns when none was, or elsewhere than unix.
pub(super) fn end_as_heard() {
    #[cfg(unix)]
    if let Some(signal) = heard() {
        unix::raise(signal);
    }
}

#[cfg(unix)]
mod unix {
    use std::process::Child;
    use std::sync::atomic::Ordering;

    use super::{HEARD, TIMES};

    /// The signals a terminal or a supervisor ends the launcher with.
    const ENDING: [libc::c_int; 3] = [libc::SIGTERM, libc::SIGINT, libc::SIGHUP];

    /// Notes the first ending signal and counts each; only atomics, which
    /// are async-signal-safe.
    extern "C" fn on_signal(signal: libc::c_int) {
        let _ = HEARD.compare_exchange(0, signal, Ordering::SeqCst, Ordering::SeqCst);
        TIMES.fetch_add(1, Ordering::SeqCst);
    }

    /// Hears each ending signal the launcher was not started ignoring; the
    /// actions it replaced.
    pub(super) fn listen() -> Vec<(libc::c_int, libc::sigaction)> {
        let mut replaced = Vec::new();
        for signal in ENDING {
            // SAFETY: plain calls on zeroed actions this function owns; the
            // handler only stores to an atomic.
            unsafe {
                let mut before: libc::sigaction = std::mem::zeroed();
                libc::sigaction(signal, std::ptr::null(), &raw mut before);
                if before.sa_sigaction == libc::SIG_IGN {
                    continue;
                }
                let mut action: libc::sigaction = std::mem::zeroed();
                action.sa_sigaction = on_signal as *const () as libc::sighandler_t;
                action.sa_flags = libc::SA_RESTART;
                libc::sigemptyset(&raw mut action.sa_mask);
                libc::sigaction(signal, &raw const action, std::ptr::null_mut());
                replaced.push((signal, before));
            }
        }
        replaced
    }

    /// Puts back the actions [`listen`] replaced.
    pub(super) fn restore(replaced: &[(libc::c_int, libc::sigaction)]) {
        for (signal, before) in replaced {
            // SAFETY: puts back an action read by `listen`.
            unsafe {
                libc::sigaction(*signal, before, std::ptr::null_mut());
            }
        }
    }

    /// Sends `signal` to `child`.
    pub(super) fn tell(child: &Child, signal: libc::c_int) {
        if let Ok(pid) = libc::pid_t::try_from(child.id()) {
            // SAFETY: `kill` takes plain integers.
            unsafe {
                libc::kill(pid, signal);
            }
        }
    }

    /// Ends the process by `signal`'s default action.
    pub(super) fn raise(signal: libc::c_int) {
        // SAFETY: plain calls; the default action of an ending signal ends
        // the process.
        unsafe {
            libc::signal(signal, libc::SIG_DFL);
            let mut unblocked: libc::sigset_t = std::mem::zeroed();
            libc::sigemptyset(&raw mut unblocked);
            libc::sigaddset(&raw mut unblocked, signal);
            libc::pthread_sigmask(
                libc::SIG_UNBLOCK,
                &raw const unblocked,
                std::ptr::null_mut(),
            );
            libc::raise(signal);
        }
    }
}

/// Ctrl+C reaches every process of the console, the agent too: while the
/// launcher waits for the agent it notes Ctrl+C and waits on; otherwise it
/// ends as Ctrl+C ends it.
#[cfg(not(any(unix, target_family = "wasm")))]
mod console {
    use std::sync::atomic::{AtomicBool, Ordering};

    use rig_harness_protocol::SIGNAL_EXIT_CODE;

    use super::HEARD;

    /// Whether a [`super::Watch`] waits for the agent.
    pub(super) static WAITING: AtomicBool = AtomicBool::new(false);

    pub(super) fn listen() {
        static SET: std::sync::Once = std::sync::Once::new();
        WAITING.store(true, Ordering::SeqCst);
        SET.call_once(|| {
            let _ = ctrlc::set_handler(|| {
                if !WAITING.load(Ordering::SeqCst) {
                    std::process::exit(SIGNAL_EXIT_CODE.into());
                }
                // Not a signal number: any nonzero value says one came.
                let _ = HEARD.compare_exchange(0, 2, Ordering::SeqCst, Ordering::SeqCst);
            });
        });
    }
}

#[cfg(test)]
mod tests;
