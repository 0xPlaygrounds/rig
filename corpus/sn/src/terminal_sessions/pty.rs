//! A program in a pseudo-terminal of its own: the pty pair, the program
//! started as the leader of a new session whose controlling terminal is
//! the pty, a thread that reads its output into the [`Screen`], and one
//! that waits for its exit. Neither rig-tools nor the harness starts
//! processes on a terminal, so the few `libc` calls a pty needs are here.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read as _, Write as _};
use std::os::fd::AsRawFd as _;
use std::os::unix::fs::OpenOptionsExt as _;
use std::os::unix::process::{CommandExt as _, ExitStatusExt as _};
use std::path::Path;
use std::process::Stdio;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use futures::channel::mpsc::UnboundedSender;
use rig_tools::process::Launcher;

use crate::terminal_sessions::screen::Screen;

/// How long output must pause for a program to count as waiting.
pub const QUIET: Duration = Duration::from_millis(300);
/// How long a closed program has after its hang-up before it is killed.
pub const GRACE: Duration = Duration::from_secs(2);
/// How long a new program that prints nothing counts as starting.
const SILENT_START: Duration = Duration::from_secs(1);
/// The most output kept since the last send.
const MAX_SINCE: usize = 1 << 20;

/// What a terminal's threads share with the app and the tool calls.
#[derive(Default)]
pub struct State {
    /// The screen, as the output drew it.
    pub screen: Screen,
    /// The output since the last send, as it came.
    pub since: Vec<u8>,
    /// When output last came.
    pub last_output: Option<Instant>,
    /// How the program ended: its code, or 128 plus the signal.
    pub exit: Option<i32>,
    /// Whether every writer closed the terminal.
    pub closed: bool,
    /// Whether the user typed in it since the agent last looked.
    pub user_typed: bool,
}

/// A program on a pseudo-terminal.
pub struct Pty {
    state: Mutex<State>,
    changed: Condvar,
    master: File,
    /// The program's pid, which is also its session's and group's id.
    pub pid: u32,
    /// Set by `term_close`, whose own thread ends it.
    pub closing: AtomicBool,
}

/// Why a wait ended.
pub enum Waited {
    /// The program waits for input.
    Quiet,
    /// `wait_for` matched.
    Matched,
    /// The program exited.
    Exited,
    /// The time ran out.
    TimedOut,
}

impl Pty {
    /// Starts `command` with `sh -c` as `launcher` starts processes, in
    /// `dir`, on a new `cols` × `rows` terminal; `ticks` hears of each
    /// change.
    pub fn start(
        launcher: &Launcher,
        command: &str,
        dir: &Path,
        (cols, rows): (u16, u16),
        ticks: UnboundedSender<()>,
    ) -> io::Result<Arc<Self>> {
        let master = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NOCTTY)
            .open("/dev/ptmx")?;
        let slave_path = slave_of(&master)?;
        resize(&master, cols, rows)?;
        let slave = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NOCTTY)
            .open(&slave_path)?;
        let mut sh = launcher.command("sh");
        sh.args(["-c", command])
            .current_dir(dir)
            .env("TERM", "xterm")
            .env("COLUMNS", cols.to_string())
            .env("LINES", rows.to_string())
            .stdin(Stdio::from(slave.try_clone()?))
            .stdout(Stdio::from(slave.try_clone()?))
            .stderr(Stdio::from(slave));
        // SAFETY: runs in the forked child before `exec`, after the
        // launcher's `setsid`, and only makes stdin, the pty, the new
        // session's controlling terminal with an async-signal-safe ioctl.
        unsafe {
            sh.pre_exec(|| {
                if libc::ioctl(0, libc::TIOCSCTTY, 0) == -1 {
                    Err(io::Error::last_os_error())
                } else {
                    Ok(())
                }
            });
        }
        let mut child = sh.spawn()?;
        // The parent's slave copies go with the command, so reading ends
        // once the program and what it started closed the terminal.
        drop(sh);
        let pty = Arc::new(Self {
            state: Mutex::new(State {
                screen: Screen::new(cols, rows),
                ..State::default()
            }),
            changed: Condvar::new(),
            master: master.try_clone()?,
            pid: child.id(),
            closing: AtomicBool::new(false),
        });
        let reader = Arc::clone(&pty);
        let read_ticks = ticks.clone();
        let mut output = master;
        thread::Builder::new()
            .name("term-output".to_owned())
            .spawn(move || {
                let mut buffer = [0; 8192];
                loop {
                    let read = match output.read(&mut buffer) {
                        Ok(0) => break,
                        Ok(read) => read,
                        Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                        // EIO once every slave closed.
                        Err(_) => break,
                    };
                    let bytes = buffer.get(..read).unwrap_or_default();
                    reader.update(|state| {
                        state.screen.feed(bytes);
                        if state.since.len() < MAX_SINCE {
                            state.since.extend_from_slice(bytes);
                        }
                        state.last_output = Some(Instant::now());
                    });
                    read_ticks.unbounded_send(()).ok();
                }
                reader.update(|state| state.closed = true);
                read_ticks.unbounded_send(()).ok();
            })?;
        let waiter = Arc::clone(&pty);
        thread::Builder::new()
            .name("term-wait".to_owned())
            .spawn(move || {
                let code = child.wait().ok().map_or(-1, |status| {
                    status
                        .code()
                        .or_else(|| status.signal().map(|signal| 128 + signal))
                        .unwrap_or(-1)
                });
                waiter.update(|state| state.exit = Some(code));
                ticks.unbounded_send(()).ok();
            })?;
        Ok(pty)
    }

    /// The shared state, locked.
    pub fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn update(&self, change: impl FnOnce(&mut State)) {
        change(&mut self.lock());
        self.changed.notify_all();
    }

    /// Writes `bytes` as typed; `by_user` marks that the user typed them.
    pub fn send(&self, bytes: &[u8], by_user: bool) -> io::Result<()> {
        {
            let mut state = self.lock();
            if by_user {
                state.user_typed = true;
            } else {
                state.since.clear();
            }
        }
        (&self.master).write_all(bytes)
    }

    /// Waits until the program waits for input (output quiet for
    /// [`QUIET`] and the program itself, not a job it started, holds the
    /// terminal), or `wait_for` matches the output since the last send,
    /// or it exits, or `timeout` passes. Blocking: call it on a thread of
    /// its own. With `first`, quiet counts only once some output came: the
    /// first screen of a program that starts slowly.
    pub fn wait(&self, wait_for: Option<&regex::Regex>, timeout: Duration, first: bool) -> Waited {
        let start = Instant::now();
        let mut state = self.lock();
        loop {
            let now = Instant::now();
            if let Some(pattern) = wait_for
                && pattern.is_match(&Screen::plain(&state.since, state.screen.cols()))
            {
                return Waited::Matched;
            }
            if state.exit.is_some() {
                // A moment for the last output to be read.
                let end = now + QUIET;
                while !state.closed && Instant::now() < end {
                    let left = end.saturating_duration_since(Instant::now());
                    state = self.wait_change(state, left);
                }
                return Waited::Exited;
            }
            let quiet_since = state.last_output.map_or(start, |last| last.max(start));
            let quiet_left = QUIET.saturating_sub(now - quiet_since);
            // A program that prints nothing at first is waiting after a
            // second.
            let started = !first || state.last_output.is_some() || now - start >= SILENT_START;
            if wait_for.is_none() && quiet_left.is_zero() && started && self.holds_terminal() {
                return Waited::Quiet;
            }
            let left = timeout.saturating_sub(now - start);
            if left.is_zero() {
                return Waited::TimedOut;
            }
            let next = if quiet_left.is_zero() {
                // Quiet, but a job of its own runs: look again soon.
                Duration::from_millis(100)
            } else {
                quiet_left
            };
            state = self.wait_change(state, left.min(next));
        }
    }

    fn wait_change<'a>(
        &self,
        state: MutexGuard<'a, State>,
        at_most: Duration,
    ) -> MutexGuard<'a, State> {
        match self.changed.wait_timeout(state, at_most) {
            Ok((state, _)) => state,
            Err(poisoned) => poisoned.into_inner().0,
        }
    }

    /// Whether the terminal's foreground process group is the program's,
    /// not a job it started, such as a command a shell runs. A `sh -c`
    /// that did not exec its command (dash) makes no groups, so a child of
    /// it that leads the foreground group is the program, such as an
    /// interactive shell that took the terminal for itself.
    fn holds_terminal(&self) -> bool {
        // SAFETY: `tcgetpgrp` reads the foreground group of an open
        // terminal descriptor.
        let group = unsafe { libc::tcgetpgrp(self.master.as_raw_fd()) };
        let Ok(group) = u32::try_from(group) else {
            return false;
        };
        let still_sh_c = || {
            fs::read(format!("/proc/{}/cmdline", self.pid))
                .is_ok_and(|line| line.split(|&byte| byte == 0).nth(1) == Some(b"-c"))
        };
        group == self.pid || (stat_field(group, 1) == Some(i64::from(self.pid)) && still_sh_c())
    }

    /// Hangs up on the program: `SIGHUP` to its group and the foreground
    /// job, as a closed terminal does.
    pub fn hang_up(&self) {
        // SAFETY: as in `holds_terminal`.
        let foreground = unsafe { libc::tcgetpgrp(self.master.as_raw_fd()) };
        for group in [foreground, i32::try_from(self.pid).unwrap_or(0)] {
            if group > 0 {
                // SAFETY: `kill` takes plain integers; a negative pid
                // names a process group.
                unsafe {
                    libc::kill(-group, libc::SIGHUP);
                    libc::kill(-group, libc::SIGCONT);
                }
            }
        }
    }

    /// Waits at most `grace` for the program to exit.
    pub fn wait_exit(&self, grace: Duration) -> bool {
        let end = Instant::now() + grace;
        let mut state = self.lock();
        while state.exit.is_none() && Instant::now() < end {
            let left = end.saturating_duration_since(Instant::now());
            state = self.wait_change(state, left);
        }
        state.exit.is_some()
    }

    /// Kills every process of the program's session: it and everything it
    /// started, whatever group each moved to.
    pub fn kill_all(&self) {
        let Ok(session) = i32::try_from(self.pid) else {
            return;
        };
        for pid in in_session(session) {
            // SAFETY: as in `hang_up`, for one process.
            unsafe {
                libc::kill(pid, libc::SIGKILL);
            }
        }
    }

    /// Hangs up, then kills what is left after `grace`.
    pub fn close(&self, grace: Duration) {
        self.hang_up();
        self.wait_exit(grace);
        self.kill_all();
    }
}

/// The processes whose session is `session`.
pub fn in_session(session: i32) -> Vec<i32> {
    let Ok(entries) = fs::read_dir("/proc") else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter_map(|entry| {
            let pid: u32 = entry.file_name().to_str()?.parse().ok()?;
            (stat_field(pid, 3)? == i64::from(session)).then_some(i32::try_from(pid).ok()?)
        })
        .collect()
}

/// Field `index` of process `pid`'s stat after its command: 0 its state,
/// 1 its parent, 2 its group, 3 its session.
fn stat_field(pid: u32, index: usize) -> Option<i64> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let (_, rest) = stat.rsplit_once(')')?;
    rest.split_whitespace().nth(index)?.parse().ok()
}

/// Unlocks the pty `master` and names its slave.
fn slave_of(master: &File) -> io::Result<String> {
    let fd = master.as_raw_fd();
    let mut name = [0 as libc::c_char; 128];
    // SAFETY: `grantpt` and `unlockpt` take an open pty master; `ptsname_r`
    // writes a NUL-terminated name of at most `name.len()` bytes into it.
    let failed = unsafe {
        libc::grantpt(fd) != 0
            || libc::unlockpt(fd) != 0
            || libc::ptsname_r(fd, name.as_mut_ptr(), name.len()) != 0
    };
    if failed {
        return Err(io::Error::last_os_error());
    }
    let bytes: Vec<u8> = name
        .iter()
        .take_while(|&&byte| byte != 0)
        .map(|&byte| byte.to_ne_bytes().first().copied().unwrap_or_default())
        .collect();
    String::from_utf8(bytes).map_err(io::Error::other)
}

/// Tells the terminal, and the program on it, its size.
fn resize(master: &File, cols: u16, rows: u16) -> io::Result<()> {
    let size = libc::winsize {
        ws_row: rows,
        ws_col: cols,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    // SAFETY: `TIOCSWINSZ` reads one `winsize` from the pointer.
    if unsafe { libc::ioctl(master.as_raw_fd(), libc::TIOCSWINSZ, &size) } == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}
