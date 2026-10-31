//! Child processes started in a session of their own, such as `shell`
//! commands, so they never read the terminal and are killed as a group, and
//! the [`Launcher`] that says how every tool starts them. Every process the
//! agent starts goes through [`Process::start`] (or
//! [`Process::start_with_stdio`]), which lists its group in the ledger, ties
//! it to the call that started it, and hands its output over line by line;
//! it is stopped gently or killed with its whole group.
//!
//! A group stays listed until it is empty, not until its leader exits: a job
//! a command left running in the background (`sleep 60 > log 2>&1 &`) lives
//! on between calls and is killed when rig exits. `setsid` starts one in a
//! session of its own, which nothing here owns. [`record_groups`] keeps the
//! listed groups in a file, for a supervisor that outlives this process to
//! kill should it die without stopping them, as on `SIGKILL`.
//!
//! Work that [`blocking`](crate::blocking) runs is a [`Scope`]: dropping its
//! future stops the processes it started, and refuses those it would start
//! after.

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::ffi::{OsStr, OsString};
use std::io::{self, Read, Write as _};
use std::path::PathBuf;
use std::process::{
    Child, ChildStderr, ChildStdin, ChildStdout, Command, ExitStatus, Output, Stdio,
};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

/// How tools start processes: under a wrapper program (a sandbox such as
/// `bwrap … --`, `nice -n 10`, a container's `exec`), in a directory, with
/// variables set and removed. Every command it builds is [`detach`]ed. A
/// host that hands one launcher to every tool that starts processes
/// ([`Shell`](crate::Shell), a background job, a formatter) puts them all
/// under the same policy.
///
/// ```
/// use rig_tools::process::Launcher;
///
/// let launcher = Launcher::default()
///     .wrapped(["nice", "-n", "10"])
///     .in_dir("/srv/project")
///     .with_env("CI", "1")
///     .without_env(["AGENT_TOKEN"]);
/// let mut command = launcher.command("sh");
/// command.args(["-c", "make test"]);
/// assert_eq!(command.get_program(), "nice");
/// let args: Vec<_> = command.get_args().collect();
/// assert_eq!(args, ["-n", "10", "sh", "-c", "make test"]);
/// assert_eq!(command.get_current_dir(), Some(std::path::Path::new("/srv/project")));
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Launcher {
    /// The program and arguments every command runs under; empty runs the
    /// command itself.
    pub prefix: Vec<OsString>,
    /// The directory commands start in; `None` is this process's.
    pub cwd: Option<PathBuf>,
    /// Variables set for every command.
    pub env: Vec<(OsString, OsString)>,
    /// Variables no command inherits, such as those that make a program
    /// act as the agent that runs it.
    pub unset_env: Vec<OsString>,
}

impl Launcher {
    /// The same launcher, running every command under `prefix`.
    pub fn wrapped(mut self, prefix: impl IntoIterator<Item = impl Into<OsString>>) -> Self {
        self.prefix = prefix.into_iter().map(Into::into).collect();
        self
    }

    /// The same launcher, starting commands in `dir`.
    pub fn in_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cwd = Some(dir.into());
        self
    }

    /// The same launcher, setting `name` to `value` for every command.
    pub fn with_env(mut self, name: impl Into<OsString>, value: impl Into<OsString>) -> Self {
        self.env.push((name.into(), value.into()));
        self
    }

    /// The same launcher, keeping `names` from every command.
    pub fn without_env(mut self, names: impl IntoIterator<Item = impl Into<OsString>>) -> Self {
        self.unset_env.extend(names.into_iter().map(Into::into));
        self
    }

    /// A detached command that runs `program` under this launcher's policy;
    /// add the program's arguments to it.
    pub fn command(&self, program: impl AsRef<OsStr>) -> Command {
        let mut command = match self.prefix.split_first() {
            Some((wrapper, arguments)) => {
                let mut command = Command::new(wrapper);
                command.args(arguments).arg(program);
                command
            }
            None => Command::new(program),
        };
        if let Some(dir) = &self.cwd {
            command.current_dir(dir);
        }
        for name in &self.unset_env {
            command.env_remove(name);
        }
        command.envs(self.env.iter().map(|(name, value)| (name, value)));
        detach(&mut command);
        command
    }

    /// Starts `program` with `args` under this launcher's policy, as
    /// [`command`](Self::command) builds it, with stdin and output as `how`
    /// says.
    ///
    /// ```
    /// # #[cfg(unix)] {
    /// use rig_tools::process::{Launcher, OutputLine, Spawn};
    ///
    /// let how = Spawn { stdin: Some(b"hello\n".to_vec()), ..Spawn::default() };
    /// let mut process = Launcher::default().spawn("cat", ["-"], how)?;
    /// let status = process.wait_for(std::time::Duration::from_secs(5))?;
    /// assert!(status.is_some_and(|status| status.success()));
    /// let lines: Vec<OutputLine> = process.lines().iter().collect();
    /// assert_eq!(lines, [OutputLine::Out("hello\n".to_owned())]);
    /// # }
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn spawn(
        &self,
        program: impl AsRef<OsStr>,
        args: impl IntoIterator<Item = impl AsRef<OsStr>>,
        how: Spawn,
    ) -> io::Result<Process> {
        self.spawn_holding(program, args, how, false)
    }

    /// [`spawn`](Self::spawn), and with `hold` and no input, a stdin that
    /// stays open, with nothing written to it, until the process is
    /// stopped or dropped: so a transport's end on the other machine sees
    /// it close when this end goes, however it goes.
    pub(crate) fn spawn_holding(
        &self,
        program: impl AsRef<OsStr>,
        args: impl IntoIterator<Item = impl AsRef<OsStr>>,
        how: Spawn,
        hold: bool,
    ) -> io::Result<Process> {
        let mut command = self.command(program);
        command.args(args);
        Process::start_holding(command, how, hold)
    }
}

/// How [`Launcher::spawn`] connects a process.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Spawn {
    /// What the process reads on stdin, which is closed after it; `None`
    /// gives it an empty stdin.
    pub stdin: Option<Vec<u8>>,
    /// Whether stderr goes into stdout's pipe, so the two keep their order
    /// and every line arrives as [`OutputLine::Out`].
    pub merge_stderr: bool,
}

/// A piece of a process's output, as [`Process::lines`] hands it over: a
/// line with its `\n`, or the end of the output without one. A line longer
/// than 64 KB comes in pieces, never cut inside a UTF-8 character.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutputLine {
    /// From stdout.
    Out(Vec<u8>),
    /// From stderr.
    Err(Vec<u8>),
}

impl OutputLine {
    /// The bytes, whichever stream they came from.
    pub fn bytes(&self) -> &[u8] {
        match self {
            Self::Out(bytes) | Self::Err(bytes) => bytes,
        }
    }

    /// The text, whichever stream it came from, with bytes that are not
    /// UTF-8 replaced.
    pub fn text(&self) -> Cow<'_, str> {
        String::from_utf8_lossy(self.bytes())
    }
}

type Tag = fn(Vec<u8>) -> OutputLine;

/// How often a waiting [`Process`] is checked.
const POLL: Duration = Duration::from_millis(10);

/// How long [`Process::run`] reads on after the process exited, for output
/// something it left running still writes, and again after a kill.
const PIPE_GRACE: Duration = Duration::from_millis(100);

/// How long a process asked to stop (`SIGTERM`) has to end before it is
/// killed: time for a transport that hangs up the command it runs on
/// another machine when it is stopped to do so.
pub const STOP_GRACE: Duration = Duration::from_secs(2);

/// The longest piece of a line [`forward`] hands over at once.
const MAX_PIECE: usize = 64 * 1024;

impl Process {
    /// Starts `command` as the agent's own: in a session and process group
    /// of its own ([`detach`]), its group listed in the ledger until it is
    /// empty, and part of the current [`Scope`], with stdin and output as
    /// `how` says.
    ///
    /// # Errors
    ///
    /// [`Interrupted`](io::ErrorKind::Interrupted) when the current scope
    /// was stopped, else why it could not start.
    ///
    /// ```
    /// # #[cfg(unix)] {
    /// use std::process::Command;
    /// use rig_tools::process::{Process, Spawn};
    ///
    /// let mut git = Command::new("sh");
    /// git.args(["-c", "printf 'a\\0b'"]);
    /// let output = Process::start(git, Spawn::default())?.output(None)?;
    /// assert_eq!(output.stdout, b"a\0b");
    /// # }
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn start(command: Command, how: Spawn) -> io::Result<Self> {
        Self::start_holding(command, how, false)
    }

    /// [`start`](Self::start), and with `hold` and no input, a stdin that
    /// stays open, with nothing written to it, until the process is
    /// stopped or dropped: so a transport's end on the other machine sees
    /// it close when this end goes, however it goes.
    pub(crate) fn start_holding(mut command: Command, how: Spawn, hold: bool) -> io::Result<Self> {
        command.stdin(match how.stdin {
            Some(_) => Stdio::piped(),
            None if hold => Stdio::piped(),
            None => Stdio::null(),
        });
        let mut pipes: Vec<(Box<dyn Read + Send>, Tag)> = Vec::new();
        if how.merge_stderr {
            // One pipe for both keeps their order.
            let (reader, writer) = io::pipe()?;
            command.stdout(writer.try_clone()?).stderr(writer);
            pipes.push((Box::new(reader), OutputLine::Out));
        } else {
            command.stdout(Stdio::piped()).stderr(Stdio::piped());
        }
        // From here on, an error drops the process, which kills it.
        let (mut process, sender) = Self::owned(command)?;
        let (stdout, stderr) = (process.child.stdout.take(), process.child.stderr.take());
        #[cfg(unix)]
        let (stdout, stderr) = match (stdout, stderr) {
            (Some(stdout), Some(stderr)) => {
                let sender = sender.clone();
                thread::Builder::new()
                    .name("rig-process-output".to_owned())
                    .spawn(move || forward_both(stdout, stderr, &sender))?;
                (None, None)
            }
            apart => apart,
        };
        if let Some(stdout) = stdout {
            pipes.push((Box::new(stdout), OutputLine::Out));
        }
        if let Some(stderr) = stderr {
            pipes.push((Box::new(stderr), OutputLine::Err));
        }
        for (pipe, tag) in pipes {
            let sender = sender.clone();
            thread::Builder::new()
                .name("rig-process-output".to_owned())
                .spawn(move || forward(pipe, tag, &sender))?;
        }
        if let Some(bytes) = how.stdin
            && let Some(mut stdin) = process.child.stdin.take()
        {
            // Written on a thread of its own, so a child that reads slowly,
            // or not at all, never blocks the caller; closed once written.
            thread::Builder::new()
                .name("rig-process-input".to_owned())
                .spawn(move || {
                    stdin.write_all(&bytes).ok();
                })?;
        }
        Ok(process)
    }
}

/// Reads `pipe` until every writer closed it, or the receiver is gone,
/// and sends what it reads on line by line.
///
/// When the receiver stops listening while a process that left the group
/// still holds the pipe, this thread stays blocked in `read` until that
/// process writes again or exits; its next write then finds the receiver
/// gone, the read end is dropped, and the process gets `EPIPE` or
/// `SIGPIPE`, as after any closed pipe.
fn forward(mut pipe: impl Read, tag: Tag, lines: &Sender<OutputLine>) {
    let mut pending = Vec::new();
    let mut buffer = [0; 8192];
    loop {
        let read = match pipe.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        };
        pending.extend_from_slice(buffer.get(..read).unwrap_or_default());
        let mut start = 0;
        while let Some(end) = pending.get(start..).and_then(piece_end) {
            let piece = pending.get(start..start + end).unwrap_or_default();
            if lines.send(tag(piece.to_vec())).is_err() {
                return;
            }
            start += end;
        }
        pending.drain(..start);
    }
    if !pending.is_empty() {
        lines.send(tag(pending)).ok();
    }
}

/// Reads a process's stdout and stderr on one thread, whichever has data
/// first, stdout first when both have, and sends each on line by line as
/// [`forward`] does: so the lines of the two keep the order they were
/// written in, as far as one reader can tell, and a line one writes in
/// pieces is never split by the other's.
#[cfg(unix)]
fn forward_both(
    stdout: std::process::ChildStdout,
    stderr: std::process::ChildStderr,
    lines: &Sender<OutputLine>,
) {
    use std::os::fd::AsRawFd;

    let fds = [stdout.as_raw_fd(), stderr.as_raw_fd()];
    let mut pipes: [(Box<dyn Read>, Tag, Vec<u8>, bool); 2] = [
        (Box::new(stdout), OutputLine::Out, Vec::new(), true),
        (Box::new(stderr), OutputLine::Err, Vec::new(), true),
    ];
    let mut buffer = [0; 8192];
    while pipes.iter().any(|(.., open)| *open) {
        let mut polled = fds.map(|fd| libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        });
        for (poll, (.., open)) in polled.iter_mut().zip(&pipes) {
            if !*open {
                poll.fd = -1;
            }
        }
        // SAFETY: `polled` is an array of two initialized `pollfd`s that
        // outlives the call, and its length is passed with it.
        let ready = unsafe { libc::poll(polled.as_mut_ptr(), 2, -1) };
        if ready < 0 {
            if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                continue;
            }
            break;
        }
        for (poll, (pipe, tag, pending, open)) in polled.iter().zip(&mut pipes) {
            if !*open || poll.revents == 0 {
                continue;
            }
            let read = match pipe.read(&mut buffer) {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Ok(0) | Err(_) => None,
                Ok(read) => Some(read),
            };
            let Some(read) = read else {
                *open = false;
                if !pending.is_empty() && lines.send(tag(std::mem::take(pending))).is_err() {
                    return;
                }
                continue;
            };
            pending.extend_from_slice(buffer.get(..read).unwrap_or_default());
            let mut start = 0;
            while let Some(end) = pending.get(start..).and_then(piece_end) {
                let piece = pending.get(start..start + end).unwrap_or_default();
                if lines.send(tag(piece.to_vec())).is_err() {
                    return;
                }
                start += end;
            }
            pending.drain(..start);
        }
    }
}

/// Where the first piece of `bytes` ends: after its first `\n`, else after
/// [`MAX_PIECE`] bytes, not inside a character; `None` while it is shorter.
fn piece_end(bytes: &[u8]) -> Option<usize> {
    if let Some(at) = bytes.iter().position(|&byte| byte == b'\n') {
        return Some(at + 1);
    }
    let head = bytes.get(..MAX_PIECE)?;
    match std::str::from_utf8(head) {
        Err(error) if error.error_len().is_none() && error.valid_up_to() > 0 => {
            Some(error.valid_up_to())
        }
        _ => Some(head.len()),
    }
}

/// A process [`Process::start`] started, the leader of a process group of
/// its own. Dropping it while it runs kills the group and reaps it.
pub struct Process {
    child: Child,
    lines: Receiver<OutputLine>,
    group: Group,
    /// The machine it is the local end of a transport to, if it is one.
    remote: Option<String>,
    /// Whether it was [handed over](Self::release).
    released: bool,
}

impl Process {
    /// Starts `command` as [`start`](Self::start) does, with its stdin,
    /// stdout and stderr as the command sets them, for a caller that reads
    /// and writes the pipes itself ([`take_stdin`](Self::take_stdin) and
    /// the others); [`lines`](Self::lines) gets nothing.
    ///
    /// # Errors
    ///
    /// As [`start`](Self::start).
    pub fn start_with_stdio(command: Command) -> io::Result<Self> {
        Self::owned(command).map(|(process, _)| process)
    }

    /// Spawns `command` detached, lists its group and adds it to the
    /// current scope, all while the scope cannot stop, so no process is
    /// ever outside both; also the sender of its lines.
    fn owned(mut command: Command) -> io::Result<(Self, Sender<OutputLine>)> {
        detach(&mut command);
        let scope = Scope::current();
        let mut calls = scope.as_ref().map(Scope::calls);
        if calls.as_ref().is_some_and(|calls| calls.stopped) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "the call was stopped before the process started",
            ));
        }
        // The one place the agent's crates start a process.
        #[allow(clippy::disallowed_methods)]
        let spawned = command.spawn();
        // The parent's copies of a shared pipe's write end must close, or
        // reading never ends.
        drop(command);
        let child = spawned?;
        list(child.id());
        let group = Group(Arc::new(AtomicU32::new(child.id())));
        if let Some(calls) = &mut calls {
            calls.groups.retain(Group::running);
            calls.groups.push(group.clone());
        }
        let (sender, lines) = channel();
        let process = Self {
            child,
            lines,
            group,
            remote: None,
            released: false,
        };
        Ok((process, sender))
    }

    /// The process's id, which is also its group's.
    pub fn id(&self) -> u32 {
        self.child.id()
    }

    /// Its stdin, for a process [started with its own
    /// stdio](Self::start_with_stdio) that pipes it; once.
    pub fn take_stdin(&mut self) -> Option<ChildStdin> {
        self.child.stdin.take()
    }

    /// Its stdout, as [`take_stdin`](Self::take_stdin).
    pub fn take_stdout(&mut self) -> Option<ChildStdout> {
        self.child.stdout.take()
    }

    /// Its stderr, as [`take_stdin`](Self::take_stdin).
    pub fn take_stderr(&mut self) -> Option<ChildStderr> {
        self.child.stderr.take()
    }

    /// Hands it over: off the ledger and out of its scope, so nothing here
    /// stops it or its group, now, as it is dropped or when rig exits. For
    /// a process whose work must outlive the call, such as a clipboard
    /// program that leaves a process behind to serve the selection, or a
    /// URL opener that may wait for the browser it starts. It can still be
    /// waited for, which reaps it.
    pub fn release(&mut self) {
        self.released = true;
        self.group.0.store(0, Ordering::SeqCst);
        unlist(self.child.id());
    }

    /// The same process, marked as the local end of a transport that runs
    /// the command on the machine `host`, whose end there stops the command
    /// when this end stops.
    pub fn through(mut self, host: impl Into<String>) -> Self {
        self.remote = Some(host.into());
        self
    }

    /// The machine it runs the command on through a transport, if it was
    /// marked so ([`through`](Self::through)).
    pub fn remote(&self) -> Option<&str> {
        self.remote.as_deref()
    }

    /// Its output, line by line, as it comes. Lines wait here until they
    /// are received; the channel disconnects once every process holding
    /// the output closed it.
    pub fn lines(&self) -> &Receiver<OutputLine> {
        &self.lines
    }

    /// A handle that stops its group from another thread, such as the one
    /// dropping the call that waits for it.
    pub fn group(&self) -> Group {
        self.group.clone()
    }

    /// How it ended, if it has.
    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        let status = self.child.try_wait();
        if !matches!(status, Ok(None)) {
            // Reaped, or about to be: its id may be reused.
            self.group.0.store(0, Ordering::SeqCst);
            if !self.released {
                reaped(self.child.id(), false);
            }
        }
        status
    }

    /// Waits at most `timeout` for it to end, and says how it ended if it
    /// did. Its output keeps arriving at [`lines`](Self::lines) meanwhile.
    /// A timeout too long for the clock, such as `Duration::MAX`, waits
    /// until it ends.
    pub fn wait_for(&mut self, timeout: Duration) -> io::Result<Option<ExitStatus>> {
        let deadline = Instant::now().checked_add(timeout);
        // A process whose output just closed is usually a moment from
        // ending, so the first naps are short.
        let mut nap = Duration::from_millis(1);
        loop {
            if let Some(status) = self.try_wait()? {
                return Ok(Some(status));
            }
            let left = deadline.map_or(POLL, |deadline| {
                deadline.saturating_duration_since(Instant::now())
            });
            if left.is_zero() {
                return Ok(None);
            }
            thread::sleep(left.min(nap));
            nap = (nap * 2).min(POLL);
        }
    }

    /// Runs it to its end, at most `timeout` (`None` waits as long as it
    /// takes), and returns every byte it printed on stdout and on stderr
    /// (with [`Spawn::merge_stderr`], all of it on stdout), read to the
    /// pipes' end, and how it ended. A blocking call: run it on a thread of
    /// its own, such as inside [`blocking`](crate::blocking). Past `timeout`
    /// its group is [stopped](Self::stop) and the error is
    /// [`io::ErrorKind::TimedOut`].
    ///
    /// ```
    /// # #[cfg(unix)] {
    /// use rig_tools::process::{Launcher, Spawn};
    ///
    /// let process = Launcher::default().spawn("sh", ["-c", "echo hi"], Spawn::default())?;
    /// let output = process.output(Some(std::time::Duration::from_secs(5)))?;
    /// assert_eq!(output.stdout, b"hi\n");
    /// assert!(output.status.success());
    /// # }
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn output(mut self, timeout: Option<Duration>) -> io::Result<Output> {
        let deadline = timeout.and_then(|timeout| Instant::now().checked_add(timeout));
        let timed_out = || {
            io::Error::new(
                io::ErrorKind::TimedOut,
                "the process ran past its timeout and was killed",
            )
        };
        let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
        // Every byte, to the pipes' end, as `wait_with_output` reads.
        loop {
            let left = deadline.map_or(POLL, |deadline| {
                deadline.saturating_duration_since(Instant::now())
            });
            match self.lines.recv_timeout(left.min(POLL)) {
                Ok(OutputLine::Out(bytes)) => stdout.extend(bytes),
                Ok(OutputLine::Err(bytes)) => stderr.extend(bytes),
                Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) if left.is_zero() => {
                    self.stop(STOP_GRACE).ok();
                    return Err(timed_out());
                }
                Err(RecvTimeoutError::Timeout) => {}
            }
        }
        let left = deadline.map_or(Duration::MAX, |deadline| {
            deadline.saturating_duration_since(Instant::now())
        });
        match self.wait_for(left)? {
            Some(status) => Ok(Output {
                status,
                stdout,
                stderr,
            }),
            None => {
                self.stop(STOP_GRACE).ok();
                Err(timed_out())
            }
        }
    }

    /// Waits for it to end, at most until `deadline` (`None` waits as long
    /// as it takes), handing each piece of its output to `sink`; `sink`
    /// returning `false` stops the wait. A blocking call.
    ///
    /// One decision ends the output: once `sink` returned `false`, or the
    /// deadline passed, `sink` gets nothing more, while the process runs or
    /// after it exited. After an exit, something it left running that still
    /// holds its output gets a moment (100 ms) to finish, then the rest of
    /// its group is killed, as it is when `sink` stops taking output then
    /// (which ends the wait as [`Ended::Stopped`]); past the deadline or
    /// stopped while it runs, the group is [stopped](Self::stop) with
    /// [`STOP_GRACE`]. Its pipes are read to their end for that moment
    /// after a kill: a process that left the group survives the kill and may
    /// hold them for good. Something a command left running with its output
    /// elsewhere, in its group, lives on until it ends or rig exits.
    pub fn run(
        &mut self,
        deadline: Option<Instant>,
        mut sink: impl FnMut(OutputLine) -> bool,
    ) -> io::Result<Ended> {
        let mut more = true;
        let ended = loop {
            if let Some(status) = self.try_wait()? {
                break Ended::Exited(status);
            }
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break Ended::TimedOut;
            }
            if !more {
                break Ended::Stopped;
            }
            match self.lines.recv_timeout(POLL) {
                Ok(line) => more = sink(line),
                Err(RecvTimeoutError::Timeout) => {}
                // It closed its output but still runs.
                Err(RecvTimeoutError::Disconnected) => {
                    self.wait_for(POLL)?;
                }
            }
        };
        if !matches!(ended, Ended::Exited(_)) {
            self.stop(STOP_GRACE).ok();
            self.read_rest(&mut drop);
            return Ok(ended);
        }
        let more = Cell::new(more);
        let mut take = |line| {
            if more.get() && deadline.is_none_or(|deadline| Instant::now() < deadline) {
                more.set(sink(line));
            }
        };
        if !self.read_rest(&mut take) || !more.get() {
            self.kill().ok();
            self.read_rest(&mut take);
        }
        Ok(if more.get() { ended } else { Ended::Stopped })
    }

    /// Reads its output until the pipes' end, for at most [`PIPE_GRACE`],
    /// handing each piece to `take`; whether the end came.
    fn read_rest(&self, take: &mut impl FnMut(OutputLine)) -> bool {
        let end = Instant::now() + PIPE_GRACE;
        loop {
            match self
                .lines
                .recv_timeout(end.saturating_duration_since(Instant::now()))
            {
                Ok(line) => take(line),
                Err(RecvTimeoutError::Disconnected) => return true,
                Err(RecvTimeoutError::Timeout) => return false,
            }
        }
    }

    /// Closes its stdin if it is still open, asks its group to stop
    /// (`SIGTERM`), waits at most `grace` for it to end, then
    /// [kills](Self::kill) the group. Elsewhere than unix it is killed at
    /// once.
    pub fn stop(&mut self, grace: Duration) -> io::Result<ExitStatus> {
        drop(self.child.stdin.take());
        if self.try_wait()?.is_none() {
            self.group.terminate();
            self.wait_for(grace)?;
        }
        self.kill()
    }

    /// Kills its group (`SIGKILL`), also what is left of it after the
    /// process itself ended, and says how the process ended. Kill within
    /// moments of the end: once the group is empty its id is free again.
    pub fn kill(&mut self) -> io::Result<ExitStatus> {
        self.group.0.store(0, Ordering::SeqCst);
        #[cfg(unix)]
        kill_group_of(self.child.id());
        #[cfg(not(unix))]
        if self.child.try_wait()?.is_none() {
            self.child.kill()?;
        }
        // Off the ledger before the reap frees its id.
        if !self.released {
            reaped(self.child.id(), true);
        }
        self.child.wait()
    }
}

/// How [`Process::run`]'s wait ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ended {
    /// It exited.
    Exited(ExitStatus),
    /// It ran past the deadline, and its group was stopped.
    TimedOut,
    /// The sink stopped the wait, and its group was stopped.
    Stopped,
}

impl Drop for Process {
    fn drop(&mut self) {
        if self.group.0.load(Ordering::SeqCst) != 0 {
            self.kill().ok();
        }
    }
}

/// A [`Process`]'s group, to stop from another thread. It acts only while
/// the process runs: once it was reaped, its id may name another process.
#[derive(Clone, Debug)]
pub struct Group(Arc<AtomicU32>);

impl Group {
    /// Kills the group (`SIGKILL`); elsewhere than unix, nothing.
    pub fn kill(&self) {
        self.signal(Signal::Kill);
    }

    /// Asks the group to stop (`SIGTERM`); elsewhere than unix, nothing.
    pub fn terminate(&self) {
        self.signal(Signal::Terminate);
    }

    /// Whether its process still runs, unreaped: once it does not, the
    /// group is no longer this handle's to signal.
    pub fn running(&self) -> bool {
        self.leader().is_some()
    }

    /// The id of its process, while it runs.
    fn leader(&self) -> Option<u32> {
        Some(self.0.load(Ordering::SeqCst)).filter(|&id| id != 0)
    }

    fn signal(&self, signal: Signal) {
        let leader = self.0.load(Ordering::SeqCst);
        #[cfg(unix)]
        if leader != 0 {
            signal_group(leader, signal);
        }
        #[cfg(not(unix))]
        let _ = (leader, signal);
    }
}

/// Stops a [`Process`]'s group when dropped: kept in a future that waits
/// for the process on another thread, it asks the group to stop
/// (`SIGTERM`) the moment the future is dropped, such as when a call is
/// interrupted or the app quits, and kills it [`STOP_GRACE`] later if it
/// still runs. The thread waiting for the process must go on reaping it
/// meanwhile, as [`Process::run`] does. Work [`blocking`](crate::blocking)
/// runs needs none: its [`Scope`] does this for every process it starts.
pub struct KillOnDrop(pub Group);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        stop_groups(self.0.leader().into_iter().collect());
    }
}

/// Asks each of the groups `ids` to stop (`SIGTERM`) and kills those still
/// listed [`STOP_GRACE`] later: so one whose leader ended on the request
/// while a member ignores it is killed all the same.
fn stop_groups(ids: Vec<u32>) {
    if ids.is_empty() {
        return;
    }
    signal_listed(&ids, Signal::Terminate);
    let later = ids.clone();
    let spawned = thread::Builder::new()
        .name("rig-process-stop".to_owned())
        .spawn(move || {
            thread::sleep(STOP_GRACE);
            signal_listed(&later, Signal::Kill);
        });
    if spawned.is_err() {
        signal_listed(&ids, Signal::Kill);
    }
}

/// Sends `signal` to each of the groups `ids` the ledger still lists: one
/// it took off is empty or killed, and its id may name another group.
fn signal_listed(ids: &[u32], signal: Signal) {
    let ledger = ledger();
    for id in ids.iter().filter(|id| ledger.listed.contains(id)) {
        #[cfg(unix)]
        signal_group(*id, signal);
        #[cfg(not(unix))]
        let _ = (id, signal);
    }
}

/// The signals a [`Group`] sends.
#[derive(Clone, Copy)]
enum Signal {
    Terminate,
    Kill,
}

/// The processes one piece of work starts, such as a tool call that
/// [`blocking`](crate::blocking) runs: [`Process::start`] adds each one
/// started on a thread inside the scope ([`enter`](Self::enter)), and once
/// the scope is [stopped](Self::stop), stops them and starts no more. So a
/// call that was interrupted while its thread still runs launches nothing
/// after, not even an operation queued for a turn. A process whose leader
/// exited is the scope's no more: a job it left running in the background
/// lives on.
///
/// ```
/// # #[cfg(unix)] {
/// use rig_tools::process::{Launcher, Scope, Spawn};
///
/// let scope = Scope::default();
/// scope.stop();
/// let started = scope.enter(|| Launcher::default().spawn("true", [""; 0], Spawn::default()));
/// assert!(started.is_err_and(|error| error.kind() == std::io::ErrorKind::Interrupted));
/// # }
/// ```
#[derive(Clone, Debug, Default)]
pub struct Scope(Arc<Mutex<Calls>>);

#[derive(Debug, Default)]
struct Calls {
    stopped: bool,
    groups: Vec<Group>,
}

thread_local! {
    /// The scope processes started on this thread belong to.
    static CURRENT: RefCell<Option<Scope>> = const { RefCell::new(None) };
}

impl Scope {
    /// The scope of the work running on this thread, if any: for work that
    /// goes on in threads of its own to [`enter`](Self::enter) it there.
    pub fn current() -> Option<Self> {
        CURRENT.with_borrow(Clone::clone)
    }

    /// Runs `work` on this thread inside this scope, then goes back to the
    /// scope before.
    pub fn enter<R>(&self, work: impl FnOnce() -> R) -> R {
        struct Back(Option<Scope>);
        impl Drop for Back {
            fn drop(&mut self) {
                CURRENT.set(self.0.take());
            }
        }
        let _back = Back(CURRENT.replace(Some(self.clone())));
        work()
    }

    /// Stops it: asks the groups of its processes still running to stop
    /// (`SIGTERM`), kills them [`STOP_GRACE`] later, and refuses every
    /// process started in it from now on.
    pub fn stop(&self) {
        let mut calls = self.calls();
        calls.stopped = true;
        let groups = std::mem::take(&mut calls.groups);
        drop(calls);
        stop_groups(groups.iter().filter_map(Group::leader).collect());
    }

    fn calls(&self) -> MutexGuard<'_, Calls> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The variable every process [`Process::start`] starts carries while
/// [`record_groups`] keeps a ledger: the ledger's path.
pub const LEDGER_VARIABLE: &str = "RIG_PROCESS_LEDGER";

/// How often groups whose leader exited are checked for members.
const SWEEP: Duration = Duration::from_secs(1);

/// The groups the agent owns. A group is listed from its start until it is
/// killed or found empty; `left` holds the listed ones whose leader exited
/// with members still in them, checked every [`SWEEP`].
struct Ledger {
    /// The file [`record_groups`] mirrors `listed` in.
    path: Option<PathBuf>,
    listed: BTreeSet<u32>,
    left: BTreeSet<u32>,
    sweeping: bool,
}

static LEDGER: Mutex<Ledger> = Mutex::new(Ledger {
    path: None,
    listed: BTreeSet::new(),
    left: BTreeSet::new(),
    sweeping: false,
});

fn ledger() -> MutexGuard<'static, Ledger> {
    LEDGER.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Ledger {
    /// Mirrors the list in the file, if one is kept; a file that cannot be
    /// written lists nothing new.
    fn write(&self) {
        let Some(path) = &self.path else {
            return;
        };
        let text: String = self.listed.iter().map(|id| format!("{id}\n")).collect();
        let written = path.with_extension("tmp");
        std::fs::write(&written, text)
            .and_then(|()| std::fs::rename(&written, path))
            .ok();
    }

    /// Lists `id` as a group left behind by its leader, and checks it
    /// from now on until it is empty.
    fn keep_left(&mut self, id: u32) {
        self.listed.insert(id);
        self.left.insert(id);
        if !std::mem::replace(&mut self.sweeping, true) {
            let swept = thread::Builder::new()
                .name("rig-process-sweep".to_owned())
                .spawn(sweep);
            if swept.is_err() {
                self.sweeping = false;
            }
        }
    }
}

/// From now on mirrors the groups the agent owns in the file at `path`,
/// one id per line, rewritten whole on each change, and marks every
/// process [`Process::start`] starts with [`LEDGER_VARIABLE`] set to
/// `path`. `adopted` are groups an earlier agent listed there and left
/// running on purpose, such as across a reload: owned from now on, as
/// groups whose leader exited. A supervisor that outlives this process,
/// such as rig's launcher, kills the groups listed once it ended, each only
/// while one of its processes still carries the mark, so an id that names
/// another process by then is left alone.
pub fn record_groups(path: impl Into<PathBuf>, adopted: impl IntoIterator<Item = u32>) {
    let mut ledger = ledger();
    ledger.path = Some(path.into());
    for id in adopted {
        ledger.keep_left(id);
    }
    ledger.write();
}

/// Ends the groups the agent owns whose leader exited, such as a job a
/// command left running in the background, as the agent exits for good:
/// asks each to stop (`SIGTERM`), waits at most [`STOP_GRACE`] for them to
/// empty, then kills what is left (`SIGKILL`). A group whose leader still
/// runs is its [`Process`]'s, which kills it as it is dropped. A job
/// started with `setsid` is in a session of its own and is not owned. A
/// blocking call.
pub fn end_left() {
    let ids: Vec<u32> = ledger().left.iter().copied().collect();
    signal_listed(&ids, Signal::Terminate);
    let deadline = Instant::now() + STOP_GRACE;
    while ids.iter().any(|&id| has_members(id)) && Instant::now() < deadline {
        thread::sleep(POLL);
    }
    signal_listed(&ids, Signal::Kill);
    for id in ids {
        unlist(id);
    }
}

/// The ledger's path, while [`record_groups`] keeps one.
fn ledger_path() -> Option<PathBuf> {
    ledger().path.clone()
}

/// Lists the group a process just started leads.
fn list(id: u32) {
    let mut ledger = ledger();
    if ledger.listed.insert(id) {
        ledger.write();
    }
}

/// Takes the group `id` off the list.
fn unlist(id: u32) {
    let mut ledger = ledger();
    ledger.left.remove(&id);
    if ledger.listed.remove(&id) {
        ledger.write();
    }
}

/// Takes the group of `id`, whose leader is reaped, off the list once it
/// is empty: now when it was `killed` or has no member left, else as soon
/// as a sweep finds it empty.
fn reaped(id: u32, killed: bool) {
    if killed || !has_members(id) {
        unlist(id);
    } else {
        ledger().keep_left(id);
    }
}

/// Checks the groups whose leader exited every [`SWEEP`], taking each off
/// the list once it is empty, until none is left.
fn sweep() {
    loop {
        thread::sleep(SWEEP);
        let mut ledger = ledger();
        let empty: Vec<u32> = ledger
            .left
            .iter()
            .copied()
            .filter(|&id| !has_members(id))
            .collect();
        for id in &empty {
            ledger.left.remove(id);
            ledger.listed.remove(id);
        }
        if !empty.is_empty() {
            ledger.write();
        }
        if ledger.left.is_empty() {
            ledger.sweeping = false;
            return;
        }
    }
}

/// Whether the process group `id` has a member. The kernel never gives a
/// new process the id of a group that exists, so while it does, `id` names
/// that group. Elsewhere than unix there are no groups.
fn has_members(id: u32) -> bool {
    #[cfg(unix)]
    if let Ok(group) = i32::try_from(id) {
        // SAFETY: `kill` takes plain integers; signal 0 checks only that
        // the group exists and may be signalled.
        let sent = unsafe { libc::kill(-group, 0) };
        return sent == 0 || io::Error::last_os_error().raw_os_error() == Some(libc::EPERM);
    }
    let _ = id;
    false
}

/// What a command a tool starts finds for programs that would wait for a
/// person: an editor that fails at once saying what to pass instead, and
/// pagers that print.
pub(crate) const NON_INTERACTIVE: [(&str, &str); 8] = [
    ("GIT_TERMINAL_PROMPT", "0"),
    ("GIT_EDITOR", NO_EDITOR),
    ("GIT_SEQUENCE_EDITOR", NO_EDITOR),
    ("EDITOR", NO_EDITOR),
    ("VISUAL", NO_EDITOR),
    ("PAGER", "cat"),
    ("GIT_PAGER", "cat"),
    ("GH_PAGER", "cat"),
];

/// The editor [`NON_INTERACTIVE`] names, run with `sh` by the programs
/// that start one.
const NO_EDITOR: &str = "echo 'no editor here: pass -m or --no-edit' >&2; false";

/// Starts `command` in a new session, which also makes it the leader of a
/// new process group, with nobody at its terminal: with no controlling
/// terminal, a program that opens `/dev/tty` for a prompt (ssh, git, sudo)
/// fails at once instead of drawing over the view and stopping on a
/// terminal read, and an editor or pager a program would start (`git
/// commit` without `-m`) fails or prints at once. A variable the command
/// already sets keeps its value. It also carries [`LEDGER_VARIABLE`] while
/// a ledger is kept. Detaching twice is detaching once.
pub fn detach(command: &mut Command) {
    let set: BTreeSet<OsString> = command
        .get_envs()
        .map(|(name, _)| name.to_os_string())
        .collect();
    for (name, value) in NON_INTERACTIVE {
        if !set.contains(OsStr::new(name)) {
            command.env(name, value);
        }
    }
    if let Some(ledger) = ledger_path() {
        command.env(LEDGER_VARIABLE, ledger);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // SAFETY: the closure runs in the forked child before `exec` and
        // only calls `getsid`, `getpid` and `setsid`, which are
        // async-signal-safe.
        unsafe {
            command.pre_exec(|| {
                if libc::getsid(0) == libc::getpid() || libc::setsid() != -1 {
                    Ok(())
                } else {
                    Err(std::io::Error::last_os_error())
                }
            });
        }
    }
}

/// Runs `command` in the foreground until it exits: in this process's own
/// group, on its terminal, with the environment the agent has. For a
/// program a person works in, such as their editor; it is waited for, so
/// it has its owner, the caller. Not [detached](detach) and not listed:
/// its group is the agent's own.
///
/// # Errors
///
/// Why it could not start.
pub fn foreground(command: &mut Command) -> io::Result<ExitStatus> {
    #[allow(clippy::disallowed_methods)]
    command.status()
}

/// Kills the process group `child` leads, created by [`detach`];
/// elsewhere, the child alone.
#[cfg(unix)]
pub fn kill_group(child: &mut Child) {
    kill_group_of(child.id());
}

/// Kills the process group led by the process `leader`. Its id names that
/// group while the leader is not reaped or any member lives: the kernel
/// does not reuse the id of a process group that exists. Once the leader
/// was reaped and the group emptied, the id is free again, so callers kill
/// within moments of the reap.
#[cfg(unix)]
pub fn kill_group_of(leader: u32) {
    signal_group(leader, Signal::Kill);
}

/// Sends `signal` to the process group led by the process `leader`.
#[cfg(unix)]
fn signal_group(leader: u32, signal: Signal) {
    let signal = match signal {
        Signal::Terminate => libc::SIGTERM,
        Signal::Kill => libc::SIGKILL,
    };
    if let Ok(group) = i32::try_from(leader) {
        // SAFETY: `kill` takes plain integers; a negative pid names the
        // process group the child leads, created by `detach`.
        unsafe {
            libc::kill(-group, signal);
        }
    }
}

#[cfg(not(unix))]
pub fn kill_group(child: &mut Child) {
    child.kill().ok();
}

#[cfg(test)]
mod tests;
