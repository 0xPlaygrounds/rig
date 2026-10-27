//! Child processes started in a session of their own, such as `shell`
//! commands, so they never read the terminal and are killed as a group, and
//! the [`Launcher`] that says how every tool starts them. A [`Process`] it
//! [spawns](Launcher::spawn) takes its input, hands its output over line
//! by line, and is stopped gently or killed with its whole group.

use std::ffi::{OsStr, OsString};
use std::io::{self, Read, Write as _};
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Mutex, MutexGuard};
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
        command.args(args).stdin(match how.stdin {
            Some(_) => Stdio::piped(),
            None if hold => Stdio::piped(),
            None => Stdio::null(),
        });
        let (sender, lines) = channel();
        let mut pipes: Vec<(Box<dyn Read + Send>, Tag)> = Vec::new();
        if how.merge_stderr {
            // One pipe for both keeps their order.
            let (reader, writer) = io::pipe()?;
            command.stdout(writer.try_clone()?).stderr(writer);
            pipes.push((Box::new(reader), OutputLine::Out));
        } else {
            command.stdout(Stdio::piped()).stderr(Stdio::piped());
        }
        let spawned = command.spawn();
        // The parent's copies of a shared pipe's write end must close, or
        // reading never ends.
        drop(command);
        let child = spawned?;
        let group = Group(Arc::new(Mutex::new(GroupState {
            leader: child.id(),
            stopping: false,
        })));
        // From here on, an error drops the process, which kills it.
        let mut process = Process {
            child,
            lines,
            group,
            remote: None,
        };
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
/// than 64 KB comes in pieces. Bytes that are not UTF-8 are replaced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutputLine {
    /// From stdout.
    Out(String),
    /// From stderr.
    Err(String),
}

impl OutputLine {
    /// The text, whichever stream it came from.
    pub fn text(&self) -> &str {
        match self {
            Self::Out(text) | Self::Err(text) => text,
        }
    }
}

type Tag = fn(String) -> OutputLine;

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
            if lines
                .send(tag(String::from_utf8_lossy(piece).into_owned()))
                .is_err()
            {
                return;
            }
            start += end;
        }
        pending.drain(..start);
    }
    if !pending.is_empty() {
        lines
            .send(tag(String::from_utf8_lossy(&pending).into_owned()))
            .ok();
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
                if !pending.is_empty() {
                    let text = String::from_utf8_lossy(pending).into_owned();
                    pending.clear();
                    if lines.send(tag(text)).is_err() {
                        return;
                    }
                }
                continue;
            };
            pending.extend_from_slice(buffer.get(..read).unwrap_or_default());
            let mut start = 0;
            while let Some(end) = pending.get(start..).and_then(piece_end) {
                let piece = pending.get(start..start + end).unwrap_or_default();
                if lines
                    .send(tag(String::from_utf8_lossy(piece).into_owned()))
                    .is_err()
                {
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

/// A process [`Launcher::spawn`] started, the leader of a process group of
/// its own. Dropping it while it runs kills the group and reaps it.
pub struct Process {
    child: Child,
    lines: Receiver<OutputLine>,
    group: Group,
    /// The machine it is the local end of a transport to, if it is one.
    remote: Option<String>,
}

impl Process {
    /// The process's id, which is also its group's.
    pub fn id(&self) -> u32 {
        self.child.id()
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
        let mut group = self.group.state();
        #[cfg(unix)]
        if group.leader != 0 && group.stopping {
            if !exited_unreaped(group.leader)? {
                return Ok(None);
            }
            // The unreaped leader reserves the id until all descendants have
            // been signalled, even when they closed both output pipes.
            signal_group(group.leader, Signal::Kill);
        }
        let status = self.child.try_wait();
        if !matches!(status, Ok(None)) {
            group.leader = 0;
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
    /// takes), and returns what it printed on stdout and on stderr (with
    /// [`Spawn::merge_stderr`], all of it on stdout) and how it ended. A
    /// blocking call: run it on a thread of its own, such as inside
    /// [`blocking`](crate::blocking). Past `timeout` the error is
    /// [`io::ErrorKind::TimedOut`]; otherwise as [`run`](Self::run).
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
        let (mut stdout, mut stderr) = (Vec::new(), Vec::new());
        let ended = self.run(deadline, |line| {
            match line {
                OutputLine::Out(text) => stdout.extend(text.into_bytes()),
                OutputLine::Err(text) => stderr.extend(text.into_bytes()),
            }
            true
        })?;
        match ended {
            Ended::Exited(status) => Ok(Output {
                status,
                stdout,
                stderr,
            }),
            Ended::TimedOut | Ended::Stopped => Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "the process ran past its timeout and was killed",
            )),
        }
    }

    /// Waits for it to end, at most until `deadline` (`None` waits as long
    /// as it takes), handing each piece of its output to `sink`; `sink`
    /// returning `false` stops the wait. A blocking call.
    ///
    /// After an exit, something it left running that still holds its
    /// output gets a moment (100 ms) to finish, then the rest of its group
    /// is killed; past the deadline or stopped, the group is
    /// [stopped](Self::stop) with [`STOP_GRACE`]. After a kill, reading
    /// goes on for that moment: a process that left the group survives the
    /// kill and may hold the pipe for good.
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
            self.read_rest(&mut sink);
        } else if !self.read_rest(&mut sink) {
            self.kill().ok();
            self.read_rest(&mut sink);
        }
        Ok(ended)
    }

    /// Hands output to `sink` until the pipe's end, for at most
    /// [`PIPE_GRACE`]; whether the end came.
    fn read_rest(&self, sink: &mut impl FnMut(OutputLine) -> bool) -> bool {
        let deadline = Instant::now() + PIPE_GRACE;
        loop {
            match self
                .lines
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            {
                Ok(line) => {
                    sink(line);
                }
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
        let mut group = self.group.state();
        #[cfg(unix)]
        kill_group_of(self.child.id());
        group.leader = 0;
        #[cfg(not(unix))]
        if self.child.try_wait()?.is_none() {
            self.child.kill()?;
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
        let active = self.group.state().leader != 0;
        if active {
            self.kill().ok();
        }
    }
}

/// A [`Process`]'s group, to stop from another thread. It acts only while
/// the process runs: once it was reaped, its id may name another process.
#[derive(Clone, Debug)]
pub struct Group(Arc<Mutex<GroupState>>);

#[derive(Debug)]
struct GroupState {
    leader: u32,
    stopping: bool,
}

/// Observes an exit without releasing the leader's process id.
#[cfg(unix)]
fn exited_unreaped(leader: u32) -> io::Result<bool> {
    // SAFETY: waitid initializes the zeroed siginfo and WNOWAIT leaves
    // the child available for std::process::Child to reap.
    unsafe {
        let mut info: libc::siginfo_t = std::mem::zeroed();
        if libc::waitid(
            libc::P_PID,
            leader as libc::id_t,
            &mut info,
            libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
        ) == -1
        {
            return Err(io::Error::last_os_error());
        }
        Ok(info.si_pid() != 0)
    }
}

impl Group {
    fn state(&self) -> MutexGuard<'_, GroupState> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Kills the group (`SIGKILL`); elsewhere than unix, nothing.
    pub fn kill(&self) {
        self.signal(Signal::Kill);
    }

    /// Asks the group to stop (`SIGTERM`); elsewhere than unix, nothing.
    pub fn terminate(&self) {
        self.signal(Signal::Terminate);
    }

    fn signal(&self, signal: Signal) {
        let mut state = self.state();
        state.stopping = true;
        let leader = state.leader;
        #[cfg(unix)]
        if leader != 0 {
            signal_group(leader, signal);
        }
        #[cfg(not(unix))]
        let _ = (leader, signal);
    }
}

/// Stops a [`Process`]'s group when dropped: kept in a future that waits
/// for the process on another thread ([`blocking`](crate::blocking)), it
/// asks the group to stop (`SIGTERM`) the moment the future is dropped,
/// such as when a call is interrupted or the app quits, and kills it
/// [`STOP_GRACE`] later if it still runs. The thread waiting for the
/// process must go on reaping it meanwhile, as [`Process::run`] does.
pub struct KillOnDrop(pub Group);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        self.0.terminate();
        let group = self.0.clone();
        let later = thread::Builder::new()
            .name("rig-process-stop".to_owned())
            .spawn(move || {
                thread::sleep(STOP_GRACE);
                group.kill();
            });
        if later.is_err() {
            self.0.kill();
        }
    }
}

/// The signals a [`Group`] sends.
#[derive(Clone, Copy)]
enum Signal {
    Terminate,
    Kill,
}

/// Starts `command` in a new session, which also makes it the leader of a
/// new process group. With no controlling terminal, a program that opens
/// `/dev/tty` for a prompt (ssh, git, sudo) fails at once instead of
/// drawing over the view and stopping on a terminal read. Git is also told
/// not to prompt.
pub fn detach(command: &mut Command) {
    command.env("GIT_TERMINAL_PROMPT", "0");
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // SAFETY: the closure runs in the forked child before `exec` and
        // only calls `setsid`, which is async-signal-safe.
        unsafe {
            command.pre_exec(|| {
                if libc::setsid() == -1 {
                    Err(std::io::Error::last_os_error())
                } else {
                    Ok(())
                }
            });
        }
    }
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
