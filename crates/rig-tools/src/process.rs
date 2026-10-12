//! Child processes started in a session of their own, such as `shell`
//! commands, so they never read the terminal and are killed as a group, and
//! the [`Launcher`] that says how every tool starts them. A [`Process`] it
//! [spawns](Launcher::spawn) takes its input, hands its output over line
//! by line, and is stopped gently or killed with its whole group.

use std::ffi::{OsStr, OsString};
use std::io::{self, Read, Write as _};
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
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
        let mut command = self.command(program);
        command.args(args).stdin(match how.stdin {
            Some(_) => Stdio::piped(),
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
        let group = Group(Arc::new(AtomicU32::new(child.id())));
        // From here on, an error drops the process, which kills it.
        let mut process = Process {
            child,
            lines,
            group,
        };
        if let Some(stdout) = process.child.stdout.take() {
            pipes.push((Box::new(stdout), OutputLine::Out));
        }
        if let Some(stderr) = process.child.stderr.take() {
            pipes.push((Box::new(stderr), OutputLine::Err));
        }
        for (pipe, tag) in pipes {
            let sender = sender.clone();
            thread::Builder::new()
                .name("rig-process-output".to_owned())
                .spawn(move || forward(pipe, tag, &sender))?;
        }
        if let (Some(mut stdin), Some(bytes)) = (process.child.stdin.take(), how.stdin) {
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
}

impl Process {
    /// The process's id, which is also its group's.
    pub fn id(&self) -> u32 {
        self.child.id()
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
        }
        status
    }

    /// Waits at most `timeout` for it to end, and says how it ended if it
    /// did. Its output keeps arriving at [`lines`](Self::lines) meanwhile.
    /// A timeout too long for the clock, such as `Duration::MAX`, waits
    /// until it ends.
    pub fn wait_for(&mut self, timeout: Duration) -> io::Result<Option<ExitStatus>> {
        let deadline = Instant::now().checked_add(timeout);
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
            thread::sleep(left.min(POLL));
        }
    }

    /// Asks its group to stop (`SIGTERM`), waits at most `grace` for it to
    /// end, then [kills](Self::kill) the group. Elsewhere than unix it is
    /// killed at once.
    pub fn stop(&mut self, grace: Duration) -> io::Result<ExitStatus> {
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
        self.child.wait()
    }
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

/// Kills a [`Process`]'s group when dropped: kept in a future that waits
/// for the process on another thread ([`blocking`](crate::blocking)), it
/// kills the process the moment the future is dropped, such as when a call
/// is interrupted or the app quits.
pub struct KillOnDrop(pub Group);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        self.0.kill();
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
